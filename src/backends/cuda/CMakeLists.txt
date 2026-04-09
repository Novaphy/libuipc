set(_uipc_cuda_architectures "${UIPC_CUDA_ARCHITECTURES}")
set(_uipc_cuda_standard "20")

if(UIPC_MUDA_USE_COREX)
    if(UIPC_COREX_CUDA_ARCHITECTURES)
        set(_uipc_cuda_architectures "${UIPC_COREX_CUDA_ARCHITECTURES}")
    endif()

    set(_uipc_cuda_standard "${UIPC_COREX_CUDA_STANDARD}")

    if(UIPC_COREX_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${UIPC_COREX_CUDA_COMPILER}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES ${_uipc_cuda_architectures})
uipc_info("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}.")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# CUDA Backend
# ------------------------------------------------------------------------------
uipc_add_backend(cuda)

# basic setup
target_link_libraries(cuda PUBLIC
    muda::muda
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
    uipc_geometry
)

# add CCCL include directory for CUDA 13.0 compatibility
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_compile_features(cuda PUBLIC cxx_std_${_uipc_cuda_standard})
target_include_directories(cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(cuda PROPERTIES
    # Default to non-RDC. Corex/clang-CUDA does not accept nvcc's -rdc=true,
    # and most of our kernels do not require separable compilation.
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_ARCHITECTURES ${_uipc_cuda_architectures}
    CUDA_STANDARD_REQUIRED ON
    CUDA_STANDARD ${_uipc_cuda_standard}
)

if(UIPC_MUDA_USE_COREX)
    target_link_libraries(cuda PUBLIC dl)
    target_compile_definitions(cuda PUBLIC UIPC_COREX_CUDA10_COMPAT=1)
    # Clang/Corex may not define __CUDA_ARCH__ during device compilation; force
    # MUDA_GENERIC to __host__ __device__ so device kernels can call shared helpers.
    target_compile_definitions(cuda PRIVATE MUDA_FORCE_HD_GENERIC=1)
    # Corex toolchain path uses clang CUDA frontend; disable RDC for compatibility.
    set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
endif()

# Non-Corex toolchains (NVCC) may still benefit from RDC.
if(NOT UIPC_MUDA_USE_COREX AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" OR CMAKE_CUDA_COMPILER_ID STREQUAL "ILUVATAR")
    # Corex clang CUDA frontend does not accept nvcc-style '-rdc=true' / '--diag-suppress'.
    set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    target_compile_options(cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-x ivcore>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++${_uipc_cuda_standard}>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=20012,1388,27,174,1394,997,1866,69,177,554,20014,2361,20011,940,55,221,1028>
    )
else()
    target_compile_options(cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++${_uipc_cuda_standard}>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,1388,27,174,1394,997,1866,69,177,554,20014,2361,20011,940,55,221,1028>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# all files in this directory
file(GLOB_RECURSE SOURCES "*.cu" "*.h" "*.hpp" "*.cpp" "*.inl")

target_sources(cuda PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(cuda)

# ------------------------------------------------------------------------------
# setup source group for the IDE
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_GROUP_FILES "*.h" "*.hpp" "*.cpp" "*.cu" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})