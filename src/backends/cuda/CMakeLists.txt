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
#
# NOTE on uipc_io:
# The sanity-check sources under src/backends/cuda/sanity_check/* (compiled
# only on the NVIDIA path; filtered out on Corex by the `_corex.{cu,h,...}`
# / `/backends/cuda/sanity_check/` rules below) call
# `uipc::geometry::SimplicialComplexIO::{ctor,write}` to dump debug meshes
# when a sanity check fails. Those symbols live in `uipc_io.dll` (see
# src/io/CMakeLists.txt). On Linux the resulting `.so` is allowed to leave
# undefined references to be resolved at load time, so the missing link
# edge silently worked. On Windows MSVC every `__imp_*` reference must be
# satisfied at link time by an explicit import library, so the missing
# `uipc_io` link edge surfaces as
#   LNK2019 / LNK1120 unresolved external symbol
#       __imp_??0SimplicialComplexIO@geometry@uipc@@QEAA@XZ
#       __imp_?write@SimplicialComplexIO@geometry@uipc@@QEAAX...
# in uipc_backend_cuda.dll. Add the dependency explicitly so both
# platforms link cleanly.
target_link_libraries(cuda PUBLIC
    muda::muda
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
    uipc_geometry
    uipc_io
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
    # Corex 4.4+ exposes cudaMallocAsync/cudaFreeAsync even though CUDART_VERSION
    # still reports 10020. When enabled, propagate UIPC_COREX_ASYNC_MEMORY=1 so
    # the muda headers and the cuda backend swap the sync compat path for the
    # native async APIs (see external/muda/src/muda/launch/details/memory.inl
    # and external/muda/src/muda/buffer/details/device_var.inl).
    if(UIPC_COREX_USE_ASYNC_MEMORY)
        target_compile_definitions(cuda PUBLIC UIPC_COREX_ASYNC_MEMORY=1)
    endif()
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

# ------------------------------------------------------------------------------
# Source filter for two-environment build
#   - Always: drop *_corex.{cu,h,cpp,hpp,inl} overlay files from the direct
#     compile list — those are include-only sidecars used by the switcher headers
#     /sources to pull in corex-specific implementations under
#     #if UIPC_COREX_CUDA10_COMPAT.
#   - NVIDIA path (UIPC_MUDA_USE_COREX=OFF):
#       drop corex-only sources (corex_matrix_converter_kernels, backends/cuda/details/)
#       so this build is byte-equivalent to building /root/src verbatim.
#   - Corex path (UIPC_MUDA_USE_COREX=ON):
#       drop sources that depend on NVCC-only features the Corex toolchain cannot compile
#       (FEM external force module + backends/cuda/sanity_check kernels).
# ------------------------------------------------------------------------------
list(FILTER SOURCES EXCLUDE REGEX "_corex\\.(cu|h|hpp|cpp|inl)$")

if(NOT UIPC_MUDA_USE_COREX)
    list(FILTER SOURCES EXCLUDE REGEX "/algorithm/corex_matrix_converter_kernels\\.(cu|h)$")
    list(FILTER SOURCES EXCLUDE REGEX "/backends/cuda/details/")
endif()

if(UIPC_MUDA_USE_COREX)
    list(FILTER SOURCES EXCLUDE REGEX "/backends/cuda/sanity_check/")
endif()

# Note: the previously-needed exclusion rules for
# finite_element/finite_element_external_force_{manager,reporter}.{cu,h},
# finite_element/finite_element_external_vertex_force.cu, and
# finite_element/constraints/finite_element_external_vertex_force_constraint.{cu,h}
# have been removed: those files were dead code (referenced nonexistent fields
# on FiniteElementMethod::Impl, used nowhere) and the source files themselves
# have been deleted from the tree.

target_sources(cuda PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(cuda)

# ------------------------------------------------------------------------------
# setup source group for the IDE
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_GROUP_FILES "*.h" "*.hpp" "*.cpp" "*.cu" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})