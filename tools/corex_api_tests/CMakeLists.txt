# Standalone:
#   cmake -S tools/corex_api_tests -B build_corex_tests -DCMAKE_CUDA_ARCHITECTURES=ivcore11
#   cmake --build build_corex_tests -j

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(corex_api_tests LANGUAGES CXX CUDA)
endif()

find_package(CUDAToolkit REQUIRED)

set(COMMON_CUDA_PROPS
    CUDA_SEPARABLE_COMPILATION OFF
)

function(corex_add_test name)
    add_executable(${name} ${ARGN})
    set_target_properties(${name} PROPERTIES ${COMMON_CUDA_PROPS})
    target_compile_features(${name} PRIVATE cxx_std_17)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
endfunction()

# C — cuBLAS (CSV cublas*)
corex_add_test(corex_cublas_inventory corex_cublas_inventory.cu)
target_link_libraries(corex_cublas_inventory PRIVATE CUDA::cudart CUDA::cublas)

# D — cuSPARSE (CSV cusparse*)
corex_add_test(corex_cusparse_inventory corex_cusparse_inventory.cu)
target_link_libraries(corex_cusparse_inventory PRIVATE CUDA::cudart CUDA::cusparse)

# E — cuSOLVER (CSV cusolver*)
corex_add_test(corex_cusolver_inventory corex_cusolver_inventory.cu)
target_link_libraries(corex_cusolver_inventory PRIVATE CUDA::cudart CUDA::cusolver CUDA::cublas)

# A — Runtime extended (CSV cuda_runtime minus graph)
corex_add_test(corex_runtime_extended corex_runtime_extended.cu)
target_link_libraries(corex_runtime_extended PRIVATE CUDA::cudart)

# B — CUDA Graph
corex_add_test(corex_cuda_graph corex_cuda_graph.cu)
target_link_libraries(corex_cuda_graph PRIVATE CUDA::cudart)

# F — Cross-buffer double subtraction bug diagnostic
corex_add_test(corex_cross_buffer_subtract corex_cross_buffer_subtract.cu)
target_link_libraries(corex_cross_buffer_subtract PRIVATE CUDA::cudart)
