# Can be configured standalone:
#   cmake -S tools/corex_cuda_api_probe -B build_probe -DCMAKE_CUDA_ARCHITECTURES=native
#   cmake --build build_probe
# Or as part of the full libuipc tree via add_subdirectory(tools).

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(corex_cuda_api_probe LANGUAGES CXX CUDA)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(corex_cuda_api_probe main.cu)

set_target_properties(corex_cuda_api_probe PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_features(corex_cuda_api_probe PRIVATE cxx_std_17)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
endif()

target_link_libraries(corex_cuda_api_probe PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
)

target_include_directories(corex_cuda_api_probe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
