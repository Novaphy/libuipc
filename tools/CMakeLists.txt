# Developer utilities (CUDA API inventory is script-only; probe is optional build target).
option(UIPC_BUILD_CUDA_API_PROBE "Build corex_cuda_api_probe (small CUDA math-library smoke test)" ON)
option(UIPC_BUILD_COREX_API_TESTS "Build corex_api_tests (CSV-aligned multi-binary Corex API checks)" ON)

if(UIPC_WITH_CUDA_BACKEND AND UIPC_BUILD_CUDA_API_PROBE)
    add_subdirectory(corex_cuda_api_probe)
endif()

if(UIPC_WITH_CUDA_BACKEND AND UIPC_BUILD_COREX_API_TESTS)
    add_subdirectory(corex_api_tests)
endif()
