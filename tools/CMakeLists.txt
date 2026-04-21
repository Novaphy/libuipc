# Developer utilities (CUDA API inventory is script-only; probe is optional build target).
#
# corex_cuda_api_probe and corex_api_tests deliberately exercise the
# Iluvatar Corex CUDA-math libraries via *legacy* cuSPARSE/cuBLAS entry
# points (e.g. cusparseDcsrmv) that NVIDIA's CUDA 12.x has removed.
# They are diagnostic tools intended to be run on a Corex toolchain
# only, so their build is gated on UIPC_MUDA_USE_COREX. Defaults are
# OFF so an NVIDIA-only consumer of this repo never has to think about
# them; flip them on explicitly when working on the Corex port.
option(UIPC_BUILD_CUDA_API_PROBE  "Build corex_cuda_api_probe (Corex-only CUDA math-library smoke test)" OFF)
option(UIPC_BUILD_COREX_API_TESTS "Build corex_api_tests (Corex-only CSV-aligned multi-binary API checks)" OFF)

if(UIPC_WITH_CUDA_BACKEND AND UIPC_MUDA_USE_COREX AND UIPC_BUILD_CUDA_API_PROBE)
    add_subdirectory(corex_cuda_api_probe)
endif()

if(UIPC_WITH_CUDA_BACKEND AND UIPC_MUDA_USE_COREX AND UIPC_BUILD_COREX_API_TESTS)
    add_subdirectory(corex_api_tests)
endif()
