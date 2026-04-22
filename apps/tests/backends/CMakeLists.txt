if(UIPC_WITH_CUDA_BACKEND)
    # The apps/tests/backends/cuda target inherits CMake's default CUDA flags
    # (assumes nvcc + RDC + native arch). On the Corex path the main `cuda`
    # backend target injects clang-cuda specific flags (`-x ivcore`,
    # `-std=c++${UIPC_COREX_CUDA_STANDARD}`, RDC OFF, UIPC_COREX_CUDA10_COMPAT,
    # etc.) but those do not propagate to this independent test target, so
    # building it under Corex fails on host C++17/20 headers and missing
    # compat macros. Equivalent device-API coverage on Corex is provided by
    # tools/corex_api_tests/, so skip this NVIDIA-specific subdir under Corex.
    if(UIPC_MUDA_USE_COREX)
        message(STATUS "[uipc] skipping apps/tests/backends/cuda under Corex; "
                       "use tools/corex_api_tests/ instead")
    else()
        add_subdirectory(cuda)
    endif()
endif()