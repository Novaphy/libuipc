file(GLOB SOURCES "*.cpp" "*.h" "details/*.inl")
if(UIPC_MUDA_USE_COREX)
    # NVIDIA-only: distance_diagnoser_feature is registered only on the NVIDIA path.
    list(FILTER SOURCES EXCLUDE REGEX "/distance_diagnoser_feature\\.(cpp|h)$")
endif()
target_sources(pyuipc PRIVATE ${SOURCES})
