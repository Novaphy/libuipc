file(GLOB SOURCE "*.cpp" "*.h")
if(UIPC_MUDA_USE_COREX)
    # optimal_transform.cpp uses Eigen mixed double/float ops that hit
    # YOU_MIXED_DIFFERENT_NUMERIC_TYPES__... static_assert under Float=float.
    list(FILTER SOURCE EXCLUDE REGEX "/optimal_transform\\.cpp$")
endif()
uipc_add_test(geometry ${SOURCE})
if(UIPC_WITH_VDB_SUPPORT)
    target_compile_definitions(geometry PRIVATE UIPC_TEST_WITH_VDB_SUPPORT=1)
endif()