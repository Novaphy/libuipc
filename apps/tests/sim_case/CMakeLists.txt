file(GLOB SOURCE "*.cpp" "*.h")
if(UIPC_MUDA_USE_COREX)
    # Two upstream sim_case files do not match the Corex (Float=float) ABI:
    #   79_abd_fixed_joint.cpp     — Eigen mixed double/float static_assert
    #   80_abd_spherical_joint.cpp — calls apply_to(... span<Vec3>, int) overload
    #                                that is not declared in the corex header
    #                                (only 4-arg and 6-arg variants exist).
    # Skip them so the rest of the 85 sim_case tests still build under Corex.
    list(FILTER SOURCE EXCLUDE REGEX "/(79_abd_fixed_joint|80_abd_spherical_joint)\\.cpp$")
endif()
uipc_add_test(sim_case ${SOURCE})
target_link_libraries(sim_case PRIVATE uipc::backends)

# Serialize PDB writes on MSVC to avoid C1041 when many .cpp files compile in parallel
if(MSVC)
    target_compile_options(sim_case PRIVATE /FS)
endif()