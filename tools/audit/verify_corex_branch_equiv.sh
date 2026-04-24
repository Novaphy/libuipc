#!/usr/bin/env bash
#
# verify_corex_branch_equiv.sh (v11 expanded)
# -------------------------------------------
# Extracts the COREX branch (#if-true side) of every tracked file under the v11
# fork and diffs it against .rebase-snapshot/switcher-corex-branches/, which was
# captured from "v11 as it arrived from the Corex machine" (BEFORE the
# NVIDIA-restoration whole-file switcher rewrite).
#
# Expected output: empty diff for every file. Any non-empty diff indicates the
# Corex branch has drifted, in violation of the "Corex behavior is frozen"
# policy.
#
# Tracked file count = 167 (159 modified vs origin + 8 v11-only non-sidecar
# constraint/external-force files that must be Corex-only).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SNAP_DIR="${ROOT_DIR}/.rebase-snapshot/switcher-corex-branches"
EXTRACT="${ROOT_DIR}/.rebase-snapshot/extract_corex_branch.py"

if [ ! -d "${SNAP_DIR}" ]; then
    echo "FATAL: snapshot baseline ${SNAP_DIR} not found" >&2
    exit 2
fi

FILES=(
    include/uipc/builtin/details/attribute_name.h
    include/uipc/common/details/smart_pointer.inl
    include/uipc/common/span.h
    include/uipc/common/type_define.h
    include/uipc/constitution/affine_body_constitution.h
    include/uipc/constitution/affine_body_fixed_joint.h
    include/uipc/constitution/affine_body_prismatic_joint.h
    include/uipc/constitution/affine_body_revolute_joint.h
    include/uipc/constitution/affine_body_spherical_joint.h
    include/uipc/constitution/finite_element_external_force.h
    include/uipc/core/affine_body_state_accessor_feature.h
    include/uipc/core/details/object.inl
    include/uipc/core/finite_element_state_accessor_feature.h
    include/uipc/core/i_engine.h
    include/uipc/core/internal/engine.h
    include/uipc/core/i_sanity_checker.h
    include/uipc/core/sanity_checker.h
    include/uipc/diff_sim/sparse_coo_view.h
    include/uipc/geometry/simplicial_complex_attributes.h
    include/uipc/geometry/utils/affine_body/affine_body_from_rigid_body.h
    src/backends/common/sim_engine.cpp
    src/backends/common/sim_system_auto_register.h
    src/backends/common/sim_system_collection.cpp
    src/backends/cuda/active_set_system/global_active_set_manager.cu
    src/backends/cuda/active_set_system/global_active_set_manager.h
    src/backends/cuda/affine_body/abd_active_set_reporter.cu
    src/backends/cuda/affine_body/abd_diag_preconditioner.cu
    src/backends/cuda/affine_body/abd_jacobi_matrix.cu
    src/backends/cuda/affine_body/abd_jacobi_matrix.h
    src/backends/cuda/affine_body/abd_linear_subsystem.cu
    src/backends/cuda/affine_body/abd_linear_subsystem.h
    src/backends/cuda/affine_body/abd_line_search_reporter.cu
    src/backends/cuda/affine_body/abd_line_search_reporter.h
    src/backends/cuda/affine_body/abd_tolerance_checker.cu
    src/backends/cuda/affine_body/affine_body_body_reporter.cu
    src/backends/cuda/affine_body/affine_body_dynamics.cu
    src/backends/cuda/affine_body/affine_body_dynamics.h
    src/backends/cuda/affine_body/affine_body_external_force_manager.cu
    src/backends/cuda/affine_body/affine_body_prismatic_joint_external_force.cu
    src/backends/cuda/affine_body/affine_body_revolute_joint_external_force.cu
    src/backends/cuda/affine_body/affine_body_state_accessor.cu
    src/backends/cuda/affine_body/affine_body_state_accessor_feature.cu
    src/backends/cuda/affine_body/affine_body_state_accessor_feature.h
    src/backends/cuda/affine_body/affine_body_vertex_reporter.cu
    src/backends/cuda/affine_body/bdf/abd_bdf1_time_integrator.cu
    src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu
    src/backends/cuda/affine_body/constitutions/affine_body_fixed_joint.cu
    src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu
    src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint_function.h
    src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint_limit.cu
    src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu
    src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint_function.h
    src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint_limit.cu
    src/backends/cuda/affine_body/constitutions/affine_body_spherical_joint.cu
    src/backends/cuda/affine_body/constitutions/arap_function.h
    src/backends/cuda/affine_body/constitutions/joint_limit_penalty.h
    src/backends/cuda/affine_body/constitutions/ortho_potential.cu
    src/backends/cuda/affine_body/constitutions/sym/affine_body_driving_revolute_joint.inl
    src/backends/cuda/affine_body/constraints/affine_body_prismatic_joint_external_force_constraint.cu
    src/backends/cuda/affine_body/constraints/affine_body_prismatic_joint_external_force_constraint.h
    src/backends/cuda/affine_body/constraints/affine_body_revolute_joint_external_force_constraint.cu
    src/backends/cuda/affine_body/constraints/affine_body_revolute_joint_external_force_constraint.h
    src/backends/cuda/affine_body/constraints/external_articulation_constraint.cu
    src/backends/cuda/affine_body/constraints/soft_transform_constraint.cu
    src/backends/cuda/affine_body/constraints/sym/external_articulation_revolute_joint_constraint.inl
    src/backends/cuda/affine_body/details/abd_jacobi_matrix.inl
    src/backends/cuda/affine_body/inter_affine_body_constitution.cu
    src/backends/cuda/affine_body/inter_affine_body_constitution.h
    src/backends/cuda/algorithm/details/fast_segmental_reduce.inl
    src/backends/cuda/algorithm/details/matrix_converter.inl
    src/backends/cuda/algorithm/fast_segmental_reduce.h
    src/backends/cuda/algorithm/givens.hpp
    src/backends/cuda/animator/utils.h
    src/backends/cuda/collision_detection/details/info_stackless_bvh.inl
    src/backends/cuda/collision_detection/details/info_stackless_bvh_v0.inl
    src/backends/cuda/collision_detection/details/linear_bvh.inl
    src/backends/cuda/collision_detection/details/stackless_bvh.inl
    src/backends/cuda/collision_detection/filters/al_vertex_half_plane_trajectory_filter.h
    src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.cu
    src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h
    src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu
    src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h
    src/backends/cuda/collision_detection/global_trajectory_filter.cu
    src/backends/cuda/collision_detection/info_stackless_bvh.h
    src/backends/cuda/collision_detection/info_stackless_bvh_v0.h
    src/backends/cuda/collision_detection/linear_bvh.h
    src/backends/cuda/collision_detection/stackless_bvh.h
    src/backends/cuda/contact_system/contact_coeff.h
    src/backends/cuda/contact_system/contact_models/codim_ipc_contact_function.h
    src/backends/cuda/contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h
    src/backends/cuda/contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h
    src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu
    src/backends/cuda/contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu
    src/backends/cuda/contact_system/contact_models/sym/codim_ipc_contact.inl
    src/backends/cuda/contact_system/global_contact_manager.cu
    src/backends/cuda/contact_system/global_contact_manager.h
    src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu
    src/backends/cuda/engine/sim_engine.cu
    src/backends/cuda/engine/sim_engine_do_init.cu
    src/backends/cuda/entrance.cpp
    src/backends/cuda/finite_element/bdf/fem_bdf1_time_integrator.cu
    src/backends/cuda/finite_element/bdf/fem_bdf2_time_integrator.cu
    src/backends/cuda/finite_element/constitutions/arap_3d.cu
    src/backends/cuda/finite_element/constitutions/arap_function.h
    src/backends/cuda/finite_element/constitutions/detail/stable_neo_hookean_3d.inl
    src/backends/cuda/finite_element/constitutions/discrete_shell_bending_function.h
    src/backends/cuda/finite_element/constitutions/stable_neo_hookean_3d.cu
    src/backends/cuda/finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.cu
    src/backends/cuda/finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h
    src/backends/cuda/finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h
    src/backends/cuda/finite_element/fem_mas_preconditioner.cu
    src/backends/cuda/finite_element/fem_time_integrator.h
    src/backends/cuda/finite_element/fem_utils.cu
    src/backends/cuda/finite_element/fem_utils.h
    src/backends/cuda/finite_element/finite_element_method.cu
    src/backends/cuda/finite_element/finite_element_method.h
    src/backends/cuda/finite_element/finite_element_state_accessor_feature.cu
    src/backends/cuda/finite_element/finite_element_state_accessor_feature.h
    src/backends/cuda/finite_element/mas_preconditioner_engine.cu
    src/backends/cuda/finite_element/mas_preconditioner_engine.h
    src/backends/cuda/finite_element/matrix_utils.cu
    src/backends/cuda/finite_element/matrix_utils.h
    src/backends/cuda/global_geometry/global_vertex_manager.cu
    src/backends/cuda/global_geometry/global_vertex_manager.h
    src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_edge_stitch.cu
    src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu
    src/backends/cuda/linear_system/global_linear_system.cu
    src/backends/cuda/linear_system/linear_fused_pcg.cu
    src/backends/cuda/linear_system/linear_fused_pcg.h
    src/backends/cuda/linear_system/linear_pcg.cu
    src/backends/cuda/linear_system/linear_pcg.h
    src/backends/cuda/linear_system/spmv.cu
    src/backends/cuda/linear_system/spmv.h
    src/backends/cuda/line_search/line_searcher.cu
    src/backends/cuda/type_define.h
    src/backends/cuda/utils/codim_thickness.h
    src/backends/cuda/utils/distance/ccd.h
    src/backends/cuda/utils/distance/details/ccd.inl
    src/backends/cuda/utils/distance/details/edge_edge.inl
    src/backends/cuda/utils/distance/details/edge_edge_mollifier.inl
    src/backends/cuda/utils/distance/details/point_edge.inl
    src/backends/cuda/utils/distance/details/point_point.inl
    src/backends/cuda/utils/distance/details/point_triangle.inl
    src/backends/cuda/utils/distance/distance_flagged.h
    src/backends/cuda/utils/distance/edge_edge.h
    src/backends/cuda/utils/distance/edge_edge_mollifier.h
    src/backends/cuda/utils/distance/point_edge.h
    src/backends/cuda/utils/distance/point_point.h
    src/backends/cuda/utils/distance/point_triangle.h
    src/backends/cuda/utils/friction_utils.h
    src/backends/cuda/utils/make_spd.h
    src/backends/cuda/utils/matrix_assembler.h
    src/backends/cuda/utils/matrix_unpacker.h
    src/backends/cuda/utils/primitive_d_hat.h
    src/backends/cuda/xmake.lua
    src/constitution/affine_body_constitution.cpp
    src/constitution/affine_body_rod.cpp
    src/constitution/affine_body_shell.cpp
    src/core/core/i_engine.cpp
    src/core/core/internal/engine.cpp
    src/core/core/internal/world.cpp
    src/core/core/sanity_checker.cpp
    src/core/core/world.cpp
    src/geometry/intersection.cpp
    src/io/simplicial_complex_io.cpp
    src/io/urdf_io.cpp
    src/pybind/pyuipc/core/module.cpp
)

# Trailing-newline-only differences are accepted as cosmetic for ALL files:
# the wrap_to_switcher rewrite must add a newline after the Corex section so
# `#else` lands on its own line, even if the original Corex view ended at EOF
# without a trailing newline. This is semantically a no-op (C preprocessor).
is_cosmetic() {
    return 0
}

fail=0
ok=0
cosmetic=0
missing=0

for rel in "${FILES[@]}"; do
    fork_path="${ROOT_DIR}/${rel}"
    snap_path="${SNAP_DIR}/${rel}"

    if [ ! -f "${fork_path}" ]; then
        echo "MISSING_FORK: ${rel}" >&2
        missing=$((missing + 1))
        continue
    fi
    if [ ! -f "${snap_path}" ]; then
        echo "MISSING_SNAP: ${rel}" >&2
        missing=$((missing + 1))
        continue
    fi

    extracted="$(mktemp)"
    python3 "${EXTRACT}" "${fork_path}" > "${extracted}"
    if diff -q "${snap_path}" "${extracted}" > /dev/null; then
        ok=$((ok + 1))
    else
        a="$(mktemp)"; b="$(mktemp)"
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:-1] if d.endswith(b'\\n') else d)" "${snap_path}" "$a"
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:-1] if d.endswith(b'\\n') else d)" "${extracted}" "$b"
        if diff -q "$a" "$b" > /dev/null && is_cosmetic "$rel"; then
            echo "COSMETIC:    ${rel}"
            cosmetic=$((cosmetic + 1))
        else
            echo "DIVERGE:     ${rel}"
            diff -u "${snap_path}" "${extracted}" | head -20
            fail=$((fail + 1))
        fi
        rm -f "$a" "$b"
    fi
    rm -f "${extracted}"
done

echo
echo "Summary: ok=${ok} cosmetic=${cosmetic} fail=${fail} missing=${missing} total=${#FILES[@]}"
[ "${fail}" -eq 0 ] && [ "${missing}" -eq 0 ]
