#!/usr/bin/env bash
#
# verify_nvidia_branch_equiv.sh
# ----------------------------
# For every modified .cu / .h / .hpp / .cpp / .inl file under src/ in the
# libuipc fork that we have intentionally branch-gated for Corex, extract the
# "NVIDIA branch" view (UIPC_COREX_CUDA10_COMPAT undefined) and diff it against
# the corresponding file in /root/src.
#
# Expected output: empty diff for every file. Any non-empty diff indicates a
# Corex-specific change that has not yet been wrapped behind
# UIPC_COREX_CUDA10_COMPAT, in violation of the dual-branch policy documented
# in docs/corex-compat-policy.md.
#
# Usage:
#     bash tools/audit/verify_nvidia_branch_equiv.sh
#
# Exit code 0 if every file is byte-equivalent to /root/src, 1 otherwise.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
REF_DIR="/root/src"
EXTRACT="${ROOT_DIR}/tools/audit/extract_nvidia_branch.py"

if [ ! -d "${REF_DIR}" ]; then
    echo "FATAL: reference baseline ${REF_DIR} not found" >&2
    exit 2
fi

# Files that are intentionally dual-branched for Corex (NVIDIA branch must
# match /root/src byte-for-byte). Keep this list in sync with
# docs/corex-compat-audit-2026-04-17.md §Phase 2.
FILES=(
    # cuda backend switcher-overlay primaries
    src/backends/cuda/linear_system/linear_pcg.cu
    src/backends/cuda/linear_system/linear_pcg.h
    src/backends/cuda/linear_system/linear_fused_pcg.cu
    src/backends/cuda/linear_system/linear_fused_pcg.h
    src/backends/cuda/type_define.h
    src/backends/cuda/contact_system/contact_coeff.h
    src/backends/cuda/affine_body/abd_jacobi_matrix.cu
    src/backends/cuda/affine_body/abd_jacobi_matrix.h
    src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu
    src/backends/cuda/collision_detection/info_stackless_bvh.h
    src/backends/cuda/collision_detection/info_stackless_bvh_v0.h
    src/backends/cuda/collision_detection/linear_bvh.h
    src/backends/cuda/collision_detection/stackless_bvh.h
    src/backends/cuda/finite_element/fem_utils.h
    src/backends/cuda/finite_element/matrix_utils.h
    src/backends/cuda/finite_element/mas_preconditioner_engine.cu
    src/backends/cuda/finite_element/mas_preconditioner_engine.h

    # uipc_core switcher-overlay primaries (dylib v2/v3 split)
    src/core/core/i_engine.cpp
    src/core/core/internal/engine.cpp
    src/core/core/internal/world.cpp
    src/core/core/sanity_checker.cpp
    src/core/core/world.cpp
)

fail=0
ok=0
missing=0

for rel in "${FILES[@]}"; do
    fork_path="${ROOT_DIR}/${rel}"
    ref_path="${REF_DIR}/${rel#src/}"

    if [ ! -f "${fork_path}" ]; then
        echo "MISSING_FORK: ${rel}" >&2
        missing=$((missing + 1))
        continue
    fi
    if [ ! -f "${ref_path}" ]; then
        echo "MISSING_REF:  ${rel}" >&2
        missing=$((missing + 1))
        continue
    fi

    extracted="$(mktemp)"
    python3 "${EXTRACT}" "${fork_path}" > "${extracted}"
    if diff -q "${ref_path}" "${extracted}" > /dev/null; then
        ok=$((ok + 1))
    else
        echo "DIVERGE:     ${rel}"
        diff -u "${ref_path}" "${extracted}" | head -40
        fail=$((fail + 1))
    fi
    rm -f "${extracted}"
done

echo
echo "Summary: ok=${ok} fail=${fail} missing=${missing} total=${#FILES[@]}"
[ "${fail}" -eq 0 ] && [ "${missing}" -eq 0 ]
