#!/usr/bin/env bash
#
# verify_corex_branch_equiv.sh
# ---------------------------
# Inverse of verify_nvidia_branch_equiv.sh.
#
# For every switcher-overlay file gated on UIPC_COREX_CUDA10_COMPAT, extract the
# COREX branch (the #if-true side) and diff it against the pre-rebase snapshot
# captured under .rebase-snapshot/switcher-corex-branches/.
#
# Expected output: empty diff for every file. Any non-empty diff indicates that
# the corex branch has been altered by an upstream rebase, in violation of the
# "corex behavior is frozen" policy from docs/corex-compat-policy.md.
#
# Usage:
#     bash tools/audit/verify_corex_branch_equiv.sh
#
# Exit code 0 if every file is byte-equivalent to the snapshot, 1 otherwise.
#
# baseline: .rebase-snapshot/switcher-corex-branches/ captured at rebase 2026-04-22

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SNAP_DIR="${ROOT_DIR}/.rebase-snapshot/switcher-corex-branches"
EXTRACT="${ROOT_DIR}/.rebase-snapshot/extract_corex_branch.py"

if [ ! -d "${SNAP_DIR}" ]; then
    echo "FATAL: snapshot baseline ${SNAP_DIR} not found" >&2
    exit 2
fi

# Same 22 switcher files as verify_nvidia_branch_equiv.sh.
FILES=(
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
    src/core/core/i_engine.cpp
    src/core/core/internal/engine.cpp
    src/core/core/internal/world.cpp
    src/core/core/sanity_checker.cpp
    src/core/core/world.cpp
)

# Files with accepted cosmetic-only diffs (e.g. trailing-newline after EOF inside
# the #if branch, byte-different but semantically identical to C preprocessor).
COSMETIC_ALLOW=(
    src/backends/cuda/type_define.h
)

is_cosmetic() {
    local target="$1"
    for f in "${COSMETIC_ALLOW[@]}"; do
        [ "$f" = "$target" ] && return 0
    done
    return 1
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
        # Strip trailing newline from both sides; if then equal, treat as cosmetic
        a="$(mktemp)"; b="$(mktemp)"
        # Use Python to strip exactly one trailing newline if present
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:-1] if d.endswith(b'\\n') else d)" "${snap_path}" "$a"
        python3 -c "import sys; d=open(sys.argv[1],'rb').read(); open(sys.argv[2],'wb').write(d[:-1] if d.endswith(b'\\n') else d)" "${extracted}" "$b"
        if diff -q "$a" "$b" > /dev/null && is_cosmetic "$rel"; then
            echo "COSMETIC:    ${rel} (trailing-newline only, semantically identical)"
            cosmetic=$((cosmetic + 1))
        else
            echo "DIVERGE:     ${rel}"
            diff -u "${snap_path}" "${extracted}" | head -40
            fail=$((fail + 1))
        fi
        rm -f "$a" "$b"
    fi
    rm -f "${extracted}"
done

echo
echo "Summary: ok=${ok} cosmetic=${cosmetic} fail=${fail} missing=${missing} total=${#FILES[@]}"
[ "${fail}" -eq 0 ] && [ "${missing}" -eq 0 ]
