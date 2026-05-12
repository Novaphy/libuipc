#!/usr/bin/env python3
"""rewrap_all.py
Re-apply the #if UIPC_COREX_CUDA10_COMPAT wrap to all 22 switcher files so that:
  * NVIDIA-branch extract matches libuipc-origin byte-for-byte
  * COREX-branch extract matches the pre-rebase snapshot byte-for-byte

Run from libuipc-port/ root.
"""
import sys
from pathlib import Path

PORT = Path('.').resolve()
SNAP = PORT / '.rebase-snapshot' / 'switcher-corex-branches'

# (relative path, corex_block_text)
# For 20 simple switchers, corex_block is just an #include of the sidecar.
# Indent style ("#include" vs "#  include") is chosen to match pre-rebase snapshot.
SIMPLE = [
    ('src/backends/cuda/linear_system/linear_pcg.cu',                       '#  include "linear_pcg_corex.cu.inc"'),
    ('src/backends/cuda/linear_system/linear_pcg.h',                        '#include "linear_pcg_corex.h"'),
    ('src/backends/cuda/linear_system/linear_fused_pcg.cu',                 '#include "linear_fused_pcg_corex.cu"'),
    ('src/backends/cuda/linear_system/linear_fused_pcg.h',                  '#include "linear_fused_pcg_corex.h"'),
    ('src/backends/cuda/affine_body/abd_jacobi_matrix.cu',                  '#include "abd_jacobi_matrix_corex.cu"'),
    ('src/backends/cuda/affine_body/abd_jacobi_matrix.h',                   '#include "abd_jacobi_matrix_corex.h"'),
    ('src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu', '#include "affine_body_revolute_joint_corex.cu"'),
    ('src/backends/cuda/collision_detection/info_stackless_bvh.h',          '#include "info_stackless_bvh_corex.h"'),
    ('src/backends/cuda/collision_detection/info_stackless_bvh_v0.h',       '#include "info_stackless_bvh_v0_corex.h"'),
    ('src/backends/cuda/collision_detection/linear_bvh.h',                  '#include "linear_bvh_corex.h"'),
    ('src/backends/cuda/collision_detection/stackless_bvh.h',               '#include "stackless_bvh_corex.h"'),
    ('src/backends/cuda/finite_element/fem_utils.h',                        '#include "fem_utils_corex.h"'),
    ('src/backends/cuda/finite_element/matrix_utils.h',                     '#include "matrix_utils_corex.h"'),
    ('src/backends/cuda/finite_element/mas_preconditioner_engine.cu',       '#include "mas_preconditioner_engine_corex.cu"'),
    ('src/backends/cuda/finite_element/mas_preconditioner_engine.h',        '#include "mas_preconditioner_engine_corex.h"'),
    ('src/core/core/i_engine.cpp',                                          '#  include "i_engine_corex.cpp.inc"'),
    ('src/core/core/internal/engine.cpp',                                   '#  include "engine_corex.cpp.inc"'),
    ('src/core/core/internal/world.cpp',                                    '#  include "world_corex.cpp.inc"'),
    ('src/core/core/sanity_checker.cpp',                                    '#  include "sanity_checker_corex.cpp.inc"'),
    ('src/core/core/world.cpp',                                             '#  include "world_corex.cpp.inc"'),
]

# These two have inline corex content in the snapshot, not just an #include.
INLINE_FROM_SNAPSHOT = [
    'src/backends/cuda/type_define.h',
    'src/backends/cuda/contact_system/contact_coeff.h',
]


def wrap_simple(rel: str, corex_block: str) -> None:
    fp = PORT / rel
    raw = fp.read_bytes()
    text = raw.decode('utf-8')
    has_trailing_nl = raw.endswith(b'\n')
    body = text[:-1] if has_trailing_nl else text
    out = (
        f'#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT\n'
        f'{corex_block}\n'
        f'#else\n'
        f'{body}\n'
        f'#endif'
    )
    if has_trailing_nl:
        out += '\n'
    fp.write_text(out, encoding='utf-8')


def wrap_inline(rel: str) -> None:
    """Wrap entire file: snapshot content in #if branch, origin content in #else branch.
    Both branches have their own #pragma once on the first line of the body, which is fine."""
    fp = PORT / rel
    snap_fp = SNAP / rel
    if not snap_fp.exists():
        sys.exit(f'snapshot missing: {snap_fp}')

    origin_raw = fp.read_bytes()
    origin_text = origin_raw.decode('utf-8')
    origin_has_nl = origin_raw.endswith(b'\n')
    origin_body = origin_text[:-1] if origin_has_nl else origin_text

    snap_raw = snap_fp.read_bytes()
    snap_text = snap_raw.decode('utf-8')
    snap_has_nl = snap_raw.endswith(b'\n')
    snap_body = snap_text[:-1] if snap_has_nl else snap_text

    out = (
        f'#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT\n'
        f'{snap_body}\n'
        f'#else\n'
        f'{origin_body}\n'
        f'#endif'
    )
    if origin_has_nl:
        out += '\n'
    fp.write_text(out, encoding='utf-8')


def main():
    for rel, blk in SIMPLE:
        wrap_simple(rel, blk)
        print(f'wrapped simple: {rel}')
    for rel in INLINE_FROM_SNAPSHOT:
        wrap_inline(rel)
        print(f'wrapped inline: {rel}')
    print(f'\nDone. {len(SIMPLE)} simple + {len(INLINE_FROM_SNAPSHOT)} inline = {len(SIMPLE)+len(INLINE_FROM_SNAPSHOT)} files.')


if __name__ == '__main__':
    main()
