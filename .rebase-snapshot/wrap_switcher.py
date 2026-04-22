#!/usr/bin/env python3
"""wrap_switcher.py
Wrap an origin file body with `#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
<sidecar_include>
#else
<origin body byte-for-byte>
#endif`

Used by the rebase tooling.

Modes:
  --include <header>   Insert `#  include "<header>"` in the corex branch.
  --inline-corex <file> Read corex body verbatim from <file>.

Place a `#pragma once` outside the wrap if the origin file starts with one
(common for headers): we keep the leading `#pragma once` line outside, then
wrap everything below.
"""
import argparse
import sys
from pathlib import Path


def wrap(origin_path: Path, corex_block: str, output_path: Path) -> None:
    raw = origin_path.read_bytes()
    text = raw.decode('utf-8')
    has_trailing_nl = raw.endswith(b'\n')

    lines = text.split('\n')
    if has_trailing_nl and lines and lines[-1] == '':
        lines.pop()

    # NOTE on structure: we keep the OLD port layout
    #   #if ... corex
    #     <corex_block>
    #   #else
    #     <full origin body, including any leading #pragma once>
    #   #endif
    # so that:
    #   * extract_nvidia_branch.py (audit) sees the full origin body byte-for-byte
    #   * extract_corex_branch.py sees just the corex_block, matching pre-rebase snapshot
    body = '\n'.join(lines)

    out = []
    out.append('#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT')
    out.append(corex_block.rstrip('\n'))
    out.append('#else')
    out.append(body)
    out.append('#endif')

    result = '\n'.join(out)
    if has_trailing_nl:
        result += '\n'
    output_path.write_text(result, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', required=True, help='path to origin file (will be wrapped in place)')
    ap.add_argument('--include', help='sidecar header to include in corex branch')
    ap.add_argument('--inline-corex-file', help='file containing the corex body verbatim')
    args = ap.parse_args()

    origin = Path(args.origin)
    if args.include:
        # No indent — matches pre-rebase port style ('#include "..."')
        corex_block = f'#include "{args.include}"'
    elif args.inline_corex_file:
        corex_block = Path(args.inline_corex_file).read_text(encoding='utf-8')
    else:
        sys.exit('must pass --include or --inline-corex-file')

    wrap(origin, corex_block, origin)


if __name__ == '__main__':
    main()
