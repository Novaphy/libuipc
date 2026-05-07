#!/usr/bin/env python3
"""wrap_to_switcher.py

Universal whole-file switcher rewrite for the v11 NVIDIA-restoration project.

Given a path inside the v11 fork (e.g. src/backends/cuda/foo.cu), rewrite that
file in-place as:

    #if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
    <Corex view of current v11 content (via extract_corex_branch.py)>
    #else
    <byte-equivalent libuipc-origin file content>
    #endif

This single mechanical transform handles all three structural buckets
(W = already a whole-file switcher, F = flat / no COREX guards, I = inline
guard mix) - in every case extract_corex on the rewritten file reproduces the
original Corex view (modulo trailing-newline cosmetics) and extract_nvidia
reproduces the origin file bytes.

Trailing-newline policy
-----------------------
The C preprocessor needs `#if`, `#else`, `#endif` to live on their own lines,
so this script always inserts a `\n` between each section's body and the next
directive. The OVERALL file's trailing-newline state is set to match the
ORIGIN file's trailing-newline state, because that drives extract_nvidia's
output trailing-nl (which has to byte-match origin).

Some Corex baseline snapshots may have been captured without a trailing
newline (e.g. F-bucket files whose original raw content ended at EOF without
a final \\n). After the rewrite the Corex view will gain a final \\n where
needed for directive separation; the audit script accepts that as cosmetic.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / '.rebase-snapshot'))

import extract_corex_branch  # noqa: E402

IF_LINE = "#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT\n"
ELSE_LINE = "#else\n"
ENDIF_LINE = "#endif"  # newline appended later iff origin had one


def wrap(rel_path: str, root: Path, ref_root: Path) -> bytes:
    fork_path = root / rel_path
    if not fork_path.is_file():
        raise FileNotFoundError(fork_path)

    corex_view = extract_corex_branch.extract(fork_path)

    ref_path = ref_root / rel_path
    if ref_path.is_file():
        nvidia_bytes = ref_path.read_bytes()
        nv_has_trailing_nl = nvidia_bytes.endswith(b'\n')
        nvidia_view = nvidia_bytes.decode('utf-8', errors='replace')
    else:
        nvidia_view = ''
        nv_has_trailing_nl = True

    # Each section needs to end with a newline so the next #directive starts
    # on its own line.
    if corex_view and not corex_view.endswith('\n'):
        corex_view += '\n'
    if nvidia_view and not nvidia_view.endswith('\n'):
        nvidia_view += '\n'

    out = IF_LINE + corex_view + ELSE_LINE + nvidia_view + ENDIF_LINE
    if nv_has_trailing_nl:
        out += '\n'
    return out.encode('utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rel_path')
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--ref-root', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    new_bytes = wrap(args.rel_path, Path(args.root), Path(args.ref_root))
    if args.dry_run:
        sys.stdout.buffer.write(new_bytes)
    else:
        target = Path(args.root) / args.rel_path
        target.write_bytes(new_bytes)
        print(f"WRAPPED: {args.rel_path}")


if __name__ == '__main__':
    main()
