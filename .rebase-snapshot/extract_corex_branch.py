#!/usr/bin/env python3
"""extract_corex_branch.py
Inverse of extract_nvidia_branch.py. Reads a switcher-overlay file with
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT  ... #else ... #endif
and emits the COREX branch (the #if-true side), stripping out the #else NVIDIA
upstream body.

Used by the rebase tooling to verify corex branches are byte-for-byte preserved
across the rebase.
"""
import re
import sys
from pathlib import Path

IF_RE = re.compile(r'^\s*#\s*if\s+defined\s*\(\s*UIPC_COREX_CUDA10_COMPAT\s*\)\s*&&\s*UIPC_COREX_CUDA10_COMPAT\s*$')
ELSE_RE = re.compile(r'^\s*#\s*else\b')
ELIF_RE = re.compile(r'^\s*#\s*elif\b')
ENDIF_RE = re.compile(r'^\s*#\s*endif\b')
NESTED_IF_RE = re.compile(r'^\s*#\s*if(n?def)?\b')


def extract(path: Path) -> str:
    out = []
    text = path.read_text(encoding='utf-8', errors='replace').splitlines(keepends=True)
    i = 0
    while i < len(text):
        line = text[i]
        if IF_RE.match(line):
            i += 1
            depth = 1
            in_then = True
            while i < len(text) and depth > 0:
                l = text[i]
                if NESTED_IF_RE.match(l):
                    depth += 1
                    if in_then:
                        out.append(l)
                elif ENDIF_RE.match(l) and depth > 0:
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                    elif in_then:
                        out.append(l)
                elif depth == 1 and (ELSE_RE.match(l) or ELIF_RE.match(l)):
                    in_then = False
                elif in_then:
                    out.append(l)
                i += 1
        else:
            out.append(line)
            i += 1
    return ''.join(out)


def main():
    if len(sys.argv) != 2:
        print('usage: extract_corex_branch.py <file>', file=sys.stderr)
        sys.exit(2)
    p = Path(sys.argv[1])
    sys.stdout.write(extract(p))


if __name__ == '__main__':
    main()
