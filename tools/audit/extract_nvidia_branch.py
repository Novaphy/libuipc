#!/usr/bin/env python3
"""Extract the NVIDIA-branch view of a source file.

Simulates "UIPC_COREX_CUDA10_COMPAT is undefined / 0":
    * #if (defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT)
      #if UIPC_COREX_CUDA10_COMPAT
      #ifdef UIPC_COREX_CUDA10_COMPAT
      #ifndef UIPC_COREX_CUDA10_COMPAT  (kept; opposite branch)
      blocks are evaluated as false (kept the #else / #elif body).
    * Other #if directives are passed through untouched.
    * The skip state is inherited: anything inside a skipped block, including
      nested #if directives, is dropped.
    * Blank-line and trailing-newline state is preserved byte-for-byte so that
      the output can be diffed against /root/src.

Used by tools/audit/verify_nvidia_branch_equiv.sh.
"""
import re
import sys


def extract(path):
    with open(path) as f:
        lines = f.readlines()

    out = []
    stack = []  # list of {'kind': 'corex'|'corex-ndef'|'other', 'mode': 'keep'|'skip'}

    def currently_skipping():
        return any(e['mode'] == 'skip' for e in stack)

    for ln in lines:
        s = ln.rstrip('\n')
        m_if = re.match(r'^\s*#\s*if(def|ndef)?\b(.*)$', s)
        m_elif = re.match(r'^\s*#\s*elif\b(.*)$', s)
        m_else = re.match(r'^\s*#\s*else\b', s)
        m_end = re.match(r'^\s*#\s*endif\b', s)

        if m_if:
            kind_kw = m_if.group(1) or ''
            cond = m_if.group(2)
            is_corex = bool(re.search(r'UIPC_COREX_CUDA10_COMPAT', cond))
            parent_skip = currently_skipping()
            if is_corex:
                if kind_kw == 'ndef':
                    # #ifndef UIPC_COREX_CUDA10_COMPAT -> NVIDIA-true
                    stack.append({'kind': 'corex-ndef', 'mode': 'keep'})
                else:
                    # #if / #ifdef UIPC_COREX_CUDA10_COMPAT -> NVIDIA-false
                    stack.append({'kind': 'corex', 'mode': 'skip'})
            else:
                stack.append({'kind': 'other', 'mode': 'keep'})
                if not parent_skip:
                    out.append(s)
            continue

        if m_else:
            top = stack[-1]
            if top['kind'] == 'corex':
                top['mode'] = 'keep'
            elif top['kind'] == 'corex-ndef':
                top['mode'] = 'skip'
            else:
                if not any(e['mode'] == 'skip' for e in stack[:-1]):
                    out.append(s)
            continue

        if m_elif:
            top = stack[-1]
            if top['kind'] in ('corex', 'corex-ndef'):
                # We don't really model #elif on the corex macro; mark skip
                # to be safe. This is sufficient for our codebase, which only
                # uses #if / #ifdef / #ifndef + #else.
                top['mode'] = 'skip'
            else:
                if not any(e['mode'] == 'skip' for e in stack[:-1]):
                    out.append(s)
            continue

        if m_end:
            top = stack.pop()
            if top['kind'] in ('corex', 'corex-ndef'):
                pass  # the directive itself disappears
            else:
                if not currently_skipping():
                    out.append(s)
            continue

        # regular source line
        if currently_skipping():
            continue
        out.append(s)

    with open(path, 'rb') as f:
        raw = f.read()
    has_trailing_nl = raw.endswith(b'\n')

    text = '\n'.join(out)
    if has_trailing_nl:
        text += '\n'
    return text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('usage: extract_nvidia_branch.py <file>\n')
        sys.exit(2)
    sys.stdout.write(extract(sys.argv[1]))
