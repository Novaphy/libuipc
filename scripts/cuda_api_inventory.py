#!/usr/bin/env python3
"""
Scan libuipc CUDA backend + vendored muda for explicit CUDA Runtime / library API calls.

Outputs:
  - cuda_api_inventory.csv  (symbol, category, file, line)
  - cuda_api_inventory.md   (deduplicated summary by category)

Usage:
  python3 scripts/cuda_api_inventory.py [--root /path/to/libuipc] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# (category_name, regex for full identifier before '(')
# Order matters: more specific prefixes first.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cublas", re.compile(r"\b(cublas[A-Z][a-zA-Z0-9]*)\s*\(")),
    ("cusparse", re.compile(r"\b(cusparse[A-Z][a-zA-Z0-9]*)\s*\(")),
    ("cusolver", re.compile(r"\b(cusolver[A-Z][a-zA-Z0-9]*)\s*\(")),
    ("curand", re.compile(r"\b(curand[A-Z][a-zA-Z0-9]*)\s*\(")),
    ("cufft", re.compile(r"\b(cufft[A-Z][a-zA-Z0-9]*)\s*\(")),
    ("nvrtc", re.compile(r"\b(nvrtc[A-Z][a-zA-Z0-9]*)\s*\(")),
    # CUDA Driver API (rare in this tree but capture if present)
    ("cuda_driver", re.compile(r"\b(cu[A-Z][a-zA-Z0-9]{2,})\s*\(")),
    ("cuda_runtime", re.compile(r"\b(cuda[A-Z][a-zA-Z0-9]*)\s*\(")),
]

EXTENSIONS = {".cu", ".cuh", ".h", ".hpp", ".inl"}


def should_scan(path: Path) -> bool:
    """Production scope: CUDA backend + vendored muda library sources only (not muda example/test)."""
    if path.suffix.lower() not in EXTENSIONS:
        return False
    s = str(path).replace("\\", "/")
    if "src/backends/cuda" in s:
        return True
    if "external/muda/src/" in s or "/external/muda/src/" in s:
        return True
    return False


def categorize_symbol(sym: str, category: str) -> str:
    if category == "cuda_driver" and sym.startswith("cuda"):
        return "cuda_runtime"
    return category


def scan_file(path: Path, rows: list[dict[str, str]]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"warn: skip {path}: {e}", file=sys.stderr)
        return
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        if line.lstrip().startswith("//"):
            continue
        for cat, pat in PATTERNS:
            for m in pat.finditer(line):
                sym = m.group(1)
                cat2 = categorize_symbol(sym, cat)
                rows.append(
                    {
                        "symbol": sym,
                        "category": cat2,
                        "file": str(path),
                        "line": str(lineno),
                    }
                )


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDA API static inventory for libuipc + muda")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="libuipc repository root",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/tools/cuda_api_inventory_out)",
    )
    args = ap.parse_args()
    root: Path = args.root.resolve()
    out_dir: Path = (args.out_dir or (root / "tools" / "cuda_api_inventory_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if not should_scan(p):
            continue
        scan_file(p, rows)

    csv_path = out_dir / "cuda_api_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "category", "file", "line"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["category"], x["symbol"], x["file"], int(x["line"]))):
            w.writerow(r)

    # Deduplicate symbols per category
    by_cat: dict[str, set[str]] = defaultdict(set)
    ref_count: dict[str, int] = defaultdict(int)
    for r in rows:
        sym = r["symbol"]
        cat = r["category"]
        by_cat[cat].add(sym)
        ref_count[sym] += 1

    def is_cuda_graph_symbol(sym: str) -> bool:
        if sym.startswith("cudaGraph") or sym.startswith("cudaGraphExec"):
            return True
        if sym in ("cudaStreamBeginCapture", "cudaStreamEndCapture"):
            return True
        return False

    graph_syms = sorted(s for s in ref_count if is_cuda_graph_symbol(s))

    md_path = out_dir / "cuda_api_inventory.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# CUDA API static inventory (production)\n\n")
        f.write(
            f"Generated from `{root}` (paths: `src/backends/cuda`, `external/muda/src` only).\n\n"
        )
        f.write("**Note:** Conditional branches and macros may list symbols not present in a given build.\n\n")
        f.write(f"- Total reference rows: **{len(rows)}**\n")
        f.write(f"- Unique symbols: **{len(ref_count)}**\n\n")

        f.write("## cuda_graph (subset of cuda_runtime)\n\n")
        f.write(
            "CUDA Graph and stream-capture related symbols; cross-check separately against your stack.\n\n"
        )
        f.write(f"- Unique in this category: **{len(graph_syms)}**\n\n")
        for s in graph_syms:
            f.write(f"- `{s}` (refs: {ref_count[s]})\n")
        f.write("\n")

        for cat in sorted(by_cat.keys()):
            syms = sorted(by_cat[cat])
            f.write(f"## {cat} ({len(syms)} unique)\n\n")
            for s in syms:
                f.write(f"- `{s}` (refs: {ref_count[s]})\n")
            f.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
