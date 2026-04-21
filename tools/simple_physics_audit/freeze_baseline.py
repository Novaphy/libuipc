#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a simple-scene run directory as a parity baseline.")
    parser.add_argument("--run-dir", required=True, help="Run output dir containing config/objs")
    parser.add_argument("--baseline-root", required=True, help="Destination baseline root")
    parser.add_argument("--baseline-name", required=True, help="Baseline name under root")
    parser.add_argument("--label", required=True, help="Human-readable baseline label")
    parser.add_argument("--source-type", required=True, choices=["nvidia", "upstream", "corex-legacy", "corex-current"])
    parser.add_argument("--notes", default="", help="Optional notes")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.baseline_root).resolve() / args.baseline_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy core artifacts
    copy_if_exists(run_dir / "config.json", out_dir / "config.json")
    copy_if_exists(run_dir / "systems.json", out_dir / "systems.json")
    copy_if_exists(run_dir / "run.log", out_dir / "run.log")

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for obj in sorted(run_dir.glob("scene_surface_*.obj")):
        shutil.copy2(obj, frame_dir / obj.name)

    manifest = {
        "label": args.label,
        "source_type": args.source_type,
        "run_dir": str(run_dir),
        "baseline_dir": str(out_dir),
        "frame_count": len(list(frame_dir.glob("scene_surface_*.obj"))),
        "notes": args.notes,
    }
    (out_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"baseline frozen: {out_dir}")


if __name__ == "__main__":
    main()

