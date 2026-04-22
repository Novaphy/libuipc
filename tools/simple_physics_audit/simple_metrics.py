#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _load_vertices(path: str) -> np.ndarray:
    vs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vs, dtype=np.float64)


def _frame_index(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def _safe_normal(tet4: np.ndarray) -> np.ndarray:
    c = tet4.mean(axis=0)
    rel = tet4 - c
    n = np.cross(rel[1] - rel[0], rel[2] - rel[0])
    ln = np.linalg.norm(n)
    if ln <= 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return n / ln


def _point_in_tet(p: np.ndarray, tet: np.ndarray, eps: float = 1e-6) -> bool:
    a, b, c, d = tet
    m = np.column_stack((b - a, c - a, d - a))
    try:
        uvw = np.linalg.solve(m, p - a)
    except np.linalg.LinAlgError:
        return False
    u, v, w = uvw
    t = 1.0 - u - v - w
    return u >= -eps and v >= -eps and w >= -eps and t >= -eps


def compute_metrics(frames_dir: str) -> dict:
    frame_files = sorted(glob.glob(str(Path(frames_dir) / "scene_surface_*.obj")))
    frames = []
    inside_frames = []
    for fp in frame_files:
        vs = _load_vertices(fp)
        if vs.shape[0] < 8:
            continue
        frame = _frame_index(fp)
        moving = vs[:4]
        fixed = vs[4:8]
        com = moving.mean(axis=0)
        n = _safe_normal(moving)
        y_gap = float(moving[:, 1].min() - fixed[:, 1].max())
        inside = False
        for p in moving:
            if _point_in_tet(p, fixed):
                inside = True
                break
        if not inside:
            for p in fixed:
                if _point_in_tet(p, moving):
                    inside = True
                    break
        if inside:
            inside_frames.append(frame)
        frames.append(
            {
                "frame": frame,
                "y_gap": y_gap,
                "com": com.tolist(),
                "normal": n.tolist(),
                "tetra_overlap_proxy": int(inside),
            }
        )
    if not frames:
        raise RuntimeError(f"no valid frames found in {frames_dir}")
    return {
        "frames_dir": str(Path(frames_dir).resolve()),
        "frame_count": len(frames),
        "first_frame": frames[0]["frame"],
        "last_frame": frames[-1]["frame"],
        "min_y_gap": min(f["y_gap"] for f in frames),
        "first_negative_y_gap": next((f["frame"] for f in frames if f["y_gap"] < 0.0), None),
        "tetra_overlap_frame_count": len(inside_frames),
        "tetra_overlap_frames_head": inside_frames[:20],
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract simple-scene physical metrics from OBJ frames.")
    parser.add_argument("--frames-dir", required=True, help="Directory with scene_surface_*.obj")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    metrics = compute_metrics(args.frames_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote metrics: {out}")


if __name__ == "__main__":
    main()

