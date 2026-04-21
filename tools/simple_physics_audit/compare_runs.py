#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _vec(d: dict, key: str) -> np.ndarray:
    return np.asarray(d[key], dtype=np.float64)


def compare_metrics(ref: dict, cur: dict) -> dict:
    ref_by_frame = {f["frame"]: f for f in ref["frames"]}
    cur_by_frame = {f["frame"]: f for f in cur["frames"]}
    common = sorted(set(ref_by_frame) & set(cur_by_frame))
    if not common:
        raise RuntimeError("no common frames between reference and current metrics")

    com_err = []
    n_ang = []
    y_gap_diff = []
    for f in common:
        rf = ref_by_frame[f]
        cf = cur_by_frame[f]
        rc = _vec(rf, "com")
        cc = _vec(cf, "com")
        rn = _vec(rf, "normal")
        cn = _vec(cf, "normal")
        com_err.append(float(np.linalg.norm(cc - rc)))
        dot = float(np.clip(np.dot(rn, cn), -1.0, 1.0))
        n_ang.append(float(np.degrees(np.arccos(dot))))
        y_gap_diff.append(float(cf["y_gap"] - rf["y_gap"]))

    # First visible divergence frame by thresholds
    first_div = None
    for i, f in enumerate(common):
        if com_err[i] > 1e-3 or n_ang[i] > 0.2 or abs(y_gap_diff[i]) > 1e-3:
            first_div = f
            break

    return {
        "common_frame_count": len(common),
        "frame_start": common[0],
        "frame_end": common[-1],
        "com_err_l2_max": float(max(com_err)),
        "com_err_l2_mean": float(np.mean(com_err)),
        "normal_angle_deg_max": float(max(n_ang)),
        "normal_angle_deg_mean": float(np.mean(n_ang)),
        "y_gap_diff_abs_max": float(max(abs(x) for x in y_gap_diff)),
        "y_gap_diff_mean": float(np.mean(y_gap_diff)),
        "first_divergence_frame_by_threshold": first_div,
        "reference_tetra_overlap_frame_count": int(ref["tetra_overlap_frame_count"]),
        "current_tetra_overlap_frame_count": int(cur["tetra_overlap_frame_count"]),
        "reference_first_negative_y_gap": ref["first_negative_y_gap"],
        "current_first_negative_y_gap": cur["first_negative_y_gap"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two simple-scene metrics JSON files.")
    parser.add_argument("--reference", required=True, help="Reference metrics JSON")
    parser.add_argument("--current", required=True, help="Current metrics JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    ref = _load_json(args.reference)
    cur = _load_json(args.current)
    out = compare_metrics(ref, cur)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote comparison: {output}")


if __name__ == "__main__":
    main()

