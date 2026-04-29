#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from simple_metrics import compute_metrics


def _frame_map(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(f["frame"]): f for f in metrics["frames"]}


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    for frame in metrics["frames"]:
        values = [frame["y_gap"], *frame["com"], *frame["normal"]]
        if any(not math.isfinite(float(v)) for v in values):
            return False
    return True


def _estimate_gravity(metrics: dict[str, Any], dt: float, max_sample_frame: int) -> float | None:
    frames = [f for f in metrics["frames"] if int(f["frame"]) <= max_sample_frame]
    frames = sorted(frames, key=lambda f: int(f["frame"]))
    if len(frames) < 3:
        return None

    samples: list[float] = []
    for a, b, c in zip(frames, frames[1:], frames[2:]):
        fa, fb, fc = int(a["frame"]), int(b["frame"]), int(c["frame"])
        if fb - fa != 1 or fc - fb != 1:
            continue
        ya = float(a["com"][1])
        yb = float(b["com"][1])
        yc = float(c["com"][1])
        samples.append((yc - 2.0 * yb + ya) / (dt * dt))
    if not samples:
        return None
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    if la <= 1e-12 or lb <= 1e-12:
        return 0.0
    dot = float(np.clip(np.dot(a / la, b / lb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    frame_count = int(metrics["frame_count"])
    last_frame = int(metrics["last_frame"])
    first_negative = metrics["first_negative_y_gap"]
    overlap_count = int(metrics["tetra_overlap_frame_count"])

    if frame_count < args.min_frames:
        failures.append(f"frame_count {frame_count} < min_frames {args.min_frames}")
    if last_frame < args.min_last_frame:
        failures.append(f"last_frame {last_frame} < min_last_frame {args.min_last_frame}")
    if overlap_count > args.max_overlap_frames:
        failures.append(
            f"tetra_overlap_frame_count {overlap_count} > max_overlap_frames {args.max_overlap_frames}"
        )
    if not _finite_metrics(metrics):
        failures.append("non-finite value found in COM/normal/y_gap metrics")

    if first_negative is None:
        warnings.append("no negative y_gap frame observed; contact window may not be reached")
        gravity_sample_end = min(last_frame, args.freefall_sample_frames)
    else:
        gravity_sample_end = max(2, min(int(first_negative) - 2, args.freefall_sample_frames))

    gravity_estimate = _estimate_gravity(metrics, args.dt, gravity_sample_end)
    if gravity_estimate is None:
        warnings.append("unable to estimate free-fall acceleration")
    elif abs(gravity_estimate - args.expected_gravity_y) > args.gravity_tolerance:
        failures.append(
            "free-fall acceleration estimate "
            f"{gravity_estimate:.6g} differs from expected {args.expected_gravity_y:.6g} "
            f"by more than {args.gravity_tolerance:.6g}"
        )

    post_contact_com_delta = None
    post_contact_normal_angle_deg = None
    if first_negative is not None:
        by_frame = _frame_map(metrics)
        start_frame = int(first_negative)
        end_frame = min(last_frame, start_frame + args.post_contact_window)
        if start_frame in by_frame and end_frame in by_frame:
            start = by_frame[start_frame]
            end = by_frame[end_frame]
            c0 = np.asarray(start["com"], dtype=np.float64)
            c1 = np.asarray(end["com"], dtype=np.float64)
            n0 = np.asarray(start["normal"], dtype=np.float64)
            n1 = np.asarray(end["normal"], dtype=np.float64)
            post_contact_com_delta = float(np.linalg.norm(c1 - c0))
            post_contact_normal_angle_deg = _angle_deg(n0, n1)

            if post_contact_com_delta < args.min_post_contact_com_delta:
                failures.append(
                    f"post-contact COM delta {post_contact_com_delta:.6g} "
                    f"< {args.min_post_contact_com_delta:.6g}"
                )
            if post_contact_normal_angle_deg < args.min_post_contact_normal_angle_deg:
                warnings.append(
                    f"post-contact normal angle {post_contact_normal_angle_deg:.6g} "
                    f"< {args.min_post_contact_normal_angle_deg:.6g}"
                )
        else:
            warnings.append(
                f"cannot evaluate post-contact window [{start_frame}, {end_frame}] from available frames"
            )

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "checks": {
            "frame_count": frame_count,
            "last_frame": last_frame,
            "min_y_gap": metrics["min_y_gap"],
            "first_negative_y_gap": first_negative,
            "tetra_overlap_frame_count": overlap_count,
            "gravity_estimate_y": gravity_estimate,
            "post_contact_com_delta": post_contact_com_delta,
            "post_contact_normal_angle_deg": post_contact_normal_angle_deg,
        },
        "thresholds": {
            "min_frames": args.min_frames,
            "min_last_frame": args.min_last_frame,
            "max_overlap_frames": args.max_overlap_frames,
            "dt": args.dt,
            "expected_gravity_y": args.expected_gravity_y,
            "gravity_tolerance": args.gravity_tolerance,
            "post_contact_window": args.post_contact_window,
            "min_post_contact_com_delta": args.min_post_contact_com_delta,
            "min_post_contact_normal_angle_deg": args.min_post_contact_normal_angle_deg,
        },
    }


def run_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.run_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.demo,
        "--backend",
        "cuda",
        "--scene",
        "simple",
        "--frames",
        str(args.frames),
        "--gpu",
        str(args.gpu),
        "--output_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    for key in (
        "UIPC_COREX_TRACE_LINEAR_SYSTEM",
        "UIPC_COREX_TRACE_SIMPLEX_FILTER",
        "UIPC_COREX_TRACE_FILTER_ACTIVE_DIAG",
        "UIPC_COREX_TRACE_CONTACT_TYPE_ENERGY",
        "UIPC_COREX_TRACE_CONTACT_TYPE_GRAD",
        "UIPC_COREX_TRACE_BARRIER_DBDD",
    ):
        env.pop(key, None)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=args.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"corex_demo timed out after {exc.timeout} seconds; see {log_path}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"corex_demo failed with exit code {proc.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/evaluate the CoreX simple-scene correctness gate.")
    parser.add_argument("--frames-dir", required=True, help="Directory containing or receiving scene_surface_*.obj")
    parser.add_argument("--metrics-output", required=True, help="Output metrics JSON")
    parser.add_argument("--report-output", required=True, help="Output gate report JSON")
    parser.add_argument("--run-demo", action="store_true", help="Run corex_demo before evaluating frames")
    parser.add_argument("--demo", default="./Release/bin/corex_demo", help="corex_demo executable")
    parser.add_argument("--cwd", default=".", help="Working directory for --run-demo")
    parser.add_argument("--run-log", default="/tmp/simple_gate_run.log", help="Log path for --run-demo")
    parser.add_argument("--frames", type=int, default=120, help="Frames to run when --run-demo is set")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for --run-demo")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout seconds for --run-demo")
    parser.add_argument("--min-frames", type=int, default=90)
    parser.add_argument("--min-last-frame", type=int, default=89)
    parser.add_argument("--max-overlap-frames", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--expected-gravity-y", type=float, default=-9.8)
    parser.add_argument("--gravity-tolerance", type=float, default=2.5)
    parser.add_argument("--freefall-sample-frames", type=int, default=20)
    parser.add_argument("--post-contact-window", type=int, default=30)
    parser.add_argument("--min-post-contact-com-delta", type=float, default=0.03)
    parser.add_argument("--min-post-contact-normal-angle-deg", type=float, default=0.1)
    args = parser.parse_args()

    if args.run_demo:
        try:
            run_demo(args)
        except Exception as exc:
            report_path = Path(args.report_output)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "passed": False,
                "failures": [str(exc)],
                "warnings": [],
                "checks": {"run_demo": "failed"},
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("simple gate: FAIL")
            print(f"report:  {report_path}")
            print(f"FAIL: {exc}", file=sys.stderr)
            sys.exit(1)

    metrics = compute_metrics(args.frames_dir)
    report = evaluate(metrics, args)

    metrics_path = Path(args.metrics_output)
    report_path = Path(args.report_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "PASS" if report["passed"] else "FAIL"
    print(f"simple gate: {status}")
    print(f"metrics: {metrics_path}")
    print(f"report:  {report_path}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    for failure in report["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
