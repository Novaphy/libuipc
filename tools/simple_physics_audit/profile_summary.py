#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRAME_TIMING_RE = re.compile(
    r"frame\s+(?P<frame>\d+)\s+timings:\s+advance=(?P<advance>\d+)ms\s+sync=(?P<sync>\d+)ms\s+retrieve=(?P<retrieve>\d+)ms\s+write_obj=(?P<write>\d+)ms"
)
PCG_RE = re.compile(
    r"LinearPCG:\s+frame=(?P<frame>\d+)\s+newton_iter=(?P<newton>\d+)\s+dof=(?P<dof>\d+)\s+max_iter=(?P<max_iter>\d+)\s+->\s+iters=(?P<iters>\d+)"
)
NEWTON_RE = re.compile(r"Newton Iteration Converged with Iteration Count:\s+(?P<count>\d+)")
TRIPLET_RE = re.compile(
    r"GlobalLinearSystem has (?P<dof>\d+) DoFs, Unique Triplet Count:\s+(?P<triplets>\d+)"
)
CANDIDATE_RE = re.compile(
    r"SimplexTrajectoryFilter PTs:\s+(?P<pt>\d+), EEs:\s+(?P<ee>\d+), PEs:\s+(?P<pe>\d+), PPs:\s+(?P<pp>\d+)"
)
COREX_PHASE_RE = re.compile(
    r"\[corex_phase\]\s+category=(?P<category>\S+)\s+name=(?P<name>\S+)\s+"
    r"frame=(?P<frame>-?\d+)\s+newton=(?P<newton>-?\d+)\s+iter=(?P<iter>-?\d+)\s+"
    r"elapsed_ms=(?P<elapsed>[0-9.]+)"
)
ABD_ENERGY_AB_RE = re.compile(
    r"\[corex_abd_energy_ab\]\s+kind=(?P<kind>\S+)\s+host=(?P<host>[-+0-9.eE]+)\s+"
    r"gpu=(?P<gpu>[-+0-9.eE]+)\s+abs=(?P<abs>[-+0-9.eE]+)\s+rel=(?P<rel>[-+0-9.eE]+)"
)


def _stats(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "sum": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "sum": sum(values),
    }


def summarize_log(path: Path) -> dict:
    frame_timings = []
    pcg_iters = []
    newton_counts = []
    triplets = []
    candidate_totals = []
    corex_phases: dict[str, list[float]] = {}
    pcg_by_frame: dict[int, list[int]] = {}
    pcg_by_frame_newton: list[dict[str, int]] = []
    abd_energy_ab: dict[str, dict[str, list[float]]] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := FRAME_TIMING_RE.search(line):
            frame_timings.append(
                {
                    "frame": int(m.group("frame")),
                    "advance_ms": int(m.group("advance")),
                    "sync_ms": int(m.group("sync")),
                    "retrieve_ms": int(m.group("retrieve")),
                    "write_obj_ms": int(m.group("write")),
                }
            )
            continue
        if m := PCG_RE.search(line):
            frame = int(m.group("frame"))
            newton = int(m.group("newton"))
            iters = int(m.group("iters"))
            pcg_iters.append(iters)
            pcg_by_frame.setdefault(frame, []).append(iters)
            pcg_by_frame_newton.append({"frame": frame, "newton": newton, "iters": iters})
            continue
        if m := NEWTON_RE.search(line):
            newton_counts.append(int(m.group("count")))
            continue
        if m := TRIPLET_RE.search(line):
            triplets.append(int(m.group("triplets")))
            continue
        if m := CANDIDATE_RE.search(line):
            candidate_totals.append(
                int(m.group("pt")) + int(m.group("ee")) + int(m.group("pe")) + int(m.group("pp"))
            )
            continue
        if m := COREX_PHASE_RE.search(line):
            key = f"{m.group('category')}.{m.group('name')}"
            corex_phases.setdefault(key, []).append(float(m.group("elapsed")))
            continue
        if m := ABD_ENERGY_AB_RE.search(line):
            kind = m.group("kind")
            values = abd_energy_ab.setdefault(
                kind, {"host": [], "gpu": [], "abs": [], "rel": []}
            )
            values["host"].append(float(m.group("host")))
            values["gpu"].append(float(m.group("gpu")))
            values["abs"].append(float(m.group("abs")))
            values["rel"].append(float(m.group("rel")))

    advances = [x["advance_ms"] for x in frame_timings]
    syncs = [x["sync_ms"] for x in frame_timings]
    retrieves = [x["retrieve_ms"] for x in frame_timings]
    writes = [x["write_obj_ms"] for x in frame_timings]

    return {
        "log": str(path),
        "frame_timing": {
            "frames_reported": len(frame_timings),
            "advance_ms": _stats(advances),
            "sync_ms": _stats(syncs),
            "retrieve_ms": _stats(retrieves),
            "write_obj_ms": _stats(writes),
        },
        "pcg_iters": _stats(pcg_iters),
        "pcg_by_frame": {
            str(frame): _stats(values) for frame, values in sorted(pcg_by_frame.items())
        },
        "pcg_by_frame_newton": pcg_by_frame_newton,
        "newton_iters": _stats(newton_counts),
        "unique_triplets": _stats(triplets),
        "simplex_candidate_totals": _stats(candidate_totals),
        "corex_phase_ms": {key: _stats(values) for key, values in sorted(corex_phases.items())},
        "abd_energy_ab": {
            kind: {metric: _stats(values) for metric, values in metrics.items()}
            for kind, metrics in sorted(abd_energy_ab.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize corex_demo run logs.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = summarize_log(Path(args.log))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote profile summary: {output}")


if __name__ == "__main__":
    main()
