#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TYPE_NAMES = ("PT", "EE", "PE", "PP")


SPD_BEGIN_RE = re.compile(
    r"\[corex_spd_contact_begin\] total=(?P<total>\d+) PT=(?P<PT>\d+) "
    r"EE=(?P<EE>\d+) PE=(?P<PE>\d+) PP=(?P<PP>\d+) "
    r"gradient_only=(?P<gradient_only>\d+) enabled=(?P<enabled>\d+)"
)
SPD_RE = re.compile(
    r"\[corex_spd_contact\] type=(?P<type>\w+) count=(?P<count>\d+) "
    r"projected=(?P<projected>\d+) projected_rate=(?P<projected_rate>[-+0-9.eE]+) "
    r"gershgorin_lower_avg=(?P<lower>[-+0-9.eE]+) "
    r"correction_avg=(?P<corr_avg>[-+0-9.eE]+) "
    r"correction_max=(?P<corr_max>[-+0-9.eE]+)"
    r"(?: diag_abs_avg=(?P<diag_abs_avg>[-+0-9.eE]+) "
    r"correction_diag_ratio_avg=(?P<corr_diag_ratio_avg>[-+0-9.eE]+) "
    r"correction_diag_ratio_max=(?P<corr_diag_ratio_max>[-+0-9.eE]+))?"
)
PCG_COST_RE = re.compile(
    r"\[corex_pcg_cost\] frame=(?P<frame>\d+) newton=(?P<newton>\d+) iter=(?P<iter>\d+) "
    r"spmv_ms=(?P<spmv_ms>[-+0-9.eE]+) spmv_sync_ms=(?P<spmv_sync_ms>[-+0-9.eE]+) "
    r"precond_ms=(?P<precond_ms>[-+0-9.eE]+) dotnorm_ms=(?P<dotnorm_ms>[-+0-9.eE]+) "
    r"spmv_pct=(?P<spmv_pct>[-+0-9.eE]+) spmv_sync_pct=(?P<spmv_sync_pct>[-+0-9.eE]+) "
    r"precond_pct=(?P<precond_pct>[-+0-9.eE]+) dotnorm_pct=(?P<dotnorm_pct>[-+0-9.eE]+) "
    r"per_iter_ms=(?P<per_iter_ms>[-+0-9.eE]+) "
    r"skip_spmv_sync=(?P<skip_spmv_sync>\d+) fused_rz_norm=(?P<fused_rz_norm>\d+) "
    r"reduce2=(?P<reduce2>\d+)"
)
LINEAR_PCG_RE = re.compile(
    r"LinearPCG: frame=(?P<frame>\d+) newton_iter=(?P<newton>\d+) dof=(?P<dof>\d+) "
    r"max_iter=(?P<max_iter>\d+) -> iters=(?P<iter>\d+)"
)
STRUCT_BLOCK_RE = re.compile(
    r"\[corex_precond_struct_block\] bodies=(?P<bodies>\d+) "
    r"accepted_t=(?P<accepted_t>\d+) accepted_a0=(?P<accepted_a0>\d+) "
    r"accepted_a1=(?P<accepted_a1>\d+) accepted_a2=(?P<accepted_a2>\d+) "
    r"total_blocks=(?P<total_blocks>\d+)"
)


def as_ints(match):
    return {k: int(v) for k, v in match.groupdict().items()}


def as_numbers(match):
    out = {}
    for key, value in match.groupdict().items():
        if key in {"frame", "newton", "iter", "skip_spmv_sync", "fused_rz_norm", "reduce2"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def empty_spd_bucket():
    bucket = {
        "begin_total": 0,
        "begin_gradient_only": 0,
        "begin_enabled": 0,
    }
    for name in TYPE_NAMES:
        bucket[f"{name}_begin_count"] = 0
        bucket[f"{name}_count"] = 0
        bucket[f"{name}_projected"] = 0
        bucket[f"{name}_projected_rate"] = 0.0
        bucket[f"{name}_lower_avg"] = 0.0
        bucket[f"{name}_corr_avg"] = 0.0
        bucket[f"{name}_corr_max"] = 0.0
        bucket[f"{name}_diag_abs_avg"] = 0.0
        bucket[f"{name}_corr_diag_ratio_avg"] = 0.0
        bucket[f"{name}_corr_diag_ratio_max"] = 0.0
    return bucket


def parse_log(path):
    pending_spd = []
    pending_struct = []
    rows = []
    pcg_by_key = {}

    with path.open(errors="ignore") as f:
        current_spd = None
        for line in f:
            if m := SPD_BEGIN_RE.search(line):
                current_spd = empty_spd_bucket()
                begin = as_ints(m)
                current_spd["begin_total"] = begin["total"]
                current_spd["begin_gradient_only"] = begin["gradient_only"]
                current_spd["begin_enabled"] = begin["enabled"]
                for name in TYPE_NAMES:
                    current_spd[f"{name}_begin_count"] = begin[name]
                pending_spd.append(current_spd)
                continue

            if m := SPD_RE.search(line):
                if current_spd is None:
                    current_spd = empty_spd_bucket()
                    pending_spd.append(current_spd)
                typ = m.group("type")
                if typ in TYPE_NAMES:
                    current_spd[f"{typ}_count"] = int(m.group("count"))
                    current_spd[f"{typ}_projected"] = int(m.group("projected"))
                    current_spd[f"{typ}_projected_rate"] = float(m.group("projected_rate"))
                    current_spd[f"{typ}_lower_avg"] = float(m.group("lower"))
                    current_spd[f"{typ}_corr_avg"] = float(m.group("corr_avg"))
                    current_spd[f"{typ}_corr_max"] = float(m.group("corr_max"))
                    if m.group("diag_abs_avg") is not None:
                        current_spd[f"{typ}_diag_abs_avg"] = float(m.group("diag_abs_avg"))
                        current_spd[f"{typ}_corr_diag_ratio_avg"] = float(m.group("corr_diag_ratio_avg"))
                        current_spd[f"{typ}_corr_diag_ratio_max"] = float(m.group("corr_diag_ratio_max"))
                continue

            if m := STRUCT_BLOCK_RE.search(line):
                pending_struct.append(as_ints(m))
                continue

            pcg = None
            if m := PCG_COST_RE.search(line):
                pcg = as_numbers(m)
                pcg_by_key[(pcg["frame"], pcg["newton"])] = pcg
                continue
            elif m := LINEAR_PCG_RE.search(line):
                pcg = as_ints(m)
                existing = pcg_by_key.get((pcg["frame"], pcg["newton"]), {})
                pcg = {**existing, **pcg}

            if pcg is not None:
                spd = pending_spd.pop(0) if pending_spd else empty_spd_bucket()
                struct = pending_struct.pop(0) if pending_struct else {}
                row = {
                    "frame": pcg["frame"],
                    "newton": pcg["newton"],
                    "pcg_iter": pcg["iter"],
                }
                row.update(spd)
                row.update(struct)
                for key, value in pcg.items():
                    if key not in {"frame", "newton", "iter"}:
                        row[f"pcg_{key}"] = value
                rows.append(row)

    return rows


def summarize(rows):
    def type_snapshot(row, typ):
        return {
            "count": row.get(f"{typ}_count", 0),
            "projected": row.get(f"{typ}_projected", 0),
            "corr_avg": row.get(f"{typ}_corr_avg", 0.0),
            "corr_max": row.get(f"{typ}_corr_max", 0.0),
            "corr_diag_ratio_avg": row.get(f"{typ}_corr_diag_ratio_avg", 0.0),
            "corr_diag_ratio_max": row.get(f"{typ}_corr_diag_ratio_max", 0.0),
        }

    def row_outlier(row, typ):
        return {
            "log": row.get("log"),
            "frame": row.get("frame"),
            "newton": row.get("newton"),
            "pcg_iter": row.get("pcg_iter"),
            "begin_total": row.get("begin_total"),
            "pcg_per_iter_ms": row.get("pcg_per_iter_ms"),
            "pcg_dotnorm_pct": row.get("pcg_dotnorm_pct"),
            "pcg_precond_pct": row.get("pcg_precond_pct"),
            "type": typ,
            **type_snapshot(row, typ),
        }

    summary = {
        "rows": len(rows),
        "pcg_iter_sum": sum(r.get("pcg_iter", 0) for r in rows),
        "pcg_iter_max": max((r.get("pcg_iter", 0) for r in rows), default=0),
        "nonzero_spd_rows": sum(1 for r in rows if r.get("begin_total", 0) > 0),
    }
    for name in TYPE_NAMES:
        summary[f"{name}_count_sum"] = sum(r.get(f"{name}_count", 0) for r in rows)
        summary[f"{name}_projected_sum"] = sum(r.get(f"{name}_projected", 0) for r in rows)
        weighted_corr = sum(r.get(f"{name}_corr_avg", 0.0) * r.get(f"{name}_count", 0) for r in rows)
        count = summary[f"{name}_count_sum"]
        summary[f"{name}_corr_weighted_avg"] = weighted_corr / count if count else 0.0
        summary[f"{name}_corr_max"] = max((r.get(f"{name}_corr_max", 0.0) for r in rows), default=0.0)
        weighted_ratio = sum(
            r.get(f"{name}_corr_diag_ratio_avg", 0.0) * r.get(f"{name}_count", 0) for r in rows
        )
        summary[f"{name}_corr_diag_ratio_weighted_avg"] = weighted_ratio / count if count else 0.0
        summary[f"{name}_corr_diag_ratio_max"] = max(
            (r.get(f"{name}_corr_diag_ratio_max", 0.0) for r in rows), default=0.0
        )
    for key in ("spmv_ms", "spmv_sync_ms", "precond_ms", "dotnorm_ms"):
        full_key = f"pcg_{key}"
        summary[f"{key}_sum"] = sum(float(r.get(full_key, 0.0)) for r in rows)
    measured = sum(summary[f"{key}_sum"] for key in ("spmv_ms", "spmv_sync_ms", "precond_ms", "dotnorm_ms"))
    if measured > 0:
        for key in ("spmv_ms", "spmv_sync_ms", "precond_ms", "dotnorm_ms"):
            summary[f"{key[:-3]}_pct"] = 100.0 * summary[f"{key}_sum"] / measured

    top = sorted(
        rows,
        key=lambda r: (r.get("pcg_iter", 0), max(r.get(f"{t}_corr_max", 0.0) for t in TYPE_NAMES)),
        reverse=True,
    )[:10]
    summary["top_pcg_iter_rows"] = [
        {
            "log": r.get("log"),
            "frame": r.get("frame"),
            "newton": r.get("newton"),
            "pcg_iter": r.get("pcg_iter"),
            "begin_total": r.get("begin_total"),
            "PT_corr_max": r.get("PT_corr_max"),
            "EE_corr_max": r.get("EE_corr_max"),
            "PE_corr_max": r.get("PE_corr_max"),
            "PP_corr_max": r.get("PP_corr_max"),
            "PT_corr_diag_ratio_max": r.get("PT_corr_diag_ratio_max"),
            "EE_corr_diag_ratio_max": r.get("EE_corr_diag_ratio_max"),
            "PE_corr_diag_ratio_max": r.get("PE_corr_diag_ratio_max"),
            "PP_corr_diag_ratio_max": r.get("PP_corr_diag_ratio_max"),
            "dotnorm_pct": r.get("pcg_dotnorm_pct"),
        }
        for r in top
    ]
    summary["outliers_by_type"] = {
        typ: [
            row_outlier(r, typ)
            for r in sorted(
                rows,
                key=lambda row: (
                    row.get(f"{typ}_corr_diag_ratio_max", 0.0),
                    row.get("pcg_iter", 0),
                    row.get(f"{typ}_corr_max", 0.0),
                ),
                reverse=True,
            )[:10]
            if r.get(f"{typ}_count", 0) > 0
        ]
        for typ in TYPE_NAMES
    }
    summary["high_iter_contact_leaders"] = []
    for r in top:
        leader = max(TYPE_NAMES, key=lambda typ: r.get(f"{typ}_corr_diag_ratio_max", 0.0))
        summary["high_iter_contact_leaders"].append(row_outlier(r, leader))

    frame_buckets = defaultdict(list)
    for r in rows:
        frame_buckets[(r.get("log"), r.get("frame"))].append(r)
    frame_rows = []
    for (log, frame), bucket in frame_buckets.items():
        if frame is None:
            continue
        frame_row = {
            "log": log,
            "frame": frame,
            "pcg_iter_sum": sum(r.get("pcg_iter", 0) for r in bucket),
            "pcg_iter_max": max((r.get("pcg_iter", 0) for r in bucket), default=0),
            "newton_count": len(bucket),
            "begin_total_max": max((r.get("begin_total", 0) for r in bucket), default=0),
        }
        for typ in TYPE_NAMES:
            frame_row[f"{typ}_corr_diag_ratio_max"] = max(
                (r.get(f"{typ}_corr_diag_ratio_max", 0.0) for r in bucket), default=0.0
            )
            frame_row[f"{typ}_corr_max"] = max(
                (r.get(f"{typ}_corr_max", 0.0) for r in bucket), default=0.0
            )
        frame_rows.append(frame_row)
    summary["top_frames_by_pcg_iter"] = sorted(
        frame_rows,
        key=lambda r: (r["pcg_iter_sum"], r["pcg_iter_max"]),
        reverse=True,
    )[:10]
    summary["top_frames_by_contact_ratio"] = sorted(
        frame_rows,
        key=lambda r: max(r[f"{typ}_corr_diag_ratio_max"] for typ in TYPE_NAMES),
        reverse=True,
    )[:10]

    pe_rows = [r for r in rows if r.get("PE_count", 0) > 0]
    summary["pe_specific"] = {
        "top_by_ratio": [
            row_outlier(r, "PE")
            for r in sorted(
                pe_rows,
                key=lambda row: (
                    row.get("PE_corr_diag_ratio_max", 0.0),
                    row.get("pcg_iter", 0),
                    row.get("PE_count", 0),
                ),
                reverse=True,
            )[:15]
        ],
        "top_by_weighted_ratio": [
            {
                **row_outlier(r, "PE"),
                "weighted_ratio_signal": r.get("PE_corr_diag_ratio_avg", 0.0)
                * r.get("PE_count", 0),
            }
            for r in sorted(
                pe_rows,
                key=lambda row: (
                    row.get("PE_corr_diag_ratio_avg", 0.0) * row.get("PE_count", 0),
                    row.get("pcg_iter", 0),
                ),
                reverse=True,
            )[:15]
        ],
        "top_by_pcg_iter": [
            row_outlier(r, "PE")
            for r in sorted(
                pe_rows,
                key=lambda row: (
                    row.get("pcg_iter", 0),
                    row.get("PE_corr_diag_ratio_max", 0.0),
                    row.get("PE_count", 0),
                ),
                reverse=True,
            )[:15]
        ],
    }

    window_buckets = defaultdict(list)
    for r in pe_rows:
        frame = r.get("frame")
        if frame is None:
            continue
        window_start = (int(frame) // 10) * 10
        window_buckets[(r.get("log"), window_start)].append(r)
    pe_windows = []
    for (log, window_start), bucket in window_buckets.items():
        pe_count = sum(r.get("PE_count", 0) for r in bucket)
        weighted_ratio = sum(
            r.get("PE_corr_diag_ratio_avg", 0.0) * r.get("PE_count", 0) for r in bucket
        )
        pe_windows.append(
            {
                "log": log,
                "frame_window": f"{window_start}-{window_start + 9}",
                "rows": len(bucket),
                "pcg_iter_sum": sum(r.get("pcg_iter", 0) for r in bucket),
                "pcg_iter_max": max((r.get("pcg_iter", 0) for r in bucket), default=0),
                "PE_count_sum": pe_count,
                "PE_corr_diag_ratio_weighted_avg": weighted_ratio / pe_count
                if pe_count
                else 0.0,
                "PE_corr_diag_ratio_max": max(
                    (r.get("PE_corr_diag_ratio_max", 0.0) for r in bucket), default=0.0
                ),
                "PE_corr_max": max((r.get("PE_corr_max", 0.0) for r in bucket), default=0.0),
            }
        )
    summary["pe_specific"]["top_windows_by_pcg_iter"] = sorted(
        pe_windows,
        key=lambda r: (r["pcg_iter_sum"], r["pcg_iter_max"]),
        reverse=True,
    )[:15]
    summary["pe_specific"]["top_windows_by_weighted_ratio"] = sorted(
        pe_windows,
        key=lambda r: (r["PE_corr_diag_ratio_weighted_avg"], r["PE_count_sum"]),
        reverse=True,
    )[:15]
    return summary


def write_csv(rows, path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CoreX contact SPD and PCG diagnostics.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    all_rows = []
    summaries = {}
    for log in args.logs:
        rows = parse_log(log)
        for row in rows:
            row["log"] = str(log)
        all_rows.extend(rows)
        summaries[str(log)] = summarize(rows)

    result = {
        "logs": summaries,
        "combined": summarize(all_rows),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(all_rows, args.csv)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
