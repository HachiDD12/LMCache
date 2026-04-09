#!/usr/bin/env python3
"""
Visualize LMCache KV-cache profiling results.

Usage:
    python scripts/visualize_kv_profile.py [lmcache_profile.jsonl]

Reads the JSONL file produced by the LMCache profiler (enabled via
LMCACHE_PROFILE=1) and produces:

  1. A per-phase summary table printed to stdout.
  2. A stacked bar chart (PNG) breaking down time per operation.
  3. A timeline scatter plot (PNG) showing operation latency over time.

Both images are saved next to the input file.

Dependencies:  only stdlib + matplotlib (optional for plots).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# vLLM's chunked-prefill scheduler dispatches the same logical request to
# LMCache once per prefill step, prepending the step index to the request_id
# (e.g. "0_<hash>", "1_<hash>", ...). Strip the prefix so all steps of one
# logical request aggregate under a single key in the per-request table.
_CHUNKED_PREFILL_PREFIX_RE = re.compile(r'^(\d+)_([0-9a-f]{16}(?:#\d+)?)$')


def normalize_req_id(req_id):
    if not req_id:
        return req_id
    m = _CHUNKED_PREFILL_PREFIX_RE.match(req_id)
    if m:
        return m.group(2)
    return req_id


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_phase_table(records: list[dict]) -> None:
    """Aggregate all phase timings across operations and print a summary."""
    phase_data: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for phase, ms in rec.get("phases", {}).items():
            phase_data[phase].append(ms)

    print()
    print("=" * 80)
    print("  LMCache KV-Cache Profile — Phase Summary")
    print("=" * 80)
    header = (
        f"  {'Phase':<45} {'N':>5} {'Avg ms':>9} "
        f"{'P50 ms':>9} {'P95 ms':>9} {'Max ms':>9} {'Total ms':>10}"
    )
    print(header)
    print("-" * 80)

    for phase in sorted(phase_data):
        vals = sorted(phase_data[phase])
        n = len(vals)
        avg = sum(vals) / n
        total = sum(vals)
        mx = max(vals)
        p50 = vals[n // 2]
        p95 = vals[int(n * 0.95)] if n >= 20 else mx
        print(
            f"  {phase:<45} {n:>5} {avg:>9.2f} "
            f"{p50:>9.2f} {p95:>9.2f} {mx:>9.2f} {total:>10.1f}"
        )
    print("=" * 80)


def print_op_table(records: list[dict]) -> None:
    """Print per-operation-type summary."""
    op_data: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        op = rec.get("op", "unknown")
        total = sum(rec.get("phases", {}).values())
        op_data[op].append(total)

    print()
    print("=" * 80)
    print("  LMCache KV-Cache Profile — Operation Summary")
    print("=" * 80)
    header = f"  {'Operation':<25} {'N':>5} {'Avg ms':>9} {'P50 ms':>9} {'Max ms':>9}"
    print(header)
    print("-" * 80)

    for op in sorted(op_data):
        vals = sorted(op_data[op])
        n = len(vals)
        avg = sum(vals) / n
        mx = max(vals)
        p50 = vals[n // 2]
        print(f"  {op:<25} {n:>5} {avg:>9.2f} {p50:>9.2f} {mx:>9.2f}")
    print("=" * 80)


def print_req_id_table(records: list[dict], top_n: int = 20) -> None:
    """Print a per-request summary keyed by the stable content-derived req_id.

    The server now stamps each request with a stable hash that threads
    through vLLM and LMCache, so profile records carry it as a top-level
    field. This aggregates total phase time per request and prints the top
    ``top_n`` slowest requests — the landing zone for outlier drill-down.

    Records bearing a vLLM chunked-prefill step prefix (e.g. ``0_<hash>``)
    are normalized to their bare hash so every prefill step of one logical
    request lands in the same row.
    """
    per_req: dict[str, float] = defaultdict(float)
    per_req_n: dict[str, int] = defaultdict(int)
    per_req_ops: dict[str, set] = defaultdict(set)

    normalized = 0
    for rec in records:
        raw = rec.get("req_id")
        if not raw:
            continue
        req_id = normalize_req_id(raw)
        if req_id != raw:
            normalized += 1
        total = sum(rec.get("phases", {}).values())
        per_req[req_id] += total
        per_req_n[req_id] += 1
        per_req_ops[req_id].add(rec.get("op", "unknown"))

    if not per_req:
        print()
        print("(no req_id-tagged records; older run without stable IDs?)")
        return

    ordered = sorted(per_req.items(), key=lambda kv: kv[1], reverse=True)

    print()
    print("=" * 96)
    print(f"  LMCache KV-Cache Profile — Per-Request Totals (top {min(top_n, len(ordered))})")
    print("=" * 96)
    print(f"  {'req_id':<36} {'Records':>8} {'Ops':<32} {'Total ms':>12}")
    print("-" * 96)
    for req_id, total in ordered[:top_n]:
        ops_str = ",".join(sorted(per_req_ops[req_id]))
        if len(ops_str) > 30:
            ops_str = ops_str[:27] + "..."
        print(f"  {req_id:<36} {per_req_n[req_id]:>8} {ops_str:<32} {total:>12.2f}")
    print("=" * 96)
    print(f"  Total unique req_ids seen: {len(per_req)}")
    if normalized:
        print(
            f"  ({normalized} chunked-prefill step records were collapsed "
            f"into their bare hash)"
        )


def plot_stacked_bar(records: list[dict], out_path: str) -> None:
    """Stacked bar chart showing average phase breakdown per operation type."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots.")
        return

    op_phases: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        op = rec.get("op", "unknown")
        for phase, ms in rec.get("phases", {}).items():
            short = phase.split(".", 1)[-1] if "." in phase else phase
            op_phases[op][short].append(ms)

    fig, axes = plt.subplots(
        1, len(op_phases), figsize=(6 * len(op_phases), 6), squeeze=False
    )

    for ax, (op, phases) in zip(axes[0], sorted(op_phases.items())):
        labels = sorted(phases.keys())
        avgs = [sum(phases[l]) / len(phases[l]) for l in labels]
        bottom = 0.0
        for label, avg in zip(labels, avgs):
            ax.bar(op, avg, bottom=bottom, label=label)
            bottom += avg
        ax.set_ylabel("Avg time (ms)")
        ax.set_title(f"{op} breakdown")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved stacked bar chart -> {out_path}")


def plot_timeline(records: list[dict], out_path: str) -> None:
    """Scatter plot of per-operation latency over wall-clock time."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    op_colors = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for rec in records:
        op = rec.get("op", "unknown")
        ts = rec.get("wall_ts", 0)
        total_ms = sum(rec.get("phases", {}).values())

        if op not in op_colors:
            op_colors[op] = color_cycle[len(op_colors) % len(color_cycle)]
        ax.scatter(ts, total_ms, c=op_colors[op], label=op, alpha=0.6, s=20)

    handles = {}
    for op, c in op_colors.items():
        handles[op] = plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=op)
    ax.legend(handles=list(handles.values()), fontsize=8)

    ax.set_xlabel("Wall time (unix)")
    ax.set_ylabel("Total latency (ms)")
    ax.set_title("KV Cache operation latency over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved timeline plot    -> {out_path}")


def write_summary_txt(records: list[dict], out_path: str) -> None:
    """Write the op + phase tables to a text file (same content as stdout)."""
    import io

    buf = io.StringIO()
    _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

    def file_print(*args, **kwargs):
        kwargs["file"] = buf
        _orig_print(*args, **kwargs)

    # Temporarily redirect print output into buf
    import builtins

    old_print = builtins.print
    builtins.print = file_print
    try:
        print_op_table(records)
        print_phase_table(records)
        print_req_id_table(records)
    finally:
        builtins.print = old_print

    Path(out_path).write_text(buf.getvalue())
    old_print(f"  Saved text summary     -> {out_path}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "lmcache_profile.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(path).exists():
        print(f"Error: {path} not found.")
        print("Run your workload with LMCACHE_PROFILE=1 first.")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print("No records found in the profile file.")
        sys.exit(1)

    print(f"Loaded {len(records)} profiling records from {path}")

    print_op_table(records)
    print_phase_table(records)
    print_req_id_table(records)

    stem = Path(path).stem
    parent = Path(out_dir) if out_dir else Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)

    write_summary_txt(records, str(parent / f"{stem}_summary.txt"))
    plot_stacked_bar(records, str(parent / f"{stem}_breakdown.png"))
    plot_timeline(records, str(parent / f"{stem}_timeline.png"))


if __name__ == "__main__":
    main()
