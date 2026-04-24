#!/usr/bin/env python3
"""
Thesis evidence figures: small-request speedup variance analysis.

Produces publication-quality plots that substantiate the claims in
findings_small_request_variance.md:

  Fig 1  Speedup distribution by token bucket (violin + box)
  Fig 2  Same, filtered to high-hit requests (hit_ratio >= 0.5)
  Fig 3  Fragmentation rate by token bucket
  Fig 4  Per-call retrieve_layer phase breakdown (blend vs no-blend)
  Fig 5  Speedup vs total_tokens colored by fragmentation class

Usage:
    python3 analyze_variance.py \
        --blend-profile <blend.jsonl> \
        --noblend-profile <noblend.jsonl> \
        --comparison-csv <lmcache_comparison_data.csv> \
        --blend-err <blend.err> \
        --output-dir <dir>
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNKED_PREFILL_RE = re.compile(r'^(\d+)_(\d+|[0-9a-f]{16}(?:#\d+)?)$')


def norm(rid):
    if not rid:
        return rid
    m = _CHUNKED_PREFILL_RE.match(rid)
    return m.group(2) if m else rid


def set_plot_fonts():
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 24,
    })


TOKEN_BUCKETS = [
    ("<2k",    0,     2000),
    ("2-5k",   2000,  5000),
    ("5-10k",  5000,  10000),
    ("10-30k", 10000, 30000),
    (">30k",   30000, float("inf")),
]


def bucket_label(tok):
    for label, lo, hi in TOKEN_BUCKETS:
        if lo <= tok < hi:
            return label
    return ">30k"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_comparison_csv(path):
    """Returns list of dicts with keys: req_id, blend_ttft, noblend_ttft, speedup, hit_ratio."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "req_id": r["Request_ID"],
                "blend_ttft": float(r[next(k for k in r if "Blend" in k and "TTFT" in k)]),
                "noblend_ttft": float(r[next(k for k in r if "No Blend" in k and "TTFT" in k)] or 0),
                "speedup": float(r["Speedup"]),
                "hit_ratio": float(r.get("Hit_Ratio", 0) or 0),
            })
    return rows


def load_profile_ops(path):
    """Returns (per_req_ops, per_req_tokens, per_call_phases).

    per_req_ops:    {req_id: int}  -- number of retrieve_layer calls
    per_req_tokens: {req_id: int}  -- num_tokens field from the record
    per_call_phases: list of dicts, one per retrieve_layer record, each
                     mapping phase_group -> ms. Groups: token_db, storage_get,
                     gpu_onload, other.
    """
    per_req_ops = Counter()
    per_req_tokens = {}
    per_call_phases_blend = []  # [{group: ms, ...}, ...]
    per_call_req_id = []

    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            rid = norm(r.get("req_id") or "")
            if not rid:
                continue
            if r.get("op") == "retrieve_layer":
                per_req_ops[rid] += 1
                per_req_tokens[rid] = r.get("num_tokens", 0)
                phases = r.get("phases", {})
                grouped = {"token_db": 0.0, "storage_get": 0.0, "gpu_onload": 0.0, "other": 0.0}
                for ph, ms in phases.items():
                    if "token_db" in ph:
                        grouped["token_db"] += ms
                    elif "storage_get" in ph:
                        grouped["storage_get"] += ms
                    elif "gpu_onload" in ph:
                        grouped["gpu_onload"] += ms
                    else:
                        grouped["other"] += ms
                per_call_phases_blend.append(grouped)
                per_call_req_id.append(rid)

    return per_req_ops, per_req_tokens, per_call_phases_blend, per_call_req_id


def load_cache_data(err_path):
    """Parse .err to get {req_id: (total_tokens, hit_tokens)}. Normalizes prefixes."""
    pat = re.compile(r'Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: \d+')
    data = {}
    seen = set()
    with open(err_path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            rid = norm(m.group(1).strip())
            if rid in seen:
                continue
            seen.add(rid)
            data[rid] = (int(m.group(2)), int(m.group(3)))
    return data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_speedup_by_bucket(rows, cache_data, output_dir, suffix="", title_extra="",
                          hit_ratio_filter=None):
    """Fig 1/2: Violin + box plot of speedup distribution by token bucket."""
    set_plot_fonts()
    fig, ax = plt.subplots(figsize=(12, 7))

    bucket_data = {label: [] for label, _, _ in TOKEN_BUCKETS}
    for r in rows:
        tok = cache_data.get(r["req_id"], (0, 0))[0]
        if tok == 0:
            continue
        if hit_ratio_filter is not None and r["hit_ratio"] < hit_ratio_filter:
            continue
        bl = bucket_label(tok)
        bucket_data[bl].append(r["speedup"])

    labels = [label for label, _, _ in TOKEN_BUCKETS if bucket_data[label]]
    data = [bucket_data[l] for l in labels]
    counts = [len(d) for d in data]

    if not data:
        plt.close(fig)
        return

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#7fcdbb')
        pc.set_alpha(0.5)

    bp = ax.boxplot(data, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='#2c7fb8', alpha=0.7),
                    medianprops=dict(color='orange', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.4))

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, counts)])
    ax.set_xlabel("Total Prompt Tokens")
    ax.set_ylabel("Speedup (No Blend TTFT / Blend TTFT)")
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Breakeven (1.0x)')
    ax.legend(loc='upper right')

    title = "Speedup Distribution by Prompt Length"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        plt.savefig(output_dir / f"speedup_by_bucket{suffix}{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig speedup_by_bucket{suffix}: {len(labels)} buckets, {sum(counts)} points")


def fig_fragmentation_rate(blend_ops, blend_tokens, output_dir):
    """Fig 3: Fragmentation rate (% 36-op) by token bucket."""
    set_plot_fonts()
    fig, ax = plt.subplots(figsize=(10, 6))

    bucket_counts = {label: {"4": 0, "36": 0, "other": 0} for label, _, _ in TOKEN_BUCKETS}
    for rid, ops in blend_ops.items():
        tok = blend_tokens.get(rid, 0)
        if tok == 0:
            continue
        bl = bucket_label(tok)
        if ops == 4:
            bucket_counts[bl]["4"] += 1
        elif ops == 36:
            bucket_counts[bl]["36"] += 1
        else:
            bucket_counts[bl]["other"] += 1

    labels = [l for l, _, _ in TOKEN_BUCKETS
              if bucket_counts[l]["4"] + bucket_counts[l]["36"] + bucket_counts[l]["other"] > 0]
    frag_pcts = []
    nonfrag_pcts = []
    other_pcts = []
    totals = []
    for l in labels:
        c = bucket_counts[l]
        total = c["4"] + c["36"] + c["other"]
        totals.append(total)
        frag_pcts.append(100 * c["36"] / total if total else 0)
        nonfrag_pcts.append(100 * c["4"] / total if total else 0)
        other_pcts.append(100 * c["other"] / total if total else 0)

    x = np.arange(len(labels))
    w = 0.6
    ax.bar(x, nonfrag_pcts, w, label='1 prefill step (4 ops)', color='#2c7fb8', alpha=0.8)
    ax.bar(x, frag_pcts, w, bottom=nonfrag_pcts, label='9 prefill steps (36 ops)', color='#d95f02', alpha=0.8)
    if any(p > 0 for p in other_pcts):
        bottoms = [a + b for a, b in zip(nonfrag_pcts, frag_pcts)]
        ax.bar(x, other_pcts, w, bottom=bottoms, label='Other', color='#999999', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={t})" for l, t in zip(labels, totals)])
    ax.set_xlabel("Total Prompt Tokens")
    ax.set_ylabel("Percentage of Requests")
    ax.set_title("Chunked-Prefill Fragmentation Rate by Prompt Length", fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Annotate fragmentation percentages
    for i, fp in enumerate(frag_pcts):
        if fp > 3:
            ax.text(i, nonfrag_pcts[i] + fp / 2, f"{fp:.0f}%",
                    ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        plt.savefig(output_dir / f"fragmentation_rate{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig fragmentation_rate: {len(labels)} buckets, {sum(totals)} requests")


def fig_phase_breakdown(blend_phases, noblend_phases, output_dir):
    """Fig 4: Grouped bar chart of per-call phase cost (blend vs no-blend)."""
    set_plot_fonts()
    fig, ax = plt.subplots(figsize=(10, 7))

    groups = ["token_db", "storage_get", "gpu_onload"]
    group_labels = ["Token DB\nLookup", "Storage Get\n(CPU/Disk)", "GPU Onload\n(KV Transfer)"]

    def means(phase_list, grp):
        vals = [p[grp] for p in phase_list if p[grp] > 0]
        return np.mean(vals) if vals else 0.0

    blend_means = [means(blend_phases, g) for g in groups]
    noblend_means = [means(noblend_phases, g) for g in groups]

    x = np.arange(len(groups))
    w = 0.35
    bars_b = ax.bar(x - w/2, blend_means, w, label='Blend', color='#d95f02', alpha=0.85)
    bars_n = ax.bar(x + w/2, noblend_means, w, label='No Blend', color='#2c7fb8', alpha=0.85)

    # Annotate values
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    for bar in bars_n:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Add total bars as a separate annotation
    blend_total = sum(blend_means)
    noblend_total = sum(noblend_means)
    ax.text(0.98, 0.95,
            f"Total per call:\n  Blend: {blend_total:.1f} ms\n  No Blend: {noblend_total:.1f} ms\n"
            f"  Ratio: {blend_total/noblend_total:.2f}x",
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean Time per retrieve_layer Call (ms)")
    ax.set_title("Per-Call Phase Cost Breakdown: Blend vs No Blend", fontweight='bold')
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        plt.savefig(output_dir / f"phase_breakdown{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig phase_breakdown: blend n={len(blend_phases)}, noblend n={len(noblend_phases)}")


def fig_speedup_vs_tokens_by_fragmentation(rows, cache_data, blend_ops, output_dir):
    """Fig 5: Scatter of speedup vs total_tokens, colored by fragmentation class."""
    set_plot_fonts()
    fig, ax = plt.subplots(figsize=(13, 7))

    x4, y4 = [], []
    x36, y36 = [], []
    xother, yother = [], []

    for r in rows:
        rid = r["req_id"]
        tok = cache_data.get(rid, (0, 0))[0]
        if tok == 0:
            continue
        ops = blend_ops.get(rid, 0)
        if ops == 4:
            x4.append(tok)
            y4.append(r["speedup"])
        elif ops == 36:
            x36.append(tok)
            y36.append(r["speedup"])
        else:
            xother.append(tok)
            yother.append(r["speedup"])

    if x4:
        ax.scatter(x4, y4, alpha=0.55, s=40, c='#2c7fb8', edgecolors='none',
                   label=f'1 prefill step (n={len(x4)})', zorder=3)
    if x36:
        ax.scatter(x36, y36, alpha=0.7, s=55, c='#d95f02', edgecolors='black', linewidths=0.5,
                   label=f'9 prefill steps (n={len(x36)})', marker='D', zorder=4)
    if xother:
        ax.scatter(xother, yother, alpha=0.3, s=30, c='gray', edgecolors='none',
                   label=f'Other (n={len(xother)})', zorder=2)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Breakeven')
    ax.set_xlabel("Total Prompt Tokens")
    ax.set_ylabel("Speedup (No Blend TTFT / Blend TTFT)")
    ax.set_title("Speedup vs Prompt Length, Colored by Fragmentation Class", fontweight='bold')
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        plt.savefig(output_dir / f"speedup_vs_tokens_fragmentation{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig speedup_vs_tokens_fragmentation: 4-op={len(x4)}, 36-op={len(x36)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thesis evidence figures: small-request speedup variance analysis"
    )
    parser.add_argument("--blend-profile", required=True,
                        help="Path to blend run LMCache profiler JSONL")
    parser.add_argument("--noblend-profile", required=True,
                        help="Path to no-blend run LMCache profiler JSONL")
    parser.add_argument("--comparison-csv", required=True,
                        help="Path to lmcache_comparison_data.csv from compare_lmcache_runs.py")
    parser.add_argument("--blend-err", required=True,
                        help="Path to blend run .err file (for total_tokens)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save figures")
    parser.add_argument("--speedup-cutoff", type=float, default=5.0,
                        help="Hard speedup cutoff for no-outlier versions "
                             "(requests with speedup > cutoff are excluded). "
                             "Default: 5.0")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(exist_ok=True)

    print("Loading data...")
    rows = load_comparison_csv(Path(args.comparison_csv).expanduser())
    print(f"  CSV: {len(rows)} rows")

    cache_data = load_cache_data(Path(args.blend_err).expanduser())
    print(f"  Cache data (.err): {len(cache_data)} logical requests")

    blend_ops, blend_tokens, blend_phases, _ = load_profile_ops(
        Path(args.blend_profile).expanduser()
    )
    print(f"  Blend profile: {sum(blend_ops.values())} retrieve_layer records, "
          f"{len(blend_ops)} unique req_ids")

    noblend_ops, _, noblend_phases, _ = load_profile_ops(
        Path(args.noblend_profile).expanduser()
    )
    print(f"  No-blend profile: {sum(noblend_ops.values())} retrieve_layer records, "
          f"{len(noblend_ops)} unique req_ids")

    # Enrich rows with total_tokens from cache_data
    for r in rows:
        r["total_tokens"] = cache_data.get(r["req_id"], (0, 0))[0]

    print("\nGenerating figures...")

    # Fig 1: Speedup distribution by token bucket (all requests)
    fig_speedup_by_bucket(
        rows, cache_data, output_dir,
        suffix="", title_extra="(all requests)"
    )

    # Fig 2: Same but filtered to hit_ratio >= 0.5
    fig_speedup_by_bucket(
        rows, cache_data, output_dir,
        suffix="_high_hit",
        title_extra="(cache hit ratio >= 50% only)",
        hit_ratio_filter=0.5,
    )

    # Fig 3: Fragmentation rate by token bucket
    fig_fragmentation_rate(blend_ops, blend_tokens, output_dir)

    # Fig 4: Per-call phase cost breakdown
    fig_phase_breakdown(blend_phases, noblend_phases, output_dir)

    # Fig 5: Speedup vs total_tokens colored by fragmentation class
    fig_speedup_vs_tokens_by_fragmentation(rows, cache_data, blend_ops, output_dir)

    # --- No-outlier versions (speedup cutoff) ---
    cutoff = args.speedup_cutoff
    rows_filtered = [r for r in rows if r["speedup"] <= cutoff]
    dropped = len(rows) - len(rows_filtered)
    no_dir = output_dir / "no_outlier"
    no_dir.mkdir(exist_ok=True)

    print(f"\nGenerating no-outlier figures (speedup <= {cutoff}x, "
          f"dropped {dropped}/{len(rows)} requests)...")

    fig_speedup_by_bucket(
        rows_filtered, cache_data, no_dir,
        suffix="_no_outlier",
        title_extra=f"(all requests, speedup <= {cutoff}x)",
    )

    fig_speedup_by_bucket(
        rows_filtered, cache_data, no_dir,
        suffix="_high_hit_no_outlier",
        title_extra=f"(hit ratio >= 50%, speedup <= {cutoff}x)",
        hit_ratio_filter=0.5,
    )

    fig_speedup_vs_tokens_by_fragmentation(
        rows_filtered, cache_data, blend_ops, no_dir,
    )

    # Print numerical summaries for the thesis table
    def _print_summaries(label_prefix, data_rows):
        for label, lo, hi in TOKEN_BUCKETS:
            bucket_rows = [r for r in data_rows if lo <= r["total_tokens"] < hi]
            if not bucket_rows:
                continue
            speeds = [r["speedup"] for r in bucket_rows]
            n = len(speeds)
            speeds_s = sorted(speeds)
            mean = sum(speeds) / n
            var = sum((s - mean) ** 2 for s in speeds) / max(n - 1, 1)
            stddev = var ** 0.5
            iqr = np.percentile(speeds, 75) - np.percentile(speeds, 25)
            above1 = sum(1 for s in speeds if s > 1.0)
            print(f"\n  {label_prefix}{label} (n={n}):")
            print(f"    mean={mean:.3f}  median={speeds_s[n//2]:.3f}  "
                  f"std={stddev:.3f}  IQR={iqr:.3f}")
            print(f"    min={speeds_s[0]:.3f}  max={speeds_s[-1]:.3f}")
            print(f"    above breakeven: {above1}/{n} ({100*above1/n:.0f}%)")

            frag = [r["speedup"] for r in bucket_rows if blend_ops.get(r["req_id"], 0) == 36]
            nonfrag = [r["speedup"] for r in bucket_rows if blend_ops.get(r["req_id"], 0) == 4]
            if frag:
                print(f"    36-op (fragmented): n={len(frag)}  "
                      f"mean={sum(frag)/len(frag):.3f}  std={np.std(frag):.3f}")
            if nonfrag:
                print(f"     4-op (normal):     n={len(nonfrag)}  "
                      f"mean={sum(nonfrag)/len(nonfrag):.3f}  std={np.std(nonfrag):.3f}")

    print("\n" + "=" * 70)
    print("  NUMERICAL SUMMARIES FOR THESIS (all requests)")
    print("=" * 70)
    _print_summaries("", rows)

    print("\n" + "=" * 70)
    print(f"  NUMERICAL SUMMARIES FOR THESIS (no outliers, speedup <= {cutoff}x)")
    print("=" * 70)
    _print_summaries("[filtered] ", rows_filtered)

    print("\n" + "=" * 70)
    print("Done. Figures saved to", output_dir)
    print(f"  No-outlier figures saved to {no_dir}")


if __name__ == "__main__":
    main()
