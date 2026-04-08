#!/usr/bin/env python3
"""
Script to compare two LMCache runs, using one as baseline for TTFT comparison.
Extracts cache data and TTFT from both runs and creates comparison plots.
Skips cache-related plots if cache data is not available.
"""

import json
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse
from scipy.stats import gaussian_kde

# Default polynomial degree for 3D plot fitting
POLYNOMIAL_DEGREE = 3

def set_plot_fonts():
    """Set larger font sizes for all plot elements."""
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 28,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 28
    })

def extract_cache_data(err_file_path):
    """
    Extract request ID, total tokens, and LMCache hit tokens from the error log file.

    The server stamps each request with a stable, content-derived ``req_id``
    that flows into the LMCache ``Reqid:`` log line. We preserve it as the
    primary join key and also return a positional view so legacy logs keyed
    by order of appearance still work.

    Args:
        err_file_path (str): Path to the lmcache.err file

    Returns:
        tuple: (cache_data_by_id, cache_data_by_pos)
    """
    cache_data_by_id = {}
    cache_data_by_pos = {}

    pattern = r'Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: \d+'
    pat = re.compile(pattern)

    request_counter = 0

    try:
        with open(err_file_path, 'r') as f:
            for line in f:
                match = pat.search(line)
                if match:
                    req_id_str = match.group(1).strip()
                    total_tokens = int(match.group(2))
                    hit_tokens = int(match.group(3))
                    record = (total_tokens, hit_tokens)
                    cache_data_by_id[req_id_str] = record
                    cache_data_by_pos[request_counter] = record
                    request_counter += 1
    except FileNotFoundError:
        print(f"Warning: Could not find file {err_file_path}")
        return {}, {}
    except Exception as e:
        print(f"Warning: Error reading {err_file_path}: {e}")
        return {}, {}

    if cache_data_by_pos:
        print(f"Extracted cache data for {len(cache_data_by_pos)} requests from {err_file_path}"
              f" ({len(cache_data_by_id)} unique req_ids)")
    return cache_data_by_id, cache_data_by_pos


def extract_prefix_cache_data(err_file_path):
    """
    Extract prefix cache hit tokens from the error log file.

    Format: ``[INFO] Prefix cache hit tokens: X``. The stats line itself
    has no req_id, so we rely on positional alignment. Returned tuple
    mirrors :func:`extract_cache_data` for call-site consistency; the
    ``by_id`` view is always empty.

    Args:
        err_file_path (str): Path to the .err file

    Returns:
        tuple: ({}, {position: prefix_hit_tokens})
    """
    prefix_cache_data_by_pos = {}

    pattern = r'\[INFO\] Prefix cache hit tokens: (\d+)'
    pat = re.compile(pattern)

    request_counter = 0

    try:
        with open(err_file_path, 'r') as f:
            for line in f:
                match = pat.search(line)
                if match:
                    prefix_hit_tokens = int(match.group(1))
                    prefix_cache_data_by_pos[request_counter] = prefix_hit_tokens
                    request_counter += 1
    except FileNotFoundError:
        print(f"Warning: Could not find file {err_file_path}")
        return {}, {}
    except Exception as e:
        print(f"Warning: Error reading {err_file_path}: {e}")
        return {}, {}

    if prefix_cache_data_by_pos:
        print(f"Extracted prefix cache data for {len(prefix_cache_data_by_pos)} requests from {err_file_path}")
    return {}, prefix_cache_data_by_pos

def extract_ttft_data(out_file_path):
    """
    Extract TTFT from the output log file.

    Captures the optional ``(req_id=...)`` marker the server now appends to
    ``[MOCK] Request N completed`` and ``[INFO] TTFT: ... seconds`` so
    downstream joins can key on the stable content hash. Falls back to a
    positional counter for logs that predate the change.

    Args:
        out_file_path (str): Path to the .out file

    Returns:
        tuple: (ttft_data_by_id, ttft_data_by_pos)
    """
    ttft_data_by_id = {}
    ttft_data_by_pos = {}

    primary_pattern = (
        r'@@@@@ Response # of choices: (\d+)\s*'
        r'\[MOCK\] Request (\d+) completed(?:\s*\(req_id=([^)]+)\))?:'
        r'.*?TTFT: ([0-9.]+)s\s*Speedup: ([0-9.]+)x'
    )
    primary_pat = re.compile(primary_pattern, re.DOTALL)

    fallback_pattern = r'\[INFO\] TTFT: ([0-9.]+) seconds(?:\s*\(req_id=([^)]+)\))?'
    fallback_pat = re.compile(fallback_pattern)

    request_counter = 0

    try:
        with open(out_file_path, 'r') as f:
            content = f.read()

            primary_matches = primary_pat.findall(content)
            used_fallback = False

            if primary_matches:
                for match in primary_matches:
                    req_id_str = (match[2] or "").strip() or None
                    ttft = float(match[3])
                    if req_id_str:
                        ttft_data_by_id[req_id_str] = ttft
                    ttft_data_by_pos[request_counter] = ttft
                    request_counter += 1
            else:
                used_fallback = True
                fallback_matches = fallback_pat.findall(content)
                for match in fallback_matches:
                    ttft = float(match[0])
                    req_id_str = (match[1] or "").strip() or None
                    if req_id_str:
                        ttft_data_by_id[req_id_str] = ttft
                    ttft_data_by_pos[request_counter] = ttft
                    request_counter += 1

    except FileNotFoundError:
        print(f"Error: Could not find file {out_file_path}")
        return {}, {}
    except Exception as e:
        print(f"Error reading {out_file_path}: {e}")
        return {}, {}

    print(f"Extracted TTFT data for {len(ttft_data_by_pos)} requests from {out_file_path}"
          f" ({len(ttft_data_by_id)} unique req_ids)")
    if used_fallback:
        print(f"  Note: Used fallback pattern")
    return ttft_data_by_id, ttft_data_by_pos


def _select_join_view(left, right):
    """Pick the strongest join between two ``(by_id, by_pos)`` tuples.

    Returns ``(left_view, right_view, mode)``. ``mode`` is ``"req_id"`` if
    both sides have overlapping stable hashes, else ``"position"``.
    """
    left_by_id, left_by_pos = left
    right_by_id, right_by_pos = right

    if left_by_id and right_by_id:
        if set(left_by_id) & set(right_by_id):
            return left_by_id, right_by_id, "req_id"
        print("Warning: both sides have req_ids but they do not overlap; "
              "falling back to positional join.")
    return left_by_pos, right_by_pos, "position"


def _sorted_common_ids(view_a, view_b):
    common = set(view_a.keys()) & set(view_b.keys())
    # Ints come first (positional), then strings, each sorted internally.
    return sorted(common, key=lambda k: (isinstance(k, str), k))


def correlate_lmcache_data(cache_data, ttft_data):
    """
    Correlate cache data with TTFT data.

    Both arguments are ``(by_id, by_pos)`` tuples. Joins on stable req_id
    strings when both sides have them; otherwise falls back to positional.

    Returns:
        tuple: (hit_ratios, request_ids, total_tokens_list, hit_tokens_list, ttfts)
    """
    hit_ratios = []
    request_ids = []
    total_tokens_list = []
    hit_tokens_list = []
    ttfts = []

    cache_by_id, cache_by_pos = cache_data
    ttft_by_id, ttft_by_pos = ttft_data

    # If no cache data, just return TTFT data
    if not cache_by_id and not cache_by_pos:
        # Prefer req_id keys if present
        view = ttft_by_id if ttft_by_id else ttft_by_pos
        for req_id in sorted(view.keys(), key=lambda k: (isinstance(k, str), k)):
            ttfts.append(view[req_id])
            request_ids.append(req_id)
        return [], request_ids, [], [], ttfts

    cache_view, ttft_view, mode = _select_join_view(cache_data, ttft_data)
    common_ids = _sorted_common_ids(cache_view, ttft_view)

    if not common_ids:
        print("Warning: No matching request IDs found between cache and TTFT data")
        return [], [], [], [], []

    print(f"Found {len(common_ids)} matching request IDs (join mode: {mode})")

    for req_id in common_ids:
        total_tokens, hit_tokens = cache_view[req_id]
        ttft = ttft_view[req_id]

        hit_ratio = hit_tokens / total_tokens if total_tokens > 0 else 0

        hit_ratios.append(hit_ratio)
        request_ids.append(req_id)
        total_tokens_list.append(total_tokens)
        hit_tokens_list.append(hit_tokens)
        ttfts.append(ttft)

    return hit_ratios, request_ids, total_tokens_list, hit_tokens_list, ttfts

def calculate_speedup_comparison(run1_ttft_data, run2_ttft_data):
    """
    Calculate speedup by comparing TTFT between two runs.
    Speedup = run2_ttft / run1_ttft (higher means run1 is faster)

    Both arguments are ``(by_id, by_pos)`` tuples from :func:`extract_ttft_data`.
    Joins on stable req_id strings whenever both runs have them so a slow
    request in run1 can be directly compared to the same request in run2
    even if the two runs completed their requests in different orders.

    Returns:
        tuple: (request_ids, speedups, run1_ttfts, run2_ttfts)
    """
    run1_view, run2_view, mode = _select_join_view(run1_ttft_data, run2_ttft_data)
    common_ids = _sorted_common_ids(run1_view, run2_view)

    if not common_ids:
        print("Warning: No matching request IDs found between the two runs")
        return [], [], [], []

    print(f"Found {len(common_ids)} matching request IDs for TTFT comparison (join mode: {mode})")

    request_ids = []
    speedups = []
    run1_ttfts = []
    run2_ttfts = []

    for req_id in common_ids:
        run1_ttft = run1_view[req_id]
        run2_ttft = run2_view[req_id]

        # Calculate speedup: run2_ttft / run1_ttft
        # Higher values mean run1 is faster (better)
        if run1_ttft > 0 and run2_ttft > 0:
            speedup = run2_ttft / run1_ttft
        else:
            continue

        request_ids.append(req_id)
        speedups.append(speedup)
        run1_ttfts.append(run1_ttft)
        run2_ttfts.append(run2_ttft)

    return request_ids, speedups, run1_ttfts, run2_ttfts

def create_ttft_comparison_scatter(run1_ttfts, run2_ttfts, request_ids,
                                   run1_label="Run 1", run2_label="Run 2 (Baseline)",
                                   output_file=None):
    """Create scatter plot comparing TTFT between two runs."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    plt.scatter(run1_ttfts, run2_ttfts, alpha=0.7, s=50, c='blue', edgecolors='black')
    
    # Add diagonal line (y=x)
    max_ttft = max(max(run1_ttfts), max(run2_ttfts))
    min_ttft = min(min(run1_ttfts), min(run2_ttfts))
    plt.plot([min_ttft, max_ttft], [min_ttft, max_ttft], 'r--', alpha=0.8, linewidth=2, 
            label='Equal TTFT (y=x)')
    
    # Add trend line
    if len(run1_ttfts) > 1:
        z = np.polyfit(run1_ttfts, run2_ttfts, 1)
        p = np.poly1d(z)
        plt.plot(run1_ttfts, p(run1_ttfts), "g--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    plt.xlabel(f'{run1_label} TTFT (seconds)', fontsize=24)
    plt.ylabel(f'{run2_label} TTFT (seconds)', fontsize=24)
    plt.title(f'TTFT Comparison: {run1_label} vs {run2_label}', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"TTFT comparison scatter plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

def create_speedup_vs_hit_ratio_plot(speedups, hit_ratios, request_ids, output_file=None):
    """Create scatter plot of speedup vs hit ratio."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    plt.scatter(hit_ratios, speedups, alpha=0.7, s=50, c='darkgreen', edgecolors='black')
    
    # Add trend line
    if len(hit_ratios) > 1:
        z = np.polyfit(hit_ratios, speedups, 1)
        p = np.poly1d(z)
        plt.plot(hit_ratios, p(hit_ratios), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    # Add horizontal line at speedup = 1.0 (no improvement)
    plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.7, linewidth=2, 
               label='No improvement (1.0x)')
    
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=24)
    plt.ylabel('Speedup (Baseline TTFT / Run TTFT)', fontsize=24)
    plt.title('Speedup vs Cache Hit Ratio', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Speedup vs hit ratio plot saved to {output_file}")
    else:
        plt.show()
    plt.close()


def create_speedup_vs_hit_ratio_log_plot(speedups, hit_ratios, request_ids, output_file=None):
    """Create scatter plot of speedup vs hit ratio with log-scaled y axis."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))

    plt.scatter(hit_ratios, speedups, alpha=0.7, s=50, c='darkgreen', edgecolors='black')

    if len(hit_ratios) > 1:
        z = np.polyfit(hit_ratios, speedups, 1)
        p = np.poly1d(z)
        plt.plot(hit_ratios, p(hit_ratios), "r--", alpha=0.8, linewidth=2,
                label=f'Trend line (slope: {z[0]:.3f})')

    plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.7, linewidth=2,
               label='No improvement (1.0x)')

    plt.yscale('log')
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=24)
    plt.ylabel('Speedup (Baseline TTFT / Run TTFT) (log scale)', fontsize=24)
    plt.title('Speedup vs Cache Hit Ratio (Log Scale)', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=20)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Speedup vs hit ratio (log) plot saved to {output_file}")
    else:
        plt.show()
    plt.close()


def create_speedup_vs_extra_hit_ratio_plot(speedups, extra_hit_ratios, request_ids,
                                           run1_label="Run 1", run2_label="Run 2",
                                           output_file=None):
    """
    Create scatter plot of speedup vs extra cache hit ratio.
    
    Extra hit ratio = (blend_hit_tokens - prefix_hit_tokens) / total_tokens
    This shows how much additional cache benefit blending provides over standard prefix caching.
    
    Args:
        speedups: List of speedup values (run2_ttft / run1_ttft)
        extra_hit_ratios: List of extra hit ratios ((blend_hits - prefix_hits) / total)
        request_ids: List of request IDs
        run1_label: Label for run1 (the blending run)
        run2_label: Label for run2 (the prefix cache baseline)
        output_file: Path to save the plot (optional)
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    plt.scatter(extra_hit_ratios, speedups, alpha=0.7, s=50, c='purple', edgecolors='black')
    
    # Add trend line
    if len(extra_hit_ratios) > 1:
        z = np.polyfit(extra_hit_ratios, speedups, 1)
        p = np.poly1d(z)
        x_sorted = sorted(extra_hit_ratios)
        plt.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    # Add horizontal line at speedup = 1.0 (no improvement)
    plt.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
               label='No improvement (1.0x)')
    
    # Add vertical line at x = 0 (no extra cache benefit)
    plt.axvline(x=0.0, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label='No extra cache benefit (0%)')
    
    plt.xlabel('Extra Cache Hit Ratio\n(Blend Hit - Prefix Hit) / Total Tokens', fontsize=24)
    plt.ylabel(f'Speedup ({run2_label} / {run1_label})', fontsize=24)
    plt.title(f'Speedup vs Extra Cache Benefit from Blending', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18, loc='best')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Speedup vs extra hit ratio plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

def create_speedup_histogram(speedups, run1_label="Run 1", run2_label="Run 2 (Baseline)", output_file=None):
    """Create histogram of speedup values."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    n, bins, patches = plt.hist(speedups, bins=30, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black', linewidth=1.2)
    
    # Create smoothed distribution curve
    if len(speedups) > 1:
        x_range = np.linspace(min(speedups), max(speedups), 200)
        kde = gaussian_kde(speedups)
        density = kde(x_range)
        plt.plot(x_range, density, 'r-', linewidth=3, label='Smoothed Distribution')
    
    # Add statistics
    mean_speedup = np.mean(speedups)
    median_speedup = np.median(speedups)
    
    plt.axvline(mean_speedup, color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_speedup:.2f}x')
    plt.axvline(median_speedup, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_speedup:.2f}x')
    plt.axvline(1.0, color='blue', linestyle='-', linewidth=2, alpha=0.7,
               label='No improvement (1.0x)')
    
    plt.xlabel(f'Speedup ({run2_label} / {run1_label})', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.title(f'Speedup Distribution ({run2_label} / {run1_label})', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Speedup histogram saved to {output_file}")
    else:
        plt.show()
    plt.close()

def detect_outliers(speedups, threshold_multiplier=1.5, include_under=False):
    """
    Detect speedup outliers using the IQR method.

    When ``include_under`` is False (default for legacy callers) only the
    high tail — extremely good speedup — is returned. When True, the low
    tail (Q1 - k*IQR) is also returned, tagged so callers can separate
    over-performers from under-performers.

    Args:
        speedups (list): List of speedup values
        threshold_multiplier (float): Multiplier for IQR threshold (default: 1.5)
        include_under (bool): Also detect under-performers.

    Returns:
        If ``include_under`` is False: list of indices (upper-tail only).
        If ``include_under`` is True: list of (index, "over"|"under") tuples.
    """
    if len(speedups) < 4:
        return [] if not include_under else []

    speedups_array = np.array(speedups)
    q1 = np.percentile(speedups_array, 25)
    q3 = np.percentile(speedups_array, 75)
    iqr = q3 - q1

    upper_bound = q3 + threshold_multiplier * iqr
    lower_bound = q1 - threshold_multiplier * iqr

    upper_idx = np.where(speedups_array > upper_bound)[0].tolist()

    if not include_under:
        return upper_idx

    lower_idx = np.where(speedups_array < lower_bound)[0].tolist()
    # Sorted by index so the two classes interleave in natural request order.
    tagged = [(i, "over") for i in upper_idx] + [(i, "under") for i in lower_idx]
    tagged.sort(key=lambda t: t[0])
    return tagged

def filter_outliers(speedups, *arrays, threshold_multiplier=1.5):
    """
    Remove outlier entries (by speedup IQR) from speedups and any
    parallel arrays.  Returns the filtered versions in the same order.
    """
    outlier_set = set(detect_outliers(speedups, threshold_multiplier))
    if not outlier_set:
        return (speedups, *arrays)
    filtered = tuple(
        [v for i, v in enumerate(a) if i not in outlier_set]
        for a in (speedups, *arrays)
    )
    removed = len(outlier_set)
    print(f"  Removed {removed} outlier(s) for no-outlier plots "
          f"(threshold: {threshold_multiplier}x IQR)")
    return filtered


def load_profile_records(profile_file):
    """Load an LMCache profiler JSONL file into ``{req_id: [records]}``.

    Each record is a dict of the form emitted by ``lmcache/v1/profiling.py``:
    ``{"op": ..., "req_id": ..., "phases": {...}, "wall_ts": ..., ...}``.
    Records without a ``req_id`` field (orphaned phases, legacy logs) are
    collected under the ``None`` key so callers can still count them.
    """
    by_req_id = defaultdict(list)
    if profile_file is None:
        return by_req_id
    path = Path(profile_file).expanduser()
    if not path.exists():
        print(f"  Warning: profile file not found: {path}")
        return by_req_id
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_req_id[rec.get("req_id")].append(rec)
    except Exception as e:
        print(f"  Warning: failed to read profile file {path}: {e}")
    return by_req_id


def _phase_medians(records_by_req_id):
    """Build ``{(op, phase): median_ms}`` across every request in a run.

    Used as a comparison baseline so outlier phases get flagged with a
    ratio-to-median marker, turning the per-outlier report from "here are
    some numbers" into "here is where this request diverges from the rest".
    """
    buckets = defaultdict(list)
    for req_id, recs in records_by_req_id.items():
        if req_id is None:
            continue
        for rec in recs:
            op = rec.get("op", "unknown")
            for phase, ms in rec.get("phases", {}).items():
                buckets[(op, phase)].append(ms)
    return {k: float(np.median(v)) for k, v in buckets.items() if v}


def _aggregate_op_totals(records):
    """Group a list of records by ``op``, summing phase totals per group.

    Returns a list of ``(op, total_ms, record_count, phase_dict)`` tuples
    sorted by total descending so the biggest contributors land on top of
    the per-outlier report.
    """
    grouped = defaultdict(lambda: {"total": 0.0, "count": 0, "phases": defaultdict(float), "phase_counts": defaultdict(int)})
    for rec in records:
        op = rec.get("op", "unknown")
        phases = rec.get("phases", {})
        rec_total = sum(phases.values())
        grouped[op]["total"] += rec_total
        grouped[op]["count"] += 1
        for phase, ms in phases.items():
            grouped[op]["phases"][phase] += ms
            grouped[op]["phase_counts"][phase] += 1
    out = []
    for op, data in grouped.items():
        out.append((op, data["total"], data["count"], dict(data["phases"]), dict(data["phase_counts"])))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _format_phase_line(op, phase, summed_ms, phase_count, medians):
    """Build the phase row, annotating with a ratio-to-median marker.

    Because a single request can produce several records of the same op
    (e.g. per-layer store), we report ``sum`` and ``avg`` on the phase and
    compare ``avg`` to the run-wide median for that (op, phase).
    """
    avg = summed_ms / phase_count if phase_count else summed_ms
    median = medians.get((op, phase))
    marker = ""
    if median and median > 0:
        ratio = avg / median
        if ratio >= 1.5:
            marker = f"  [x{ratio:.2f} vs median {median:.2f}ms]   <<< HOT"
        elif ratio <= 0.5:
            marker = f"  [x{ratio:.2f} vs median {median:.2f}ms]   (cold)"
        else:
            marker = f"  (median {median:.2f}ms)"
    if phase_count > 1:
        return (f"      {phase:<40} sum={summed_ms:9.2f}ms "
                f"avg={avg:8.2f}ms (n={phase_count}){marker}")
    return f"      {phase:<40} {summed_ms:9.2f}ms{marker}"


def write_outlier_profile_report(
    tagged_outliers,
    request_ids,
    speedups,
    run1_ttfts,
    run2_ttfts,
    hit_ratios,
    total_tokens,
    hit_tokens,
    run1_records_by_id,
    run2_records_by_id,
    run1_label,
    run2_label,
    output_dir,
    iqr_threshold,
):
    """Write a per-outlier profiler breakdown keyed by stable req_id.

    Produces ``speedup_outliers_profile.txt`` in ``output_dir``. For each
    outlier (good or bad) the report prints the speedup / TTFT / cache
    summary followed by the op/phase breakdown from the supplied profile
    JSONL files. Phases whose average is >=1.5x the run-wide median get a
    ``<<< HOT`` marker so bottlenecks in outliers are immediately visible.
    """
    if not tagged_outliers:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "speedup_outliers_profile.txt"

    has_cache = bool(hit_ratios)

    run1_medians = _phase_medians(run1_records_by_id)
    run2_medians = _phase_medians(run2_records_by_id)

    run1_total_records = sum(len(v) for k, v in run1_records_by_id.items() if k is not None)
    run2_total_records = sum(len(v) for k, v in run2_records_by_id.items() if k is not None)
    run1_orphan = len(run1_records_by_id.get(None, []))
    run2_orphan = len(run2_records_by_id.get(None, []))

    over_count = sum(1 for _, cls in tagged_outliers if cls == "over")
    under_count = sum(1 for _, cls in tagged_outliers if cls == "under")

    with open(out_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("SPEEDUP OUTLIER PROFILE REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Speedup definition: {run2_label} TTFT / {run1_label} TTFT\n")
        f.write(f"IQR threshold multiplier: {iqr_threshold}\n")
        f.write(f"Over-performers (upper tail): {over_count}\n")
        f.write(f"Under-performers (lower tail): {under_count}\n\n")
        f.write(
            f"run1 profile records loaded: {run1_total_records}"
            f" (orphaned without req_id: {run1_orphan})\n"
        )
        f.write(
            f"run2 profile records loaded: {run2_total_records}"
            f" (orphaned without req_id: {run2_orphan})\n"
        )
        f.write("\n")
        f.write(
            "Phase rows annotated with '<<< HOT' are >=1.5x the run-wide "
            "median for that (op, phase) pair -- likely bottleneck driver.\n"
        )
        f.write(
            "Phase rows with 'sum/avg (n=...)' summarise multiple records of\n"
            "the same op type (e.g. per-layer store/retrieve) for this request.\n"
        )
        f.write("\n" + "=" * 100 + "\n\n")

        for idx, cls in tagged_outliers:
            req_id = request_ids[idx]
            ttft1 = run1_ttfts[idx]
            ttft2 = run2_ttfts[idx]
            speedup = speedups[idx]

            label = "OVER-PERFORMER" if cls == "over" else "UNDER-PERFORMER"
            f.write("-" * 100 + "\n")
            f.write(f"REQUEST: {req_id}    [{label}]\n")
            f.write("-" * 100 + "\n")
            f.write(f"  {run1_label} TTFT: {ttft1:.6f} s\n")
            f.write(f"  {run2_label} TTFT: {ttft2:.6f} s\n")
            f.write(f"  Speedup: {speedup:.4f}x  ({((speedup - 1.0) * 100):+.2f}% vs run2)\n")

            if has_cache and idx < len(hit_ratios):
                f.write(
                    f"  Cache hit ratio: {hit_ratios[idx]:.4f}"
                )
                if total_tokens and idx < len(total_tokens):
                    f.write(f"   total_tokens={total_tokens[idx]}")
                if hit_tokens and idx < len(hit_tokens):
                    f.write(f"   hit_tokens={hit_tokens[idx]}")
                f.write("\n")

            f.write("\n")
            for run_label, records_by_id, medians in (
                (run1_label, run1_records_by_id, run1_medians),
                (run2_label, run2_records_by_id, run2_medians),
            ):
                f.write(f"  Profile breakdown [{run_label}]\n")
                recs = records_by_id.get(req_id)
                if not recs:
                    if not records_by_id:
                        f.write("    (no profile file provided for this run)\n\n")
                    else:
                        f.write(f"    (no profiler records found for req_id={req_id})\n\n")
                    continue

                ops = _aggregate_op_totals(recs)
                req_total = sum(t for _, t, _, _, _ in ops)
                f.write(
                    f"    records for this req_id: {len(recs)}  |  "
                    f"summed phase time: {req_total:.2f} ms\n"
                )
                for op, total, n, phases, phase_counts in ops:
                    f.write(
                        f"    Op {op:<32} n={n:<3} "
                        f"total={total:9.2f} ms\n"
                    )
                    for phase, summed_ms in sorted(phases.items(), key=lambda kv: kv[1], reverse=True):
                        pc = phase_counts.get(phase, 1)
                        f.write(
                            _format_phase_line(op, phase, summed_ms, pc, medians) + "\n"
                        )
                f.write("\n")

        f.write("=" * 100 + "\n")

    print(f"\nOutlier profile report written to {out_file}")
    print(
        f"  {over_count} over-performer(s), {under_count} under-performer(s); "
        f"run1 recs={run1_total_records}, run2 recs={run2_total_records}"
    )


def write_outlier_details(outlier_indices, request_ids, run1_ttfts, run2_ttfts,
                         speedups, hit_ratios, total_tokens, hit_tokens,
                         run1_label, run2_label, output_dir):
    """
    Write details of outlier data points to a file.
    
    Args:
        outlier_indices (list): Indices of outlier speedup values
        request_ids (list): List of request IDs
        run1_ttfts (list): List of run1 TTFT values
        run2_ttfts (list): List of run2 TTFT values
        speedups (list): List of speedup values
        hit_ratios (list): List of hit ratios (may be empty)
        total_tokens (list): List of total tokens (may be empty)
        hit_tokens (list): List of hit tokens (may be empty)
        run1_label (str): Label for run1
        run2_label (str): Label for run2
        output_dir (Path): Output directory path
    """
    if not outlier_indices:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    outlier_file = output_dir / 'speedup_outliers.txt'
    
    has_cache_data = hit_ratios and len(hit_ratios) > 0
    
    with open(outlier_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTREMELY GOOD SPEEDUP OUTLIERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total outliers detected: {len(outlier_indices)}\n")
        f.write(f"Outlier threshold: Values above Q3 + 1.5*IQR\n")
        f.write(f"Speedup calculation: {run2_label} TTFT / {run1_label} TTFT\n\n")
        f.write("-" * 80 + "\n\n")
        
        for idx in outlier_indices:
            req_id = request_ids[idx]
            ttft1 = run1_ttfts[idx]
            ttft2 = run2_ttfts[idx]
            speedup = speedups[idx]
            
            f.write(f"Request ID: {req_id}\n")
            f.write(f"  {run1_label} TTFT: {ttft1:.6f} seconds\n")
            f.write(f"  {run2_label} TTFT: {ttft2:.6f} seconds\n")
            f.write(f"  Speedup: {speedup:.4f}x\n")
            f.write(f"  Speedup percentage: {((speedup - 1.0) * 100):.2f}% improvement\n")
            
            # Add cache data if available
            if has_cache_data and idx < len(hit_ratios):
                f.write(f"  Cache Hit Ratio: {hit_ratios[idx]:.6f}\n")
                if idx < len(total_tokens):
                    f.write(f"  Total Tokens: {total_tokens[idx]}\n")
                if idx < len(hit_tokens):
                    f.write(f"  Hit Tokens: {hit_tokens[idx]}\n")
            
            f.write("-" * 80 + "\n\n")
    
    print(f"\nOutlier details written to {outlier_file}")
    print(f"  Found {len(outlier_indices)} extremely good speedup outliers")

def main():
    parser = argparse.ArgumentParser(description='Compare two LMCache runs')
    parser.add_argument('--run1-err-file', default=None,
                       help='Path to first run .err file (optional, for cache data)')
    parser.add_argument('--run1-out-file', required=True,
                       help='Path to first run .out file')
    parser.add_argument('--run2-err-file', default=None,
                       help='Path to second run .err file (optional, for cache data)')
    parser.add_argument('--run2-out-file', required=True,
                       help='Path to second run .out file (used as baseline for TTFT)')
    parser.add_argument('--run1-label', default='Run 1',
                       help='Label for first run (default: Run 1)')
    parser.add_argument('--run2-label', default='Run 2 (Baseline)',
                       help='Label for second run (default: Run 2 (Baseline))')
    parser.add_argument('--output-dir', default=None,
                       help='Directory to save all plots (optional)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting and only show data summary')
    parser.add_argument('--plot-no-outlier', action='store_true',
                       help='Also plot no-outlier versions of all plots '
                            '(outliers removed using IQR method on speedup)')
    parser.add_argument('--outlier-threshold', type=float, default=1.5,
                       help='IQR multiplier for outlier detection (default: 1.5)')
    parser.add_argument('--run1-profile-file', default=None,
                       help='Path to run1 LMCache profiler JSONL '
                            '(enables per-outlier phase breakdown)')
    parser.add_argument('--run2-profile-file', default=None,
                       help='Path to run2 LMCache profiler JSONL '
                            '(enables per-outlier phase breakdown)')

    args = parser.parse_args()
    
    # Expand user paths
    run1_err_file = Path(args.run1_err_file).expanduser() if args.run1_err_file else None
    run1_out_file = Path(args.run1_out_file).expanduser()
    run2_err_file = Path(args.run2_err_file).expanduser() if args.run2_err_file else None
    run2_out_file = Path(args.run2_out_file).expanduser()
    
    print("LMCache Run Comparison Analysis")
    print("=" * 40)
    print(f"Run 1:")
    if run1_err_file:
        print(f"  Error log: {run1_err_file}")
    print(f"  Output log: {run1_out_file}")
    print(f"Run 2 (Baseline):")
    if run2_err_file:
        print(f"  Error log: {run2_err_file}")
    print(f"  Output log: {run2_out_file}")
    print()
    
    # Extract data. All extractors return (by_id, by_pos) tuples so
    # downstream joins can prefer stable content-derived req_id strings
    # and fall back to positional alignment for legacy logs.
    empty = ({}, {})
    run1_cache_data = extract_cache_data(run1_err_file) if run1_err_file else empty
    run1_ttft_data = extract_ttft_data(run1_out_file)
    run2_cache_data = extract_cache_data(run2_err_file) if run2_err_file else empty
    run2_ttft_data = extract_ttft_data(run2_out_file)

    # Extract prefix cache data from run2 (only available in runs with prefix cache stats enabled)
    run2_prefix_cache_data = extract_prefix_cache_data(run2_err_file) if run2_err_file else empty

    def _empty_tuple(t):
        return not t[0] and not t[1]

    if _empty_tuple(run1_ttft_data) or _empty_tuple(run2_ttft_data):
        print("Failed to extract TTFT data from one or both output files. Exiting.")
        return
    
    # Correlate cache data with TTFT for each run
    run1_hit_ratios, run1_request_ids, run1_total_tokens, run1_hit_tokens, run1_ttfts = \
        correlate_lmcache_data(run1_cache_data, run1_ttft_data)
    run2_hit_ratios, run2_request_ids, run2_total_tokens, run2_hit_tokens, run2_ttfts = \
        correlate_lmcache_data(run2_cache_data, run2_ttft_data)
    
    # Calculate speedup comparison
    speedup_request_ids, speedups, run1_ttfts_common, run2_ttfts_common = \
        calculate_speedup_comparison(run1_ttft_data, run2_ttft_data)
    
    if not speedups:
        print("No matching TTFT data found between runs. Exiting.")
        return

    # Build run1 cache lookup keyed by whichever ID flavour correlate returned.
    # This replaces the old pattern of treating ``req_id`` as a positional
    # index into the cache arrays, which no longer holds when IDs are stable
    # content hashes coming from run1's .err log.
    run1_hit_ratio_map = dict(zip(run1_request_ids, run1_hit_ratios)) if run1_hit_ratios else {}
    run1_total_tokens_map = dict(zip(run1_request_ids, run1_total_tokens)) if run1_total_tokens else {}
    run1_hit_tokens_map = dict(zip(run1_request_ids, run1_hit_tokens)) if run1_hit_tokens else {}

    def _aligned_hit_ratios(ids):
        return [run1_hit_ratio_map.get(rid, 0) for rid in ids]
    
    # Print summary statistics
    print("\nData Summary:")
    print("-" * 20)
    print(f"Total requests with TTFT data: {len(speedups)}")
    if run1_hit_ratios:
        print(f"Run 1 requests with cache data: {len(run1_hit_ratios)}")
    if run2_hit_ratios:
        print(f"Run 2 requests with cache data: {len(run2_hit_ratios)}")
    
    print(f"\n{args.run1_label} TTFT Statistics:")
    print(f"  Average: {np.mean(run1_ttfts_common):.3f}s")
    print(f"  Median: {np.median(run1_ttfts_common):.3f}s")
    print(f"  Min: {np.min(run1_ttfts_common):.3f}s")
    print(f"  Max: {np.max(run1_ttfts_common):.3f}s")
    
    print(f"\n{args.run2_label} TTFT Statistics:")
    print(f"  Average: {np.mean(run2_ttfts_common):.3f}s")
    print(f"  Median: {np.median(run2_ttfts_common):.3f}s")
    print(f"  Min: {np.min(run2_ttfts_common):.3f}s")
    print(f"  Max: {np.max(run2_ttfts_common):.3f}s")
    
    print(f"\nSpeedup Statistics ({args.run2_label} / {args.run1_label}):")
    print(f"  Average: {np.mean(speedups):.2f}x")
    print(f"  Median: {np.median(speedups):.2f}x")
    print(f"  Min: {np.min(speedups):.2f}x")
    print(f"  Max: {np.max(speedups):.2f}x")
    
    if run1_hit_ratios and len(run1_hit_ratios) > 0:
        print(f"\n{args.run1_label} Cache Statistics:")
        print(f"  Average hit ratio: {np.mean(run1_hit_ratios):.3f}")
        print(f"  Median hit ratio: {np.median(run1_hit_ratios):.3f}")
        if len(run1_hit_ratios) > 1:
            speedup_correlation = np.corrcoef(run1_hit_ratios[:len(speedups)], speedups)[0, 1]
            print(f"  Correlation with speedup: {speedup_correlation:.3f}")
    
    # Calculate extra hit ratios if we have both blend cache data (run1) and prefix cache data (run2)
    extra_hit_ratios = []
    extra_hit_request_ids = []
    extra_hit_speedups = []

    # Prefix cache data is position-only (the [INFO] Prefix cache hit tokens
    # log line carries no req_id), so the extra-hit join stays positional.
    run1_cache_view = run1_cache_data[1]
    run2_prefix_view = run2_prefix_cache_data[1]

    if run1_cache_view and run2_prefix_view:
        print(f"\nCalculating extra cache hit ratios...")
        print(f"  Run 1 (blend) cache entries: {len(run1_cache_view)}")
        print(f"  Run 2 (prefix cache) entries: {len(run2_prefix_view)}")

        # Positional join; speedup_request_ids may be either string or int.
        # Only intersect with speedup ids when they are positional ints
        # (matching the join mode extra-hit relies on).
        speedup_set = set(
            r for r in speedup_request_ids if isinstance(r, int)
        ) or set(range(len(speedup_request_ids)))
        common_ids = set(run1_cache_view.keys()) & set(run2_prefix_view.keys()) & speedup_set
        print(f"  Common request IDs: {len(common_ids)}")

        for req_id in sorted(common_ids):
            total_tokens, blend_hit_tokens = run1_cache_view[req_id]
            prefix_hit_tokens = run2_prefix_view[req_id]
            
            # Extra hit ratio = (blend_hits - prefix_hits) / total_tokens
            if total_tokens > 0:
                extra_hit_ratio = (blend_hit_tokens - prefix_hit_tokens) / total_tokens
                extra_hit_ratios.append(extra_hit_ratio)
                extra_hit_request_ids.append(req_id)
                
                # Find corresponding speedup
                if req_id in speedup_request_ids:
                    idx = list(speedup_request_ids).index(req_id)
                    extra_hit_speedups.append(speedups[idx])
        
        if extra_hit_ratios:
            print(f"\nExtra Cache Hit Statistics (Blend - Prefix Cache):")
            print(f"  Requests with extra hit data: {len(extra_hit_ratios)}")
            print(f"  Average extra hit ratio: {np.mean(extra_hit_ratios):.3f}")
            print(f"  Median extra hit ratio: {np.median(extra_hit_ratios):.3f}")
            print(f"  Min extra hit ratio: {np.min(extra_hit_ratios):.3f}")
            print(f"  Max extra hit ratio: {np.max(extra_hit_ratios):.3f}")
            if len(extra_hit_ratios) > 1 and len(extra_hit_speedups) == len(extra_hit_ratios):
                extra_correlation = np.corrcoef(extra_hit_ratios, extra_hit_speedups)[0, 1]
                print(f"  Correlation with speedup: {extra_correlation:.3f}")
    
    # Detect and write outlier details. The legacy .txt keeps the upper-tail
    # report that existing thesis artifacts depend on; the new profile-aware
    # report walks BOTH tails (over- and under-performers) and joins them
    # against the LMCache profiler JSONL files so each outlier gets a
    # phase-level breakdown rather than just speedup/TTFT numbers.
    outlier_indices = detect_outliers(speedups, args.outlier_threshold)
    tagged_outliers = detect_outliers(
        speedups, args.outlier_threshold, include_under=True
    )

    if tagged_outliers and args.output_dir:
        # Align cache data with speedups for outlier reporting, using the
        # map-based lookups that work for both stable-id and positional runs.
        aligned_hit_ratios = [run1_hit_ratio_map.get(rid, 0) for rid in speedup_request_ids]
        aligned_total_tokens = [run1_total_tokens_map.get(rid, 0) for rid in speedup_request_ids]
        aligned_hit_tokens = [run1_hit_tokens_map.get(rid, 0) for rid in speedup_request_ids]

        # Legacy upper-tail-only text file (preserved for existing artifacts)
        if outlier_indices:
            write_outlier_details(
                outlier_indices, speedup_request_ids, run1_ttfts_common, run2_ttfts_common,
                speedups, aligned_hit_ratios, aligned_total_tokens, aligned_hit_tokens,
                args.run1_label, args.run2_label, args.output_dir)

        # New two-tailed profiler-aware report
        run1_records_by_id = load_profile_records(args.run1_profile_file)
        run2_records_by_id = load_profile_records(args.run2_profile_file)
        if not run1_records_by_id and not run2_records_by_id:
            print("\nNote: no --run1-profile-file / --run2-profile-file given. "
                  "Outlier report will still be written, but without phase breakdowns. "
                  "Pass the LMCACHE_PROFILE_OUTPUT JSONL path(s) to get them.")
        write_outlier_profile_report(
            tagged_outliers=tagged_outliers,
            request_ids=speedup_request_ids,
            speedups=speedups,
            run1_ttfts=run1_ttfts_common,
            run2_ttfts=run2_ttfts_common,
            hit_ratios=aligned_hit_ratios,
            total_tokens=aligned_total_tokens,
            hit_tokens=aligned_hit_tokens,
            run1_records_by_id=run1_records_by_id,
            run2_records_by_id=run2_records_by_id,
            run1_label=args.run1_label,
            run2_label=args.run2_label,
            output_dir=args.output_dir,
            iqr_threshold=args.outlier_threshold,
        )
    elif tagged_outliers:
        over = sum(1 for _, c in tagged_outliers if c == "over")
        under = sum(1 for _, c in tagged_outliers if c == "under")
        print(
            f"\nWarning: {over} over-performer and {under} under-performer "
            f"outliers detected but no output directory specified."
        )
        print("  Specify --output-dir to save outlier details.")
    
    # Create plots
    if not args.no_plot:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            pdf_dir = output_dir / 'pdf'
            pdf_dir.mkdir(exist_ok=True)
            
            # TTFT comparison plots (always available)
            create_ttft_comparison_scatter(
                run1_ttfts_common, run2_ttfts_common, speedup_request_ids,
                args.run1_label, args.run2_label,
                output_dir / 'ttft_comparison_scatter.png')
            create_speedup_histogram(
                speedups, args.run1_label, args.run2_label,
                output_dir / 'speedup_histogram.png')
            
            # Cache-related plots (only if cache data exists)
            if run1_hit_ratios and len(run1_hit_ratios) > 0:
                aligned_hit_ratios = _aligned_hit_ratios(speedup_request_ids)

                if len(aligned_hit_ratios) == len(speedups):
                    create_speedup_vs_hit_ratio_plot(
                        speedups, aligned_hit_ratios, speedup_request_ids,
                        output_dir / 'speedup_vs_hit_ratio.png')
                    create_speedup_vs_hit_ratio_log_plot(
                        speedups, aligned_hit_ratios, speedup_request_ids,
                        output_dir / 'speedup_vs_hit_ratio_log.png')

            # Extra cache hit ratio plot (blend vs prefix cache)
            if extra_hit_ratios and len(extra_hit_ratios) > 0 and len(extra_hit_speedups) == len(extra_hit_ratios):
                create_speedup_vs_extra_hit_ratio_plot(
                    extra_hit_speedups, extra_hit_ratios, extra_hit_request_ids,
                    args.run1_label, args.run2_label,
                    output_dir / 'speedup_vs_extra_hit_ratio.png')
            
            # Save PDF plots
            create_ttft_comparison_scatter(
                run1_ttfts_common, run2_ttfts_common, speedup_request_ids,
                args.run1_label, args.run2_label,
                pdf_dir / 'ttft_comparison_scatter.pdf')
            create_speedup_histogram(
                speedups, args.run1_label, args.run2_label,
                pdf_dir / 'speedup_histogram.pdf')
            
            if run1_hit_ratios and len(run1_hit_ratios) > 0:
                aligned_hit_ratios = _aligned_hit_ratios(speedup_request_ids)

                if len(aligned_hit_ratios) == len(speedups):
                    create_speedup_vs_hit_ratio_plot(
                        speedups, aligned_hit_ratios, speedup_request_ids,
                        pdf_dir / 'speedup_vs_hit_ratio.pdf')
                    create_speedup_vs_hit_ratio_log_plot(
                        speedups, aligned_hit_ratios, speedup_request_ids,
                        pdf_dir / 'speedup_vs_hit_ratio_log.pdf')

            # Extra cache hit ratio PDF plot
            if extra_hit_ratios and len(extra_hit_ratios) > 0 and len(extra_hit_speedups) == len(extra_hit_ratios):
                create_speedup_vs_extra_hit_ratio_plot(
                    extra_hit_speedups, extra_hit_ratios, extra_hit_request_ids,
                    args.run1_label, args.run2_label,
                    pdf_dir / 'speedup_vs_extra_hit_ratio.pdf')
            
            # --- No-outlier versions ---
            if args.plot_no_outlier:
                no_dir = output_dir / 'no_outlier'
                no_dir.mkdir(exist_ok=True)
                no_pdf_dir = no_dir / 'pdf'
                no_pdf_dir.mkdir(exist_ok=True)
                thr = args.outlier_threshold

                f_speedups, f_run1, f_run2, f_ids = filter_outliers(
                    speedups, run1_ttfts_common, run2_ttfts_common,
                    speedup_request_ids, threshold_multiplier=thr)

                for dest in (no_dir, no_pdf_dir):
                    ext = '.pdf' if dest == no_pdf_dir else '.png'
                    create_ttft_comparison_scatter(
                        f_run1, f_run2, f_ids,
                        args.run1_label, args.run2_label,
                        dest / f'ttft_comparison_scatter_no_outlier{ext}')
                    create_speedup_histogram(
                        f_speedups, args.run1_label, args.run2_label,
                        dest / f'speedup_histogram_no_outlier{ext}')

                    if run1_hit_ratios and len(run1_hit_ratios) > 0:
                        f_aligned = _aligned_hit_ratios(f_ids)
                        if len(f_aligned) == len(f_speedups):
                            create_speedup_vs_hit_ratio_plot(
                                f_speedups, f_aligned, f_ids,
                                dest / f'speedup_vs_hit_ratio_no_outlier{ext}')
                            create_speedup_vs_hit_ratio_log_plot(
                                f_speedups, f_aligned, f_ids,
                                dest / f'speedup_vs_hit_ratio_log_no_outlier{ext}')

                    if extra_hit_ratios and len(extra_hit_ratios) > 0 and len(extra_hit_speedups) == len(extra_hit_ratios):
                        extra_map = dict(zip(extra_hit_request_ids, zip(extra_hit_speedups, extra_hit_ratios)))
                        f_extra_s, f_extra_r, f_extra_ids = [], [], []
                        for rid in f_ids:
                            if rid in extra_map:
                                s, r = extra_map[rid]
                                f_extra_s.append(s)
                                f_extra_r.append(r)
                                f_extra_ids.append(rid)
                        if f_extra_r:
                            # Re-filter the extra arrays by the same speedup threshold
                            f_extra_s2, f_extra_r2, f_extra_ids2 = filter_outliers(
                                f_extra_s, f_extra_r, f_extra_ids,
                                threshold_multiplier=thr)
                            create_speedup_vs_extra_hit_ratio_plot(
                                f_extra_s2, f_extra_r2, f_extra_ids2,
                                args.run1_label, args.run2_label,
                                dest / f'speedup_vs_extra_hit_ratio_no_outlier{ext}')

                print(f"No-outlier plots saved to {no_dir}")
                print(f"No-outlier PDF plots saved to {no_pdf_dir}")

            print(f"\nAll plots saved to {output_dir}")
            print(f"PDF plots saved to {pdf_dir}")
        else:
            # Create plots interactively
            create_ttft_comparison_scatter(
                run1_ttfts_common, run2_ttfts_common, speedup_request_ids,
                args.run1_label, args.run2_label)
            create_speedup_histogram(speedups, args.run1_label, args.run2_label)

            if run1_hit_ratios and len(run1_hit_ratios) > 0:
                aligned_hit_ratios = _aligned_hit_ratios(speedup_request_ids)

                if len(aligned_hit_ratios) == len(speedups):
                    create_speedup_vs_hit_ratio_plot(
                        speedups, aligned_hit_ratios, speedup_request_ids)
                    create_speedup_vs_hit_ratio_log_plot(
                        speedups, aligned_hit_ratios, speedup_request_ids)

            # Extra cache hit ratio interactive plot
            if extra_hit_ratios and len(extra_hit_ratios) > 0 and len(extra_hit_speedups) == len(extra_hit_ratios):
                create_speedup_vs_extra_hit_ratio_plot(
                    extra_hit_speedups, extra_hit_ratios, extra_hit_request_ids,
                    args.run1_label, args.run2_label)

            # --- No-outlier versions (interactive) ---
            if args.plot_no_outlier:
                thr = args.outlier_threshold
                f_speedups, f_run1, f_run2, f_ids = filter_outliers(
                    speedups, run1_ttfts_common, run2_ttfts_common,
                    speedup_request_ids, threshold_multiplier=thr)

                create_ttft_comparison_scatter(
                    f_run1, f_run2, f_ids,
                    args.run1_label, args.run2_label)
                create_speedup_histogram(
                    f_speedups, args.run1_label, args.run2_label)

                if run1_hit_ratios and len(run1_hit_ratios) > 0:
                    f_aligned = _aligned_hit_ratios(f_ids)
                    if len(f_aligned) == len(f_speedups):
                        create_speedup_vs_hit_ratio_plot(
                            f_speedups, f_aligned, f_ids)
                        create_speedup_vs_hit_ratio_log_plot(
                            f_speedups, f_aligned, f_ids)

                if extra_hit_ratios and len(extra_hit_ratios) > 0 and len(extra_hit_speedups) == len(extra_hit_ratios):
                    extra_map = dict(zip(extra_hit_request_ids, zip(extra_hit_speedups, extra_hit_ratios)))
                    f_extra_s, f_extra_r, f_extra_ids = [], [], []
                    for rid in f_ids:
                        if rid in extra_map:
                            s, r = extra_map[rid]
                            f_extra_s.append(s)
                            f_extra_r.append(r)
                            f_extra_ids.append(rid)
                    if f_extra_r:
                        f_extra_s2, f_extra_r2, f_extra_ids2 = filter_outliers(
                            f_extra_s, f_extra_r, f_extra_ids,
                            threshold_multiplier=thr)
                        create_speedup_vs_extra_hit_ratio_plot(
                            f_extra_s2, f_extra_r2, f_extra_ids2,
                            args.run1_label, args.run2_label)
    
    # Save data to CSV
    csv_file = Path(args.output_dir) / 'lmcache_comparison_data.csv' if args.output_dir else Path('lmcache_comparison_data.csv')
    try:
        import pandas as pd
        df = pd.DataFrame({
            'Request_ID': speedup_request_ids,
            f'{args.run1_label}_TTFT': run1_ttfts_common,
            f'{args.run2_label}_TTFT': run2_ttfts_common,
            'Speedup': speedups
        })
        
        if run1_hit_ratios and len(run1_hit_ratios) > 0:
            df['Hit_Ratio'] = _aligned_hit_ratios(speedup_request_ids)
        
        # Add extra hit ratio data if available
        if extra_hit_ratios and len(extra_hit_ratios) > 0:
            # Create a mapping from request_id to extra hit ratio
            extra_hit_map = dict(zip(extra_hit_request_ids, extra_hit_ratios))
            aligned_extra_hit_ratios = [extra_hit_map.get(req_id, np.nan) for req_id in speedup_request_ids]
            df['Extra_Hit_Ratio'] = aligned_extra_hit_ratios
        
        df.to_csv(csv_file, index=False)
        print(f"\nData saved to {csv_file} for further analysis")
    except ImportError:
        print("\nPandas not available. Skipping CSV export.")

if __name__ == "__main__":
    main()

