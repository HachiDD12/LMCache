#!/usr/bin/env python3
"""
Script to compare two LMCache runs, using one as baseline for TTFT comparison.
Extracts cache data and TTFT from both runs and creates comparison plots.
Skips cache-related plots if cache data is not available.
"""

import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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
    Uses an atomic counter (0-indexed) to align with speedup_data.
    
    Args:
        err_file_path (str): Path to the lmcache.err file
        
    Returns:
        dict: Dictionary mapping request index (0-indexed) to (total_tokens, hit_tokens)
    """
    cache_data = {}
    
    # Pattern to match: Reqid: <any characters>, Total tokens X, LMCache hit tokens: Y, need to load: Z
    pattern = r'Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: \d+'
    pat = re.compile(pattern)
    
    # Atomic counter starting at 0
    request_counter = 0
    
    try:
        with open(err_file_path, 'r') as f:
            for line in f:
                match = pat.search(line)
                if match:
                    total_tokens = int(match.group(2))
                    hit_tokens = int(match.group(3))
                    cache_data[request_counter] = (total_tokens, hit_tokens)
                    request_counter += 1
    except FileNotFoundError:
        print(f"Warning: Could not find file {err_file_path}")
        return {}
    except Exception as e:
        print(f"Warning: Error reading {err_file_path}: {e}")
        return {}
    
    if cache_data:
        print(f"Extracted cache data for {len(cache_data)} requests from {err_file_path}")
    return cache_data


def extract_prefix_cache_data(err_file_path):
    """
    Extract prefix cache hit tokens from the error log file.
    These are printed by our prefix_cache_stats module before each LMCache request.
    Format: [INFO] Prefix cache hit tokens: X
    
    Uses an atomic counter (0-indexed) to align with other data.
    
    Args:
        err_file_path (str): Path to the .err file
        
    Returns:
        dict: Dictionary mapping request index (0-indexed) to prefix_hit_tokens
    """
    prefix_cache_data = {}
    
    # Pattern to match: [INFO] Prefix cache hit tokens: X
    pattern = r'\[INFO\] Prefix cache hit tokens: (\d+)'
    pat = re.compile(pattern)
    
    # Atomic counter starting at 0
    request_counter = 0
    
    try:
        with open(err_file_path, 'r') as f:
            for line in f:
                match = pat.search(line)
                if match:
                    prefix_hit_tokens = int(match.group(1))
                    prefix_cache_data[request_counter] = prefix_hit_tokens
                    request_counter += 1
    except FileNotFoundError:
        print(f"Warning: Could not find file {err_file_path}")
        return {}
    except Exception as e:
        print(f"Warning: Error reading {err_file_path}: {e}")
        return {}
    
    if prefix_cache_data:
        print(f"Extracted prefix cache data for {len(prefix_cache_data)} requests from {err_file_path}")
    return prefix_cache_data

def extract_ttft_data(out_file_path):
    """
    Extract TTFT from the output log file.
    Uses a monotonic counter (0-indexed) for alignment.
    Supports two formats:
    1. Full format: '[MOCK] Request N completed: ... TTFT: X.XXs Speedup: X.XXx'
    2. Fallback format: '[INFO] TTFT: X.XX seconds'
    
    Args:
        out_file_path (str): Path to the .out file
        
    Returns:
        dict: Dictionary mapping request index (0-indexed) to ttft
    """
    ttft_data = {}
    
    # Primary pattern: Full format with [MOCK] Request and Speedup
    primary_pattern = r'@@@@@ Response # of choices: (\d+)\s*\[MOCK\] Request (\d+) completed:.*?TTFT: ([0-9.]+)s\s*Speedup: ([0-9.]+)x'
    primary_pat = re.compile(primary_pattern, re.DOTALL)
    
    # Fallback pattern: Just TTFT line (format: [INFO] TTFT: X.XX seconds)
    fallback_pattern = r'\[INFO\] TTFT: ([0-9.]+) seconds'
    fallback_pat = re.compile(fallback_pattern)
    
    # Monotonic counter starting at 0
    request_counter = 0
    
    try:
        with open(out_file_path, 'r') as f:
            content = f.read()
            
            # Try primary pattern first
            primary_matches = primary_pat.findall(content)
            used_fallback = False
            
            if primary_matches:
                for match in primary_matches:
                    ttft = float(match[2])
                    ttft_data[request_counter] = ttft
                    request_counter += 1
            else:
                # Fallback: Extract just TTFT lines
                used_fallback = True
                fallback_matches = fallback_pat.findall(content)
                for match in fallback_matches:
                    ttft = float(match)
                    ttft_data[request_counter] = ttft
                    request_counter += 1
                
    except FileNotFoundError:
        print(f"Error: Could not find file {out_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading {out_file_path}: {e}")
        return {}
    
    print(f"Extracted TTFT data for {len(ttft_data)} requests from {out_file_path}")
    if used_fallback:
        print(f"  Note: Used fallback pattern")
    return ttft_data

def correlate_lmcache_data(cache_data, ttft_data):
    """
    Correlate cache data with TTFT data based on request indices.
    
    Args:
        cache_data (dict): Cache hit data from error log (may be empty)
        ttft_data (dict): TTFT data from output log
        
    Returns:
        tuple: (hit_ratios, request_ids, total_tokens_list, hit_tokens_list, ttfts)
               Returns empty lists for cache-related data if cache_data is empty
    """
    hit_ratios = []
    request_ids = []
    total_tokens_list = []
    hit_tokens_list = []
    ttfts = []
    
    # If no cache data, just return TTFT data
    if not cache_data:
        for req_id in sorted(ttft_data.keys()):
            ttft = ttft_data[req_id]
            ttfts.append(ttft)
            request_ids.append(req_id)
        return [], request_ids, [], [], ttfts
    
    # Find common request IDs between the two datasets
    common_ids = set(cache_data.keys()) & set(ttft_data.keys())
    
    if not common_ids:
        print("Warning: No matching request IDs found between cache and TTFT data")
        return [], [], [], [], []
    
    print(f"Found {len(common_ids)} matching request IDs")
    
    for req_id in sorted(common_ids):
        total_tokens, hit_tokens = cache_data[req_id]
        ttft = ttft_data[req_id]
        
        # Calculate hit ratio (hit tokens / total tokens)
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
    
    Args:
        run1_ttft_data (dict): TTFT data from first run
        run2_ttft_data (dict): TTFT data from second run (baseline)
        
    Returns:
        tuple: (request_ids, speedups, run1_ttfts, run2_ttfts) for requests with both data
    """
    # Find common request IDs
    common_ids = set(run1_ttft_data.keys()) & set(run2_ttft_data.keys())
    
    if not common_ids:
        print("Warning: No matching request IDs found between the two runs")
        return [], [], [], []
    
    print(f"Found {len(common_ids)} matching request IDs for TTFT comparison")
    
    request_ids = []
    speedups = []
    run1_ttfts = []
    run2_ttfts = []
    
    for req_id in sorted(common_ids):
        run1_ttft = run1_ttft_data[req_id]
        run2_ttft = run2_ttft_data[req_id]
        
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

def detect_outliers(speedups, threshold_multiplier=1.5):
    """
    Detect outliers in speedup values using IQR method.
    Returns indices of outliers (extremely good speedup values).
    
    Args:
        speedups (list): List of speedup values
        threshold_multiplier (float): Multiplier for IQR threshold (default: 1.5)
        
    Returns:
        list: Indices of outlier values
    """
    if len(speedups) < 4:
        return []
    
    speedups_array = np.array(speedups)
    q1 = np.percentile(speedups_array, 25)
    q3 = np.percentile(speedups_array, 75)
    iqr = q3 - q1
    
    # Upper bound for outliers (extremely good speedup)
    upper_bound = q3 + threshold_multiplier * iqr
    
    # Find indices where speedup is above upper bound
    outlier_indices = np.where(speedups_array > upper_bound)[0].tolist()
    
    return outlier_indices

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
    
    # Extract data
    run1_cache_data = extract_cache_data(run1_err_file) if run1_err_file else {}
    run1_ttft_data = extract_ttft_data(run1_out_file)
    run2_cache_data = extract_cache_data(run2_err_file) if run2_err_file else {}
    run2_ttft_data = extract_ttft_data(run2_out_file)
    
    # Extract prefix cache data from run2 (only available in runs with prefix cache stats enabled)
    run2_prefix_cache_data = extract_prefix_cache_data(run2_err_file) if run2_err_file else {}
    
    if not run1_ttft_data or not run2_ttft_data:
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
    
    if run1_cache_data and run2_prefix_cache_data:
        print(f"\nCalculating extra cache hit ratios...")
        print(f"  Run 1 (blend) cache entries: {len(run1_cache_data)}")
        print(f"  Run 2 (prefix cache) entries: {len(run2_prefix_cache_data)}")
        
        # Find common request IDs across all three datasets
        common_ids = set(run1_cache_data.keys()) & set(run2_prefix_cache_data.keys()) & set(speedup_request_ids)
        print(f"  Common request IDs: {len(common_ids)}")
        
        for req_id in sorted(common_ids):
            total_tokens, blend_hit_tokens = run1_cache_data[req_id]
            prefix_hit_tokens = run2_prefix_cache_data[req_id]
            
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
    
    # Detect and write outlier details
    outlier_indices = detect_outliers(speedups)
    if outlier_indices and args.output_dir:
        # Create mappings from request_id to cache data for run1
        run1_cache_map = {}
        if run1_hit_ratios and len(run1_hit_ratios) > 0 and len(run1_request_ids) == len(run1_hit_ratios):
            for i, req_id in enumerate(run1_request_ids):
                run1_cache_map[req_id] = {
                    'hit_ratio': run1_hit_ratios[i],
                    'total_tokens': run1_total_tokens[i] if i < len(run1_total_tokens) else 0,
                    'hit_tokens': run1_hit_tokens[i] if i < len(run1_hit_tokens) else 0
                }
        
        # Align cache data with speedups for outlier reporting
        aligned_hit_ratios = []
        aligned_total_tokens = []
        aligned_hit_tokens = []
        
        for req_id in speedup_request_ids:
            if req_id in run1_cache_map:
                aligned_hit_ratios.append(run1_cache_map[req_id]['hit_ratio'])
                aligned_total_tokens.append(run1_cache_map[req_id]['total_tokens'])
                aligned_hit_tokens.append(run1_cache_map[req_id]['hit_tokens'])
            else:
                aligned_hit_ratios.append(0)
                aligned_total_tokens.append(0)
                aligned_hit_tokens.append(0)
        
        write_outlier_details(
            outlier_indices, speedup_request_ids, run1_ttfts_common, run2_ttfts_common,
            speedups, aligned_hit_ratios, aligned_total_tokens, aligned_hit_tokens,
            args.run1_label, args.run2_label, args.output_dir)
    elif outlier_indices:
        print(f"\nWarning: {len(outlier_indices)} outliers detected but no output directory specified.")
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
                # Align hit ratios with speedups
                aligned_hit_ratios = []
                for req_id in speedup_request_ids:
                    if req_id < len(run1_hit_ratios):
                        aligned_hit_ratios.append(run1_hit_ratios[req_id])
                    else:
                        aligned_hit_ratios.append(0)
                
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
                aligned_hit_ratios = []
                for req_id in speedup_request_ids:
                    if req_id < len(run1_hit_ratios):
                        aligned_hit_ratios.append(run1_hit_ratios[req_id])
                    else:
                        aligned_hit_ratios.append(0)
                
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
                        f_aligned = [
                            run1_hit_ratios[rid] if rid < len(run1_hit_ratios) else 0
                            for rid in f_ids
                        ]
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
                aligned_hit_ratios = []
                for req_id in speedup_request_ids:
                    if req_id < len(run1_hit_ratios):
                        aligned_hit_ratios.append(run1_hit_ratios[req_id])
                    else:
                        aligned_hit_ratios.append(0)

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
                    f_aligned = [
                        run1_hit_ratios[rid] if rid < len(run1_hit_ratios) else 0
                        for rid in f_ids
                    ]
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
            aligned_hit_ratios = []
            for req_id in speedup_request_ids:
                if req_id < len(run1_hit_ratios):
                    aligned_hit_ratios.append(run1_hit_ratios[req_id])
                else:
                    aligned_hit_ratios.append(0)
            df['Hit_Ratio'] = aligned_hit_ratios
        
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

