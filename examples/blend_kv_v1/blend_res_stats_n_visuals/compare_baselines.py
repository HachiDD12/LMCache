#!/usr/bin/env python3
"""
Script to compare TTFT between two baseline instances (no cache data).
Extracts TTFT from both baseline log files and creates comparison plots.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from scipy.stats import gaussian_kde

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

def extract_ttft_data(out_file_path):
    """
    Extract TTFT from the output log file.
    Uses a monotonic counter (0-indexed) for alignment.
    Supports two formats:
    1. Full format: '[MOCK] Request N completed: ... TTFT: X.XXs Speedup: X.XXx'
    2. Fallback format: '[INFO] TTFT: X.XX seconds'
    
    Args:
        out_file_path (str): Path to the baseline .out file
        
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

def correlate_baseline_data(baseline1_data, baseline2_data):
    """
    Correlate TTFT data from two baseline instances based on request indices.
    
    Args:
        baseline1_data (dict): TTFT data from first baseline
        baseline2_data (dict): TTFT data from second baseline
        
    Returns:
        tuple: (baseline1_ttfts, baseline2_ttfts, request_ids, speedups) for requests with both data
    """
    baseline1_ttfts = []
    baseline2_ttfts = []
    request_ids = []
    speedups = []
    
    # Find common request IDs between the two datasets
    common_ids = set(baseline1_data.keys()) & set(baseline2_data.keys())
    
    if not common_ids:
        print("Warning: No matching request IDs found between the two baseline files")
        return [], [], [], []
    
    print(f"Found {len(common_ids)} matching request IDs")
    
    for req_id in sorted(common_ids):
        ttft1 = baseline1_data[req_id]
        ttft2 = baseline2_data[req_id]
        
        # Calculate speedup: baseline1_ttft / baseline2_ttft
        speedup = ttft1 / ttft2 if ttft2 > 0 else 0
        
        baseline1_ttfts.append(ttft1)
        baseline2_ttfts.append(ttft2)
        request_ids.append(req_id)
        speedups.append(speedup)
        
        # Debug output for first few entries
        if len(baseline1_ttfts) <= 5:
            print(f"Request {req_id}: Baseline1 TTFT={ttft1:.3f}s, Baseline2 TTFT={ttft2:.3f}s, "
                  f"Speedup={speedup:.2f}x")
    
    return baseline1_ttfts, baseline2_ttfts, request_ids, speedups

def create_ttft_comparison_scatter(baseline1_ttfts, baseline2_ttfts, request_ids, 
                                   baseline1_label="Baseline 1", baseline2_label="Baseline 2",
                                   output_file=None):
    """Create scatter plot comparing TTFT between two baselines."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(baseline1_ttfts, baseline2_ttfts, alpha=0.7, s=50, c='blue', edgecolors='black')
    
    # Add diagonal line (y=x)
    max_ttft = max(max(baseline1_ttfts), max(baseline2_ttfts))
    min_ttft = min(min(baseline1_ttfts), min(baseline2_ttfts))
    plt.plot([min_ttft, max_ttft], [min_ttft, max_ttft], 'r--', alpha=0.8, linewidth=2, 
            label='Equal TTFT (y=x)')
    
    # Add trend line
    if len(baseline1_ttfts) > 1:
        z = np.polyfit(baseline1_ttfts, baseline2_ttfts, 1)
        p = np.poly1d(z)
        plt.plot(baseline1_ttfts, p(baseline1_ttfts), "g--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    plt.xlabel(f'{baseline1_label} TTFT (seconds)', fontsize=24)
    plt.ylabel(f'{baseline2_label} TTFT (seconds)', fontsize=24)
    plt.title(f'TTFT Comparison: {baseline1_label} vs {baseline2_label}', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"TTFT comparison scatter plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

def create_ttft_histogram_comparison(baseline1_ttfts, baseline2_ttfts,
                                     baseline1_label="Baseline 1", baseline2_label="Baseline 2",
                                     output_file=None):
    """Create overlapping histograms comparing TTFT distributions."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create histograms
    plt.hist(baseline1_ttfts, bins=30, alpha=0.6, color='blue', edgecolor='black', 
            label=baseline1_label, density=True)
    plt.hist(baseline2_ttfts, bins=30, alpha=0.6, color='red', edgecolor='black', 
            label=baseline2_label, density=True)
    
    # Add smoothed distribution curves
    if len(baseline1_ttfts) > 1:
        x_range1 = np.linspace(min(baseline1_ttfts), max(baseline1_ttfts), 200)
        kde1 = gaussian_kde(baseline1_ttfts)
        density1 = kde1(x_range1)
        plt.plot(x_range1, density1, 'b-', linewidth=3, label=f'{baseline1_label} (smoothed)')
    
    if len(baseline2_ttfts) > 1:
        x_range2 = np.linspace(min(baseline2_ttfts), max(baseline2_ttfts), 200)
        kde2 = gaussian_kde(baseline2_ttfts)
        density2 = kde2(x_range2)
        plt.plot(x_range2, density2, 'r-', linewidth=3, label=f'{baseline2_label} (smoothed)')
    
    # Add statistics
    mean1 = np.mean(baseline1_ttfts)
    mean2 = np.mean(baseline2_ttfts)
    median1 = np.median(baseline1_ttfts)
    median2 = np.median(baseline2_ttfts)
    
    plt.axvline(mean1, color='blue', linestyle='--', linewidth=2, 
               label=f'{baseline1_label} Mean: {mean1:.3f}s')
    plt.axvline(mean2, color='red', linestyle='--', linewidth=2, 
               label=f'{baseline2_label} Mean: {mean2:.3f}s')
    plt.axvline(median1, color='blue', linestyle=':', linewidth=2, 
               label=f'{baseline1_label} Median: {median1:.3f}s')
    plt.axvline(median2, color='red', linestyle=':', linewidth=2, 
               label=f'{baseline2_label} Median: {median2:.3f}s')
    
    plt.xlabel('TTFT (seconds)', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.title(f'TTFT Distribution Comparison: {baseline1_label} vs {baseline2_label}', 
              fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"TTFT histogram comparison saved to {output_file}")
    else:
        plt.show()
    plt.close()

def create_speedup_histogram(speedups, baseline1_label="Baseline 1", baseline2_label="Baseline 2", output_file=None):
    """Create histogram of speedup values (baseline1/baseline2)."""
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create histogram
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
               label='No difference (1.0x)')
    
    plt.xlabel(f'Speedup ({baseline1_label} / {baseline2_label})', fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.title(f'Speedup Distribution ({baseline1_label} / {baseline2_label})', fontsize=28, fontweight='bold')
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

def write_outlier_details(outlier_indices, request_ids, baseline1_ttfts, baseline2_ttfts, 
                         speedups, baseline1_label, baseline2_label, output_dir):
    """
    Write details of outlier data points to a file.
    
    Args:
        outlier_indices (list): Indices of outlier speedup values
        request_ids (list): List of request IDs
        baseline1_ttfts (list): List of baseline1 TTFT values
        baseline2_ttfts (list): List of baseline2 TTFT values
        speedups (list): List of speedup values
        baseline1_label (str): Label for baseline1
        baseline2_label (str): Label for baseline2
        output_dir (Path): Output directory path
    """
    if not outlier_indices:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    outlier_file = output_dir / 'speedup_outliers.txt'
    
    with open(outlier_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTREMELY GOOD SPEEDUP OUTLIERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total outliers detected: {len(outlier_indices)}\n")
        f.write(f"Outlier threshold: Values above Q3 + 1.5*IQR\n")
        f.write(f"Speedup calculation: {baseline1_label} TTFT / {baseline2_label} TTFT\n\n")
        f.write("-" * 80 + "\n\n")
        
        for idx in outlier_indices:
            req_id = request_ids[idx]
            ttft1 = baseline1_ttfts[idx]
            ttft2 = baseline2_ttfts[idx]
            speedup = speedups[idx]
            
            f.write(f"Request ID: {req_id}\n")
            f.write(f"  {baseline1_label} TTFT: {ttft1:.6f} seconds\n")
            f.write(f"  {baseline2_label} TTFT: {ttft2:.6f} seconds\n")
            f.write(f"  Speedup: {speedup:.4f}x\n")
            f.write(f"  Speedup percentage: {((speedup - 1.0) * 100):.2f}% improvement\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"\nOutlier details written to {outlier_file}")
    print(f"  Found {len(outlier_indices)} extremely good speedup outliers")

def main():
    parser = argparse.ArgumentParser(description='Compare TTFT between two baseline instances')
    parser.add_argument('--baseline1-file', required=True,
                       help='Path to first baseline .out file')
    parser.add_argument('--baseline2-file', required=True,
                       help='Path to second baseline .out file')
    parser.add_argument('--baseline1-label', default='Baseline 1',
                       help='Label for first baseline (default: Baseline 1)')
    parser.add_argument('--baseline2-label', default='Baseline 2',
                       help='Label for second baseline (default: Baseline 2)')
    parser.add_argument('--output-dir', default=None,
                       help='Directory to save all plots (optional)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting and only show data summary')
    
    args = parser.parse_args()
    
    # Expand user paths
    baseline1_file = Path(args.baseline1_file).expanduser()
    baseline2_file = Path(args.baseline2_file).expanduser()
    
    print("Baseline Comparison Analysis")
    print("=" * 40)
    print(f"Baseline 1: {baseline1_file}")
    print(f"Baseline 2: {baseline2_file}")
    print()
    
    # Extract TTFT data from both files
    baseline1_data = extract_ttft_data(baseline1_file)
    baseline2_data = extract_ttft_data(baseline2_file)
    
    if not baseline1_data or not baseline2_data:
        print("Failed to extract data from one or both baseline files. Exiting.")
        return
    
    # Correlate the data
    baseline1_ttfts, baseline2_ttfts, request_ids, speedups = correlate_baseline_data(
        baseline1_data, baseline2_data)
    
    if not baseline1_ttfts:
        print("No correlated data found. Exiting.")
        return
    
    # Print summary statistics
    print("\nData Summary:")
    print("-" * 20)
    print(f"Total correlated requests: {len(baseline1_ttfts)}")
    print(f"\n{args.baseline1_label} Statistics:")
    print(f"  Average TTFT: {np.mean(baseline1_ttfts):.3f}s")
    print(f"  Median TTFT: {np.median(baseline1_ttfts):.3f}s")
    print(f"  Min TTFT: {np.min(baseline1_ttfts):.3f}s")
    print(f"  Max TTFT: {np.max(baseline1_ttfts):.3f}s")
    print(f"\n{args.baseline2_label} Statistics:")
    print(f"  Average TTFT: {np.mean(baseline2_ttfts):.3f}s")
    print(f"  Median TTFT: {np.median(baseline2_ttfts):.3f}s")
    print(f"  Min TTFT: {np.min(baseline2_ttfts):.3f}s")
    print(f"  Max TTFT: {np.max(baseline2_ttfts):.3f}s")
    print(f"\nSpeedup Statistics (Baseline1 / Baseline2):")
    print(f"  Average: {np.mean(speedups):.2f}x")
    print(f"  Median: {np.median(speedups):.2f}x")
    print(f"  Min: {np.min(speedups):.2f}x")
    print(f"  Max: {np.max(speedups):.2f}x")
    
    # Calculate correlation
    if len(baseline1_ttfts) > 1:
        correlation = np.corrcoef(baseline1_ttfts, baseline2_ttfts)[0, 1]
        print(f"\nCorrelation between baselines: {correlation:.3f}")
    
    # Detect and write outlier details
    outlier_indices = detect_outliers(speedups)
    if outlier_indices and args.output_dir:
        write_outlier_details(
            outlier_indices, request_ids, baseline1_ttfts, baseline2_ttfts, speedups,
            args.baseline1_label, args.baseline2_label, args.output_dir)
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
            
            # Save PNG plots
            create_ttft_comparison_scatter(
                baseline1_ttfts, baseline2_ttfts, request_ids,
                args.baseline1_label, args.baseline2_label,
                output_dir / 'ttft_comparison_scatter.png')
            create_ttft_histogram_comparison(
                baseline1_ttfts, baseline2_ttfts,
                args.baseline1_label, args.baseline2_label,
                output_dir / 'ttft_histogram_comparison.png')
            create_speedup_histogram(
                speedups, args.baseline1_label, args.baseline2_label,
                output_dir / 'speedup_histogram.png')
            
            # Save PDF plots
            create_ttft_comparison_scatter(
                baseline1_ttfts, baseline2_ttfts, request_ids,
                args.baseline1_label, args.baseline2_label,
                pdf_dir / 'ttft_comparison_scatter.pdf')
            create_ttft_histogram_comparison(
                baseline1_ttfts, baseline2_ttfts,
                args.baseline1_label, args.baseline2_label,
                pdf_dir / 'ttft_histogram_comparison.pdf')
            create_speedup_histogram(
                speedups, args.baseline1_label, args.baseline2_label,
                pdf_dir / 'speedup_histogram.pdf')
            
            print(f"\nAll plots saved to {output_dir}")
            print(f"PDF plots saved to {pdf_dir}")
        else:
            # Create plots interactively
            create_ttft_comparison_scatter(
                baseline1_ttfts, baseline2_ttfts, request_ids,
                args.baseline1_label, args.baseline2_label)
            create_ttft_histogram_comparison(
                baseline1_ttfts, baseline2_ttfts,
                args.baseline1_label, args.baseline2_label)
            create_speedup_histogram(speedups, args.baseline1_label, args.baseline2_label)
    
    # Save data to CSV
    csv_file = 'baseline_comparison_data.csv'
    try:
        import pandas as pd
        df = pd.DataFrame({
            'Request_ID': request_ids,
            f'{args.baseline1_label}_TTFT': baseline1_ttfts,
            f'{args.baseline2_label}_TTFT': baseline2_ttfts,
            'Speedup': speedups
        })
        df.to_csv(csv_file, index=False)
        print(f"\nData saved to {csv_file} for further analysis")
    except ImportError:
        print("\nPandas not available. Skipping CSV export.")

if __name__ == "__main__":
    main()

