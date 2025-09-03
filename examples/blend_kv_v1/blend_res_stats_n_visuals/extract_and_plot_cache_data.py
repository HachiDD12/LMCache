#!/usr/bin/env python3
"""
Script to extract cache hit data and speedup information from LMCache log files
and plot the correlation between speedup and cache hit ratio.
"""

import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import argparse
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy import stats
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

def extract_cache_data(err_file_path):
    """
    Extract request ID, total tokens, and LMCache hit tokens from the error log file.
    
    Args:
        err_file_path (str): Path to the lmcache.err file
        
    Returns:
        dict: Dictionary mapping request ID to (total_tokens, hit_tokens)
    """
    cache_data = {}
    
    # Pattern to match: Reqid: N, Total tokens X, LMCache hit tokens: Y, need to load: Z
    pattern = r'Reqid: (?:\d+_)?(\d+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: \d+'
    pat = re.compile(pattern)
    try:
        with open(err_file_path, 'r') as f:
            for line in f:
                match = pat.search(line)
                if match:
                    req_id = int(match.group(1))
                    total_tokens = int(match.group(2))
                    hit_tokens = int(match.group(3))
                    # if req_id in cache_data:
                    #     cache_data[req_id][0] += total_tokens
                    #     cache_data[req_id][1] += hit_tokens
                    cache_data[req_id] = (total_tokens, hit_tokens)
    except FileNotFoundError:
        print(f"Error: Could not find file {err_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading {err_file_path}: {e}")
        return {}
    
    print(f"Extracted cache data for {len(cache_data)} requests from {err_file_path}")
    return cache_data

def extract_speedup_data(out_file_path):
    """
    Extract request ID+1 and speedup from the output log file.
    
    Args:
        out_file_path (str): Path to the lmcache.out file
        
    Returns:
        dict: Dictionary mapping request ID+1 to (ttft, speedup)
    """
    speedup_data = {}
    
    # Pattern to match: '@@@@@ Response # of choices: N' followed by '[MOCK] Request N completed: ... TTFT: X.XXs Speedup: X.XXx'
    pattern = r'@@@@@ Response # of choices: (\d+)\s*\[MOCK\] Request (\d+) completed:.*?TTFT: ([0-9.]+)s\s*Speedup: ([0-9.]+)x'
    pat = re.compile(pattern, re.DOTALL)
    try:
        with open(out_file_path, 'r') as f:
            content = f.read()
            # Use re.DOTALL to match across multiple lines
            matches = pat.findall(content)
            
            for match in matches:
                num_choices = int(match[0])
                req_id = int(match[1])
                ttft = float(match[2])
                speedup = float(match[3])
                # Map to actual request index as specified in the requirements
                speedup_data[req_id - 1] = (ttft, speedup)
                
    except FileNotFoundError:
        print(f"Error: Could not find file {out_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading {out_file_path}: {e}")
        return {}
    
    print(f"Extracted speedup data for {len(speedup_data)} requests from {out_file_path}")
    return speedup_data

def correlate_data(cache_data, speedup_data):
    """
    Correlate cache data with speedup data based on request IDs.
    
    Args:
        cache_data (dict): Cache hit data from error log
        speedup_data (dict): Speedup data from output log
        
    Returns:
        tuple: (hit_ratios, speedups, request_ids, total_tokens_list, hit_tokens_list, ttfts) for plotting
    """
    hit_ratios = []
    speedups = []
    request_ids = []
    total_tokens_list = []
    hit_tokens_list = []
    ttfts = []
    
    # Find common request IDs between the two datasets
    common_ids = set(cache_data.keys()) & set(speedup_data.keys())
    
    if not common_ids:
        print("Warning: No matching request IDs found between the two log files")
        return [], [], [], [], [], []
    
    print(f"Found {len(common_ids)} matching request IDs")
    
    for req_id in sorted(common_ids):
        total_tokens, hit_tokens = cache_data[req_id]
        ttft, speedup = speedup_data[req_id]
        
        # Calculate hit ratio (hit tokens / total tokens)
        hit_ratio = hit_tokens / total_tokens if total_tokens > 0 else 0
        
        hit_ratios.append(hit_ratio)
        speedups.append(speedup)
        request_ids.append(req_id)
        total_tokens_list.append(total_tokens)
        hit_tokens_list.append(hit_tokens)
        ttfts.append(ttft)
        
        # Debug output for first few entries
        if len(hit_ratios) <= 5 or req_id == 225 or req_id == 224:
            print(f"Request {req_id}: Total={total_tokens}, Hit={hit_tokens}, "
                  f"Ratio={hit_ratio:.3f}, Speedup={speedup:.2f}x, TTFT={ttft:.3f}s")
    
    # Debug: Check final array lengths
    print(f"DEBUG: Final array lengths in correlate_data:")
    print(f"  hit_ratios: {len(hit_ratios)}")
    print(f"  speedups: {len(speedups)}")
    print(f"  request_ids: {len(request_ids)}")
    print(f"  total_tokens_list: {len(total_tokens_list)}")
    print(f"  hit_tokens_list: {len(hit_tokens_list)}")
    print(f"  ttfts: {len(ttfts)}")
    
    return hit_ratios, speedups, request_ids, total_tokens_list, hit_tokens_list, ttfts

def create_plot(hit_ratios, speedups, request_ids, output_file=None):
    """
    Create a scatter plot of speedup vs hit ratio.
    
    Args:
        hit_ratios (list): List of hit ratios
        speedups (list): List of speedup values
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(hit_ratios, speedups, alpha=0.7, s=50, c='blue', edgecolors='black')
    
    # Add trend line
    if len(hit_ratios) > 1:
        z = np.polyfit(hit_ratios, speedups, 1)
        p = np.poly1d(z)
        plt.plot(hit_ratios, p(hit_ratios), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    # Add horizontal line at speedup = 1.0 (no speedup)
    plt.axhline(y=1.0, color='green', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
    
    # Customize the plot
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=24)
    plt.ylabel('Speedup', fontsize=24)
    plt.title('Speedup vs Cache Hit Ratio', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()

def create_total_tokens_plot(total_tokens, speedups, request_ids, output_file=None):
    """
    Create a scatter plot of speedup vs total tokens.
    
    Args:
        total_tokens (list): List of total token counts
        speedups (list): List of speedup values
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    # Debug: Check array lengths
    if len(total_tokens) != len(speedups):
        print(f"ERROR: Array length mismatch in create_total_tokens_plot:")
        print(f"  total_tokens length: {len(total_tokens)}")
        print(f"  speedups length: {len(speedups)}")
        print(f"  request_ids length: {len(request_ids)}")
        return
    
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(total_tokens, speedups, alpha=0.7, s=50, c='green', edgecolors='black')
    
    # Add trend line
    if len(total_tokens) > 1:
        z = np.polyfit(total_tokens, speedups, 1)
        p = np.poly1d(z)
        plt.plot(total_tokens, p(total_tokens), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.6f})')
    
    # Add horizontal line at speedup = 1.0 (no speedup)
    plt.axhline(y=1.0, color='blue', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
    
    # Customize the plot
    plt.xlabel('Total Tokens', fontsize=24)
    plt.ylabel('Speedup', fontsize=24)
    plt.title('Speedup vs Total Tokens', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Total tokens plot saved to {output_file}")
    
    plt.show()

def create_hit_tokens_plot(hit_tokens, speedups, request_ids, output_file=None):
    """
    Create a scatter plot of speedup vs hit tokens.
    
    Args:
        hit_tokens (list): List of hit token counts
        speedups (list): List of speedup values
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(hit_tokens, speedups, alpha=0.7, s=50, c='purple', edgecolors='black')
    
    # Add trend line
    if len(hit_tokens) > 1:
        z = np.polyfit(hit_tokens, speedups, 1)
        p = np.poly1d(z)
        plt.plot(hit_tokens, p(hit_tokens), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.6f})')
    
    # Add horizontal line at speedup = 1.0 (no speedup)
    plt.axhline(y=1.0, color='blue', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
    
    # Customize the plot
    plt.xlabel('Hit Tokens', fontsize=24)
    plt.ylabel('Speedup', fontsize=24)
    plt.title('Speedup vs Hit Tokens', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Hit tokens plot saved to {output_file}")
    
    plt.show()

def create_all_plots(hit_ratios, speedups, request_ids, total_tokens, hit_tokens, ttfts, output_dir=None):
    """
    Create all plots and optionally save them to a directory.
    
    Args:
        hit_ratios (list): List of hit ratios
        speedups (list): List of speedup values
        request_ids (list): List of request IDs
        total_tokens (list): List of total token counts
        hit_tokens (list): List of hit token counts
        ttfts (list): List of TTFT values
        output_dir (str, optional): Directory to save plots
    """
    # Debug: Check data alignment before plotting
    data_lengths = [len(hit_ratios), len(speedups), len(request_ids), len(total_tokens), len(hit_tokens), len(ttfts)]
    if len(set(data_lengths)) > 1:
        print(f"ERROR: Data arrays have different lengths in create_all_plots:")
        print(f"  hit_ratios: {len(hit_ratios)}")
        print(f"  speedups: {len(speedups)}")
        print(f"  request_ids: {len(request_ids)}")
        print(f"  total_tokens: {len(total_tokens)}")
        print(f"  hit_tokens: {len(hit_tokens)}")
        print(f"  ttfts: {len(ttfts)}")
        return
    
    # Create hit ratio vs speedup plot
    create_plot(hit_ratios, speedups, request_ids)
    
    # Create total tokens vs speedup plot
    create_total_tokens_plot(total_tokens, speedups, request_ids)
    
    # Create hit tokens vs speedup plot
    create_hit_tokens_plot(hit_tokens, speedups, request_ids)
    
    # Create hit ratio vs total tokens plot
    create_hit_ratio_vs_total_tokens_plot(hit_ratios, total_tokens, request_ids)
    
    # Create histogram of hit ratios
    create_hit_ratio_histogram(hit_ratios)
    
    # Create histogram of hit tokens
    create_hit_tokens_histogram(hit_tokens)
    
    # Create histogram of speedups
    create_speedup_histogram(speedups)
    
    # Create TTFT vs hit ratio plot
    create_ttft_vs_hit_ratio_plot(ttfts, hit_ratios, request_ids)
    
    # Create TTFT vs hit tokens plot
    create_ttft_vs_hit_tokens_plot(ttfts, hit_tokens, request_ids)
    
    # Save all plots if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create PDF subdirectory
        pdf_dir = output_dir / 'pdf'
        pdf_dir.mkdir(exist_ok=True)
        
        # Save PNG plots
        create_plot(hit_ratios, speedups, request_ids, 
                   output_dir / 'hit_ratio_vs_speedup.png')
        create_total_tokens_plot(total_tokens, speedups, request_ids,
                               output_dir / 'total_tokens_vs_speedup.png')
        create_hit_tokens_plot(hit_tokens, speedups, request_ids,
                              output_dir / 'hit_tokens_vs_speedup.png')
        create_hit_ratio_vs_total_tokens_plot(hit_ratios, total_tokens, request_ids,
                                            output_dir / 'hit_ratio_vs_total_tokens.png')
        create_hit_ratio_histogram(hit_ratios, output_dir / 'hit_ratio_histogram.png')
        create_hit_tokens_histogram(hit_tokens, output_dir / 'hit_tokens_histogram.png')
        create_speedup_histogram(speedups, output_dir / 'speedup_histogram.png')
        create_ttft_vs_hit_ratio_plot(ttfts, hit_ratios, request_ids,
                                     output_dir / 'ttft_vs_hit_ratio.png')
        create_ttft_vs_hit_tokens_plot(ttfts, hit_tokens, request_ids,
                                      output_dir / 'ttft_vs_hit_tokens.png')
        
        # Save PDF plots
        create_plot(hit_ratios, speedups, request_ids, 
                   pdf_dir / 'hit_ratio_vs_speedup.pdf')
        create_total_tokens_plot(total_tokens, speedups, request_ids,
                               pdf_dir / 'total_tokens_vs_speedup.pdf')
        create_hit_tokens_plot(hit_tokens, speedups, request_ids,
                              pdf_dir / 'hit_tokens_vs_speedup.pdf')
        create_hit_ratio_vs_total_tokens_plot(hit_ratios, total_tokens, request_ids,
                                            pdf_dir / 'hit_ratio_vs_total_tokens.pdf')
        create_hit_ratio_histogram(hit_ratios, pdf_dir / 'hit_ratio_histogram.pdf')
        create_hit_tokens_histogram(hit_tokens, pdf_dir / 'hit_tokens_histogram.pdf')
        create_speedup_histogram(speedups, pdf_dir / 'speedup_histogram.pdf')
        create_ttft_vs_hit_ratio_plot(ttfts, hit_ratios, request_ids,
                                     pdf_dir / 'ttft_vs_hit_ratio.pdf')
        create_ttft_vs_hit_tokens_plot(ttfts, hit_tokens, request_ids,
                                      pdf_dir / 'ttft_vs_hit_tokens.pdf')
        
        print(f"All plots saved to {output_dir}")
        print(f"PDF plots saved to {pdf_dir}")

def create_hit_ratio_vs_total_tokens_plot(hit_ratios, total_tokens, request_ids, output_file=None):
    """
    Create a scatter plot of hit ratio vs total tokens.
    
    Args:
        hit_ratios (list): List of hit ratios
        total_tokens (list): List of total token counts
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(total_tokens, hit_ratios, alpha=0.7, s=50, c='orange', edgecolors='black')
    
    # Add trend line
    if len(total_tokens) > 1:
        z = np.polyfit(total_tokens, hit_ratios, 1)
        p = np.poly1d(z)
        plt.plot(total_tokens, p(total_tokens), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.6f})')
    
    # Add horizontal line at hit ratio = 0.5 (50% cache hit)
    plt.axhline(y=0.5, color='green', linestyle='-', alpha=0.5, label='50% cache hit')
    
    # Customize the plot
    plt.xlabel('Total Tokens', fontsize=24)
    plt.ylabel('Cache Hit Ratio', fontsize=24)
    plt.title('Cache Hit Ratio vs Total Tokens', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Hit ratio vs total tokens plot saved to {output_file}")
    
    plt.show()

def create_hit_ratio_histogram(hit_ratios, output_file=None):
    """
    Create a histogram of hit ratios.
    
    Args:
        hit_ratios (list): List of hit ratios
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create histogram with more bins for better resolution
    n_bins = min(30, len(hit_ratios) // 3)  # Adaptive binning
    n_bins = max(10, n_bins)  # At least 10 bins
    
    plt.hist(hit_ratios, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
    
    # Add vertical line for mean and median
    mean_hit_ratio = np.mean(hit_ratios)
    median_hit_ratio = np.median(hit_ratios)
    plt.axvline(x=mean_hit_ratio, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_hit_ratio:.3f}')
    plt.axvline(x=median_hit_ratio, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_hit_ratio:.3f}')
    
    # Customize the plot
    plt.xlabel('Cache Hit Ratio', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.title('Distribution of Cache Hit Ratios', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Hit ratio histogram saved to {output_file}")
    
    plt.show()

def create_hit_tokens_histogram(hit_tokens, output_file=None):
    """
    Create a histogram of hit tokens.
    
    Args:
        hit_tokens (list): List of hit token counts
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create histogram with more bins for better resolution
    n_bins = min(30, len(hit_tokens) // 3)  # Adaptive binning
    n_bins = max(10, n_bins)  # At least 10 bins
    
    plt.hist(hit_tokens, bins=n_bins, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=1)
    
    # Add vertical line for mean and median
    mean_hit_tokens = np.mean(hit_tokens)
    median_hit_tokens = np.median(hit_tokens)
    plt.axvline(x=mean_hit_tokens, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_hit_tokens:.0f}')
    plt.axvline(x=median_hit_tokens, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_hit_tokens:.0f}')
    
    # Customize the plot
    plt.xlabel('Hit Tokens', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.title('Distribution of Hit Tokens', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Hit tokens histogram saved to {output_file}")
    
    plt.show()

def create_speedup_histogram(speedups, output_file=None):
    """
    Create a histogram of speedups.
    
    Args:
        speedups (list): List of speedup values
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create histogram with more bins for better resolution
    n_bins = min(30, len(speedups) // 3)  # Adaptive binning
    n_bins = max(10, n_bins)  # At least 10 bins
    
    plt.hist(speedups, bins=n_bins, alpha=0.7, color='gold', edgecolor='black', linewidth=1)
    
    # Add vertical line for mean and median
    mean_speedup = np.mean(speedups)
    median_speedup = np.median(speedups)
    plt.axvline(x=mean_speedup, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_speedup:.2f}x')
    plt.axvline(x=median_speedup, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_speedup:.2f}x')
    
    # Add vertical line at speedup = 1.0 (no speedup)
    plt.axvline(x=1.0, color='blue', linestyle='-', linewidth=2, alpha=0.7,
                label='No speedup (1.0x)')
    
    # Customize the plot
    plt.xlabel('Speedup', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.title('Distribution of Speedups', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Speedup histogram saved to {output_file}")
    
    plt.show()

def create_ttft_vs_hit_ratio_plot(ttfts, hit_ratios, request_ids, output_file=None):
    """
    Create a scatter plot of TTFT vs hit ratio.
    
    Args:
        ttfts (list): List of TTFT values
        hit_ratios (list): List of hit ratios
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(hit_ratios, ttfts, alpha=0.7, s=50, c='crimson', edgecolors='black')
    
    # Add trend line
    if len(hit_ratios) > 1:
        z = np.polyfit(hit_ratios, ttfts, 1)
        p = np.poly1d(z)
        plt.plot(hit_ratios, p(hit_ratios), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    # Add horizontal line at average TTFT
    avg_ttft = np.mean(ttfts)
    plt.axhline(y=avg_ttft, color='blue', linestyle='-', alpha=0.5, label=f'Average TTFT: {avg_ttft:.3f}s')
    
    # Customize the plot
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=24)
    plt.ylabel('TTFT (seconds)', fontsize=24)
    plt.title('TTFT vs Cache Hit Ratio', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"TTFT vs hit ratio plot saved to {output_file}")
    
    plt.show()

def create_ttft_vs_hit_tokens_plot(ttfts, hit_tokens, request_ids, output_file=None):
    """
    Create a scatter plot of TTFT vs hit tokens.
    
    Args:
        ttfts (list): List of TTFT values
        hit_tokens (list): List of hit token counts
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(hit_tokens, ttfts, alpha=0.7, s=50, c='teal', edgecolors='black')
    
    # Add trend line
    if len(hit_tokens) > 1:
        z = np.polyfit(hit_tokens, ttfts, 1)
        p = np.poly1d(z)
        plt.plot(hit_tokens, p(hit_tokens), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.6f})')
    
    # Add horizontal line at average TTFT
    avg_ttft = np.mean(ttfts)
    plt.axhline(y=avg_ttft, color='blue', linestyle='-', alpha=0.5, label=f'Average TTFT: {avg_ttft:.3f}s')
    
    # Customize the plot
    plt.xlabel('Hit Tokens', fontsize=24)
    plt.ylabel('TTFT (seconds)', fontsize=24)
    plt.title('TTFT vs Hit Tokens', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"TTFT vs hit tokens plot saved to {output_file}")
    
    plt.show()

def calculate_actual_speedup(lmcache_data, baseline_data):
    """
    Calculate actual speedup by comparing LMCache TTFT with baseline TTFT.
    
    Args:
        lmcache_data (dict): LMCache data mapping request ID to (ttft, speedup)
        baseline_data (dict): Baseline data mapping request ID to (ttft, speedup)
        
    Returns:
        tuple: (request_ids, actual_speedups, lmcache_ttfts, baseline_ttfts) for requests with both data
    """
    # Find common request IDs between LMCache and baseline
    common_ids = set(lmcache_data.keys()) & set(baseline_data.keys())
    
    if not common_ids:
        print("Warning: No matching request IDs found between LMCache and baseline data")
        return [], [], [], []
    
    print(f"Found {len(common_ids)} matching request IDs between LMCache and baseline")
    
    request_ids = []
    actual_speedups = []
    lmcache_ttfts = []
    baseline_ttfts = []
    
    for req_id in sorted(common_ids):
        lmcache_ttft, _ = lmcache_data[req_id]
        baseline_ttft, _ = baseline_data[req_id]
        
        # Calculate actual speedup: baseline_ttft / lmcache_ttft
        # Higher values mean LMCache is faster (better)
        actual_speedup = baseline_ttft / lmcache_ttft if lmcache_ttft > 0 else 0
        
        request_ids.append(req_id)
        actual_speedups.append(actual_speedup)
        lmcache_ttfts.append(lmcache_ttft)
        baseline_ttfts.append(baseline_ttft)
        
        # Debug output for first few entries
        if len(request_ids) <= 5:
            print(f"Request {req_id}: Baseline TTFT={baseline_ttft:.3f}s, LMCache TTFT={lmcache_ttft:.3f}s, "
                  f"Actual Speedup={actual_speedup:.2f}x")
    
    return request_ids, actual_speedups, lmcache_ttfts, baseline_ttfts

def create_actual_speedup_vs_hit_ratio_plot(actual_speedups, hit_ratios, request_ids, output_file=None):
    """
    Create a scatter plot of actual speedup vs hit ratio.
    
    Args:
        actual_speedups (list): List of actual speedup values
        hit_ratios (list): List of hit ratios
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(hit_ratios, actual_speedups, alpha=0.7, s=50, c='darkgreen', edgecolors='black')
    
    # Add trend line
    if len(hit_ratios) > 1:
        z = np.polyfit(hit_ratios, actual_speedups, 1)
        p = np.poly1d(z)
        plt.plot(hit_ratios, p(hit_ratios), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.3f})')
    
    # Add horizontal line at speedup = 1.0 (no improvement)
    plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.7, linewidth=2, label='No improvement (1.0x)')
    
    # Customize the plot
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=24)
    plt.ylabel('Speedup (Base TTFT / TTFT)', fontsize=24)
    plt.title('Actual Speedup vs Cache Hit Ratio', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    

    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Actual speedup vs hit ratio plot saved to {output_file}")
    
    plt.show()

def create_3d_speedup_plot(actual_speedups, hit_ratios, total_tokens, request_ids, output_file=None):
    """
    Create a 3D scatter plot of actual speedup vs hit ratio vs total tokens with curve fitting.
    
    Args:
        actual_speedups (list): List of actual speedup values
        hit_ratios (list): List of hit ratios
        total_tokens (list): List of total token counts
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
    set_plot_fonts()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot
    scatter = ax.scatter(hit_ratios, total_tokens, actual_speedups, 
                        c=actual_speedups, cmap='viridis', alpha=0.7, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Actual Speedup', fontsize=20)
    
    # Set labels and title with increased distance from plot
    ax.set_xlabel('Cache Hit Ratio', fontsize=24, labelpad=30)
    ax.set_ylabel('Total Tokens', fontsize=24, labelpad=30)
    ax.set_zlabel('Actual Speedup', fontsize=24, labelpad=30)
    ax.set_title('3D: Actual Speedup vs Hit Ratio vs Total Tokens', fontsize=28, fontweight='bold')
    
    # Scale down the actual speedup axis to max = 25
    ax.set_zlim(0, 25)
    
    # Try to fit a surface using Gaussian Process Regression
    if len(actual_speedups) > 10:  # Need sufficient data points
        try:
            # Prepare data for Gaussian Process Regression
            x_data = np.array(hit_ratios)
            y_data = np.array(total_tokens)
            z_data = np.array(actual_speedups)
            
            # Combine x and y into a 2D feature matrix
            X_train = np.column_stack([x_data, y_data])
            
            # Define Gaussian Process kernel
            # RBF kernel for smooth variations, WhiteKernel for noise
            kernel = RBF(length_scale=[0.1, 1000], length_scale_bounds=(1e-2, 1e2)) + \
                     WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
            
            # Create and fit Gaussian Process Regressor
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                        normalize_y=True, random_state=42)
            gp.fit(X_train, z_data)
            
            # Create a grid for the surface
            x_range = np.linspace(min(x_data), max(x_data), 30)
            y_range = np.linspace(min(y_data), max(y_data), 30)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            
            # Prepare grid points for prediction
            grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
            
            # Predict mean and standard deviation
            Z_pred, Z_std = gp.predict(grid_points, return_std=True)
            Z_pred = Z_pred.reshape(X_grid.shape)
            Z_std = Z_std.reshape(X_grid.shape)
            
            # Ensure Z values are within reasonable bounds
            Z_pred = np.clip(Z_pred, 0, 25)
            
            # Plot the fitted surface
            ax.plot_surface(X_grid, Y_grid, Z_pred, alpha=0.3, color='red', linewidth=0, antialiased=True)
            
            # Optionally plot uncertainty as a semi-transparent surface
            # ax.plot_surface(X_grid, Y_grid, Z_pred + Z_std, alpha=0.1, color='gray', linewidth=0, antialiased=True)
            # ax.plot_surface(X_grid, Y_grid, Z_pred - Z_std, alpha=0.1, color='gray', linewidth=0, antialiased=True)
            
            print(f"Gaussian Process Regression completed successfully")
            print(f"Kernel parameters: {gp.kernel_}")
            print(f"Log marginal likelihood: {gp.log_marginal_likelihood():.3f}")
            
        except Exception as e:
            print(f"Could not fit Gaussian Process surface: {e}")
            # Fallback: create a simple interpolated surface
            try:
                # Create a grid for interpolation
                x_range = np.linspace(min(hit_ratios), max(hit_ratios), 30)
                y_range = np.linspace(min(total_tokens), max(total_tokens), 30)
                X, Y = np.meshgrid(x_range, y_range)
                
                # Interpolate the surface
                points = np.column_stack([hit_ratios, total_tokens])
                Z_interp = griddata(points, actual_speedups, (X, Y), method='cubic', fill_value=np.mean(actual_speedups))
                
                # Clip Z values to the same range
                Z_interp = np.clip(Z_interp, 0, 25)
                
                # Plot the interpolated surface
                ax.plot_surface(X, Y, Z_interp, alpha=0.3, color='blue', linewidth=0, antialiased=True)
                print("Used cubic interpolation for 3D surface")
                
            except Exception as e2:
                print(f"Could not create interpolated surface: {e2}")
    
    # Tilt the total tokens axis tick labels by 15 degrees counterclockwise
    ax.tick_params(axis='y', labelrotation=15)
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to {output_file}")
    
    plt.show()

def create_actual_speedup_histogram(actual_speedups, output_file=None):
    """Create histogram with smoothed distribution curve for actual speedup."""
    set_plot_fonts()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram
    n, bins, patches = ax.hist(actual_speedups, bins=30, density=True, alpha=0.7, 
                              color='skyblue', edgecolor='black', linewidth=1.2)
    
    # Create smoothed distribution curve using kernel density estimation
    if len(actual_speedups) > 1:
        # Create a range of values for the smooth curve
        x_range = np.linspace(min(actual_speedups), max(actual_speedups), 200)
        
        # Fit kernel density estimation
        kde = gaussian_kde(actual_speedups)
        density = kde(x_range)
        
        # Plot the smoothed curve
        ax.plot(x_range, density, 'r-', linewidth=3, label='Smoothed Distribution')
    
    # Add statistics
    mean_speedup = np.mean(actual_speedups)
    median_speedup = np.median(actual_speedups)
    std_speedup = np.std(actual_speedups)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_speedup, color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_speedup:.2f}x')
    ax.axvline(median_speedup, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_speedup:.2f}x')
    
    # Set labels and title
    ax.set_xlabel('Actual Speedup', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_title('Actual Speedup Distribution with Smoothed Curve', fontsize=28, fontweight='bold')
    
    # Add legend
    ax.legend(fontsize=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Actual speedup histogram saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_hit_ratio_cdf(hit_ratios, output_file=None):
    """Create CDF curve plot for hit ratio."""
    set_plot_fonts()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort the data for CDF
    sorted_ratios = np.sort(hit_ratios)
    n = len(sorted_ratios)
    
    # Calculate CDF values
    cdf_values = np.arange(1, n + 1) / n
    
    # Plot CDF
    ax.plot(sorted_ratios, cdf_values, 'b-', linewidth=3, label='Empirical CDF')
    
    # Add percentiles
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(hit_ratios, p)
        ax.axvline(value, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(value, p/100, f'{p}%', rotation=90, verticalalignment='bottom', 
                fontsize=16, color='red')
    
    # Add statistics
    mean_ratio = np.mean(hit_ratios)
    median_ratio = np.median(hit_ratios)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_ratio, color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_ratio:.3f}')
    ax.axvline(median_ratio, color='orange', linestyle='-', linewidth=2, 
               label=f'Median: {median_ratio:.3f}')
    
    # Set labels and title
    ax.set_xlabel('Cache Hit Ratio', fontsize=24)
    ax.set_ylabel('Cumulative Probability', fontsize=24)
    ax.set_title('Cumulative Distribution Function (CDF) of Cache Hit Ratio', fontsize=28, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Hit ratio CDF plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Extract and plot E2E performance data')
    parser.add_argument('--err-file', default='~/lmcache.err', 
                       help='Path to lmcache.err file (default: ~/lmcache.err)')
    parser.add_argument('--out-file', default='~/lmcache.out',
                       help='Path to lmcache.out file (default: ~/lmcache.out)')
    parser.add_argument('--baseline-file', default=None,
                       help='Path to baseline vLLM log file for comparison (optional)')
    parser.add_argument('--output-plot', default=None,
                       help='Path to save the output plot (optional)')
    parser.add_argument('--output-dir', default=None,
                       help='Directory to save all plots (optional)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting and only show data summary')
    
    args = parser.parse_args()
    
    # Expand user paths
    err_file = Path(args.err_file).expanduser()
    out_file = Path(args.out_file).expanduser()
    baseline_file = Path(args.baseline_file).expanduser() if args.baseline_file else None
    
    print("Performance Analysis")
    print("=" * 40)
    print(f"Error log: {err_file}")
    print(f"Output log: {out_file}")
    if baseline_file:
        print(f"Baseline log: {baseline_file}")
    print()
    
    # Extract data from both log files
    cache_data = extract_cache_data(err_file)
    speedup_data = extract_speedup_data(out_file)
    
    # Extract baseline data if provided
    baseline_data = None
    if baseline_file:
        baseline_data = extract_speedup_data(baseline_file)
        if not baseline_data:
            print("Warning: Failed to extract baseline data. Continuing without baseline comparison.")
            baseline_data = None
    
    if not cache_data or not speedup_data:
        print("Failed to extract data from one or both log files. Exiting.")
        return
    
    # Correlate the data
    hit_ratios, speedups, request_ids, total_tokens, hit_tokens, ttfts = correlate_data(cache_data, speedup_data)
    
    if not hit_ratios:
        print("No correlated data found. Exiting.")
        return
    
    # Debug: Check data alignment
    print(f"DEBUG: Data array lengths after correlation:")
    print(f"  hit_ratios: {len(hit_ratios)}")
    print(f"  speedups: {len(speedups)}")
    print(f"  request_ids: {len(request_ids)}")
    print(f"  total_tokens: {len(total_tokens)}")
    print(f"  hit_tokens: {len(hit_tokens)}")
    print(f"  ttfts: {len(ttfts)}")
    
    # Print summary statistics
    print("\nData Summary:")
    print("-" * 20)
    print(f"Total correlated requests: {len(hit_ratios)}")
    print(f"Average hit ratio: {np.mean(hit_ratios):.3f}")
    print(f"Average speedup: {np.mean(speedups):.2f}x")
    print(f"Median hit ratio: {np.median(hit_ratios):.3f}")
    print(f"Median speedup: {np.median(speedups):.2f}x")
    print(f"Max speedup: {np.max(speedups):.2f}x")
    print(f"Min speedup: {np.min(speedups):.2f}x")
    print(f"Max hit ratio: {np.max(hit_ratios):.3f}")
    print(f"Min hit ratio: {np.min(hit_ratios):.3f}")
    print(f"Average total tokens: {np.mean(total_tokens):.0f}")
    print(f"Median total tokens: {np.median(total_tokens):.0f}")
    print(f"Average hit tokens: {np.mean(hit_tokens):.0f}")
    print(f"Average TTFT: {np.mean(ttfts):.3f}s")
    print(f"Median TTFT: {np.median(ttfts):.3f}s")
    
    # Calculate correlation coefficients
    hit_ratio_correlation = np.corrcoef(hit_ratios, speedups)[0, 1]
    total_tokens_correlation = np.corrcoef(total_tokens, speedups)[0, 1]
    hit_tokens_correlation = np.corrcoef(hit_tokens, speedups)[0, 1]
    hit_ratio_vs_total_tokens_correlation = np.corrcoef(hit_ratios, total_tokens)[0, 1]
    ttft_vs_hit_ratio_correlation = np.corrcoef(ttfts, hit_ratios)[0, 1]
    ttft_vs_hit_tokens_correlation = np.corrcoef(ttfts, hit_tokens)[0, 1]
    
    print(f"\nCorrelation Coefficients:")
    print(f"Hit ratio vs Speedup: {hit_ratio_correlation:.3f}")
    print(f"Total tokens vs Speedup: {total_tokens_correlation:.3f}")
    print(f"Hit tokens vs Speedup: {hit_tokens_correlation:.3f}")
    print(f"Hit ratio vs Total tokens: {hit_ratio_vs_total_tokens_correlation:.3f}")
    print(f"TTFT vs Hit ratio: {ttft_vs_hit_ratio_correlation:.3f}")
    print(f"TTFT vs Hit tokens: {ttft_vs_hit_tokens_correlation:.3f}")
    
    # Calculate actual speedup from baseline if baseline data is available
    actual_speedup_data = None
    if baseline_data:
        print("\nBaseline Comparison:")
        print("-" * 20)
        actual_req_ids, actual_speedups, lmcache_ttfts, baseline_ttfts = calculate_actual_speedup(speedup_data, baseline_data)
        
        if actual_speedups:
            # Filter cache data to only include requests with actual speedup data
            actual_cache_data = {req_id: cache_data[req_id] for req_id in actual_req_ids if req_id in cache_data}
            actual_hit_ratios = []
            actual_hit_tokens = []
            
            for req_id in actual_req_ids:
                if req_id in actual_cache_data:
                    total_tokens_req, hit_tokens_req = actual_cache_data[req_id]
                    hit_ratio = hit_tokens_req / total_tokens_req if total_tokens_req > 0 else 0
                    actual_hit_ratios.append(hit_ratio)
                    actual_hit_tokens.append(hit_tokens_req)
                else:
                    # If no cache data, use 0 values
                    actual_hit_ratios.append(0)
                    actual_hit_tokens.append(0)
            
            actual_speedup_data = {
                'request_ids': actual_req_ids,
                'actual_speedups': actual_speedups,
                'hit_ratios': actual_hit_ratios,
                'hit_tokens': actual_hit_tokens,
                'lmcache_ttfts': lmcache_ttfts,
                'baseline_ttfts': baseline_ttfts
            }
            
            print(f"Actual Speedup Statistics:")
            print(f"  Average: {np.mean(actual_speedups):.2f}x")
            print(f"  Median: {np.median(actual_speedups):.2f}x")
            print(f"  Min: {np.min(actual_speedups):.2f}x")
            print(f"  Max: {np.max(actual_speedups):.2f}x")
            
            # Calculate correlation between actual speedup and hit ratio
            if len(actual_hit_ratios) > 1:
                actual_speedup_correlation = np.corrcoef(actual_hit_ratios, actual_speedups)[0, 1]
                print(f"  Correlation with hit ratio: {actual_speedup_correlation:.3f}")
        else:
            print("No matching data found between LMCache and baseline for actual speedup calculation.")
    
    # Create and show the plots
    if not args.no_plot:
        if args.output_dir:
            # If output directory is specified, create all plots and save them
            create_all_plots(hit_ratios, speedups, request_ids, total_tokens, hit_tokens, ttfts, args.output_dir)
        elif args.output_plot:
            # If output plot is specified, create all plots and save them to the plot's directory
            output_dir = Path(args.output_plot).parent
            create_all_plots(hit_ratios, speedups, request_ids, total_tokens, hit_tokens, ttfts, output_dir)
        else:
            # Create all plots interactively
            create_all_plots(hit_ratios, speedups, request_ids, total_tokens, hit_tokens, ttfts)
        
        # Create actual speedup plot if baseline data is available
        if actual_speedup_data:
            print("\nCreating actual speedup vs hit ratio plot...")
            create_actual_speedup_vs_hit_ratio_plot(
                actual_speedup_data['actual_speedups'],
                actual_speedup_data['hit_ratios'],
                actual_speedup_data['request_ids']
            )
            
            # Create 3D plot
            print("\nCreating 3D speedup plot...")
            # Get total tokens for the actual speedup data
            actual_total_tokens = []
            for req_id in actual_speedup_data['request_ids']:
                if req_id in cache_data:
                    total_tokens_req, _ = cache_data[req_id]
                    actual_total_tokens.append(total_tokens_req)
                else:
                    actual_total_tokens.append(0)
            
            create_3d_speedup_plot(
                actual_speedup_data['actual_speedups'],
                actual_speedup_data['hit_ratios'],
                actual_total_tokens,
                actual_speedup_data['request_ids']
            )
            
            # Create actual speedup histogram with smoothed distribution
            print("\nCreating actual speedup histogram...")
            create_actual_speedup_histogram(actual_speedup_data['actual_speedups'])
            
            # Create hit ratio CDF plot
            print("\nCreating hit ratio CDF plot...")
            create_hit_ratio_cdf(actual_speedup_data['hit_ratios'])
            
            # Save actual speedup plot if output directory is specified
            if args.output_dir or args.output_plot:
                output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_plot).parent
                pdf_dir = output_dir / 'pdf'
                pdf_dir.mkdir(exist_ok=True)
                
                # Save PNG
                create_actual_speedup_vs_hit_ratio_plot(
                    actual_speedup_data['actual_speedups'],
                    actual_speedup_data['hit_ratios'],
                    actual_speedup_data['request_ids'],
                    output_dir / 'actual_speedup_vs_hit_ratio.png'
                )
                
                # Save PDF
                create_actual_speedup_vs_hit_ratio_plot(
                    actual_speedup_data['actual_speedups'],
                    actual_speedup_data['hit_ratios'],
                    actual_speedup_data['request_ids'],
                    pdf_dir / 'actual_speedup_vs_hit_ratio.pdf'
                )
                
                # Save 3D plot PNG
                create_3d_speedup_plot(
                    actual_speedup_data['actual_speedups'],
                    actual_speedup_data['hit_ratios'],
                    actual_total_tokens,
                    actual_speedup_data['request_ids'],
                    output_dir / '3d_speedup_plot.png'
                )
                
                # Save 3D plot PDF
                create_3d_speedup_plot(
                    actual_speedup_data['actual_speedups'],
                    actual_speedup_data['hit_ratios'],
                    actual_total_tokens,
                    actual_speedup_data['request_ids'],
                    pdf_dir / '3d_speedup_plot.pdf'
                )
                
                # Save actual speedup histogram PNG
                create_actual_speedup_histogram(
                    actual_speedup_data['actual_speedups'],
                    output_dir / 'actual_speedup_histogram.png'
                )
                
                # Save actual speedup histogram PDF
                create_actual_speedup_histogram(
                    actual_speedup_data['actual_speedups'],
                    pdf_dir / 'actual_speedup_histogram.pdf'
                )
                
                # Save hit ratio CDF PNG
                create_hit_ratio_cdf(
                    actual_speedup_data['hit_ratios'],
                    output_dir / 'hit_ratio_cdf.png'
                )
                
                # Save hit ratio CDF PDF
                create_hit_ratio_cdf(
                    actual_speedup_data['hit_ratios'],
                    pdf_dir / 'hit_ratio_cdf.pdf'
                )
    
    # Save data to CSV for further analysis
    csv_file = 'cache_performance_data.csv'
    try:
        import pandas as pd
        df = pd.DataFrame({
            'Request_ID': request_ids,
            'Hit_Ratio': hit_ratios,
            'Speedup': speedups,
            'Total_Tokens': total_tokens,
            'Hit_Tokens': hit_tokens,
            'TTFT': ttfts
        })
        
        # Add actual speedup data if available
        if actual_speedup_data:
            # Create a mapping for actual speedup data
            actual_speedup_dict = dict(zip(actual_speedup_data['request_ids'], actual_speedup_data['actual_speedups']))
            actual_hit_ratio_dict = dict(zip(actual_speedup_data['request_ids'], actual_speedup_data['hit_ratios']))
            lmcache_ttft_dict = dict(zip(actual_speedup_data['request_ids'], actual_speedup_data['lmcache_ttfts']))
            baseline_ttft_dict = dict(zip(actual_speedup_data['request_ids'], actual_speedup_data['baseline_ttfts']))
            
            df['Actual_Speedup'] = df['Request_ID'].map(actual_speedup_dict)
            df['Actual_Hit_Ratio'] = df['Request_ID'].map(actual_hit_ratio_dict)
            df['LMCache_TTFT'] = df['Request_ID'].map(lmcache_ttft_dict)
            df['Baseline_TTFT'] = df['Request_ID'].map(baseline_ttft_dict)
        
        df.to_csv(csv_file, index=False)
        print(f"\nData saved to {csv_file} for further analysis")
    except ImportError:
        print("\nPandas not available. Skipping CSV export.")

if __name__ == "__main__":
    main()
