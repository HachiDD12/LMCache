#!/usr/bin/env python3
"""
Script to extract cache hit data and speedup information from LMCache log files
and plot the correlation between speedup and cache hit ratio.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

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
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Cache Hit Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    avg_speedup = np.mean(speedups)
    avg_hit_ratio = np.mean(hit_ratios)
    plt.text(0.02, 0.98, f'Average Speedup: {avg_speedup:.2f}x\nAverage Hit Ratio: {avg_hit_ratio:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Annotate some interesting points
    if len(hit_ratios) > 0:
        # Find highest speedup
        max_speedup_idx = np.argmax(speedups)
        plt.annotate(f'Max: {speedups[max_speedup_idx]:.2f}x\nReq {request_ids[max_speedup_idx]}',
                    xy=(hit_ratios[max_speedup_idx], speedups[max_speedup_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Find highest hit ratio
        max_hit_ratio_idx = np.argmax(hit_ratios)
        plt.annotate(f'Hit: {hit_ratios[max_hit_ratio_idx]:.3f}\nReq {request_ids[max_hit_ratio_idx]}',
                    xy=(hit_ratios[max_hit_ratio_idx], speedups[max_hit_ratio_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
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
    plt.xlabel('Total Tokens', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Total Tokens', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    avg_speedup = np.mean(speedups)
    avg_total_tokens = np.mean(total_tokens)
    plt.text(0.02, 0.98, f'Average Speedup: {avg_speedup:.2f}x\nAverage Total Tokens: {avg_total_tokens:.0f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Annotate some interesting points
    if len(total_tokens) > 0:
        # Find highest speedup
        max_speedup_idx = np.argmax(speedups)
        plt.annotate(f'Max: {speedups[max_speedup_idx]:.2f}x\nReq {request_ids[max_speedup_idx]}',
                    xy=(total_tokens[max_speedup_idx], speedups[max_speedup_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Find request with most tokens
        max_tokens_idx = np.argmax(total_tokens)
        plt.annotate(f'Tokens: {total_tokens[max_tokens_idx]:,}\nReq {request_ids[max_tokens_idx]}',
                    xy=(total_tokens[max_tokens_idx], speedups[max_tokens_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
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
    plt.xlabel('Hit Tokens', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Hit Tokens', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    avg_speedup = np.mean(speedups)
    avg_hit_tokens = np.mean(hit_tokens)
    plt.text(0.02, 0.98, f'Average Speedup: {avg_speedup:.2f}x\nAverage Hit Tokens: {avg_hit_tokens:.0f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # Annotate some interesting points
    if len(hit_tokens) > 0:
        # Find highest speedup
        max_speedup_idx = np.argmax(speedups)
        plt.annotate(f'Max: {speedups[max_speedup_idx]:.2f}x\nReq {request_ids[max_speedup_idx]}',
                    xy=(hit_tokens[max_speedup_idx], speedups[max_speedup_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Find request with most hit tokens
        max_hit_tokens_idx = np.argmax(hit_tokens)
        plt.annotate(f'Hit Tokens: {hit_tokens[max_hit_tokens_idx]:,}\nReq {max_hit_tokens_idx}',
                    xy=(hit_tokens[max_hit_tokens_idx], speedups[max_hit_tokens_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
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
        
        print(f"All plots saved to {output_dir}")

def create_hit_ratio_vs_total_tokens_plot(hit_ratios, total_tokens, request_ids, output_file=None):
    """
    Create a scatter plot of hit ratio vs total tokens.
    
    Args:
        hit_ratios (list): List of hit ratios
        total_tokens (list): List of total token counts
        request_ids (list): List of request IDs for annotation
        output_file (str, optional): Path to save the plot
    """
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
    plt.xlabel('Total Tokens', fontsize=12)
    plt.ylabel('Cache Hit Ratio', fontsize=12)
    plt.title('Cache Hit Ratio vs Total Tokens', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    avg_hit_ratio = np.mean(hit_ratios)
    avg_total_tokens = np.mean(total_tokens)
    plt.text(0.02, 0.98, f'Average Hit Ratio: {avg_hit_ratio:.3f}\nAverage Total Tokens: {avg_total_tokens:.0f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
    
    # Annotate some interesting points
    if len(hit_ratios) > 0:
        # Find highest hit ratio
        max_hit_ratio_idx = np.argmax(hit_ratios)
        plt.annotate(f'Hit: {hit_ratios[max_hit_ratio_idx]:.3f}\nReq {request_ids[max_hit_ratio_idx]}',
                    xy=(total_tokens[max_hit_ratio_idx], hit_ratios[max_hit_ratio_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Find request with most tokens
        max_tokens_idx = np.argmax(total_tokens)
        plt.annotate(f'Tokens: {total_tokens[max_tokens_idx]:,}\nReq {request_ids[max_tokens_idx]}',
                    xy=(total_tokens[max_tokens_idx], hit_ratios[max_tokens_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
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
    plt.xlabel('Cache Hit Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Cache Hit Ratios', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    plt.text(0.02, 0.98, f'Total Requests: {len(hit_ratios)}\nStd Dev: {np.std(hit_ratios):.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
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
    plt.xlabel('Hit Tokens', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Hit Tokens', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    plt.text(0.02, 0.98, f'Total Requests: {len(hit_tokens)}\nStd Dev: {np.std(hit_tokens):.0f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
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
    plt.xlabel('Speedup', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Speedups', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    plt.text(0.02, 0.98, f'Total Requests: {len(speedups)}\nStd Dev: {np.std(speedups):.2f}x',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    
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
    plt.xlabel('Cache Hit Ratio (Hit Tokens / Total Tokens)', fontsize=12)
    plt.ylabel('TTFT (seconds)', fontsize=12)
    plt.title('TTFT vs Cache Hit Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    avg_hit_ratio = np.mean(hit_ratios)
    plt.text(0.02, 0.98, f'Average Hit Ratio: {avg_hit_ratio:.3f}\nAverage TTFT: {avg_ttft:.3f}s',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Annotate some interesting points
    if len(hit_ratios) > 0:
        # Find lowest TTFT
        min_ttft_idx = np.argmin(ttfts)
        plt.annotate(f'Min TTFT: {ttfts[min_ttft_idx]:.3f}s\nReq {request_ids[min_ttft_idx]}',
                    xy=(hit_ratios[min_ttft_idx], ttfts[min_ttft_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Find highest hit ratio
        max_hit_ratio_idx = np.argmax(hit_ratios)
        plt.annotate(f'Hit: {hit_ratios[max_hit_ratio_idx]:.3f}\nReq {request_ids[max_hit_ratio_idx]}',
                    xy=(hit_ratios[max_hit_ratio_idx], ttfts[max_hit_ratio_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
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
    plt.xlabel('Hit Tokens', fontsize=12)
    plt.ylabel('TTFT (seconds)', fontsize=12)
    plt.title('TTFT vs Hit Tokens', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    avg_hit_tokens = np.mean(hit_tokens)
    plt.text(0.02, 0.98, f'Average Hit Tokens: {avg_hit_tokens:.0f}\nAverage TTFT: {avg_ttft:.3f}s',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightseagreen', alpha=0.8))
    
    # Annotate some interesting points
    if len(hit_tokens) > 0:
        # Find lowest TTFT
        min_ttft_idx = np.argmin(ttfts)
        plt.annotate(f'Min TTFT: {ttfts[min_ttft_idx]:.3f}s\nReq {request_ids[min_ttft_idx]}',
                    xy=(hit_tokens[min_ttft_idx], ttfts[min_ttft_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Find request with most hit tokens
        max_hit_tokens_idx = np.argmax(hit_tokens)
        plt.annotate(f'Hit Tokens: {hit_tokens[max_hit_tokens_idx]:,}\nReq {request_ids[max_hit_tokens_idx]}',
                    xy=(hit_tokens[max_hit_tokens_idx], ttfts[max_hit_tokens_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"TTFT vs hit tokens plot saved to {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extract and plot E2E performance data')
    parser.add_argument('--err-file', default='~/lmcache.err', 
                       help='Path to lmcache.err file (default: ~/lmcache.err)')
    parser.add_argument('--out-file', default='~/lmcache.out',
                       help='Path to lmcache.out file (default: ~/lmcache.out)')
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
    
    print("Performance Analysis")
    print("=" * 40)
    print(f"Error log: {err_file}")
    print(f"Output log: {out_file}")
    print()
    
    # Extract data from both log files
    cache_data = extract_cache_data(err_file)
    speedup_data = extract_speedup_data(out_file)
    
    if not cache_data or not speedup_data:
        print("Failed to extract data from one or both log files. Exiting.")
        return
    
    # Correlate the data
    hit_ratios, speedups, request_ids, total_tokens, hit_tokens, ttfts = correlate_data(cache_data, speedup_data)
    
    if not hit_ratios:
        print("No correlated data found. Exiting.")
        return
    
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
        df.to_csv(csv_file, index=False)
        print(f"\nData saved to {csv_file} for further analysis")
    except ImportError:
        print("\nPandas not available. Skipping CSV export.")

if __name__ == "__main__":
    main()
