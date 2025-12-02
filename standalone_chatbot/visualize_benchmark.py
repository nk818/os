#!/usr/bin/env python3
"""
Visualize Benchmark Results
Creates charts comparing optimized vs baseline performance
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_comparison_charts(results: dict, output_dir: str = "benchmark_plots"):
    """Create visualization charts from benchmark results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    baseline = results["baseline"]
    optimized = results["optimized"]
    improvements = results["improvements"]
    
    # 1. Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Total Time (s)", "Tokens/Second"]
    baseline_values = [baseline["total_time"], baseline["tokens_per_second"]]
    optimized_values = [optimized["total_time"], optimized["tokens_per_second"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color='#4ecdc4')
    
    ax.set_ylabel('Value')
    ax.set_title('Time Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_comparison.png", dpi=150)
    print(f"✅ Saved: {output_dir}/time_comparison.png")
    plt.close()
    
    # 2. Token Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Input Tokens", "Output Tokens", "Total Tokens"]
    baseline_values = [baseline["input_tokens"], baseline["output_tokens"], baseline["total_tokens"]]
    optimized_values = [optimized["input_tokens"], optimized["output_tokens"], optimized["total_tokens"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color='#4ecdc4')
    
    ax.set_ylabel('Tokens')
    ax.set_title('Token Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_comparison.png", dpi=150)
    print(f"✅ Saved: {output_dir}/token_comparison.png")
    plt.close()
    
    # 3. Memory Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Peak Memory (MB)", "Average Memory (MB)"]
    baseline_values = [baseline["peak_memory_mb"], baseline["average_memory_mb"]]
    optimized_values = [optimized["peak_memory_mb"], optimized["average_memory_mb"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color='#4ecdc4')
    
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_comparison.png", dpi=150)
    print(f"✅ Saved: {output_dir}/memory_comparison.png")
    plt.close()
    
    # 4. Improvement Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Time\nReduction", "Token\nReduction", "Memory\nReduction", "Throughput\nImprovement"]
    improvement_values = [
        improvements["time_reduction_percent"],
        improvements["token_reduction_percent"],
        improvements["memory_reduction_percent"],
        improvements["throughput_improvement_percent"],
    ]
    
    colors = ['#95e1d3' if v > 0 else '#f38181' for v in improvement_values]
    bars = ax.bar(categories, improvement_values, color=colors)
    
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Optimization Improvements')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvements.png", dpi=150)
    print(f"✅ Saved: {output_dir}/improvements.png")
    plt.close()
    
    # 5. Comprehensive Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Time metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Baseline', 'Optimized'], [baseline["total_time"], optimized["total_time"]], 
            color=['#ff6b6b', '#4ecdc4'])
    ax1.set_title('Total Time (s)')
    ax1.set_ylabel('Seconds')
    ax1.grid(axis='y', alpha=0.3)
    
    # Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['Baseline', 'Optimized'], [baseline["tokens_per_second"], optimized["tokens_per_second"]], 
            color=['#ff6b6b', '#4ecdc4'])
    ax2.set_title('Throughput (tokens/s)')
    ax2.set_ylabel('Tokens/Second')
    ax2.grid(axis='y', alpha=0.3)
    
    # Token reduction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(['Total Tokens'], [baseline["total_tokens"]], color='#ff6b6b', label='Baseline')
    ax3.bar(['Total Tokens'], [optimized["total_tokens"]], color='#4ecdc4', label='Optimized', 
            bottom=[0], alpha=0.7)
    ax3.set_title('Total Tokens')
    ax3.set_ylabel('Tokens')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Memory
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(['Peak', 'Average'], [baseline["peak_memory_mb"], baseline["average_memory_mb"]], 
            color='#ff6b6b', alpha=0.7, label='Baseline')
    ax4.bar(['Peak', 'Average'], [optimized["peak_memory_mb"], optimized["average_memory_mb"]], 
            color='#4ecdc4', alpha=0.7, label='Optimized')
    ax4.set_title('Memory Usage (MB)')
    ax4.set_ylabel('MB')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Improvements
    ax5 = fig.add_subplot(gs[1, 1:])
    categories = ['Time', 'Tokens', 'Memory', 'Throughput']
    values = [
        improvements["time_reduction_percent"],
        improvements["token_reduction_percent"],
        improvements["memory_reduction_percent"],
        improvements["throughput_improvement_percent"],
    ]
    colors = ['#95e1d3' if v > 0 else '#f38181' for v in values]
    bars = ax5.bar(categories, values, color=colors)
    ax5.set_title('Improvement Summary (%)')
    ax5.set_ylabel('Improvement %')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    # Configuration
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    config_text = f"""
    Model: {baseline['model_name']}
    AdaptiVocab: {'✅' if optimized['adaptivocab'] else '❌'}
    LaRoSA: {'✅' if optimized['larosa_sparsity'] > 0 else '❌'} ({optimized['larosa_sparsity']*100:.0f}% sparsity)
    vAttention: {'✅' if optimized['vattention'] else '❌'}
    """
    ax6.text(0.1, 0.5, config_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Benchmark Comparison Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(f"{output_dir}/comprehensive_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/comprehensive_dashboard.png")
    plt.close()
    
    print(f"\n📊 All visualizations saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize Benchmark Results")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_results.json",
        help="Input JSON file with benchmark results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_plots",
        help="Output directory for plots (default: benchmark_plots)",
    )
    
    args = parser.parse_args()
    
    print("📊 Loading benchmark results...")
    results = load_benchmark_results(args.input)
    
    print("📈 Creating visualizations...")
    create_comparison_charts(results, args.output_dir)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()

