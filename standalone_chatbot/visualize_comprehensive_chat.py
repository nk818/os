#!/usr/bin/env python3
"""
Visualize comprehensive chat engine results
Shows all optimization methods with time, token, memory, and efficiency metrics
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

def load_results(results_file: str = "comprehensive_results.json"):
    """Load results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        # Filter out None results
        return {k: v for k, v in results.items() if v is not None}
    except FileNotFoundError:
        print(f"❌ File not found: {results_file}")
        print("   Run comprehensive_chat_engine.py first to generate results")
        return None

def create_visualizations(results: dict):
    """Create comprehensive visualizations."""
    output_dir = Path("comprehensive_chat_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    methods = list(results.keys())
    baseline = results["baseline"]["metrics"]
    
    # Prepare data arrays
    times = [results[m]["metrics"]["total_time"] for m in methods]
    gen_times = [results[m]["metrics"]["generation_time"] for m in methods]
    input_tokens = [results[m]["metrics"]["input_tokens"] for m in methods]
    output_tokens = [results[m]["metrics"]["output_tokens"] for m in methods]
    total_tokens = [results[m]["metrics"]["total_tokens"] for m in methods]
    peak_memory = [results[m]["metrics"]["peak_memory_mb"] for m in methods]
    tokens_per_sec = [results[m]["metrics"]["tokens_per_second"] for m in methods]
    efficiency_scores = [results[m]["metrics"]["efficiency_score"] for m in methods]
    
    # Calculate improvements relative to baseline
    time_improvements = [((baseline["total_time"] - t) / baseline["total_time"] * 100) for t in times]
    token_diffs = [t - baseline["total_tokens"] for t in total_tokens]
    token_pcts = [((baseline["total_tokens"] - t) / baseline["total_tokens"] * 100) for t in total_tokens]
    memory_diffs = [m - baseline["peak_memory_mb"] for m in peak_memory]
    memory_pcts = [((baseline["peak_memory_mb"] - m) / baseline["peak_memory_mb"] * 100) if baseline["peak_memory_mb"] > 0 else 0 for m in peak_memory]
    
    # 1. Time Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Chat Engine - Time Metrics', fontsize=16, fontweight='bold')
    
    # Total time
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, times, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Total Time Comparison', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    for bar, time_val, imp in zip(bars1, times, time_improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s\n({imp:+.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # Generation time
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, gen_times, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Generation Time Comparison', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars2, gen_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s',
                ha='center', va='bottom', fontsize=9)
    
    # Time improvement
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, time_improvements, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('Time Improvement vs Baseline', fontsize=13, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    for bar, imp in zip(bars3, time_improvements):
        height = bar.get_height()
        color = 'green' if imp > 0 else 'red'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top', fontsize=10, color=color, fontweight='bold')
    
    # Tokens per second
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, tokens_per_sec, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax4.set_ylabel('Tokens/Second', fontsize=12)
    ax4.set_title('Throughput (Tokens/Second)', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    for bar, tps in zip(bars4, tokens_per_sec):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{tps:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/time_metrics.png")
    plt.close()
    
    # 2. Token Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Chat Engine - Token Metrics', fontsize=16, fontweight='bold')
    
    # Total tokens
    ax1 = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax1.bar(x - width/2, input_tokens, width, label='Input', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, output_tokens, width, label='Output', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Tokens', fontsize=12)
    ax1.set_title('Input vs Output Tokens', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Total tokens comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, total_tokens, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax2.set_ylabel('Total Tokens', fontsize=12)
    ax2.set_title('Total Token Count', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar, tokens, diff in zip(bars2, total_tokens, token_diffs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{tokens}\n({diff:+d})',
                ha='center', va='bottom', fontsize=9)
    
    # Token reduction
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, token_pcts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax3.set_ylabel('Token Reduction (%)', fontsize=12)
    ax3.set_title('Token Reduction vs Baseline', fontsize=13, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars3, token_pcts):
        height = bar.get_height()
        color = 'green' if pct > 0 else 'red'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:+.1f}%',
                ha='center', va='bottom' if pct > 0 else 'top', fontsize=10, color=color, fontweight='bold')
    
    # Token breakdown
    ax4 = axes[1, 1]
    x = np.arange(len(methods))
    width = 0.25
    ax4.bar(x - width, input_tokens, width, label='Input', color='#3498db', alpha=0.8)
    ax4.bar(x, output_tokens, width, label='Output', color='#e74c3c', alpha=0.8)
    ax4.bar(x + width, total_tokens, width, label='Total', color='#2ecc71', alpha=0.8)
    ax4.set_ylabel('Tokens', fontsize=12)
    ax4.set_title('Token Breakdown by Method', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "token_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/token_metrics.png")
    plt.close()
    
    # 3. Memory Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comprehensive Chat Engine - Memory Metrics', fontsize=16, fontweight='bold')
    
    # Peak memory
    ax1 = axes[0]
    bars1 = ax1.bar(methods, peak_memory, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax1.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax1.set_title('Peak Memory Usage', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    for bar, mem, diff in zip(bars1, peak_memory, memory_diffs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f} MB\n({diff:+.1f})',
                ha='center', va='bottom', fontsize=9)
    
    # Memory reduction
    ax2 = axes[1]
    bars2 = ax2.bar(methods, memory_pcts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax2.set_ylabel('Memory Reduction (%)', fontsize=12)
    ax2.set_title('Memory Reduction vs Baseline', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars2, memory_pcts):
        height = bar.get_height()
        color = 'green' if pct > 0 else 'red'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:+.1f}%',
                ha='center', va='bottom' if pct > 0 else 'top', fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "memory_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/memory_metrics.png")
    plt.close()
    
    # 4. Efficiency Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comprehensive Chat Engine - Efficiency Metrics', fontsize=16, fontweight='bold')
    
    # Efficiency scores
    ax1 = axes[0]
    bars1 = ax1.bar(methods, efficiency_scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax1.set_ylabel('Efficiency Score', fontsize=12)
    ax1.set_title('Combined Efficiency Score (Higher is Better)', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    for bar, score in zip(bars1, efficiency_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Tokens per second
    ax2 = axes[1]
    bars2 = ax2.bar(methods, tokens_per_sec, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax2.set_ylabel('Tokens/Second', fontsize=12)
    ax2.set_title('Throughput Efficiency', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar, tps in zip(bars2, tokens_per_sec):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{tps:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/efficiency_metrics.png")
    plt.close()
    
    # 5. Comprehensive Dashboard
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    fig.suptitle('Comprehensive Chat Engine - All Methods Comparison Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Total time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(methods, times, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Total Time', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total tokens
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(methods, total_tokens, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax2.set_ylabel('Tokens')
    ax2.set_title('Total Tokens', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Peak memory
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(methods, peak_memory, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('Peak Memory', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Time improvement
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(methods, time_improvements, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Time Improvement', fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Token reduction
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(methods, token_pcts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax5.set_ylabel('Reduction (%)')
    ax5.set_title('Token Reduction', fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Memory reduction
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(methods, memory_pcts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax6.set_ylabel('Reduction (%)')
    ax6.set_title('Memory Reduction', fontweight='bold')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Throughput
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.bar(methods, tokens_per_sec, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax7.set_ylabel('Tokens/sec')
    ax7.set_title('Throughput', fontweight='bold')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Efficiency score
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.bar(methods, efficiency_scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax8.set_ylabel('Score')
    ax8.set_title('Efficiency Score', fontweight='bold')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Combined improvement
    ax9 = fig.add_subplot(gs[2, 2])
    # Normalize all improvements to 0-100 scale for comparison
    normalized_improvements = []
    for i, method in enumerate(methods):
        # Combine time, token, and memory improvements
        time_imp = max(0, time_improvements[i])  # Only positive improvements
        token_imp = max(0, token_pcts[i])
        mem_imp = max(0, memory_pcts[i])
        combined = (time_imp + token_imp + mem_imp) / 3
        normalized_improvements.append(combined)
    
    ax9.bar(methods, normalized_improvements, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax9.set_ylabel('Combined Improvement (%)')
    ax9.set_title('Overall Improvement Score', fontweight='bold')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_dir / "comprehensive_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/comprehensive_dashboard.png")
    plt.close()
    
    print(f"\n📊 All visualizations saved to: {output_dir}/")
    print("   • time_metrics.png")
    print("   • token_metrics.png")
    print("   • memory_metrics.png")
    print("   • efficiency_metrics.png")
    print("   • comprehensive_dashboard.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize comprehensive chat engine results")
    parser.add_argument("--input", type=str, default="comprehensive_results.json", help="Input JSON file")
    
    args = parser.parse_args()
    
    print("📊 Creating Comprehensive Chat Visualizations...")
    print("="*70)
    
    results = load_results(args.input)
    if results:
        create_visualizations(results)
        print("\n✅ Visualization complete!")
    else:
        print("\n❌ Could not load results. Please run comprehensive_chat_engine.py first.")

