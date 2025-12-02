#!/usr/bin/env python3
"""
Visualize Multi-Query Comparison Results
Shows input/output pairs, quality metrics, and performance across queries
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)


def load_results(results_file: str = "multi_query_results.json") -> Dict:
    """Load results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"❌ File not found: {results_file}")
        print("   Run multi_query_comparison.py first to generate results")
        return None


def create_visualizations(results: Dict, output_folder: str = "multi_query_plots"):
    """Create comprehensive visualizations for multi-query results."""
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    query_ids = list(results.keys())
    methods = set()
    for query_data in results.values():
        methods.update(query_data['methods'].keys())
    methods = sorted(list(methods))
    
    # Prepare data arrays
    query_names = []
    method_names = []
    input_texts = []
    output_texts = []
    output_lengths = []
    tokens = []
    times = []
    memories = []
    efficiency_scores = []
    
    for query_id, query_data in results.items():
        query_name = query_data.get('category', query_id)
        input_text = query_data['query_input']
        
        for method, data in query_data['methods'].items():
            query_names.append(query_id)  # Use query_id instead of category for consistency
            method_names.append(method)
            input_texts.append(input_text)
            output_texts.append(data['output'])
            output_lengths.append(len(data['output']))
            
            metrics = data['metrics']
            tokens.append(metrics['total_tokens'])
            times.append(metrics['total_time'])
            memories.append(metrics['peak_memory_mb'])
            efficiency_scores.append(metrics['efficiency_score'])
    
    # 1. Output Quality Comparison (Length)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Query Output Quality Comparison', fontsize=16, fontweight='bold')
    
    # Output length by query and method
    ax1 = axes[0, 0]
    query_method_data = {}
    for i, (q, m) in enumerate(zip(query_names, method_names)):
        key = (q, m)
        if key not in query_method_data:
            query_method_data[key] = []
        query_method_data[key].append(output_lengths[i])
    
    x_pos = np.arange(len(query_ids))
    width = 0.15
    for idx, method in enumerate(methods):
        values = [np.mean([query_method_data.get((q, method), [0])[0] for q in query_ids]) 
                  for q in query_ids]
        ax1.bar(x_pos + idx * width, values, width, label=method, alpha=0.8)
    
    ax1.set_xlabel('Query', fontsize=12)
    ax1.set_ylabel('Output Length (chars)', fontsize=12)
    ax1.set_title('Average Output Length by Query and Method', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(query_ids, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Token usage comparison
    ax2 = axes[0, 1]
    method_tokens = {m: [] for m in methods}
    for m, t in zip(method_names, tokens):
        method_tokens[m].append(t)
    
    method_avg_tokens = {m: np.mean(tokens) for m, tokens in method_tokens.items()}
    bars = ax2.bar(method_avg_tokens.keys(), method_avg_tokens.values(), 
                    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax2.set_ylabel('Average Tokens', fontsize=12)
    ax2.set_title('Average Token Usage by Method', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, method_avg_tokens.values()):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Time comparison
    ax3 = axes[1, 0]
    method_times = {m: [] for m in methods}
    for m, t in zip(method_names, times):
        method_times[m].append(t)
    
    method_avg_times = {m: np.mean(times) for m, times in method_times.items()}
    bars = ax3.bar(method_avg_times.keys(), method_avg_times.values(),
                    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax3.set_ylabel('Average Time (seconds)', fontsize=12)
    ax3.set_title('Average Generation Time by Method', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, method_avg_times.values()):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Memory comparison
    ax4 = axes[1, 1]
    method_memories = {m: [] for m in methods}
    for m, mem in zip(method_names, memories):
        if mem > 0:  # Filter out zero values
            method_memories[m].append(mem)
    
    method_avg_memories = {m: np.mean(mems) if mems else 0 for m, mems in method_memories.items()}
    bars = ax4.bar(method_avg_memories.keys(), method_avg_memories.values(),
                    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax4.set_ylabel('Average Memory (MB)', fontsize=12)
    ax4.set_title('Average Memory Usage by Method', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, method_avg_memories.values()):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "quality_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/quality_comparison.png")
    plt.close()
    
    # 2. Input/Output Pairs Visualization
    fig, axes = plt.subplots(len(query_ids), 1, figsize=(18, 6 * len(query_ids)))
    if len(query_ids) == 1:
        axes = [axes]
    
    fig.suptitle('Input/Output Pairs by Query and Method', fontsize=16, fontweight='bold')
    
    for query_idx, query_id in enumerate(query_ids):
        ax = axes[query_idx]
        query_data = results[query_id]
        input_text = query_data['query_input']
        
        # Create text visualization
        y_pos = 0.95
        ax.text(0.05, y_pos, f"Query: {input_text}", 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        y_pos -= 0.12
        for method in methods:
            if method in query_data['methods']:
                output = query_data['methods'][method]['output']
                metrics = query_data['methods'][method]['metrics']
                
                # Truncate long outputs
                display_output = output[:150] + "..." if len(output) > 150 else output
                
                method_text = f"{method}: {display_output}"
                ax.text(0.05, y_pos, method_text,
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
                
                # Add metrics
                metrics_text = f"Tokens: {metrics['total_tokens']}, Time: {metrics['total_time']:.1f}s, Memory: {metrics['peak_memory_mb']:.0f}MB"
                ax.text(0.05, y_pos - 0.04, metrics_text,
                       transform=ax.transAxes, fontsize=8, style='italic', color='gray')
                
                y_pos -= 0.15
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"{query_id} ({query_data.get('category', 'N/A')})", 
                    fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "input_output_pairs.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/input_output_pairs.png")
    plt.close()
    
    # 3. Performance Dashboard
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    fig.suptitle('Multi-Query Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Average tokens
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(method_avg_tokens.keys(), method_avg_tokens.values(),
            color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax1.set_ylabel('Tokens')
    ax1.set_title('Avg Tokens', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Average time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(method_avg_times.keys(), method_avg_times.values(),
            color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Avg Time', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Average memory
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(method_avg_memories.keys(), method_avg_memories.values(),
            color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('Avg Memory', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Efficiency scores
    ax4 = fig.add_subplot(gs[1, 0])
    method_efficiency = {m: [] for m in methods}
    for m, e in zip(method_names, efficiency_scores):
        method_efficiency[m].append(e)
    method_avg_efficiency = {m: np.mean(effs) for m, effs in method_efficiency.items()}
    ax4.bar(method_avg_efficiency.keys(), method_avg_efficiency.values(),
            color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'][:len(methods)])
    ax4.set_ylabel('Score')
    ax4.set_title('Avg Efficiency Score', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Tokens per query
    ax5 = fig.add_subplot(gs[1, 1])
    query_tokens = {q: [] for q in query_ids}
    for q_id, t in zip(query_names, tokens):
        if q_id in query_tokens:
            query_tokens[q_id].append(t)
    query_avg_tokens = {q: np.mean(ts) if ts else 0 for q, ts in query_tokens.items()}
    ax5.bar(query_avg_tokens.keys(), query_avg_tokens.values())
    ax5.set_ylabel('Tokens')
    ax5.set_title('Avg Tokens per Query', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # Output length distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(output_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Output Length (chars)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Output Length Distribution', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Method comparison across queries
    ax7 = fig.add_subplot(gs[2, :])
    x = np.arange(len(query_ids))
    width = 0.15
    for idx, method in enumerate(methods):
        values = [np.mean([tokens[i] for i in range(len(tokens)) 
                          if query_names[i] == q and method_names[i] == method])
                  for q in query_ids]
        ax7.bar(x + idx * width, values, width, label=method, alpha=0.8)
    
    ax7.set_xlabel('Query', fontsize=12)
    ax7.set_ylabel('Average Tokens', fontsize=12)
    ax7.set_title('Token Usage Across Queries by Method', fontsize=13, fontweight='bold')
    ax7.set_xticks(x + width * (len(methods) - 1) / 2)
    ax7.set_xticklabels(query_ids)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_dir / "performance_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/performance_dashboard.png")
    plt.close()
    
    print(f"\n📊 All visualizations saved to: {output_dir}/")
    print("   • quality_comparison.png")
    print("   • input_output_pairs.png")
    print("   • performance_dashboard.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize multi-query comparison results")
    parser.add_argument("--input", type=str, default="multi_query_results.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="multi_query_plots", help="Output folder for plots")
    
    args = parser.parse_args()
    
    print("📊 Creating Multi-Query Visualizations...")
    print("="*70)
    
    results = load_results(args.input)
    if results:
        create_visualizations(results, args.output)
        print("\n✅ Visualization complete!")
    else:
        print("\n❌ Could not load results. Please run multi_query_comparison.py first.")

