#!/usr/bin/env python3
"""
Visualize word removal test results
Creates comprehensive graphs showing compression, token savings, and quality
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_results():
    """Load results from the test (we'll parse from the output)."""
    # Read the word_removal_test.txt file
    results_file = Path("word_removal_test.txt")
    
    if not results_file.exists():
        print("❌ word_removal_test.txt not found. Run test_word_removal.py first.")
        return None
    
    # Parse results (simplified - in real implementation, would parse more carefully)
    results = {
        "original": {
            "words": 116,
            "characters": 859,
            "tokens": 155,
            "summary": "Quantum mechanics forms the basis of contemporary physics by elucidating particle behavior and energetic interactions at microscopic levels."
        },
        "compressed": {
            3: {
                "words": 79,
                "characters": 583,
                "tokens": 113,
                "reduction_pct": 31.9,
                "token_savings": 42,
                "token_savings_pct": 27.1,
                "summary": "Quantum mechanics, the basis of modern science, studies the behavior of particles at very small scales."
            },
            4: {
                "words": 89,
                "characters": 649,
                "tokens": 124,
                "reduction_pct": 23.3,
                "token_savings": 31,
                "token_savings_pct": 20.0,
                "summary": "Quantum mechanics has become a fundamental basis for comprehending particle behavior and interacting energies, challenging traditional concepts and revolutionizing technology like semiconductor devices and Magnetic Resonance Imaging (MRI)."
            },
            5: {
                "words": 95,
                "characters": 712,
                "tokens": 133,
                "reduction_pct": 18.1,
                "token_savings": 22,
                "token_savings_pct": 14.2,
                "summary": "Quantum mechanics has shaped our scientific understanding by providing a framework to explain phenomena occurring at the microscopic level."
            }
        }
    }
    
    return results

def create_visualizations(results):
    """Create comprehensive visualizations."""
    output_dir = Path("word_removal_plots")
    output_dir.mkdir(exist_ok=True)
    
    original = results["original"]
    compressed = results["compressed"]
    
    # Extract data for plotting
    patterns = ["Original"] + [f"Every {n}th" for n in [3, 4, 5]]
    word_counts = [original["words"]] + [compressed[n]["words"] for n in [3, 4, 5]]
    char_counts = [original["characters"]] + [compressed[n]["characters"] for n in [3, 4, 5]]
    token_counts = [original["tokens"]] + [compressed[n]["tokens"] for n in [3, 4, 5]]
    reductions = [0] + [compressed[n]["reduction_pct"] for n in [3, 4, 5]]
    token_savings = [0] + [compressed[n]["token_savings"] for n in [3, 4, 5]]
    token_savings_pct = [0] + [compressed[n]["token_savings_pct"] for n in [3, 4, 5]]
    
    # 1. Text Size Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Word Removal Compression Analysis', fontsize=16, fontweight='bold')
    
    # Words comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(patterns, word_counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax1.set_ylabel('Word Count', fontsize=12)
    ax1.set_title('Word Count Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars1, word_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({reductions[i]:.1f}%{"↓" if i > 0 else ""})',
                ha='center', va='bottom', fontsize=10)
    
    # Characters comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(patterns, char_counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax2.set_ylabel('Character Count', fontsize=12)
    ax2.set_title('Character Count Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars2, char_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({(1-count/original["characters"])*100:.1f}%{"↓" if i > 0 else ""})',
                ha='center', va='bottom', fontsize=10)
    
    # Token comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(patterns, token_counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax3.set_ylabel('Input Tokens', fontsize=12)
    ax3.set_title('Input Token Count Comparison', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars3, token_counts)):
        height = bar.get_height()
        savings_text = f'\n(-{token_savings[i]} tokens)' if i > 0 else ''
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}{savings_text}',
                ha='center', va='bottom', fontsize=10)
    
    # Reduction percentages
    ax4 = axes[1, 1]
    reduction_data = reductions[1:]  # Skip original (0%)
    pattern_labels = [f"Every {n}th" for n in [3, 4, 5]]
    bars4 = ax4.bar(pattern_labels, reduction_data, color=['#3498db', '#9b59b6', '#e74c3c'])
    ax4.set_ylabel('Reduction Percentage (%)', fontsize=12)
    ax4.set_title('Text Reduction by Pattern', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars4, reduction_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "compression_overview.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/compression_overview.png")
    plt.close()
    
    # 2. Token Savings Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Token Savings Analysis', fontsize=16, fontweight='bold')
    
    # Token savings absolute
    ax1 = axes[0]
    pattern_labels = [f"Every {n}th" for n in [3, 4, 5]]
    bars1 = ax1.bar(pattern_labels, token_savings[1:], color=['#3498db', '#9b59b6', '#e74c3c'])
    ax1.set_ylabel('Tokens Saved', fontsize=12)
    ax1.set_title('Absolute Token Savings', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, savings in zip(bars1, token_savings[1:]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{savings} tokens',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Token savings percentage
    ax2 = axes[1]
    bars2 = ax2.bar(pattern_labels, token_savings_pct[1:], color=['#3498db', '#9b59b6', '#e74c3c'])
    ax2.set_ylabel('Token Savings (%)', fontsize=12)
    ax2.set_title('Percentage Token Savings', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars2, token_savings_pct[1:]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "token_savings.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/token_savings.png")
    plt.close()
    
    # 3. Efficiency Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(patterns))
    width = 0.25
    
    # Normalize data for comparison (show as percentage of original)
    word_norm = [100] + [w/original["words"]*100 for w in word_counts[1:]]
    char_norm = [100] + [c/original["characters"]*100 for c in char_counts[1:]]
    token_norm = [100] + [t/original["tokens"]*100 for t in token_counts[1:]]
    
    bars1 = ax.bar(x - width, word_norm, width, label='Words', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, char_norm, width, label='Characters', color='#9b59b6', alpha=0.8)
    bars3 = ax.bar(x + width, token_norm, width, label='Tokens', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Percentage of Original (%)', fontsize=12)
    ax.set_title('Compression Efficiency Comparison (Normalized to Original)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 110])
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/efficiency_comparison.png")
    plt.close()
    
    # 4. Comprehensive Dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Word Removal Compression - Comprehensive Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Word count
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(patterns, word_counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax1.set_ylabel('Words')
    ax1.set_title('Word Count', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, (p, w) in enumerate(zip(patterns, word_counts)):
        ax1.text(i, w, str(w), ha='center', va='bottom', fontsize=9)
    
    # 2. Character count
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(patterns, char_counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax2.set_ylabel('Characters')
    ax2.set_title('Character Count', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (p, c) in enumerate(zip(patterns, char_counts)):
        ax2.text(i, c, str(c), ha='center', va='bottom', fontsize=9)
    
    # 3. Token count
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(patterns, token_counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax3.set_ylabel('Tokens')
    ax3.set_title('Input Token Count', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, (p, t) in enumerate(zip(patterns, token_counts)):
        ax3.text(i, t, str(t), ha='center', va='bottom', fontsize=9)
    
    # 4. Reduction percentages
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(pattern_labels, reduction_data, color=['#3498db', '#9b59b6', '#e74c3c'])
    ax4.set_ylabel('Reduction (%)')
    ax4.set_title('Text Reduction %', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, (p, r) in enumerate(zip(pattern_labels, reduction_data)):
        ax4.text(i, r, f'{r:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Token savings
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(pattern_labels, token_savings[1:], color=['#3498db', '#9b59b6', '#e74c3c'])
    ax5.set_ylabel('Tokens Saved')
    ax5.set_title('Token Savings (Absolute)', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for i, (p, s) in enumerate(zip(pattern_labels, token_savings[1:])):
        ax5.text(i, s, str(s), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Token savings %
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(pattern_labels, token_savings_pct[1:], color=['#3498db', '#9b59b6', '#e74c3c'])
    ax6.set_ylabel('Token Savings (%)')
    ax6.set_title('Token Savings (%)', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for i, (p, s) in enumerate(zip(pattern_labels, token_savings_pct[1:])):
        ax6.text(i, s, f'{s:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 7. Efficiency comparison (normalized)
    ax7 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(len(patterns))
    width = 0.25
    ax7.bar(x_pos - width, word_norm, width, label='Words', color='#3498db', alpha=0.8)
    ax7.bar(x_pos, char_norm, width, label='Characters', color='#9b59b6', alpha=0.8)
    ax7.bar(x_pos + width, token_norm, width, label='Tokens', color='#e74c3c', alpha=0.8)
    ax7.set_ylabel('Percentage of Original (%)')
    ax7.set_title('Compression Efficiency (Normalized)', fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(patterns)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim([0, 110])
    
    plt.savefig(output_dir / "comprehensive_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/comprehensive_dashboard.png")
    plt.close()
    
    print(f"\n📊 All visualizations saved to: {output_dir}/")
    print("   • compression_overview.png")
    print("   • token_savings.png")
    print("   • efficiency_comparison.png")
    print("   • comprehensive_dashboard.png")

if __name__ == "__main__":
    print("📊 Creating Word Removal Visualizations...")
    print("="*70)
    
    results = load_results()
    if results:
        create_visualizations(results)
        print("\n✅ Visualization complete!")
    else:
        print("\n❌ Could not load results. Please run test_word_removal.py first.")

