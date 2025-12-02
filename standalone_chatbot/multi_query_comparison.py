#!/usr/bin/env python3
"""
Multi-Query Comparison with Input/Output Tracking
Tests different queries across optimization methods and saves results
"""

import json
import time
from pathlib import Path
from typing import Dict, List
from comprehensive_chat_engine import ComprehensiveChatEngine, compare_all_methods

# Different test queries
TEST_QUERIES = [
    {
        "id": "query_1",
        "input": "What is artificial intelligence?",
        "category": "definition",
        "expected_keywords": ["intelligence", "machine", "computer", "learning"]
    },
    {
        "id": "query_2",
        "input": "Explain quantum mechanics in simple terms.",
        "category": "explanation",
        "expected_keywords": ["quantum", "particles", "physics", "energy"]
    },
    {
        "id": "query_3",
        "input": "How does machine learning work?",
        "category": "how_it_works",
        "expected_keywords": ["learning", "data", "algorithm", "model"]
    }
]


def run_multi_query_comparison(
    queries: List[Dict],
    model_name: str = "microsoft/phi-2",
    device: str = "cpu",
    quick_mode: bool = True,
    methods_to_test: List[str] = None,
) -> Dict:
    """
    Run comparison across multiple queries.
    
    Args:
        queries: List of query dictionaries with 'id', 'input', 'category'
        model_name: Model to use
        device: Device (cpu/cuda)
        quick_mode: Use shorter responses
        methods_to_test: List of methods to test (None = test all)
    
    Returns:
        Dictionary with results for each query and method
    """
    print("🔬 Multi-Query Comprehensive Comparison")
    print("="*70)
    print(f"Queries: {len(queries)}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if quick_mode:
        print("⚡ Quick mode: Using shorter responses")
    print()
    
    all_results = {}
    
    # Methods to test (subset for faster testing)
    if methods_to_test is None:
        methods_to_test = [
            "baseline",
            "simple_adaptivocab",
            "word_removal_3",
            "hybrid_kv_cache",
            "all_combined"
        ]
    
    for query_idx, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {query_idx}/{len(queries)}: {query['id']}")
        print(f"Input: {query['input']}")
        print(f"Category: {query['category']}")
        print(f"{'='*70}\n")
        
        # Run comparison for this query
        results = compare_all_methods(
            query['input'],
            model_name=model_name,
            device=device,
            quick_mode=quick_mode,
        )
        
        # Filter to only requested methods
        filtered_results = {
            k: v for k, v in results.items() 
            if k in methods_to_test and v is not None
        }
        
        # Add query metadata
        query_results = {
            "query_id": query['id'],
            "query_input": query['input'],
            "category": query['category'],
            "expected_keywords": query.get('expected_keywords', []),
            "methods": {}
        }
        
        # Extract input/output pairs and metrics
        for method, data in filtered_results.items():
            metrics = data["metrics"]
            query_results["methods"][method] = {
                "input": query['input'],
                "output": data["response"],
                "metrics": {
                    "total_time": metrics["total_time"],
                    "input_tokens": metrics["input_tokens"],
                    "output_tokens": metrics["output_tokens"],
                    "total_tokens": metrics["total_tokens"],
                    "peak_memory_mb": metrics["peak_memory_mb"],
                    "tokens_per_second": metrics["tokens_per_second"],
                    "efficiency_score": metrics["efficiency_score"],
                },
                "optimizations": {
                    "adaptivocab": metrics.get("adaptivocab", False),
                    "simple_adaptivocab": metrics.get("simple_adaptivocab", False),
                    "word_removal": metrics.get("word_removal", False),
                    "larosa": metrics.get("larosa", False),
                    "vattention": metrics.get("vattention", False),
                    "hybrid_kv_cache": metrics.get("hybrid_kv_cache", False),
                }
            }
        
        all_results[query['id']] = query_results
        
        # Print summary for this query
        print(f"\n✅ Query {query_idx} complete: {len(filtered_results)} methods tested")
    
    return all_results


def save_results(results: Dict, output_file: str = "multi_query_results.json"):
    """Save results to JSON file."""
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    return output_path


def print_summary(results: Dict):
    """Print summary of all queries."""
    print("\n" + "="*70)
    print("📊 MULTI-QUERY SUMMARY")
    print("="*70)
    
    for query_id, query_data in results.items():
        print(f"\n{query_id.upper()}: {query_data['query_input']}")
        print("-" * 70)
        
        methods = query_data['methods']
        baseline = methods.get('baseline', {})
        baseline_output = baseline.get('output', 'N/A')
        baseline_tokens = baseline.get('metrics', {}).get('total_tokens', 0)
        
        print(f"Baseline Output: {baseline_output[:100]}...")
        print(f"Baseline Tokens: {baseline_tokens}")
        
        for method, data in methods.items():
            if method == 'baseline':
                continue
            output = data.get('output', 'N/A')
            tokens = data.get('metrics', {}).get('total_tokens', 0)
            time_val = data.get('metrics', {}).get('total_time', 0)
            memory = data.get('metrics', {}).get('peak_memory_mb', 0)
            
            token_diff = tokens - baseline_tokens
            token_pct = (token_diff / baseline_tokens * 100) if baseline_tokens > 0 else 0
            
            print(f"\n{method}:")
            print(f"  Output: {output[:100]}...")
            print(f"  Tokens: {tokens} ({token_pct:+.1f}% vs baseline)")
            print(f"  Time: {time_val:.2f}s")
            print(f"  Memory: {memory:.2f} MB")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Query Comparison")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output", type=str, default="multi_query_results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", default=True, help="Quick mode")
    parser.add_argument("--no-quick", dest="quick", action="store_false", help="Disable quick mode")
    parser.add_argument("--methods", type=str, nargs="+", default=None, 
                        help="Methods to test (default: baseline, simple_adaptivocab, word_removal_3, hybrid_kv_cache, all_combined)")
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_multi_query_comparison(
        TEST_QUERIES,
        model_name=args.model,
        device=args.device,
        quick_mode=args.quick,
        methods_to_test=args.methods,
    )
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Multi-query comparison complete!")
    print(f"📊 Use visualize_multi_query.py to create visualizations")


if __name__ == "__main__":
    main()

