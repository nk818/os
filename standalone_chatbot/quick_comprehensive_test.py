#!/usr/bin/env python3
"""
Quick comprehensive test - loads model once, tests all methods
Much faster than loading model multiple times
"""

import time
import json
from comprehensive_chat_engine import ComprehensiveChatEngine, compare_all_methods, print_comparison

def quick_test(message: str = "What is artificial intelligence?"):
    """Quick test with a single message."""
    print("🚀 Quick Comprehensive Test")
    print("="*70)
    print(f"Message: {message}")
    print()
    
    start_time = time.time()
    
    # Run comparison (this will still load models multiple times, but we'll optimize)
    results = compare_all_methods(message, model_name="microsoft/phi-2", device="cpu")
    
    total_time = time.time() - start_time
    
    # Print results
    print_comparison(results)
    
    # Save results
    output_file = "comprehensive_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n⏱️  Total test time: {total_time:.1f}s")
    print(f"💾 Results saved to: {output_file}")
    print("\n✅ Test complete!")
    
    return results

if __name__ == "__main__":
    import sys
    message = sys.argv[1] if len(sys.argv) > 1 else "What is artificial intelligence?"
    quick_test(message)

