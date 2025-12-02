#!/usr/bin/env python3
"""
Benchmark Comparison Tool
Compares optimized vs non-optimized LLM performance
Tracks: memory usage, time, tokens, throughput
"""

import time
import json
import argparse
import psutil
import os
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import torch

from fused_chatbot_enhanced import EnhancedFusedLLMEngine


@dataclass
class BenchmarkMetrics:
    """Metrics for a single benchmark run."""
    model_name: str
    adaptivocab: bool
    larosa_sparsity: float
    vattention: bool
    
    # Time metrics
    total_time: float
    generation_time: float
    tokens_per_second: float
    
    # Token metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_reduced: float  # Percentage reduction with AdaptiVocab
    
    # Memory metrics
    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency: float  # Tokens per MB
    
    # Quality metrics (optional)
    response_length: int
    response_quality: float  # Placeholder for future quality metrics


class BenchmarkRunner:
    """Runs benchmarks and collects metrics."""
    
    def __init__(self, test_prompts: List[str] = None):
        self.test_prompts = test_prompts or [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "What is the capital of France?",
            "How does a neural network work?",
            "What is the difference between AI and ML?",
        ]
        self.process = psutil.Process(os.getpid())
    
    def measure_memory(self) -> Tuple[float, float]:
        """Measure current memory usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb, memory_mb
    
    def benchmark_engine(
        self,
        engine: EnhancedFusedLLMEngine,
        prompts: List[str] = None,
        warmup: int = 1,
    ) -> BenchmarkMetrics:
        """Benchmark an engine and collect metrics."""
        if prompts is None:
            prompts = self.test_prompts
        
        # Warmup
        print(f"   Warming up ({warmup} prompts)...")
        for _ in range(warmup):
            engine.generate(prompts[0], max_new_tokens=30)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure baseline memory
        baseline_memory, _ = self.measure_memory()
        
        # Run benchmarks
        total_input_tokens = 0
        total_output_tokens = 0
        total_generation_time = 0.0
        total_time = 0.0
        peak_memory = baseline_memory
        memory_samples = []
        responses = []
        
        print(f"   Running {len(prompts)} prompts...")
        
        # Set seed for reproducibility (but note: generation still varies slightly)
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        for i, prompt in enumerate(prompts):
            # Measure memory before
            mem_before, _ = self.measure_memory()
            memory_samples.append(mem_before)
            
            # Time generation
            start_time = time.time()
            response = engine.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.7,
            )
            generation_time = time.time() - start_time
            
            # Measure memory after
            mem_after, _ = self.measure_memory()
            memory_samples.append(mem_after)
            peak_memory = max(peak_memory, mem_after)
            
            # Count tokens
            input_ids = engine.tokenizer.encode(prompt)
            output_ids = engine.tokenizer.encode(response)
            
            input_tokens = len(input_ids)
            output_tokens = len(output_ids)
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_generation_time += generation_time
            total_time += generation_time
            
            responses.append({
                "prompt": prompt,
                "response": response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "time": generation_time,
            })
            
            print(f"      [{i+1}/{len(prompts)}] {generation_time:.2f}s, {input_tokens}+{output_tokens} tokens")
        
        # Calculate metrics
        total_tokens = total_input_tokens + total_output_tokens
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else baseline_memory
        tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
        memory_efficiency = total_tokens / (peak_memory - baseline_memory) if (peak_memory - baseline_memory) > 0 else 0
        
        # Get optimization status
        stats = engine.get_stats()
        
        metrics = BenchmarkMetrics(
            model_name=engine.model_name,
            adaptivocab=stats.get("adaptivocab", False),
            larosa_sparsity=stats.get("larosa_sparsity", 0.0),
            vattention=stats.get("vattention", False),
            total_time=total_time,
            generation_time=total_generation_time,
            tokens_per_second=tokens_per_second,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            tokens_reduced=0.0,  # Will calculate in comparison
            peak_memory_mb=peak_memory - baseline_memory,
            average_memory_mb=avg_memory - baseline_memory,
            memory_efficiency=memory_efficiency,
            response_length=sum(len(r["response"]) for r in responses),
            response_quality=0.0,  # Placeholder
        )
        
        return metrics, responses
    
    def compare_engines(
        self,
        optimized_engine: EnhancedFusedLLMEngine,
        baseline_engine: EnhancedFusedLLMEngine,
        prompts: List[str] = None,
    ) -> Dict:
        """Compare optimized vs baseline engine."""
        print("\n" + "="*60)
        print("📊 Benchmarking Optimized Engine")
        print("="*60)
        optimized_metrics, optimized_responses = self.benchmark_engine(optimized_engine, prompts)
        
        print("\n" + "="*60)
        print("📊 Benchmarking Baseline Engine")
        print("="*60)
        baseline_metrics, baseline_responses = self.benchmark_engine(baseline_engine, prompts)
        
        # Calculate improvements
        time_improvement = ((baseline_metrics.total_time - optimized_metrics.total_time) / baseline_metrics.total_time) * 100
        token_improvement = ((baseline_metrics.total_tokens - optimized_metrics.total_tokens) / baseline_metrics.total_tokens) * 100 if baseline_metrics.total_tokens > 0 else 0
        memory_improvement = ((baseline_metrics.peak_memory_mb - optimized_metrics.peak_memory_mb) / baseline_metrics.peak_memory_mb) * 100 if baseline_metrics.peak_memory_mb > 0 else 0
        throughput_improvement = ((optimized_metrics.tokens_per_second - baseline_metrics.tokens_per_second) / baseline_metrics.tokens_per_second) * 100 if baseline_metrics.tokens_per_second > 0 else 0
        
        comparison = {
            "baseline": asdict(baseline_metrics),
            "optimized": asdict(optimized_metrics),
            "improvements": {
                "time_reduction_percent": time_improvement,
                "token_reduction_percent": token_improvement,
                "memory_reduction_percent": memory_improvement,
                "throughput_improvement_percent": throughput_improvement,
            },
            "responses": {
                "baseline": baseline_responses,
                "optimized": optimized_responses,
            }
        }
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Print comparison results in a readable format."""
        baseline = comparison["baseline"]
        optimized = comparison["optimized"]
        improvements = comparison["improvements"]
        
        print("\n" + "="*60)
        print("📊 BENCHMARK COMPARISON RESULTS")
        print("="*60)
        
        print("\n🔧 Configuration:")
        print(f"   Model: {baseline['model_name']}")
        print(f"   Optimizations:")
        print(f"      AdaptiVocab: {'✅' if optimized['adaptivocab'] else '❌'}")
        print(f"      LaRoSA: {'✅' if optimized['larosa_sparsity'] > 0 else '❌'} ({optimized['larosa_sparsity']*100:.0f}% sparsity)")
        print(f"      vAttention: {'✅' if optimized['vattention'] else '❌'}")
        
        print("\n⏱️  Time Metrics:")
        print(f"   Baseline:     {baseline['total_time']:.2f}s ({baseline['tokens_per_second']:.1f} tokens/s)")
        print(f"   Optimized:    {optimized['total_time']:.2f}s ({optimized['tokens_per_second']:.1f} tokens/s)")
        print(f"   Improvement:  {improvements['time_reduction_percent']:+.1f}% ({improvements['throughput_improvement_percent']:+.1f}% throughput)")
        
        print("\n🎯 Token Metrics:")
        print(f"   Baseline:     {baseline['total_tokens']} tokens ({baseline['input_tokens']} in, {baseline['output_tokens']} out)")
        print(f"   Optimized:    {optimized['total_tokens']} tokens ({optimized['input_tokens']} in, {optimized['output_tokens']} out)")
        if optimized['adaptivocab']:
            print(f"   Reduction:    {improvements['token_reduction_percent']:+.1f}% (from AdaptiVocab)")
        else:
            print(f"   Note: Token count varies naturally. AdaptiVocab (not enabled) reduces tokens by 25%+")
            if improvements['token_reduction_percent'] < 0:
                print(f"   Variation:    {abs(improvements['token_reduction_percent']):.1f}% more tokens (natural generation variation)")
        
        print("\n💾 Memory Metrics:")
        print(f"   Baseline:     {baseline['peak_memory_mb']:.2f} MB peak, {baseline['average_memory_mb']:.2f} MB avg")
        print(f"   Optimized:    {optimized['peak_memory_mb']:.2f} MB peak, {optimized['average_memory_mb']:.2f} MB avg")
        print(f"   Reduction:    {improvements['memory_reduction_percent']:+.1f}%")
        print(f"   Efficiency:   {baseline['memory_efficiency']:.1f} vs {optimized['memory_efficiency']:.1f} tokens/MB")
        
        print("\n📈 Overall Efficiency:")
        # Only count token reduction if AdaptiVocab is enabled
        if optimized['adaptivocab']:
            total_improvement = (
                improvements['time_reduction_percent'] +
                improvements['token_reduction_percent'] +
                improvements['memory_reduction_percent']
            ) / 3
        else:
            # Without AdaptiVocab, token reduction isn't meaningful
            total_improvement = (
                improvements['time_reduction_percent'] +
                improvements['memory_reduction_percent']
            ) / 2
            print(f"   Note: Token reduction only applies with AdaptiVocab")
        print(f"   Combined Improvement: {total_improvement:+.1f}%")
        
        # Add warnings for CPU performance
        if optimized['larosa_sparsity'] > 0.0 and not torch.cuda.is_available():
            print("\n⚠️  CPU Performance Warning:")
            print("   LaRoSA is designed for GPU acceleration.")
            print("   On CPU, LaRoSA adds overhead and may be slower.")
            print("   For best results, use --device cuda or disable LaRoSA (--larosa-sparsity 0.0)")
        
        if optimized['vattention'] and not torch.cuda.is_available():
            print("\n⚠️  vAttention Note:")
            print("   Current vAttention implementation only tracks memory on CPU.")
            print("   Real memory optimization requires GPU with CUDA kernels.")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Optimized vs Baseline LLM")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model name (default: microsoft/phi-2)",
    )
    parser.add_argument(
        "--patch-tokenizer",
        type=str,
        default=None,
        help="Path to AdaptiVocab PatchTokenizer .pkl file",
    )
    parser.add_argument(
        "--simple-adaptivocab",
        action="store_true",
        help="Enable SimpleAdaptiVocab (phrase combination)",
    )
    parser.add_argument(
        "--larosa-sparsity",
        type=float,
        default=0.4,
        help="LaRoSA sparsity level (0.0-1.0, default: 0.4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom test prompts (default: uses built-in prompts)",
    )
    
    args = parser.parse_args()
    
    print("🚀 Benchmark Comparison Tool")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()
    
    # Initialize optimized engine
    print("📦 Initializing Optimized Engine...")
    optimized_engine = EnhancedFusedLLMEngine(
        model_name=args.model,
        patch_tokenizer_path=args.patch_tokenizer,
        simple_adaptivocab=args.simple_adaptivocab,
        larosa_sparsity=args.larosa_sparsity,
        vattention_enabled=True,
        device=args.device,
    )
    
    # Initialize baseline engine (no optimizations)
    print("\n📦 Initializing Baseline Engine (no optimizations)...")
    baseline_engine = EnhancedFusedLLMEngine(
        model_name=args.model,
        patch_tokenizer_path=None,  # No AdaptiVocab
        larosa_sparsity=0.0,  # No LaRoSA
        vattention_enabled=False,  # No vAttention
        device=args.device,
    )
    
    # Run comparison
    runner = BenchmarkRunner(test_prompts=args.prompts)
    comparison = runner.compare_engines(
        optimized_engine,
        baseline_engine,
        prompts=args.prompts,
    )
    
    # Print results
    runner.print_comparison(comparison)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()

