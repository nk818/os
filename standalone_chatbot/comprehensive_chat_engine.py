#!/usr/bin/env python3
"""
Comprehensive Chat Engine with All Optimization Methods
Integrates: AdaptiVocab, LaRoSA, vAttention, Word Removal, SimpleAdaptiVocab
Tracks: Time, Tokens, Memory, Throughput, Efficiency
"""

import time
import psutil
import os
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from fused_chatbot_enhanced import EnhancedFusedLLMEngine
from simple_adaptivocab import SimpleAdaptiVocab
from test_word_removal import remove_every_nth_word_advanced


@dataclass
class ChatMetrics:
    """Comprehensive metrics for a chat interaction."""
    # Time metrics
    total_time: float
    tokenization_time: float
    generation_time: float
    postprocessing_time: float
    
    # Token metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: float
    output_tokens_per_second: float
    
    # Memory metrics
    peak_memory_mb: float
    average_memory_mb: float
    memory_per_token: float
    
    # Text metrics
    input_length: int
    output_length: int
    input_words: int
    output_words: int
    
    # Optimization flags
    adaptivocab: bool
    simple_adaptivocab: bool
    word_removal: bool
    larosa: bool
    vattention: bool
    
    # Efficiency metrics
    tokens_per_mb: float
    words_per_second: float
    efficiency_score: float  # Combined efficiency metric


class ComprehensiveChatEngine:
    """
    Enhanced chat engine with all optimization methods and comprehensive metrics.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        patch_tokenizer_path: Optional[str] = None,
        simple_adaptivocab: bool = False,
        word_removal_n: Optional[int] = None,  # None = disabled, 3/4/5 = every nth word
        larosa_sparsity: float = 0.0,
        vattention_enabled: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize comprehensive chat engine.
        
        Args:
            model_name: HuggingFace model name
            patch_tokenizer_path: Path to AdaptiVocab PatchTokenizer
            simple_adaptivocab: Enable SimpleAdaptiVocab (phrase combination)
            word_removal_n: Remove every nth word (3, 4, or 5)
            larosa_sparsity: LaRoSA sparsity level (0.0-1.0)
            vattention_enabled: Enable vAttention tracking
            device: Device to use (cpu/cuda)
        """
        self.model_name = model_name
        self.word_removal_n = word_removal_n
        self.baseline_memory = self._get_memory_usage()
        
        # Initialize engine
        self.engine = EnhancedFusedLLMEngine(
            model_name=model_name,
            patch_tokenizer_path=patch_tokenizer_path,
            simple_adaptivocab=simple_adaptivocab,
            larosa_sparsity=larosa_sparsity,
            vattention_enabled=vattention_enabled,
            device=device,
        )
        
        # Get optimization status
        stats = self.engine.get_stats()
        self.optimizations = {
            "adaptivocab": stats.get("adaptivocab", False),
            "simple_adaptivocab": stats.get("simple_adaptivocab", False),
            "word_removal": word_removal_n is not None,
            "larosa": stats.get("larosa", False),
            "vattention": stats.get("vattention", False),
        }
        
        print(f"\n✅ Comprehensive Chat Engine initialized")
        print(f"   Optimizations: {sum(self.optimizations.values())} enabled")
        for opt, enabled in self.optimizations.items():
            if enabled:
                value = f" (every {word_removal_n}th)" if opt == "word_removal" else ""
                print(f"   • {opt}: ✅{value}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def chat_with_metrics(
        self,
        message: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        return_metrics: bool = True,
    ) -> Tuple[str, Optional[ChatMetrics]]:
        """
        Chat with comprehensive metrics tracking.
        
        Returns:
            Tuple of (response, metrics)
        """
        # Start timing
        total_start = time.perf_counter()
        memory_samples = []
        
        # Pre-process input with word removal if enabled
        original_message = message
        if self.word_removal_n:
            word_removal_start = time.perf_counter()
            message = remove_every_nth_word_advanced(message, self.word_removal_n)
            word_removal_time = time.perf_counter() - word_removal_start
            print(f"   📝 Word removal: {len(original_message.split())} → {len(message.split())} words ({word_removal_time*1000:.1f}ms)")
        else:
            word_removal_time = 0.0
        
        # Measure memory before
        mem_before = self._get_memory_usage()
        memory_samples.append(mem_before)
        
        # Tokenization
        tokenization_start = time.perf_counter()
        input_ids = self.engine.tokenizer.encode(message)
        tokenization_time = time.perf_counter() - tokenization_start
        input_tokens = len(input_ids)
        
        # Measure memory after tokenization
        mem_after_token = self._get_memory_usage()
        memory_samples.append(mem_after_token)
        
        # Generation
        generation_start = time.perf_counter()
        response = self.engine.generate(
            message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        generation_time = time.perf_counter() - generation_start
        
        # Measure memory after generation
        mem_after_gen = self._get_memory_usage()
        memory_samples.append(mem_after_gen)
        
        # Post-processing
        postprocessing_start = time.perf_counter()
        output_ids = self.engine.tokenizer.encode(response)
        postprocessing_time = time.perf_counter() - postprocessing_start
        output_tokens = len(output_ids)
        
        total_time = time.perf_counter() - total_start
        
        # Calculate metrics
        peak_memory = max(memory_samples) - self.baseline_memory
        avg_memory = sum(memory_samples) / len(memory_samples) - self.baseline_memory
        total_tokens = input_tokens + output_tokens
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        output_tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        # Text metrics
        input_words = len(message.split())
        output_words = len(response.split())
        words_per_second = output_words / generation_time if generation_time > 0 else 0
        
        # Efficiency metrics
        memory_per_token = peak_memory / total_tokens if total_tokens > 0 else 0
        tokens_per_mb = total_tokens / peak_memory if peak_memory > 0 else 0
        
        # Combined efficiency score (higher is better)
        # Factors: tokens/sec, tokens/MB, output quality (word count)
        efficiency_score = (
            tokens_per_second * 0.4 +  # Speed
            tokens_per_mb * 0.3 +      # Memory efficiency
            output_words * 0.3         # Output quality
        )
        
        metrics = ChatMetrics(
            total_time=total_time,
            tokenization_time=tokenization_time,
            generation_time=generation_time,
            postprocessing_time=postprocessing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            output_tokens_per_second=output_tokens_per_second,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            memory_per_token=memory_per_token,
            input_length=len(message),
            output_length=len(response),
            input_words=input_words,
            output_words=output_words,
            adaptivocab=self.optimizations["adaptivocab"],
            simple_adaptivocab=self.optimizations["simple_adaptivocab"],
            word_removal=self.optimizations["word_removal"],
            larosa=self.optimizations["larosa"],
            vattention=self.optimizations["vattention"],
            tokens_per_mb=tokens_per_mb,
            words_per_second=words_per_second,
            efficiency_score=efficiency_score,
        )
        
        if return_metrics:
            return response, metrics
        return response, None
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "optimizations": self.optimizations,
            "model": self.model_name,
            "device": self.engine.device,
        }


def compare_all_methods(
    message: str,
    model_name: str = "microsoft/phi-2",
    device: str = "cpu",
    quick_mode: bool = True,
) -> Dict:
    """
    Compare all optimization methods on the same message.
    
    Args:
        message: Input message
        model_name: Model to use
        device: Device (cpu/cuda)
        quick_mode: If True, use shorter max_new_tokens for faster testing
    
    Returns:
        Dictionary with results for each method
    """
    print("🔬 Comprehensive Method Comparison")
    print("="*70)
    print(f"Message: {message[:100]}...")
    if quick_mode:
        print("⚡ Quick mode: Using shorter responses for faster testing")
    print()
    
    max_tokens = 30 if quick_mode else 100
    
    results = {}
    
    # 1. Baseline (no optimizations)
    print("📊 Testing Baseline (no optimizations)...")
    baseline_engine = ComprehensiveChatEngine(
        model_name=model_name,
        device=device,
    )
    baseline_response, baseline_metrics = baseline_engine.chat_with_metrics(
        message, max_new_tokens=max_tokens
    )
    results["baseline"] = {
        "response": baseline_response,
        "metrics": asdict(baseline_metrics),
    }
    print(f"   ✅ Baseline: {baseline_metrics.total_time:.2f}s, {baseline_metrics.total_tokens} tokens")
    
    # 2. SimpleAdaptiVocab only
    print("\n📊 Testing SimpleAdaptiVocab...")
    simple_av_engine = ComprehensiveChatEngine(
        model_name=model_name,
        simple_adaptivocab=True,
        device=device,
    )
    simple_av_response, simple_av_metrics = simple_av_engine.chat_with_metrics(
        message, max_new_tokens=max_tokens
    )
    results["simple_adaptivocab"] = {
        "response": simple_av_response,
        "metrics": asdict(simple_av_metrics),
    }
    print(f"   ✅ SimpleAdaptiVocab: {simple_av_metrics.total_time:.2f}s, {simple_av_metrics.total_tokens} tokens")
    
    # 3. Word Removal (every 3rd word)
    print("\n📊 Testing Word Removal (every 3rd word)...")
    word_removal_engine = ComprehensiveChatEngine(
        model_name=model_name,
        word_removal_n=3,
        device=device,
    )
    word_removal_response, word_removal_metrics = word_removal_engine.chat_with_metrics(
        message, max_new_tokens=max_tokens
    )
    results["word_removal_3"] = {
        "response": word_removal_response,
        "metrics": asdict(word_removal_metrics),
    }
    print(f"   ✅ Word Removal: {word_removal_metrics.total_time:.2f}s, {word_removal_metrics.total_tokens} tokens")
    
    # 4. LaRoSA only (skip on CPU as it's slower)
    if device == "cuda" or not quick_mode:
        print("\n📊 Testing LaRoSA (40% sparsity)...")
        larosa_engine = ComprehensiveChatEngine(
            model_name=model_name,
            larosa_sparsity=0.4,
            device=device,
        )
        larosa_response, larosa_metrics = larosa_engine.chat_with_metrics(
            message, max_new_tokens=max_tokens
        )
        results["larosa"] = {
            "response": larosa_response,
            "metrics": asdict(larosa_metrics),
        }
        print(f"   ✅ LaRoSA: {larosa_metrics.total_time:.2f}s, {larosa_metrics.total_tokens} tokens")
    else:
        print("\n📊 Skipping LaRoSA (slower on CPU, use --no-quick to test)")
        results["larosa"] = None
    
    # 5. All combined (skip LaRoSA on CPU in quick mode)
    print("\n📊 Testing All Methods Combined...")
    all_combined_engine = ComprehensiveChatEngine(
        model_name=model_name,
        simple_adaptivocab=True,
        word_removal_n=3,
        larosa_sparsity=0.4 if (device == "cuda" or not quick_mode) else 0.0,
        vattention_enabled=True,
        device=device,
    )
    all_combined_response, all_combined_metrics = all_combined_engine.chat_with_metrics(
        message, max_new_tokens=max_tokens
    )
    results["all_combined"] = {
        "response": all_combined_response,
        "metrics": asdict(all_combined_metrics),
    }
    print(f"   ✅ All Combined: {all_combined_metrics.total_time:.2f}s, {all_combined_metrics.total_tokens} tokens")
    
    return results


def print_comparison(results: Dict):
    """Print comprehensive comparison of all methods."""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("="*70)
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    baseline = valid_results["baseline"]["metrics"]
    
    print("\n⏱️  TIME METRICS:")
    print("-" * 70)
    for method, data in valid_results.items():
        m = data["metrics"]
        time_improvement = ((baseline["total_time"] - m["total_time"]) / baseline["total_time"] * 100) if baseline["total_time"] > 0 else 0
        print(f"{method:20s}: {m['total_time']:6.2f}s ({m['generation_time']:5.2f}s gen) | {time_improvement:+6.1f}%")
    
    print("\n🎯 TOKEN METRICS:")
    print("-" * 70)
    for method, data in valid_results.items():
        m = data["metrics"]
        token_diff = m["total_tokens"] - baseline["total_tokens"]
        token_pct = (token_diff / baseline["total_tokens"] * 100) if baseline["total_tokens"] > 0 else 0
        print(f"{method:20s}: {m['input_tokens']:3d} in + {m['output_tokens']:3d} out = {m['total_tokens']:3d} total | {token_pct:+6.1f}%")
    
    print("\n💾 MEMORY METRICS:")
    print("-" * 70)
    for method, data in valid_results.items():
        m = data["metrics"]
        mem_diff = m["peak_memory_mb"] - baseline["peak_memory_mb"]
        mem_pct = (mem_diff / baseline["peak_memory_mb"] * 100) if baseline["peak_memory_mb"] > 0 else 0
        print(f"{method:20s}: {m['peak_memory_mb']:6.2f} MB peak | {mem_pct:+6.1f}%")
    
    print("\n⚡ EFFICIENCY METRICS:")
    print("-" * 70)
    for method, data in valid_results.items():
        m = data["metrics"]
        print(f"{method:20s}: {m['tokens_per_second']:5.2f} tok/s | {m['tokens_per_mb']:6.2f} tok/MB | Score: {m['efficiency_score']:.1f}")
    
    print("\n📝 RESPONSES:")
    print("-" * 70)
    for method, data in valid_results.items():
        response = data["response"]
        print(f"\n{method}:")
        print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Chat Engine with All Optimizations")
    parser.add_argument("--message", type=str, default="What is quantum mechanics?", help="Message to test")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output", type=str, default="comprehensive_results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", default=True, help="Quick mode (shorter responses)")
    parser.add_argument("--no-quick", dest="quick", action="store_false", help="Disable quick mode")
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_all_methods(args.message, args.model, args.device, quick_mode=args.quick)
    
    # Print results
    print_comparison(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

