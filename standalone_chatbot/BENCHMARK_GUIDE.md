# Benchmark Comparison Guide

## Overview

This tool compares the **optimized** LLM (with AdaptiVocab, LaRoSA, vAttention) against a **baseline** version (no optimizations) and tracks:

- ⏱️ **Time**: Generation time, tokens per second
- 🎯 **Tokens**: Input/output/total tokens, token reduction
- 💾 **Memory**: Peak/average memory usage, memory efficiency
- 📊 **Throughput**: Overall performance improvements

## Quick Start

### Run Benchmark

```bash
cd /Users/nk/Desktop/OS/standalone_chatbot
conda activate fused_llm
python benchmark_comparison.py --model microsoft/phi-2 --larosa-sparsity 0.4
```

This will:
1. Run the same prompts on both optimized and baseline engines
2. Collect all metrics
3. Save results to `benchmark_results.json`
4. Print a detailed comparison

### With AdaptiVocab

```bash
python benchmark_comparison.py \
    --model microsoft/phi-2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl \
    --larosa-sparsity 0.4
```

### Visualize Results

```bash
python visualize_benchmark.py --input benchmark_results.json
```

This creates charts in `benchmark_plots/`:
- `time_comparison.png` - Time and throughput
- `token_comparison.png` - Token usage
- `memory_comparison.png` - Memory usage
- `improvements.png` - Improvement percentages
- `comprehensive_dashboard.png` - All metrics in one view

## What Gets Tracked

### Time Metrics
- **Total Time**: Total generation time for all prompts
- **Generation Time**: Pure generation time (excluding setup)
- **Tokens/Second**: Throughput metric

### Token Metrics
- **Input Tokens**: Tokens in prompts
- **Output Tokens**: Tokens in responses
- **Total Tokens**: Sum of input + output
- **Token Reduction**: % reduction with AdaptiVocab

### Memory Metrics
- **Peak Memory**: Maximum memory used during generation
- **Average Memory**: Average memory usage
- **Memory Efficiency**: Tokens per MB of memory

### Quality Metrics
- **Response Length**: Total characters in responses
- **Response Quality**: (Placeholder for future metrics)

## Example Output

```
📊 BENCHMARK COMPARISON RESULTS
============================================================

🔧 Configuration:
   Model: microsoft/phi-2
   Optimizations:
      AdaptiVocab: ✅
      LaRoSA: ✅ (40% sparsity)
      vAttention: ✅

⏱️  Time Metrics:
   Baseline:     12.45s (15.2 tokens/s)
   Optimized:    8.32s (22.8 tokens/s)
   Improvement:  +33.2% (+50.0% throughput)

🎯 Token Metrics:
   Baseline:     189 tokens (95 in, 94 out)
   Optimized:    142 tokens (71 in, 71 out)
   Reduction:    +24.9%

💾 Memory Metrics:
   Baseline:     245.32 MB peak, 198.45 MB avg
   Optimized:    187.21 MB peak, 152.33 MB avg
   Reduction:    +23.7%

📈 Overall Efficiency:
   Combined Improvement: +27.3%
```

## Custom Prompts

```bash
python benchmark_comparison.py \
    --model microsoft/phi-2 \
    --prompts "What is AI?" "Explain ML" "What is Python?"
```

## Output Files

- `benchmark_results.json` - Full results in JSON format
- `benchmark_plots/` - Directory with all visualization charts

## Understanding Results

### Positive Improvements
- **Time Reduction**: Lower is better (faster generation)
- **Token Reduction**: Lower is better (fewer tokens = less cost)
- **Memory Reduction**: Lower is better (less memory usage)
- **Throughput Improvement**: Higher is better (more tokens/second)

### Combined Efficiency
The overall efficiency is the average of all improvements, giving you a single metric to compare.

## Tips

1. **Run multiple times**: Results can vary, average multiple runs
2. **Use same prompts**: For fair comparison, use identical prompts
3. **Warmup included**: First generation is excluded (warmup)
4. **Monitor memory**: Check if you have enough RAM for the model

## Next Steps

1. Run benchmark: `python benchmark_comparison.py`
2. View results: Check the printed summary
3. Visualize: `python visualize_benchmark.py`
4. Analyze: Compare different optimization levels
5. Optimize: Adjust LaRoSA sparsity, add AdaptiVocab, etc.

