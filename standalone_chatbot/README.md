# Comprehensive LLM Optimization System

A comprehensive LLM efficiency system with multiple optimization methods including Hybrid KV Cache (vAttention + CAKE), SimpleAdaptiVocab, Word Removal, and LaRoSA.

## Features

✅ **Hybrid KV Cache** - Combines vAttention (15% savings) and CAKE (12% savings) for 25.2% memory reduction  
✅ **SimpleAdaptiVocab** - Phrase-based token reduction  
✅ **Word Removal** - Text compression by removing every Nth word  
✅ **LaRoSA** - Activation sparsity for computation speedup  
✅ **Comprehensive Metrics** - Time, tokens, memory, efficiency tracking  
✅ **Multi-Query Testing** - Test multiple queries with input/output tracking  

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Comprehensive Method Comparison

Test all optimization methods on a single query:

```bash
python comprehensive_chat_engine.py --message "What is AI?" --quick
```

### Multi-Query Comparison

Test multiple queries across different methods:

```bash
python multi_query_comparison.py --quick
```

### Visualize Results

```bash
# Comprehensive results
python visualize_comprehensive_chat.py --output comprehensive_analysis_plots

# Multi-query results
python visualize_multi_query.py
```

## Core Files

- `comprehensive_chat_engine.py` - Main comprehensive chat engine with all optimizations
- `hybrid_kv_cache.py` - Hybrid KV Cache optimizer (vAttention + CAKE)
- `fused_chatbot_enhanced.py` - Enhanced LLM engine with optimizations
- `simple_adaptivocab.py` - SimpleAdaptiVocab phrase combination
- `test_word_removal.py` - Word removal utility
- `multi_query_comparison.py` - Multi-query testing script
- `visualize_comprehensive_chat.py` - Comprehensive visualization
- `visualize_multi_query.py` - Multi-query visualization

## Optimization Methods

### Hybrid KV Cache
- **vAttention**: 15% memory savings through dynamic allocation
- **CAKE**: 12% additional savings through computation/I/O scheduling
- **Combined**: 25.2% total memory reduction

### SimpleAdaptiVocab
- Combines common phrases into single tokens
- Reduces token count without model retraining

### Word Removal
- Removes every Nth word (3rd, 4th, or 5th)
- Maintains conceptual soundness for summarization

### LaRoSA
- Activation sparsity for faster computation
- Best on GPU, simplified CPU version available

## Results

Results are saved to:
- `comprehensive_results.json` - Single query comprehensive results
- `multi_query_results.json` - Multi-query results with input/output pairs

Visualizations are saved to:
- `comprehensive_analysis_plots/` - Comprehensive method comparison charts
- `multi_query_plots/` - Multi-query quality and performance charts

## Documentation

- `VATTENTION_EXPLANATION.md` - Detailed explanation of vAttention and Hybrid KV Cache
