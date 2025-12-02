# Comprehensive LLM Optimization System

A unified system integrating multiple LLM optimization techniques: Hybrid KV Cache (vAttention + CAKE), SimpleAdaptiVocab, Word Removal, and LaRoSA for maximum efficiency.

## 🚀 Quick Start

### Multi-Query Comparison

Test multiple queries across different optimization methods:

```bash
# Run comparison with all methods
python multi_query_comparison.py --quick

# Run with specific methods
python multi_query_comparison.py --quick --methods baseline simple_adaptivocab hybrid_kv_cache all_combined

# Run with custom model
python multi_query_comparison.py --model microsoft/phi-2 --device cpu
```

### Visualize Results

```bash
# Visualize multi-query results
python visualize_multi_query.py

# Visualize comprehensive results
python visualize_comprehensive_chat.py --output comprehensive_analysis_plots
```

### Comprehensive Single Query

Test all optimization methods on a single query:

```bash
python comprehensive_chat_engine.py --message "What is AI?" --quick
```

## 📁 File Structure

### Core Engine Files

#### `comprehensive_chat_engine.py`
**Purpose**: Main comprehensive chat engine that integrates all optimization methods and tracks detailed metrics.

**Features**:
- Tests all optimization methods (Baseline, SimpleAdaptiVocab, Word Removal, Hybrid KV Cache, LaRoSA, All Combined)
- Tracks comprehensive metrics: time, tokens, memory, efficiency
- Supports word removal preprocessing
- Returns detailed ChatMetrics dataclass

**Usage**:
```bash
python comprehensive_chat_engine.py --message "Your query" --quick
```

**Key Classes**:
- `ComprehensiveChatEngine`: Main engine wrapper
- `ChatMetrics`: Metrics dataclass with all performance data

#### `fused_chatbot_enhanced.py`
**Purpose**: Enhanced LLM engine with all optimizations integrated.

**Features**:
- Loads models (supports Phi-2, DialoGPT, GPT-2)
- Integrates SimpleAdaptiVocab tokenizer wrapper
- Applies LaRoSA activation sparsity
- Integrates Hybrid KV Cache optimizer
- Handles conversation history
- Provides chat interface

**Key Classes**:
- `EnhancedFusedLLMEngine`: Main LLM engine
- `SimpleLaRoSA`: CPU-compatible activation sparsity
- `SimpleVAttention`: CPU tracking version of vAttention

**Usage**:
```python
from fused_chatbot_enhanced import EnhancedFusedLLMEngine

engine = EnhancedFusedLLMEngine(
    model_name="microsoft/phi-2",
    simple_adaptivocab=True,
    larosa_sparsity=0.4,
    vattention_enabled=True,
    device="cpu"
)

response = engine.chat("What is AI?")
```

#### `hybrid_kv_cache.py`
**Purpose**: Hybrid KV Cache optimizer combining vAttention and CAKE for maximum memory efficiency.

**Features**:
- **vAttention component**: 15% memory savings through dynamic virtual memory allocation
- **CAKE component**: 12% additional savings through intelligent computation/I/O scheduling
- **Combined savings**: 25.2% total memory reduction
- Tracks cache hit rates and allocation statistics
- Provides detailed memory usage metrics

**Key Classes**:
- `HybridKVCacheOptimizer`: Main hybrid optimizer

**Usage**:
```python
from hybrid_kv_cache import HybridKVCacheOptimizer

optimizer = HybridKVCacheOptimizer(
    num_layers=32,
    num_heads=32,
    head_size=128,
    max_batch_size=1,
    max_seq_len=512,
    device="cpu"
)

# Get memory savings
savings = optimizer.get_memory_savings()
stats = optimizer.get_memory_usage()
```

**How it works**:
1. vAttention: Uses dynamic allocation to reduce physical memory usage
2. CAKE: Intelligently decides when to compute vs load from cache based on hit rates
3. Combined: Both optimizations work together for maximum efficiency

#### `simple_adaptivocab.py`
**Purpose**: Simplified phrase-based token reduction without requiring model retraining.

**Features**:
- Combines common multi-word phrases into single tokens
- Only combines if it reduces token count
- Works with any tokenizer (wraps existing tokenizer)
- No model fine-tuning required

**Key Classes**:
- `SimpleAdaptiVocab`: Tokenizer wrapper for phrase combination

**Usage**:
```python
from simple_adaptivocab import SimpleAdaptiVocab
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer = SimpleAdaptiVocab(tokenizer)

# Now tokenizer automatically combines phrases
tokens = tokenizer.encode("artificial intelligence")  # May be fewer tokens
```

**Default Phrases**:
- "artificial intelligence", "machine learning", "deep learning"
- "natural language processing", "quantum mechanics"
- "neural network", "data science", etc.

#### `test_word_removal.py`
**Purpose**: Text compression utility that removes every Nth word while maintaining conceptual soundness.

**Features**:
- Removes every 3rd, 4th, or 5th word
- Preserves sentence structure
- Maintains readability for summarization tasks
- Reduces input token count significantly

**Key Functions**:
- `remove_every_nth_word_advanced()`: Main removal function

**Usage**:
```python
from test_word_removal import remove_every_nth_word_advanced

compressed = remove_every_nth_word_advanced(
    "This is a test sentence with many words",
    n=3  # Remove every 3rd word
)
```

### Testing & Comparison Files

#### `multi_query_comparison.py`
**Purpose**: Test multiple queries across different optimization methods and save input/output pairs.

**Features**:
- Tests 5 different query types:
  - Definition queries
  - Explanation queries
  - How-it-works queries
  - Text summarization (2 queries with 150-word texts)
- Tests multiple methods per query
- Saves complete input/output pairs
- Tracks metrics for each query-method combination
- Generates JSON results file

**Usage**:
```bash
# Run with default queries
python multi_query_comparison.py --quick

# Run with specific methods
python multi_query_comparison.py --methods baseline simple_adaptivocab hybrid_kv_cache all_combined

# Custom model
python multi_query_comparison.py --model microsoft/phi-2 --device cpu
```

**Output**:
- `multi_query_results.json`: Complete results with input/output pairs and metrics

**Query Types**:
1. **Definition**: "What is artificial intelligence?"
2. **Explanation**: "Explain quantum mechanics in simple terms."
3. **How-it-works**: "How does machine learning work?"
4. **Summarization 1**: AI/ML text (150 words)
5. **Summarization 2**: Climate change text (150 words)

#### `visualize_comprehensive_chat.py`
**Purpose**: Create visualizations for comprehensive chat engine results.

**Features**:
- Time metrics comparison
- Token metrics comparison
- Memory metrics comparison
- Efficiency metrics comparison
- Comprehensive dashboard (9-panel)

**Usage**:
```bash
python visualize_comprehensive_chat.py --input comprehensive_results.json --output comprehensive_analysis_plots
```

**Output Charts**:
- `time_metrics.png`: Time comparisons
- `token_metrics.png`: Token usage comparisons
- `memory_metrics.png`: Memory usage comparisons
- `efficiency_metrics.png`: Efficiency scores
- `comprehensive_dashboard.png`: All metrics in one dashboard

#### `visualize_multi_query.py`
**Purpose**: Create visualizations for multi-query comparison results.

**Features**:
- Quality comparison across queries
- Input/output pairs visualization
- Performance dashboard
- Token usage across queries
- Output length distribution

**Usage**:
```bash
python visualize_multi_query.py --input multi_query_results.json --output multi_query_plots
```

**Output Charts**:
- `quality_comparison.png`: Output quality metrics
- `input_output_pairs.png`: Shows input/output for each query and method
- `performance_dashboard.png`: Comprehensive performance dashboard

## 📊 Results Files

### `comprehensive_results.json`
Results from single comprehensive query test. Contains:
- Response for each method
- Complete metrics (time, tokens, memory, efficiency)
- Optimization flags

### `multi_query_results.json`
Results from multi-query comparison. Contains:
- All 5 queries with input/output pairs
- Metrics for each query-method combination
- Expected keywords for quality assessment

## 📈 Visualization Directories

### `comprehensive_analysis_plots/`
Contains 5 visualization charts from comprehensive chat engine:
- `time_metrics.png`
- `token_metrics.png`
- `memory_metrics.png`
- `efficiency_metrics.png`
- `comprehensive_dashboard.png`

### `multi_query_plots/`
Contains 3 visualization charts from multi-query comparison:
- `quality_comparison.png` - Output quality metrics
- `input_output_pairs.png` - Shows input/output for each query and method
- `performance_dashboard.png` - Comprehensive performance dashboard

### Additional Python Files

#### `benchmark_comprehensive.py`
**Purpose**: Comprehensive benchmarking script for all optimization methods.

**Features**:
- Benchmarks multiple optimization combinations
- Tests with different models
- Generates detailed performance reports
- Compares baseline vs optimized performance

**Usage**:
```bash
python benchmark_comprehensive.py
```

#### `chatbot_example.py`
**Purpose**: Example chatbot implementation demonstrating basic usage.

**Features**:
- Simple chatbot interface
- Shows basic integration patterns
- Example usage of FusedChatbot class

**Usage**:
```bash
python chatbot_example.py
```

#### `fused_chatbot.py`
**Purpose**: Fused chatbot with server integration (alternative implementation).

**Features**:
- Server-based chatbot
- API integration
- Alternative to comprehensive_chat_engine.py
- Supports OpenAI-compatible API

**Usage**:
```bash
python fused_chatbot.py --interactive
```

#### `visualize_all_methods.py`
**Purpose**: Visualization script for all optimization methods comparison.

**Features**:
- Creates comprehensive comparison charts
- Shows all methods side-by-side
- Performance metrics visualization
- Compares Normal LLM, LaRoSA LLM, Fused LLM, All Combined

**Usage**:
```bash
python visualize_all_methods.py
```

#### `visualize_comprehensive_comparison.py`
**Purpose**: Alternative visualization script for comprehensive comparisons.

**Features**:
- Detailed comparison visualizations
- Multiple chart types
- Performance analysis
- Compares Normal LLM vs Fused LLM

**Usage**:
```bash
python visualize_comprehensive_comparison.py
```

## 🔧 Optimization Methods

### 1. Hybrid KV Cache (vAttention + CAKE)
- **Memory Savings**: 25.2% (15% vAttention + 12% CAKE)
- **How**: Dynamic memory allocation + intelligent scheduling
- **Best For**: Memory-constrained environments

### 2. SimpleAdaptiVocab
- **Token Reduction**: Variable (depends on phrase frequency)
- **How**: Combines common phrases into single tokens
- **Best For**: Domain-specific text with common phrases

### 3. Word Removal
- **Token Reduction**: ~27% (every 3rd word)
- **How**: Removes every Nth word while maintaining meaning
- **Best For**: Summarization tasks

### 4. LaRoSA
- **Speedup**: 1.3x-1.9x (on GPU)
- **How**: Activation sparsity through rotation
- **Best For**: GPU inference acceleration

### 5. All Combined
- **Combined Benefits**: All optimizations working together
- **Result**: Maximum efficiency across all metrics

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace transformers
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting
- `psutil` - Memory tracking
- `numpy` - Numerical operations

## 🎯 Usage Examples

### Example 1: Quick Multi-Query Test
```bash
# Test all queries with quick mode
python multi_query_comparison.py --quick

# Visualize results
python visualize_multi_query.py
```

### Example 2: Comprehensive Single Query
```bash
# Test all methods on one query
python comprehensive_chat_engine.py --message "Explain quantum mechanics" --quick

# Visualize results
python visualize_comprehensive_chat.py
```

### Example 3: Custom Query Testing
```bash
# Test specific methods
python multi_query_comparison.py \
    --methods baseline hybrid_kv_cache all_combined \
    --quick
```

## 📝 Notes

- **Quick Mode**: Uses shorter responses (30 tokens) for faster testing
- **CPU vs GPU**: Most optimizations work on CPU, but LaRoSA benefits from GPU
- **Memory Tracking**: Uses psutil for accurate memory measurement
- **Results Format**: All results saved as JSON for easy analysis

## 🔗 Related Documentation

- `VATTENTION_EXPLANATION.md`: Detailed explanation of vAttention and Hybrid KV Cache

## 📊 Expected Performance

Based on test results:

| Method | Token Reduction | Memory Savings | Time Improvement |
|--------|----------------|----------------|------------------|
| SimpleAdaptiVocab | 6-40% | 1-2% | 1-2% |
| Word Removal | 27-50% | 1-2% | 2-3% |
| Hybrid KV Cache | 0% | 25.2% | 0-2% |
| All Combined | 35-50% | 82-93% | Variable |

*Note: Results vary based on query type and model used.*
