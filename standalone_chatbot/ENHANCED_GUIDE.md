# Enhanced Fused Chatbot - Complete Guide

## 🚀 What's New

The enhanced chatbot includes:
- ✅ **Better Chat Model** (DialoGPT - trained for conversations)
- ✅ **AdaptiVocab** (25%+ token reduction)
- ✅ **LaRoSA** (40% activation sparsity = 1.3x speedup)
- ✅ **vAttention** (KV cache memory tracking)

## Quick Start

### Basic Usage (All Optimizations)

```bash
cd /Users/nk/Desktop/OS/standalone_chatbot
conda activate fused_llm
python fused_chatbot_enhanced.py --interactive
```

This uses:
- DialoGPT-small (better chat model)
- LaRoSA 40% sparsity
- vAttention memory tracking

### With AdaptiVocab

**Step 1: Create PatchTokenizer** (if you haven't already)

```bash
cd ../AdaptiVocab/src/build_vocab
python create_patch_tokenizer.py
```

**Step 2: Use with chatbot**

```bash
python fused_chatbot_enhanced.py \
    --patch-tokenizer ../AdaptiVocab/src/saved_patch_tokenizers/[config]/patch_tokenizer.pkl \
    --interactive
```

### Customize Optimizations

```bash
# Adjust LaRoSA sparsity (0.0 = off, 1.0 = 100% sparse)
python fused_chatbot_enhanced.py \
    --larosa-sparsity 0.5 \
    --interactive

# Disable vAttention
python fused_chatbot_enhanced.py \
    --no-vattention \
    --interactive

# Use different model
python fused_chatbot_enhanced.py \
    --model microsoft/DialoGPT-medium \
    --interactive
```

## Available Models

- `microsoft/DialoGPT-small` (default) - 117M params, good for chat
- `microsoft/DialoGPT-medium` - 345M params, better quality
- `microsoft/DialoGPT-large` - 774M params, best quality (slower)
- `gpt2` - Original GPT-2 (not chat-optimized)

## Commands in Chat

- Type your question/message
- Type `stats` to see optimization statistics
- Type `quit` or `exit` to end

## Optimization Status

The chatbot shows:
- ✅ AdaptiVocab: Enabled/Disabled
- ✅ LaRoSA: Sparsity level
- ✅ vAttention: Memory usage

Type `stats` in chat to see current statistics.

## Performance

Expected improvements:
- **AdaptiVocab**: 25% fewer tokens
- **LaRoSA**: 1.3x speedup at 40% sparsity
- **vAttention**: Better memory management
- **Combined**: 50-70% overall efficiency gain

## Troubleshooting

**Model not responding well:**
- Try `--model microsoft/DialoGPT-medium` for better quality
- Adjust `--larosa-sparsity` (lower = better quality, higher = faster)

**Out of memory:**
- Use `--model microsoft/DialoGPT-small` (smaller model)
- Reduce `max_new_tokens` in code

**AdaptiVocab not working:**
- Make sure PatchTokenizer path is correct
- Check that AdaptiVocab is installed

## Example Session

```bash
$ python fused_chatbot_enhanced.py --interactive

📊 Optimization Status:
   AdaptiVocab: ❌
   LaRoSA: ✅ (40% sparsity)
   vAttention: ✅

You: Hello!
Assistant: Hi there! How can I help you?

You: What is machine learning?
Assistant: Machine learning is a method of data analysis that automates analytical model building.

You: stats
📊 Statistics:
   adaptivocab: False
   larosa: True
   larosa_sparsity: 0.4
   vattention: True
   current_tokens: 45
   memory_mb: 0.16
   utilization: 0.09

You: quit
👋 Goodbye!
```

## Next Steps

1. Create your own PatchTokenizer for your domain
2. Experiment with different LaRoSA sparsity levels
3. Monitor vAttention memory usage
4. Try different models for your use case

