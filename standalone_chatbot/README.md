# Standalone Fused LLM Chatbot

A clean, standalone implementation of the Fused LLM Efficiency System that works on macOS and doesn't require CUDA or complex server setups.

## Features

✅ **AdaptiVocab Integration** - Tokenizer optimization for 25%+ token reduction  
✅ **LaRoSA Activation Sparsity** - Simplified CPU implementation  
✅ **Simple Inference Engine** - No CUDA dependencies  
✅ **Interactive Chat Interface** - Easy-to-use CLI  
✅ **macOS Compatible** - Works out of the box on Mac  

## Installation

### 1. Create Conda Environment

```bash
conda create -n standalone_chat python=3.11 -y
conda activate standalone_chat
```

### 2. Install Dependencies

```bash
cd standalone_chatbot
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (Standard Model)

```bash
python standalone_chat.py --model gpt2 --interactive
```

### With AdaptiVocab (Tokenizer Optimization)

First, create a PatchTokenizer (see AdaptiVocab documentation):

```bash
# Create PatchTokenizer
cd ../AdaptiVocab/src/build_vocab
python create_patch_tokenizer.py
```

Then use it:

```bash
python standalone_chat.py \
    --model gpt2 \
    --patch-tokenizer ../AdaptiVocab/src/saved_patch_tokenizers/[config]/patch_tokenizer.pkl \
    --interactive
```

### With LaRoSA (Activation Sparsity)

```bash
python standalone_chat.py \
    --model gpt2 \
    --larosa-sparsity 0.4 \
    --interactive
```

### Combined (All Optimizations)

```bash
python standalone_chat.py \
    --model gpt2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl \
    --larosa-sparsity 0.4 \
    --interactive
```

## Command Line Options

```
--model MODEL              Model name (default: gpt2)
--patch-tokenizer PATH    Path to AdaptiVocab PatchTokenizer .pkl file
--larosa-sparsity FLOAT   LaRoSA sparsity level (0.0-1.0, default: 0.0)
--device DEVICE           Device: cpu or cuda (default: cpu)
--interactive             Run in interactive chat mode
--prompt TEXT             Single prompt to generate from
```

## Examples

### Single Generation

```bash
python standalone_chat.py \
    --model gpt2 \
    --prompt "The future of AI is"
```

### Interactive Chat

```bash
python standalone_chat.py --model gpt2 --interactive

You: Hello! How are you?
Assistant: Hello! I'm doing well, thank you for asking. How can I help you today?

You: Tell me about machine learning
Assistant: Machine learning is a subset of artificial intelligence...
```

## Architecture

```
standalone_chatbot/
├── standalone_chat.py      # Main chatbot implementation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Components

1. **FusedLLMEngine** - Main engine that integrates all optimizations
2. **SimpleLaRoSA** - CPU-compatible activation sparsity
3. **AdaptiVocab Integration** - PatchTokenizer support
4. **Chat Interface** - Interactive conversation support

## Differences from Sarathi-Serve Version

| Feature | Sarathi-Serve | Standalone |
|---------|--------------|------------|
| CUDA Required | ✅ Yes | ❌ No |
| Complex Setup | ✅ Yes | ❌ No |
| macOS Support | ⚠️ Limited | ✅ Full |
| vAttention | ✅ Yes | ❌ Simplified |
| LaRoSA | ✅ Full | ⚠️ Simplified |
| AdaptiVocab | ✅ Yes | ✅ Yes |
| Easy to Use | ⚠️ Complex | ✅ Simple |

## Limitations

- **LaRoSA**: Simplified CPU implementation (no custom CUDA kernels)
- **vAttention**: Not included (requires CUDA)
- **Performance**: Slower than GPU-accelerated version
- **Models**: Works best with smaller models (GPT-2, etc.)

## Future Enhancements

- [ ] Add REST API server
- [ ] Improve LaRoSA implementation
- [ ] Add model quantization support
- [ ] Add streaming responses
- [ ] Add conversation export/import

## Troubleshooting

### Import Error: AdaptiVocab

If you see `AdaptiVocab not available`, make sure you've installed AdaptiVocab:

```bash
cd ../AdaptiVocab
pip install -r requirements.txt
```

### Out of Memory

For larger models, use a smaller model or reduce `max_new_tokens`:

```bash
python standalone_chat.py --model gpt2 --interactive
```

### Slow Generation

This is expected on CPU. For faster generation:
- Use smaller models (gpt2 instead of gpt2-medium)
- Reduce `max_new_tokens`
- Use GPU if available (`--device cuda`)

## License

Same as the main repository.

