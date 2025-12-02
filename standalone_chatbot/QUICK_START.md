# Quick Start Guide

## Installation (2 minutes)

```bash
# 1. Create environment
conda create -n standalone_chat python=3.11 -y
conda activate standalone_chat

# 2. Install dependencies
cd standalone_chatbot
pip install -r requirements.txt
```

## Basic Usage

### Interactive Chat

```bash
python standalone_chat.py --model gpt2 --interactive
```

### Single Prompt

```bash
python standalone_chat.py --model gpt2 --prompt "Hello, how are you?"
```

### Start API Server

```bash
python api_server.py --model gpt2 --port 8000
```

Then test with:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gpt2"
  }'
```

## With Optimizations

### AdaptiVocab (Tokenizer Optimization)

```bash
python standalone_chat.py \
    --model gpt2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl \
    --interactive
```

### LaRoSA (Activation Sparsity)

```bash
python standalone_chat.py \
    --model gpt2 \
    --larosa-sparsity 0.4 \
    --interactive
```

### Both Combined

```bash
python standalone_chat.py \
    --model gpt2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl \
    --larosa-sparsity 0.4 \
    --interactive
```

## Examples

Run the example script:

```bash
python example_usage.py
```

## Troubleshooting

### Import Error

If you see `AdaptiVocab not available`, it's okay - the chatbot will work without it, just without tokenizer optimization.

### Out of Memory

Use a smaller model or reduce generation length:

```bash
python standalone_chat.py --model gpt2 --interactive
```

### Slow Generation

This is normal on CPU. For faster generation:
- Use GPU if available: `--device cuda`
- Use smaller models
- Reduce `max_new_tokens` in the code

## Next Steps

- Read `README.md` for detailed documentation
- Try different models (gpt2, gpt2-medium, etc.)
- Create your own PatchTokenizer with AdaptiVocab
- Experiment with different LaRoSA sparsity levels

