# Quick Start Guide: Fused LLM Chatbot

Get started with the Fused LLM Chatbot in 3 steps!

## Step 1: Install Dependencies

**⚠️ Python Version Note:**
- **Python 3.11 or 3.12** (recommended) - Full PyTorch support
- **Python 3.13** - May need special setup (see below)

**For Python 3.11/3.12:**
```bash
# Install PyTorch (required for server)
pip install torch

# Install chatbot dependencies
pip install openai requests

# Install server dependencies
cd vattention/sarathi-lean
pip install -r requirements.txt
```

**For Python 3.13:**
```bash
# Use Python 3.13 compatible setup script
./setup_dependencies_py313.sh

# Or manually install latest compatible PyTorch
pip install torch  # May install 2.2.x instead of 2.3.0+
```

**Or use test mode to check what's missing:**
```bash
python fused_chatbot.py --test-server
```

**💡 Recommended: Use Python 3.11 or 3.12 for best compatibility**

For full functionality (vAttention + LaRoSA), you'll also need:
- NVIDIA GPU with CUDA
- See `vattention/README.md` for vAttention setup

## Step 2: (Optional) Create PatchTokenizer

If you want to use AdaptiVocab:

```bash
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py
```

This creates a `patch_tokenizer.pkl` file that the chatbot will auto-detect.

## Step 3: Run the Chatbot!

```bash
# Basic usage (auto-detects PatchTokenizer if available)
python fused_chatbot.py

# Or specify options
python fused_chatbot.py --model gpt2 --larosa-sparsity 0.4
```

## That's It! 🎉

The chatbot will:
1. ✅ Start the fused LLM server
2. ✅ Enable all available optimizations
3. ✅ Provide an interactive chat interface

## Example Session

```
$ python fused_chatbot.py

🚀 Starting Fused LLM Server...
   Model: gpt2
   AdaptiVocab: ✅
   vAttention: ✅
   LaRoSA: ✅
   Gstack Model: ❌ (using standard model)
⏳ Waiting for server to start...
✅ Server started successfully at http://localhost:8000

💬 Chatbot ready! Type 'quit' or 'exit' to end the conversation.

You: Hello!
🤖 Assistant: Hello! How can I help you today?

You: status
📊 Optimization Status:
============================================================
Model: gpt2
Gstack: ❌ Using standard model
AdaptiVocab: ✅ Enabled
vAttention: ✅ Enabled
LaRoSA: ✅ Enabled
   Sparsity: 40%
============================================================

You: quit
👋 Goodbye!
```

## Troubleshooting

**Server won't start?**
- Check if port 8000 is available: `lsof -i :8000`
- Try a different port: `--port 8001`

**No GPU?**
- On Mac, only AdaptiVocab will work
- Use `--no-vattention --no-larosa` to disable GPU features

**PatchTokenizer not found?**
- Create one (see Step 2) or use `--no-adaptivocab`

## More Options

```bash
# See all options
python fused_chatbot.py --help

# Use configuration file
python fused_chatbot.py --config chatbot_config.json

# Disable specific optimizations
python fused_chatbot.py --no-adaptivocab --no-larosa
```

## Next Steps

- Read [CHATBOT_README.md](CHATBOT_README.md) for detailed documentation
- See [README.md](README.md) for project overview
- Check [GSTACK_INTEGRATION.md](GSTACK_INTEGRATION.md) for Gstack details
- See [LAROSA_INTEGRATION.md](LAROSA_INTEGRATION.md) for LaRoSA details

---

Happy chatting! 🚀

