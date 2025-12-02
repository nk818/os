# Fused LLM Chatbot

A unified chatbot interface that integrates all four optimization methods:
- **Gstack**: Model growth for efficient pre-training
- **AdaptiVocab**: Domain-specific vocabulary optimization
- **vAttention**: Dynamic KV-cache memory management
- **LaRoSA**: Activation sparsity for faster inference

## 🚀 Quick Start

### Basic Usage

```bash
# Run with auto-detection (finds PatchTokenizer if available)
python fused_chatbot.py

# Specify a model
python fused_chatbot.py --model gpt2

# Use a specific PatchTokenizer
python fused_chatbot.py --patch-tokenizer path/to/patch_tokenizer.pkl

# Disable specific optimizations
python fused_chatbot.py --no-adaptivocab --no-larosa

# Use configuration file
python fused_chatbot.py --config chatbot_config.json
```

### With All Optimizations

```bash
python fused_chatbot.py \
    --model gpt2 \
    --patch-tokenizer AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[config]/patch_tokenizer.pkl \
    --larosa-sparsity 0.4
```

## 📋 Requirements

**Critical Dependencies:**
```bash
# PyTorch (REQUIRED - server won't start without it)
pip install torch

# Chatbot dependencies
pip install openai requests

# Server dependencies
cd vattention/sarathi-lean
pip install -r requirements.txt
```

**Check if dependencies are installed:**
```bash
python fused_chatbot.py --test-server
```

**For vAttention (requires GPU):**
- See `vattention/README.md` for setup instructions
- On Mac, use `--no-vattention` flag

## ⚙️ Configuration

### Command Line Options

- `--model`: Model name (default: gpt2)
- `--patch-tokenizer`: Path to PatchTokenizer for AdaptiVocab
- `--no-adaptivocab`: Disable AdaptiVocab
- `--no-vattention`: Disable vAttention (use standard attention)
- `--no-larosa`: Disable LaRoSA
- `--larosa-sparsity`: LaRoSA sparsity level (0.0-1.0, default: 0.4)
- `--gstack-model`: Path to Gstack-trained model
- `--port`: Server port (default: 8000)
- `--config`: Path to JSON configuration file

### Configuration File

Create a `chatbot_config.json` file:

```json
{
  "model_name": "gpt2",
  "patch_tokenizer_path": "path/to/patch_tokenizer.pkl",
  "attention_backend": "fa_vattn",
  "block_size": 2097152,
  "larosa_sparsity": 0.4,
  "server_host": "localhost",
  "server_port": 8000,
  "gstack_model_path": null
}
```

## 🎮 Interactive Commands

While chatting, you can use these commands:

- `quit` or `exit` - Exit the chatbot
- `clear` - Clear conversation history
- `status` - Show optimization status
- `help` - Show help message

## 📊 Optimization Status

The chatbot shows which optimizations are active:

```
📊 Optimization Status:
============================================================
Model: gpt2
Gstack: ✅ Enabled
AdaptiVocab: ✅ Enabled
vAttention: ✅ Enabled
LaRoSA: ✅ Enabled
   Sparsity: 40%
============================================================
```

## 🔧 Setup Steps

### 1. Create PatchTokenizer (AdaptiVocab)

```bash
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py
```

This creates a `patch_tokenizer.pkl` file that the chatbot can auto-detect.

### 2. (Optional) Train with Gstack

If you have a Gstack-trained model, specify it with `--gstack-model`.

### 3. Start Chatbot

```bash
python fused_chatbot.py
```

## 💡 Example Usage

```bash
# Start chatbot
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

You: Hello! What optimizations are you using?

🤖 Assistant: I'm using several optimizations to improve efficiency:
- AdaptiVocab for token reduction
- vAttention for memory management
- LaRoSA for activation sparsity

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

## ⚠️ Notes

1. **GPU Required**: vAttention and LaRoSA require NVIDIA GPU with CUDA
2. **Mac Compatibility**: On Mac, only AdaptiVocab tokenization works (no GPU support)
3. **Gstack**: Gstack is a pre-training optimization. The chatbot uses Gstack-trained models if available
4. **Auto-detection**: The chatbot automatically finds PatchTokenizer files if not specified

## 🐛 Troubleshooting

### Server won't start

**Most common issue:** The server is failing silently. The chatbot now shows server output to help diagnose.

**Quick fixes:**
1. **Test server manually:**
   ```bash
   python fused_chatbot.py --test-server
   ```

2. **Run server directly to see errors:**
   ```bash
   cd vattention/sarathi-lean
   python -m sarathi.entrypoints.openai_server.api_server
   ```

3. **Check dependencies:**
   ```bash
   cd vattention/sarathi-lean
   pip install -r requirements.txt
   ```

4. **On Mac (no GPU):**
   ```bash
   python fused_chatbot.py --no-vattention --no-larosa
   ```

5. **Use different port:**
   ```bash
   python fused_chatbot.py --port 8001
   ```

**See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.**

### PatchTokenizer not found
- Create one with AdaptiVocab: `cd AdaptiVocab/src/build_vocab && python3 create_patch_tokenizer.py`
- Or specify path with `--patch-tokenizer`
- Or disable: `--no-adaptivocab`

### Import errors
```bash
pip install openai requests
cd vattention/sarathi-lean && pip install -r requirements.txt
```

## 📚 Related Documentation

- [README.md](README.md) - Main project documentation
- [GSTACK_INTEGRATION.md](GSTACK_INTEGRATION.md) - Gstack integration details
- [LAROSA_INTEGRATION.md](LAROSA_INTEGRATION.md) - LaRoSA integration details
- [HOW_TO_USE_FUSED_MODEL.md](HOW_TO_USE_FUSED_MODEL.md) - Server setup guide

---

**Status**: Ready to use! 🚀

