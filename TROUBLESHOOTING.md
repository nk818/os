# Troubleshooting: Server Won't Start

## Common Issues and Solutions

### Issue: Server Times Out During Startup

**Symptoms:**
- Server process starts but never responds
- Timeout after 5 minutes
- No error messages visible

**Solutions:**

#### 1. Check Server Output Manually

Run the server manually to see the actual error:

```bash
cd vattention/sarathi-lean
python -m sarathi.entrypoints.openai_server.api_server
```

This will show you the full error message.

#### 2. Check Dependencies

The server requires specific dependencies:

```bash
cd vattention/sarathi-lean
pip install -r requirements.txt
```

#### 3. Check GPU Availability

vAttention requires NVIDIA GPU with CUDA:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi
```

**On Mac:** vAttention won't work. Use `--no-vattention` flag.

#### 4. Check Port Availability

Port 8000 might be in use:

```bash
# Check if port is in use
lsof -i :8000

# Use a different port
python fused_chatbot.py --port 8001
```

#### 5. Check Configuration

The server uses a YAML config file. Check if default config exists:

```bash
ls vattention/sarathi-lean/sarathi/benchmark/config/
```

#### 6. Test Server Startup

Use the test mode:

```bash
python fused_chatbot.py --test-server
```

### Issue: Import Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'torch'`
- `ModuleNotFoundError` or `ImportError`
- Missing dependencies

**Solutions:**

**Most Common: Missing PyTorch**
```bash
# Install PyTorch
pip install torch

# Or install all dependencies
cd vattention/sarathi-lean
pip install -r requirements.txt
```

**Check Dependencies:**
```bash
# Use the test mode to check dependencies
python fused_chatbot.py --test-server
```

**Install All Dependencies:**
```bash
# Install all dependencies
cd vattention/sarathi-lean
pip install -r requirements.txt

# Install chatbot dependencies
pip install openai requests
```

### Issue: GPU Not Available

**Symptoms:**
- CUDA errors
- "No CUDA devices found"

**Solutions:**

1. **Disable GPU features:**
   ```bash
   python fused_chatbot.py --no-vattention --no-larosa
   ```

2. **Use CPU-only mode** (if supported by the model)

### Issue: PatchTokenizer Not Found

**Symptoms:**
- Warning about missing PatchTokenizer
- AdaptiVocab disabled

**Solutions:**

1. **Create PatchTokenizer:**
   ```bash
   cd AdaptiVocab/src/build_vocab
   python3 create_patch_tokenizer.py
   ```

2. **Specify path manually:**
   ```bash
   python fused_chatbot.py --patch-tokenizer path/to/patch_tokenizer.pkl
   ```

3. **Disable AdaptiVocab:**
   ```bash
   python fused_chatbot.py --no-adaptivocab
   ```

### Issue: Model Not Found

**Symptoms:**
- "Model not found" error
- HuggingFace download fails

**Solutions:**

1. **Check internet connection** (for model download)

2. **Use a different model:**
   ```bash
   python fused_chatbot.py --model gpt2-medium
   ```

3. **Download model manually:**
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("gpt2")
   ```

## Diagnostic Commands

### Check Server Process

```bash
# See if server is running
ps aux | grep sarathi

# Check server logs
# (if running in foreground, you'll see logs directly)
```

### Check Dependencies

```bash
# Check Python version
python --version  # Should be 3.8+

# Check installed packages
pip list | grep -E "(torch|transformers|fastapi|uvicorn)"

# Check CUDA (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test Individual Components

```bash
# Test AdaptiVocab
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py

# Test server manually
cd vattention/sarathi-lean
python -m sarathi.entrypoints.openai_server.api_server --model_name gpt2
```

### Issue: Python 3.13 Compatibility

**Symptoms:**
- `ERROR: Could not find a version that satisfies the requirement torch>=2.3.0`
- `ERROR: No matching distribution found for torch>=2.3.0`

**Cause:**
PyTorch 2.3.0+ may not have official builds for Python 3.13 yet.

**Solutions:**

1. **Use Python 3.11 or 3.12 (Recommended):**
   ```bash
   # Create new conda environment
   conda create -n fused_llm python=3.11
   conda activate fused_llm
   
   # Then install dependencies
   ./setup_dependencies.sh
   ```

2. **Try Python 3.13 compatible setup:**
   ```bash
   ./setup_dependencies_py313.sh
   ```

3. **Install PyTorch manually (may get older version):**
   ```bash
   pip install torch  # Will install latest compatible version
   ```

4. **Try nightly builds:**
   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

## Getting Help

If none of these solutions work:

1. **Run with verbose output:**
   ```bash
   python fused_chatbot.py --test-server
   ```

2. **Check server logs:**
   - Server output is now captured and displayed
   - Look for error messages in the output

3. **Check system requirements:**
   - Python 3.8+
   - Sufficient RAM
   - GPU (for vAttention/LaRoSA)
   - Internet connection (for model download)

4. **Try minimal configuration:**
   ```bash
   python fused_chatbot.py --no-adaptivocab --no-vattention --no-larosa
   ```

## Common Error Messages

### "Server failed to start within timeout"
- **Cause:** Server is taking too long or failing silently
- **Solution:** Run server manually to see full error

### "Cannot import sarathi module"
- **Cause:** Missing dependencies or wrong Python environment
- **Solution:** Install requirements: `cd vattention/sarathi-lean && pip install -r requirements.txt`

### "Port 8000 is already in use"
- **Cause:** Another process is using the port
- **Solution:** Use `--port 8001` or kill the other process

### "CUDA out of memory"
- **Cause:** GPU doesn't have enough memory
- **Solution:** Use smaller model or disable GPU features

---

**Last Updated:** After improving error handling in fused_chatbot.py

