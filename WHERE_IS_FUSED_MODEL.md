# Where is the Fused Model?

## 📍 Location Overview

The **fused model** isn't a single file - it's a **system** that combines:
1. **AdaptiVocab** (tokenizer optimization)
2. **vAttention** (memory management)
3. **Sarathi-Serve** (serving framework)

---

## 🔧 Integration Code Location

The fused system code is integrated into:

### 1. Tokenizer Integration
**Location**: `vattention/sarathi-lean/sarathi/transformers_utils/`
- `patch_tokenizer_wrapper.py` - Wrapper for AdaptiVocab PatchTokenizer
- `tokenizer.py` - Modified to support PatchTokenizer

### 2. Engine Integration
**Location**: `vattention/sarathi-lean/sarathi/engine/`
- `base_llm_engine.py` - Engine that uses fused tokenizer

### 3. Configuration
**Location**: `vattention/sarathi-lean/sarathi/config.py`
- `ModelConfig` - Supports `patch_tokenizer_path` parameter

---

## 🚀 How to Use the Fused Model

### Step 1: Create a PatchTokenizer (AdaptiVocab)

First, you need to create a PatchTokenizer for your domain:

```bash
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py
```

This creates a `patch_tokenizer.pkl` file (usually in a saved_patch_tokenizers directory).

**Example location**: 
```
AdaptiVocab/src/saved_patch_tokenizers/[config_name]/patch_tokenizer.pkl
```

### Step 2: Start the Fused Model Server

Use Sarathi-Serve with the PatchTokenizer:

```bash
cd vattention/sarathi-lean

python -m sarathi.entrypoints.openai_server.api_server \
    --model_name meta-llama/Llama-2-7b-hf \
    --patch_tokenizer_path /path/to/patch_tokenizer.pkl \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152
```

**Key parameters:**
- `--model_name`: Base model to use
- `--patch_tokenizer_path`: Path to your PatchTokenizer (enables AdaptiVocab)
- `--model_attention_backend fa_vattn`: Enables vAttention
- `--model_block_size`: vAttention page size

### Step 3: The Model is Now "Fused"

When you start the server with these parameters:
- ✅ **AdaptiVocab** is active (via PatchTokenizer)
- ✅ **vAttention** is active (via attention backend)
- ✅ **Fused system** is running!

---

## 📂 File Structure

```
OS/
├── AdaptiVocab/
│   └── src/
│       ├── build_vocab/
│       │   └── create_patch_tokenizer.py  # Create PatchTokenizer
│       └── saved_patch_tokenizers/        # PatchTokenizer files saved here
│           └── [config_name]/
│               └── patch_tokenizer.pkl    # ← Your PatchTokenizer
│
├── vattention/
│   └── sarathi-lean/
│       └── sarathi/
│           ├── transformers_utils/
│           │   ├── patch_tokenizer_wrapper.py  # ← Integration code
│           │   └── tokenizer.py                # ← Integration code
│           ├── engine/
│           │   └── base_llm_engine.py            # ← Integration code
│           └── entrypoints/
│               └── openai_server/
│                   └── api_server.py            # ← Start server here
│
└── benchmark_comprehensive.py  # Test fused vs normal
```

---

## 🎯 Quick Start: Using the Fused Model

### Option 1: Via API Server (Recommended)

```bash
# 1. Create PatchTokenizer (if not done)
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py

# 2. Start fused model server
cd ../../vattention/sarathi-lean
python -m sarathi.entrypoints.openai_server.api_server \
    --model_name gpt2 \
    --patch_tokenizer_path ../../../AdaptiVocab/src/saved_patch_tokenizers/[your_config]/patch_tokenizer.pkl \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152
```

### Option 2: Via Benchmark Script

```bash
# Test fused model
python3 benchmark_comprehensive.py \
    --model gpt2 \
    --patch-tokenizer path/to/patch_tokenizer.pkl
```

---

## 🔍 Finding Your PatchTokenizer

If you've already created one, find it:

```bash
# Search for PatchTokenizer files
find AdaptiVocab -name "patch_tokenizer.pkl" -type f

# Or check saved_patch_tokenizers directory
ls -R AdaptiVocab/src/saved_patch_tokenizers/
```

---

## ⚠️ Important Notes

### The "Fused Model" is:
- ✅ **Code integration** (already done - in vattention/sarathi-lean)
- ✅ **Runtime system** (starts when you run the server)
- ⚠️ **Requires PatchTokenizer** (you need to create this)
- ⚠️ **Requires GPU** (for vAttention - Mac can't run this part)

### What You Have Now:
- ✅ Integration code (complete)
- ✅ Benchmark framework (complete)
- ⬜ PatchTokenizer (you need to create)
- ⬜ GPU access (for full vAttention benefits)

---

## 📝 Summary

**The fused model is not a single file** - it's:
1. **Integration code**: Already in `vattention/sarathi-lean/sarathi/`
2. **PatchTokenizer**: Create using AdaptiVocab
3. **Running system**: Start via Sarathi-Serve API server

**To use it:**
1. Create PatchTokenizer → `AdaptiVocab/src/build_vocab/create_patch_tokenizer.py`
2. Start server → `vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py`
3. Use with `--patch_tokenizer_path` and `--model_attention_backend fa_vattn`

---

**The fused model = Integration code + PatchTokenizer + Running server**




