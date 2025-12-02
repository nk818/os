# How to Use the Fused Model

## 🎯 Quick Answer

The **fused model** is not a single file - it's a **running system** that combines:
- **AdaptiVocab** (via PatchTokenizer)
- **vAttention** (via Sarathi-Serve)
- **Base LLM** (e.g., GPT-2, Llama, etc.)

---

## 📍 Where Everything Is

### 1. Integration Code (Already Done ✅)
**Location**: `vattention/sarathi-lean/sarathi/`
- `transformers_utils/patch_tokenizer_wrapper.py` - AdaptiVocab integration
- `engine/base_llm_engine.py` - Engine that uses fused system
- `config.py` - Configuration with `patch_tokenizer_path` support

### 2. PatchTokenizer (You Need to Create)
**Location**: `AdaptiVocab/src/saved_patch_tokenizers_*/[config_name]/`
- Created by: `AdaptiVocab/src/build_vocab/create_patch_tokenizer.py`
- File: `patch_tokenizer.pkl`

### 3. Server Entry Point
**Location**: `vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py`
- This is where you start the fused model

---

## 🚀 Step-by-Step: Using the Fused Model

### Step 1: Create a PatchTokenizer

```bash
cd AdaptiVocab/src/build_vocab

# Edit create_patch_tokenizer.py to set your config, then:
python3 create_patch_tokenizer.py
```

**Output location**: 
```
AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[config_name]/
├── patch_tokenizer.pkl      # ← This is what you need
├── config.pkl
├── removed_tokens.pkl
└── added_ngrams.pkl
```

### Step 2: Start the Fused Model Server

```bash
cd vattention/sarathi-lean

python -m sarathi.entrypoints.openai_server.api_server \
    --model_name gpt2 \
    --patch_tokenizer_path ../../AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[your_config]/patch_tokenizer.pkl \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152
```

**What this does:**
- Loads base model (gpt2)
- Loads PatchTokenizer (AdaptiVocab) ✅
- Uses vAttention backend ✅
- **Fused model is now running!**

### Step 3: Use the Fused Model

The server runs on `http://localhost:8000` with OpenAI-compatible API.

**Test it:**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

---

## 🔍 Finding Your PatchTokenizer

If you've already created one:

```bash
# Search for it
find AdaptiVocab -name "patch_tokenizer.pkl" -type f

# Check the saved directory
ls -R AdaptiVocab/src/saved_patch_tokenizers*/
```

**Default save location** (from constants.py):
```
AdaptiVocab/src/saved_patch_tokenizers_ngram_k_analysis/
```

Or:
```
AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/
```

---

## 📝 Example: Complete Workflow

```bash
# 1. Create PatchTokenizer
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py
# → Creates: saved_patch_tokenizers_no_ngrams_new_logs/[config]/patch_tokenizer.pkl

# 2. Note the path
PATCH_PATH="$(pwd)/../saved_patch_tokenizers_no_ngrams_new_logs/[config]/patch_tokenizer.pkl"

# 3. Start fused model
cd ../../../vattention/sarathi-lean
python -m sarathi.entrypoints.openai_server.api_server \
    --model_name gpt2 \
    --patch_tokenizer_path "$PATCH_PATH" \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152

# 4. Fused model is now running!
# - AdaptiVocab: ✅ Active (via PatchTokenizer)
# - vAttention: ✅ Active (via fa_vattn backend)
```

---

## ⚠️ Important Notes

### The Fused Model Requires:

1. **PatchTokenizer** ✅ (Create using AdaptiVocab)
2. **GPU** ⚠️ (For vAttention - Mac can't run this)
3. **Base Model** ✅ (Downloaded automatically from HuggingFace)

### On Mac (Your System):

- ✅ Can create PatchTokenizer
- ✅ Can test tokenization
- ⚠️ Cannot run vAttention (needs CUDA/NVIDIA GPU)
- ⚠️ Can test AdaptiVocab tokenization only

### For Full Fused Model:

- Need GPU (AWS, GCP, or local NVIDIA GPU)
- Then both AdaptiVocab + vAttention work together

---

## 🎯 Summary

**The fused model is:**
- **Code**: Already integrated in `vattention/sarathi-lean/`
- **Tokenizer**: Create with AdaptiVocab → `patch_tokenizer.pkl`
- **Server**: Start via `api_server.py` with both enabled
- **Running**: When server starts with `--patch_tokenizer_path` + `--model_attention_backend fa_vattn`

**To use it:**
1. Create PatchTokenizer
2. Start server with PatchTokenizer path
3. Use OpenAI-compatible API

---

**Location Summary:**
- Integration code: `vattention/sarathi-lean/sarathi/` ✅
- PatchTokenizer: `AdaptiVocab/src/saved_patch_tokenizers*/[config]/patch_tokenizer.pkl` (create it)
- Server: `vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py` ✅




