# Where is the Fused Model?

## 🎯 Quick Answer

The **fused model** is not a single file - it's a **runtime system** that combines:
1. **Base LLM** (from HuggingFace, e.g., GPT-2, Llama)
2. **AdaptiVocab PatchTokenizer** (you create this)
3. **vAttention memory management** (built into Sarathi-Serve)

---

## 📍 Component Locations

### 1. Integration Code (✅ Already Done)
**Location**: `vattention/sarathi-lean/sarathi/`

**Key Files:**
- `transformers_utils/patch_tokenizer_wrapper.py` - AdaptiVocab integration
- `transformers_utils/tokenizer.py` - Modified to load PatchTokenizer
- `engine/base_llm_engine.py` - Engine that uses fused system
- `config.py` - Configuration with `patch_tokenizer_path` support
- `entrypoints/openai_server/api_server.py` - **Start server here**

### 2. PatchTokenizer (⚠️ You Need to Create)
**Location**: `AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[config_name]/`

**Files created:**
- `patch_tokenizer.pkl` - **This is what you need**
- `config.pkl`
- `removed_tokens.pkl`
- `added_ngrams.pkl`

**How to create:**
```bash
cd AdaptiVocab/src/build_vocab
python3 create_patch_tokenizer.py
```

### 3. Base Model (✅ Auto-downloaded)
**Location**: HuggingFace cache (auto-downloaded when you specify `--model_name`)

---

## 🚀 How to Run the Fused Model

### Step 1: Create PatchTokenizer

```bash
cd AdaptiVocab/src/build_vocab

# Edit create_patch_tokenizer.py to configure:
# - original_tokenizer (e.g., 'gpt2')
# - target_corpus_name (your domain dataset)
# - num_to_add, num_to_remove

python3 create_patch_tokenizer.py
```

**Output**: `AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[config_name]/patch_tokenizer.pkl`

### Step 2: Start Fused Model Server

```bash
cd vattention/sarathi-lean

python -m sarathi.entrypoints.openai_server.api_server \
    --model_name gpt2 \
    --patch_tokenizer_path ../../AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[your_config]/patch_tokenizer.pkl \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152
```

**This starts the fused model!**
- ✅ AdaptiVocab active (via PatchTokenizer)
- ✅ vAttention active (via `fa_vattn` backend)
- ✅ Server running on `http://localhost:8000`

---

## 📂 Complete File Structure

```
OS/
├── AdaptiVocab/
│   └── src/
│       ├── build_vocab/
│       │   └── create_patch_tokenizer.py  # ← Create PatchTokenizer here
│       └── saved_patch_tokenizers_no_ngrams_new_logs/  # ← PatchTokenizer saved here
│           └── [config_name]/
│               └── patch_tokenizer.pkl    # ← Your PatchTokenizer file
│
├── vattention/
│   └── sarathi-lean/
│       └── sarathi/
│           ├── transformers_utils/
│           │   ├── patch_tokenizer_wrapper.py  # ← Integration code
│           │   └── tokenizer.py                # ← Integration code
│           ├── engine/
│           │   └── base_llm_engine.py          # ← Integration code
│           └── entrypoints/
│               └── openai_server/
│                   └── api_server.py           # ← START FUSED MODEL HERE
│
└── benchmark_comprehensive.py  # Test fused vs normal
```

---

## 🔍 Finding Your PatchTokenizer

If you've already created one:

```bash
# Search for PatchTokenizer files
find AdaptiVocab -name "patch_tokenizer.pkl" -type f

# Check the default save location
ls -R AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/
```

**Default save path** (from `create_patch_tokenizer.py`):
```
AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[config_name]/patch_tokenizer.pkl
```

Where `[config_name]` is generated from your configuration (model name, corpus, etc.)

---

## ⚠️ Important Notes

### The Fused Model is:
- ✅ **Integration code** - Already in `vattention/sarathi-lean/`
- ✅ **Runtime system** - Starts when you run the server
- ⚠️ **Requires PatchTokenizer** - You need to create this first
- ⚠️ **Requires GPU** - For full vAttention benefits (Mac can't run vAttention)

### On Mac (Your System):
- ✅ Can create PatchTokenizer
- ✅ Can test tokenization
- ⚠️ Cannot run vAttention (needs CUDA/NVIDIA GPU)
- ⚠️ Can test AdaptiVocab tokenization only

### For Full Fused Model:
- Need GPU (AWS, GCP, or local NVIDIA GPU)
- Then both AdaptiVocab + vAttention work together

---

## 📝 Summary

**The fused model location:**

1. **Integration Code**: `vattention/sarathi-lean/sarathi/` ✅ (already done)
2. **PatchTokenizer**: `AdaptiVocab/src/saved_patch_tokenizers_no_ngrams_new_logs/[config]/patch_tokenizer.pkl` (create it)
3. **Server Entry**: `vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py` ✅ (start here)

**To use it:**
1. Create PatchTokenizer → `AdaptiVocab/src/build_vocab/create_patch_tokenizer.py`
2. Start server → `vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py`
3. Use with `--patch_tokenizer_path` + `--model_attention_backend fa_vattn`

---

**The fused model = Integration code + PatchTokenizer + Running server**




