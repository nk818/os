# Adding AdaptiVocab to Benchmark

## Quick Setup

AdaptiVocab requires a pre-trained PatchTokenizer. Here's how to add it:

### Option 1: Use Existing PatchTokenizer

If you have a PatchTokenizer already created:

```bash
python benchmark_comparison.py \
    --model microsoft/phi-2 \
    --patch-tokenizer /path/to/patch_tokenizer.pkl \
    --larosa-sparsity 0.4
```

### Option 2: Create PatchTokenizer Using AdaptiVocab

To create a PatchTokenizer, you need to use the AdaptiVocab scripts:

1. **Install AdaptiVocab dependencies:**
   ```bash
   cd ../AdaptiVocab
   pip install -r requirements.txt
   ```

2. **Create PatchTokenizer:**
   ```bash
   cd src/build_vocab
   python create_patch_tokenizer.py
   ```
   
   This will create a PatchTokenizer in the AdaptiVocab saved directory.

3. **Use in benchmark:**
   ```bash
   python benchmark_comparison.py \
       --model microsoft/phi-2 \
       --patch-tokenizer /path/to/saved/patch_tokenizer.pkl \
       --larosa-sparsity 0.4
   ```

### Option 3: Test Without AdaptiVocab (Current)

The benchmark works without AdaptiVocab - it will just show:
- ✅ LaRoSA speed improvements
- ✅ vAttention memory tracking
- ❌ No token reduction (AdaptiVocab not enabled)

## Expected Results with AdaptiVocab

When AdaptiVocab is enabled, you should see:

```
✅ AdaptiVocab enabled (vocab: 50257 → 37692, 25.0% reduction)
```

And in the benchmark results:
- **Token reduction**: 25%+ fewer tokens
- **Faster generation**: Fewer tokens = faster processing
- **Lower memory**: Fewer tokens in KV cache

## Current Status

The code is **ready** to use AdaptiVocab - just provide a `--patch-tokenizer` path!

The integration handles:
- ✅ Loading PatchTokenizer from .pkl file
- ✅ Using PatchTokenizer for tokenization
- ✅ Proper decoding with PatchTokenizer
- ✅ Token counting with reduced vocabulary

## Troubleshooting

**Error: "Failed to load PatchTokenizer"**
- Make sure the .pkl file exists
- Check that AdaptiVocab is in the Python path
- Verify the PatchTokenizer was created for the same model

**Error: "No module named 'datasets'"**
- Install AdaptiVocab dependencies: `pip install datasets`
- Or use an existing PatchTokenizer file

**No token reduction shown:**
- Verify AdaptiVocab is enabled: Check for "✅ AdaptiVocab enabled" message
- The PatchTokenizer must have actually reduced vocabulary
- Check the benchmark output for "adaptivocab: true"

