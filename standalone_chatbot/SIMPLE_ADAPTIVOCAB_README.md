# SimpleAdaptiVocab

A simplified phrase-based token reduction system that combines common word pairs/phrases into single tokens.

## How It Works

1. **Input Processing**: Detects common phrases (e.g., "artificial intelligence", "machine learning")
2. **Combination**: Combines them into single tokens (e.g., "artificialintelligence", "machinelearning")
3. **Tokenization**: Tokenizes the combined text
4. **Output Processing**: Splits combined tokens back into readable phrases

## Important Note

⚠️ **This is a simplified version!**

The full AdaptiVocab system:
- Adds combined phrases as **actual vocabulary tokens** to the model
- Requires model retraining/fine-tuning
- Provides 25%+ token reduction

This simplified version:
- Only combines phrases **if it reduces token count**
- Works with existing models (no retraining)
- Provides **limited** token reduction (depends on tokenizer)

## Why Limited Benefits?

Most modern tokenizers (like GPT-2, Phi-2) use **subword tokenization**:
- "artificial" → ["art", "ificial"]
- "intelligence" → ["Ġintelligence"]
- "artificial intelligence" → 3 tokens
- "artificialintelligence" → Still ~3 tokens (no benefit)

**Real token reduction** requires:
1. Adding phrases as new vocabulary tokens
2. Updating model embeddings
3. Fine-tuning the model

## Usage

```bash
# Enable SimpleAdaptiVocab in benchmark
python benchmark_comparison.py \
    --model microsoft/phi-2 \
    --simple-adaptivocab \
    --larosa-sparsity 0.0

# Use in chatbot
python fused_chatbot_enhanced.py \
    --model microsoft/phi-2 \
    --simple-adaptivocab \
    --interactive
```

## Custom Phrases

You can provide custom phrases:

```python
from simple_adaptivocab import SimpleAdaptiVocab
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
custom_phrases = [
    "quantum computing",
    "blockchain technology",
    "your domain phrases here",
]

adaptivocab = SimpleAdaptiVocab(
    tokenizer,
    common_phrases=custom_phrases
)
```

## Expected Results

- **Token Reduction**: 0-10% (depends on text and tokenizer)
- **Works on**: Both CPU and GPU
- **No model changes**: Works with existing models
- **Best for**: Domain-specific phrases that tokenizer splits inefficiently

## For Real Token Reduction

Use the full **AdaptiVocab** system:
1. Create PatchTokenizer using AdaptiVocab scripts
2. Fine-tune model with new vocabulary
3. Get 25%+ token reduction

```bash
# Full AdaptiVocab (requires patch tokenizer file)
python benchmark_comparison.py \
    --model microsoft/phi-2 \
    --patch-tokenizer /path/to/patch_tokenizer.pkl
```

## Summary

✅ **SimpleAdaptiVocab**: Quick phrase combination, limited benefits
✅ **Full AdaptiVocab**: Real vocabulary reduction, requires setup

Choose based on your needs!

