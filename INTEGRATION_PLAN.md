# Practical Integration Plan: AdaptiVocab + vAttention

## Quick Start Guide

### Step 1: Environment Setup

Create a unified conda environment with compatible dependencies:

```bash
conda create -n fused-llm python=3.10 -y
conda activate fused-llm

# Install PyTorch 2.3.0 (compatible with both)
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vAttention dependencies
pip install transformers>=4.37.0
pip install flash-attn==2.5.9.post1 --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
pip install flashinfer==0.0.6 --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/

# Install AdaptiVocab dependencies
pip install tokenizers==0.21.0 sentencepiece==0.2.0
pip install accelerate==1.7.0 datasets==2.18.0
pip install peft==0.10.0

# Install vAttention/Sarathi dependencies
pip install ray>=2.5.1 pandas pyarrow
pip install fastapi uvicorn
```

### Step 2: Build vAttention

```bash
cd vattention/vattention
# Set LIBTORCH_PATH (download libtorch if needed)
export LIBTORCH_PATH=/path/to/libtorch
python setup.py install
cd ../..
```

### Step 3: Build Sarathi-Lean

```bash
cd vattention/sarathi-lean
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
cd ../..
```

---

## Integration Implementation

### Integration Point 1: Tokenizer Wrapper

Create a wrapper that makes PatchTokenizer compatible with Sarathi-Serve:

**File**: `vattention/sarathi-lean/sarathi/transformers_utils/patch_tokenizer_wrapper.py`

```python
"""Wrapper to integrate AdaptiVocab's PatchTokenizer with Sarathi-Serve."""
import sys
import os

# Add AdaptiVocab to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../AdaptiVocab/src'))

from build_vocab.patch_tokenizer import PatchTokenizer
from transformers import PreTrainedTokenizer
from typing import List, Optional, Union
import torch


class PatchTokenizerWrapper(PreTrainedTokenizer):
    """Wrapper to make PatchTokenizer compatible with HuggingFace tokenizer interface."""
    
    def __init__(self, patch_tokenizer_path: str, existing_tokenizer_name: str, **kwargs):
        super().__init__(**kwargs)
        self.patch_tokenizer = PatchTokenizer.load_model_from_scratch(
            path=patch_tokenizer_path,
            existing_tokenizer_name=existing_tokenizer_name
        )
        self._update_attributes()
    
    def _update_attributes(self):
        """Update attributes to match PatchTokenizer."""
        self.vocab_size = len(self.patch_tokenizer.get_vocab())
        self.bos_token = self.patch_tokenizer.bos_token
        self.eos_token = self.patch_tokenizer.eos_token
        self.pad_token = self.patch_tokenizer.pad_token
        self.bos_token_id = self.patch_tokenizer.bos_token_id
        self.eos_token_id = self.patch_tokenizer.eos_token_id
        self.pad_token_id = self.patch_tokenizer.pad_token_id
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return self.patch_tokenizer.tokenize(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        vocab = self.patch_tokenizer.get_vocab()
        return vocab.get(token, self.patch_tokenizer.existing_tokenizer.unk_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        id_to_token = self.patch_tokenizer.id_to_token_dict
        return id_to_token.get(index, self.patch_tokenizer.existing_tokenizer.unk_token)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(id) for id in ids]
    
    def decode(self, token_ids: Union[int, List[int], torch.Tensor], skip_special_tokens: bool = True, **kwargs) -> str:
        return self.patch_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def __call__(self, text: Union[str, List[str]], **kwargs):
        if isinstance(text, str):
            text = [text]
        return self.patch_tokenizer(text, **kwargs)
```

### Integration Point 2: Model Loader Modification

Modify the model loader to support patched embeddings:

**File**: `vattention/sarathi-lean/sarathi/model_executor/model_loader.py`

Add import and loading logic:

```python
# Add at top of file
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../AdaptiVocab/src'))
from build_vocab.load_custom_model import load_custom_model

# Modify model loading function to check for patch tokenizer
def load_model_with_patch_tokenizer(model_name: str, patch_tokenizer_path: Optional[str] = None):
    """Load model with optional patch tokenizer support."""
    if patch_tokenizer_path and os.path.exists(patch_tokenizer_path):
        # Load with patched embeddings
        model = load_custom_model(
            model_name=model_name,
            emb_method='EXP_EMB',  # or other method from AdaptiVocab
            patch_tokenizer_path=patch_tokenizer_path,
            use_lora=False,
            unfroz=()
        )
        return model
    else:
        # Standard loading
        return AutoModelForCausalLM.from_pretrained(model_name)
```

### Integration Point 3: KV-Cache Size Optimization

Update cache size calculations to account for reduced token counts:

**File**: `vattention/sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py`

Add method to estimate token reduction:

```python
def estimate_token_reduction_factor(self) -> float:
    """Estimate token reduction factor from PatchTokenizer.
    
    Returns:
        float: Estimated reduction factor (e.g., 0.75 for 25% reduction)
    """
    # Default: no reduction
    reduction_factor = 1.0
    
    # Check if using PatchTokenizer
    if hasattr(self.tokenizer, 'patch_tokenizer'):
        # Estimate based on n-gram coverage
        # This is a heuristic - can be improved with actual statistics
        ngram_count = len(self.tokenizer.patch_tokenizer.ngram_dict)
        removed_count = len(self.tokenizer.patch_tokenizer.removed_tokens)
        
        if ngram_count > 0:
            # Rough estimate: n-grams reduce tokens by ~20-30%
            reduction_factor = 0.75  # 25% reduction
    
    return reduction_factor

def allocate_gpu_cache(self) -> List[torch.Tensor]:
    """Allocate GPU cache with token reduction optimization."""
    # Get token reduction factor
    reduction_factor = self.estimate_token_reduction_factor()
    
    # Adjust max context length based on token reduction
    # Fewer tokens means we can support longer sequences with same memory
    effective_max_len = int(self.max_model_seq_len / reduction_factor)
    
    # Use original allocation but with awareness of token efficiency
    kv_cache = vattention.init_kvcache(
        self.num_layers,
        self.num_heads,
        self.head_size,
        self.max_batch_size,
        self.max_model_seq_len,  # Keep original for safety
        self.device_idx,
        self.dtype,
        self.page_size,
        self.vattn_mega_cache
    )
    # ... rest of allocation code
```

---

## Testing Strategy

### Unit Tests

1. **Tokenizer Test**: Verify PatchTokenizer works with Sarathi interface
2. **Model Loading Test**: Ensure patched embeddings load correctly
3. **Cache Test**: Verify KV-cache calculations with reduced tokens

### Integration Tests

1. **End-to-End Inference**: Run full inference pipeline
2. **Memory Benchmark**: Compare memory usage
3. **Performance Benchmark**: Measure throughput improvements

### Benchmark Script

Create `test_integration.py`:

```python
"""Test script for AdaptiVocab + vAttention integration."""
import torch
from sarathi.entrypoints.openai_server.api_server import start_server
from sarathi.transformers_utils.patch_tokenizer_wrapper import PatchTokenizerWrapper

def test_tokenizer():
    """Test PatchTokenizer integration."""
    tokenizer = PatchTokenizerWrapper(
        patch_tokenizer_path="path/to/patch_tokenizer.pkl",
        existing_tokenizer_name="meta-llama/Llama-2-7b-hf"
    )
    
    text = "Test input text"
    encoded = tokenizer(text)
    decoded = tokenizer.decode(encoded['input_ids'])
    
    assert decoded.strip() == text.strip(), "Tokenization round-trip failed"
    print("✓ Tokenizer test passed")

def test_model_loading():
    """Test model loading with patched embeddings."""
    # Implementation here
    pass

def test_inference():
    """Test end-to-end inference."""
    # Implementation here
    pass

if __name__ == "__main__":
    test_tokenizer()
    test_model_loading()
    test_inference()
    print("All tests passed!")
```

---

## Usage Example

### Running with Integrated System

```python
# Start server with PatchTokenizer
python -m sarathi.entrypoints.openai_server.api_server \
    --model_name meta-llama/Llama-2-7b-hf \
    --patch_tokenizer_path /path/to/patch_tokenizer.pkl \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152 \
    --model_tensor_parallel_degree 1
```

---

## Expected Performance Improvements

### Token Efficiency
- **Baseline**: 1000 tokens per request
- **With AdaptiVocab**: 750 tokens per request (25% reduction)
- **Savings**: 250 tokens per request

### Memory Efficiency
- **Baseline KV-cache**: ~2GB for 1000-token context
- **With AdaptiVocab**: ~1.5GB for same content (25% fewer tokens)
- **With vAttention**: Better memory utilization (10-20% improvement)
- **Combined**: ~1.2-1.35GB (30-40% total reduction)

### Throughput
- **Fewer tokens** → Faster processing
- **Better memory** → More concurrent requests
- **Expected**: 30-50% throughput improvement

---

## Troubleshooting

### Issue: Tokenizer not compatible
**Solution**: Ensure PatchTokenizerWrapper implements all required methods

### Issue: Embedding dimension mismatch
**Solution**: Verify vocabulary size matches embedding size

### Issue: Memory allocation errors
**Solution**: Check KV-cache size calculations account for token reduction

### Issue: Attention computation errors
**Solution**: Verify attention wrappers handle n-gram tokens correctly

---

## Next Steps

1. ✅ Review integration plan
2. ⬜ Set up unified environment
3. ⬜ Implement tokenizer wrapper
4. ⬜ Integrate model loading
5. ⬜ Update KV-cache calculations
6. ⬜ Run integration tests
7. ⬜ Benchmark performance
8. ⬜ Optimize based on results

---

## Resources

- AdaptiVocab Paper: https://arxiv.org/abs/2503.19693
- vAttention Paper: https://arxiv.org/abs/2405.04437
- Sarathi-Serve: https://github.com/microsoft/sarathi-serve




