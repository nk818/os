#!/usr/bin/env python3
"""
Simple script to create a minimal PatchTokenizer for testing AdaptiVocab.
This creates a basic patch tokenizer that reduces vocabulary by removing rare tokens.
"""

import os
import sys
import pickle
from pathlib import Path

# Add AdaptiVocab to path
sys.path.insert(0, str(Path(__file__).parent.parent / "AdaptiVocab" / "src"))

try:
    from transformers import AutoTokenizer
    from build_vocab.patch_tokenizer import PatchTokenizer
except ImportError as e:
    print(f"❌ Failed to import AdaptiVocab: {e}")
    print("   Make sure AdaptiVocab is properly installed")
    sys.exit(1)


def create_simple_patch_tokenizer(
    model_name: str = "microsoft/phi-2",
    reduction_percent: float = 0.25,  # Remove 25% of vocabulary
    output_path: str = None
):
    """
    Create a simple PatchTokenizer by removing the least frequent tokens.
    
    Args:
        model_name: HuggingFace model name
        reduction_percent: Percentage of vocabulary to remove (0.0-1.0)
        output_path: Path to save the patch tokenizer
    """
    print(f"🔧 Creating PatchTokenizer for {model_name}")
    print(f"   Target reduction: {reduction_percent*100:.1f}%")
    
    # Load original tokenizer
    original_tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_vocab_size = len(original_tokenizer)
    target_vocab_size = int(original_vocab_size * (1 - reduction_percent))
    num_to_remove = original_vocab_size - target_vocab_size
    
    print(f"   Original vocab size: {original_vocab_size}")
    print(f"   Target vocab size: {target_vocab_size}")
    print(f"   Tokens to remove: {num_to_remove}")
    
    # Get vocabulary (excluding special tokens)
    vocab = original_tokenizer.get_vocab()
    special_tokens = {
        original_tokenizer.bos_token,
        original_tokenizer.eos_token,
        original_tokenizer.pad_token,
        original_tokenizer.unk_token,
        original_tokenizer.cls_token,
        original_tokenizer.sep_token,
    }
    special_tokens = {t for t in special_tokens if t is not None}
    
    # Remove special tokens from removal list
    removable_tokens = {token: token_id for token, token_id in vocab.items() 
                       if token not in special_tokens}
    
    # Sort by token ID (rough proxy for frequency in many tokenizers)
    # Lower IDs are often more frequent
    sorted_tokens = sorted(removable_tokens.items(), key=lambda x: x[1], reverse=True)
    
    # Take the highest IDs (likely less frequent) to remove
    tokens_to_remove = sorted_tokens[:num_to_remove]
    removed_tokens_dict = {token: [token_id] for token, token_id in tokens_to_remove}
    
    print(f"   ✅ Selected {len(removed_tokens_dict)} tokens to remove")
    
    # Create PatchTokenizer
    # Note: This is a simplified version - full AdaptiVocab would add ngrams
    patch_tokenizer = PatchTokenizer(
        existing_tokenizer_name=model_name,
        removed_tokens=removed_tokens_dict,
        ngram_dict={}  # No ngrams for simple version
    )
    
    # Create a simple ID converter
    # Map removed tokens to UNK token
    unk_token_id = original_tokenizer.unk_token_id
    id_converter = {}
    
    # Keep all non-removed tokens as-is
    kept_tokens = {token: token_id for token, token_id in vocab.items() 
                   if token not in removed_tokens_dict}
    
    new_id = 0
    for token, old_id in sorted(kept_tokens.items(), key=lambda x: x[1]):
        id_converter[token] = {
            'old_indices': old_id,
            'new_index': new_id
        }
        new_id += 1
    
    # Map removed tokens to UNK
    for token in removed_tokens_dict:
        id_converter[token] = {
            'old_indices': removed_tokens_dict[token][0],
            'new_index': unk_token_id if unk_token_id is not None else 0
        }
    
    patch_tokenizer.add_converter(id_converter)
    
    # Set output path
    if output_path is None:
        output_dir = Path(__file__).parent / "patch_tokenizers"
        output_dir.mkdir(exist_ok=True)
        model_short = model_name.split("/")[-1]
        output_path = output_dir / f"{model_short}_patch_tokenizer.pkl"
    
    # Save
    patch_tokenizer.save_model(str(output_path))
    
    new_vocab_size = len(patch_tokenizer.token_to_id_dict)
    actual_reduction = (1 - new_vocab_size / original_vocab_size) * 100
    
    print(f"\n✅ PatchTokenizer created successfully!")
    print(f"   Saved to: {output_path}")
    print(f"   Vocab reduction: {original_vocab_size} → {new_vocab_size} ({actual_reduction:.1f}%)")
    print(f"\n💡 Usage:")
    print(f"   python benchmark_comparison.py \\")
    print(f"       --model {model_name} \\")
    print(f"       --patch-tokenizer {output_path} \\")
    print(f"       --larosa-sparsity 0.4")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a simple PatchTokenizer for AdaptiVocab")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--reduction",
        type=float,
        default=0.25,
        help="Percentage of vocabulary to remove (0.0-1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for patch tokenizer"
    )
    
    args = parser.parse_args()
    
    create_simple_patch_tokenizer(
        model_name=args.model,
        reduction_percent=args.reduction,
        output_path=args.output
    )

