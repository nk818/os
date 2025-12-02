#!/usr/bin/env python3
"""Quick test of SimpleAdaptiVocab"""

from transformers import AutoTokenizer
from simple_adaptivocab import SimpleAdaptiVocab

# Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Wrap with SimpleAdaptiVocab
adaptivocab = SimpleAdaptiVocab(tokenizer)

# Test phrases
test_texts = [
    "What is artificial intelligence?",
    "Explain machine learning in simple terms.",
    "How does a neural network work?",
    "What is the difference between AI and ML?",
    "Quantum mechanics is a fundamental theory in physics.",
    "Deep learning uses neural networks for pattern recognition.",
]

print("🧪 Testing SimpleAdaptiVocab\n")
print("="*60)

for text in test_texts:
    # Original
    original_tokens = tokenizer.encode(text)
    original_count = len(original_tokens)
    
    # With AdaptiVocab
    combined_tokens = adaptivocab.encode(text)
    combined_count = len(combined_tokens)
    
    # Decode back
    decoded = adaptivocab.decode(combined_tokens)
    
    # Show results
    reduction = original_count - combined_count
    reduction_pct = (reduction / original_count * 100) if original_count > 0 else 0
    
    print(f"\n📝 Text: {text}")
    print(f"   Original:  {original_count} tokens")
    print(f"   Combined:  {combined_count} tokens")
    print(f"   Saved:     {reduction} tokens ({reduction_pct:.1f}%)")
    print(f"   Decoded:   {decoded}")

print("\n" + "="*60)
print("✅ SimpleAdaptiVocab test complete!")

