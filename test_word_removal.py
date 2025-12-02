#!/usr/bin/env python3
"""
Test if removing every 3rd word maintains conceptual soundness
and if the compressed text can still be accurately summarized.
"""

import re
from fused_chatbot_enhanced import EnhancedFusedLLMEngine

# The quantum mechanics text
QUANTUM_MECHANICS_TEXT = """Quantum mechanics is the foundation of modern physics, shaping how scientists understand particles and energy. Unlike classical physics, quantum mechanics explains phenomena that occur on the smallest scales, such as electrons, photons, and atomic transitions. Many technologies rely on quantum mechanics, including lasers, semiconductors, and MRI machines. Because quantum mechanics describes probability rather than certainty, it challenges our intuition and reshapes how we think about reality. Researchers continue to explore quantum mechanics to develop quantum computers, which use the principles of quantum mechanics to perform calculations far beyond classical capabilities. As our understanding of quantum mechanics deepens, new discoveries show that quantum mechanics is not just abstract theory but a powerful framework for explaining the universe."""

def remove_every_nth_word(text: str, n: int = 3) -> str:
    """
    Remove every nth word from text.
    
    Args:
        text: Input text
        n: Remove every nth word (default: 3, so removes 3rd, 6th, 9th, etc.)
    
    Returns:
        Compressed text with every nth word removed
    """
    # Split into words, preserving punctuation
    words = re.findall(r'\S+', text)
    
    # Remove every nth word (1-indexed, so 3rd, 6th, 9th...)
    # Keep words at positions 0, 1, 3, 4, 6, 7, 9, 10... (skip 2, 5, 8...)
    kept_words = []
    for i, word in enumerate(words):
        if (i + 1) % n != 0:  # Keep if not the nth word
            kept_words.append(word)
        else:
            # Optionally mark removed words
            pass
    
    # Reconstruct text with spaces
    compressed = ' '.join(kept_words)
    
    # Try to preserve some punctuation spacing
    compressed = re.sub(r'\s+([,.!?;:])', r'\1', compressed)
    compressed = re.sub(r'\s+', ' ', compressed)
    
    return compressed

def remove_every_nth_word_advanced(text: str, n: int = 3, preserve_structure: bool = True) -> str:
    """
    Advanced word removal that tries to preserve sentence structure.
    
    Args:
        text: Input text
        n: Remove every nth word
        preserve_structure: Try to maintain sentence boundaries
    
    Returns:
        Compressed text
    """
    # Split by sentences first
    sentences = re.split(r'([.!?]\s+)', text)
    
    compressed_sentences = []
    for sentence in sentences:
        if sentence.strip() and sentence.strip() not in ['.', '!', '?']:
            # Process this sentence
            words = re.findall(r'\S+', sentence)
            kept_words = [word for i, word in enumerate(words) if (i + 1) % n != 0]
            compressed = ' '.join(kept_words)
            compressed_sentences.append(compressed)
        else:
            # Keep punctuation
            compressed_sentences.append(sentence)
    
    return ''.join(compressed_sentences)

def count_words(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r'\S+', text))

def main():
    print("🧪 Testing Word Removal for Text Compression")
    print("="*70)
    
    # Original text stats
    original_words = count_words(QUANTUM_MECHANICS_TEXT)
    original_chars = len(QUANTUM_MECHANICS_TEXT)
    
    print(f"\n📝 ORIGINAL TEXT:")
    print(f"   Words: {original_words}")
    print(f"   Characters: {original_chars}")
    print(f"\n{QUANTUM_MECHANICS_TEXT[:200]}...\n")
    
    # Test different removal patterns
    removal_patterns = [3, 4, 5]
    
    compressed_texts = {}
    
    for n in removal_patterns:
        compressed = remove_every_nth_word_advanced(QUANTUM_MECHANICS_TEXT, n)
        compressed_words = count_words(compressed)
        compressed_chars = len(compressed)
        reduction = (1 - compressed_words / original_words) * 100
        
        compressed_texts[n] = compressed
        
        print(f"\n{'─'*70}")
        print(f"📝 COMPRESSED TEXT (removing every {n}th word):")
        print(f"   Words: {compressed_words} (reduced by {reduction:.1f}%)")
        print(f"   Characters: {compressed_chars} (reduced by {(1-compressed_chars/original_chars)*100:.1f}%)")
        print(f"\n{compressed[:200]}...\n")
    
    # Initialize LLM engine
    print("\n" + "="*70)
    print("🤖 Testing Summarization with LLM")
    print("="*70)
    
    engine = EnhancedFusedLLMEngine(
        model_name="microsoft/phi-2",
        simple_adaptivocab=False,
        larosa_sparsity=0.0,
        vattention_enabled=False,
        device="cpu",
    )
    
    # Summarize original
    print("\n📊 Summarizing ORIGINAL text...")
    original_prompt = f"Please summarize the following text in 2-3 sentences:\n\n{QUANTUM_MECHANICS_TEXT}\n\nSummary:"
    original_summary = engine.generate(original_prompt, max_new_tokens=100, temperature=0.7)
    original_tokens = len(engine.tokenizer.encode(original_prompt))
    
    print(f"   Summary: {original_summary}")
    print(f"   Input tokens: {original_tokens}")
    
    # Summarize each compressed version
    summaries = {}
    for n, compressed in compressed_texts.items():
        print(f"\n📊 Summarizing COMPRESSED text (every {n}th word removed)...")
        compressed_prompt = f"Please summarize the following text in 2-3 sentences:\n\n{compressed}\n\nSummary:"
        compressed_summary = engine.generate(compressed_prompt, max_new_tokens=100, temperature=0.7)
        compressed_tokens = len(engine.tokenizer.encode(compressed_prompt))
        
        summaries[n] = compressed_summary
        
        print(f"   Summary: {compressed_summary}")
        print(f"   Input tokens: {compressed_tokens} (saved {original_tokens - compressed_tokens} tokens)")
    
    # Compare summaries
    print("\n" + "="*70)
    print("📊 SUMMARY COMPARISON")
    print("="*70)
    
    print("\n📝 ORIGINAL TEXT SUMMARY:")
    print(f"   {original_summary}")
    print(f"   Input tokens: {original_tokens}")
    
    for n in removal_patterns:
        compressed_words = count_words(compressed_texts[n])
        reduction = (1 - compressed_words / original_words) * 100
        compressed_tokens = len(engine.tokenizer.encode(f"Please summarize the following text in 2-3 sentences:\n\n{compressed_texts[n]}\n\nSummary:"))
        token_savings = original_tokens - compressed_tokens
        
        print(f"\n📝 COMPRESSED TEXT SUMMARY (every {n}th word removed, {reduction:.1f}% reduction):")
        print(f"   {summaries[n]}")
        print(f"   Input tokens: {compressed_tokens} (saved {token_savings} tokens, {token_savings/original_tokens*100:.1f}%)")
    
    # Quality assessment
    print("\n" + "="*70)
    print("🎯 QUALITY ASSESSMENT")
    print("="*70)
    print("\nQuestions to consider:")
    print("   • Do the compressed summaries capture the same concepts?")
    print("   • Is meaning preserved despite word removal?")
    print("   • How much compression is possible before losing meaning?")
    print("   • Is this a viable optimization technique?")
    
    # Save results
    output_file = "word_removal_test.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("WORD REMOVAL TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("ORIGINAL TEXT:\n")
        f.write("-"*70 + "\n")
        f.write(QUANTUM_MECHANICS_TEXT + "\n\n")
        f.write(f"Words: {original_words}, Characters: {original_chars}\n\n")
        
        f.write("ORIGINAL SUMMARY:\n")
        f.write("-"*70 + "\n")
        f.write(original_summary + "\n\n")
        f.write(f"Input tokens: {original_tokens}\n\n")
        
        for n in removal_patterns:
            compressed = compressed_texts[n]
            compressed_words = count_words(compressed)
            reduction = (1 - compressed_words / original_words) * 100
            compressed_tokens = len(engine.tokenizer.encode(f"Please summarize the following text in 2-3 sentences:\n\n{compressed}\n\nSummary:"))
            
            f.write("="*70 + "\n")
            f.write(f"COMPRESSED TEXT (every {n}th word removed, {reduction:.1f}% reduction)\n")
            f.write("="*70 + "\n")
            f.write(compressed + "\n\n")
            f.write(f"Words: {compressed_words}, Characters: {len(compressed)}\n")
            f.write(f"Input tokens: {compressed_tokens} (saved {original_tokens - compressed_tokens} tokens)\n\n")
            f.write("SUMMARY:\n")
            f.write("-"*70 + "\n")
            f.write(summaries[n] + "\n\n")
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()

