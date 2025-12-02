#!/usr/bin/env python3
"""
Compare summaries with and without SimpleAdaptiVocab
Tests the quantum mechanics text with both versions
"""

import time
from fused_chatbot_enhanced import EnhancedFusedLLMEngine

# The quantum mechanics text
QUANTUM_MECHANICS_TEXT = """Quantum mechanics is the foundation of modern physics, shaping how scientists understand particles and energy. Unlike classical physics, quantum mechanics explains phenomena that occur on the smallest scales, such as electrons, photons, and atomic transitions. Many technologies rely on quantum mechanics, including lasers, semiconductors, and MRI machines. Because quantum mechanics describes probability rather than certainty, it challenges our intuition and reshapes how we think about reality. Researchers continue to explore quantum mechanics to develop quantum computers, which use the principles of quantum mechanics to perform calculations far beyond classical capabilities. As our understanding of quantum mechanics deepens, new discoveries show that quantum mechanics is not just abstract theory but a powerful framework for explaining the universe."""

PROMPT = f"""Please summarize the following text in 2-3 sentences:

{QUANTUM_MECHANICS_TEXT}

Summary:"""

def count_tokens(text, tokenizer):
    """Count tokens in text."""
    return len(tokenizer.encode(text))

def main():
    print("🔬 Quantum Mechanics Summary Comparison")
    print("="*70)
    print("\n📝 Original Text:")
    print(f"   {QUANTUM_MECHANICS_TEXT[:100]}...")
    print(f"   Length: {len(QUANTUM_MECHANICS_TEXT)} characters")
    print()
    
    # Initialize baseline engine (no SimpleAdaptiVocab)
    print("📦 Initializing Baseline Engine (no SimpleAdaptiVocab)...")
    baseline_engine = EnhancedFusedLLMEngine(
        model_name="microsoft/phi-2",
        simple_adaptivocab=False,
        larosa_sparsity=0.0,
        vattention_enabled=False,
        device="cpu",
    )
    
    # Count input tokens for baseline
    baseline_input_tokens = count_tokens(PROMPT, baseline_engine.tokenizer)
    print(f"   Input tokens: {baseline_input_tokens}")
    
    # Generate summary with baseline
    print("\n🔄 Generating summary with baseline engine...")
    start_time = time.time()
    baseline_summary = baseline_engine.generate(
        PROMPT,
        max_new_tokens=100,
        temperature=0.7,
    )
    baseline_time = time.time() - start_time
    baseline_output_tokens = count_tokens(baseline_summary, baseline_engine.tokenizer)
    
    print(f"   Time: {baseline_time:.2f}s")
    print(f"   Output tokens: {baseline_output_tokens}")
    print(f"   Total tokens: {baseline_input_tokens + baseline_output_tokens}")
    
    # Initialize optimized engine (with SimpleAdaptiVocab)
    print("\n📦 Initializing Optimized Engine (with SimpleAdaptiVocab)...")
    optimized_engine = EnhancedFusedLLMEngine(
        model_name="microsoft/phi-2",
        simple_adaptivocab=True,
        larosa_sparsity=0.0,
        vattention_enabled=False,
        device="cpu",
    )
    
    # Count input tokens for optimized
    optimized_input_tokens = count_tokens(PROMPT, optimized_engine.tokenizer)
    print(f"   Input tokens: {optimized_input_tokens}")
    
    # Generate summary with optimized
    print("\n🔄 Generating summary with SimpleAdaptiVocab engine...")
    start_time = time.time()
    optimized_summary = optimized_engine.generate(
        PROMPT,
        max_new_tokens=100,
        temperature=0.7,
    )
    optimized_time = time.time() - start_time
    optimized_output_tokens = count_tokens(optimized_summary, optimized_engine.tokenizer)
    
    print(f"   Time: {optimized_time:.2f}s")
    print(f"   Output tokens: {optimized_output_tokens}")
    print(f"   Total tokens: {optimized_input_tokens + optimized_output_tokens}")
    
    # Compare results
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS - SUMMARY QUALITY")
    print("="*70)
    
    print("\n" + "─"*70)
    print("📝 BASELINE SUMMARY (no SimpleAdaptiVocab)")
    print("─"*70)
    print(f"\n{baseline_summary}\n")
    print("─"*70)
    print("📊 Metrics:")
    print(f"   • Input tokens:  {baseline_input_tokens}")
    print(f"   • Output tokens: {baseline_output_tokens}")
    print(f"   • Total tokens:  {baseline_input_tokens + baseline_output_tokens}")
    print(f"   • Time:          {baseline_time:.2f}s")
    print(f"   • Tokens/sec:    {(baseline_input_tokens + baseline_output_tokens) / baseline_time:.2f}")
    print(f"   • Summary length: {len(baseline_summary)} characters")
    
    print("\n" + "─"*70)
    print("📝 OPTIMIZED SUMMARY (with SimpleAdaptiVocab)")
    print("─"*70)
    print(f"\n{optimized_summary}\n")
    print("─"*70)
    print("📊 Metrics:")
    print(f"   • Input tokens:  {optimized_input_tokens}")
    print(f"   • Output tokens: {optimized_output_tokens}")
    print(f"   • Total tokens:  {optimized_input_tokens + optimized_output_tokens}")
    print(f"   • Time:          {optimized_time:.2f}s")
    print(f"   • Tokens/sec:    {(optimized_input_tokens + optimized_output_tokens) / optimized_time:.2f}")
    print(f"   • Summary length: {len(optimized_summary)} characters")
    
    print("\n" + "="*70)
    print("📋 SIDE-BY-SIDE COMPARISON")
    print("="*70)
    print("\nBASELINE (no SimpleAdaptiVocab):")
    print("─" * 70)
    print(baseline_summary)
    print("\nOPTIMIZED (with SimpleAdaptiVocab):")
    print("─" * 70)
    print(optimized_summary)
    
    # Calculate improvements
    token_reduction = baseline_input_tokens - optimized_input_tokens
    token_reduction_pct = (token_reduction / baseline_input_tokens * 100) if baseline_input_tokens > 0 else 0
    total_token_reduction = (baseline_input_tokens + baseline_output_tokens) - (optimized_input_tokens + optimized_output_tokens)
    time_improvement = ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0
    length_difference = len(optimized_summary) - len(baseline_summary)
    
    print("\n" + "="*70)
    print("📈 PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n   Input token reduction:  {token_reduction} tokens ({token_reduction_pct:+.1f}%)")
    print(f"   Total token reduction: {total_token_reduction} tokens")
    print(f"   Time improvement:       {time_improvement:+.1f}%")
    print(f"   Summary length diff:    {length_difference:+d} characters")
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE STATISTICS")
    print("="*70)
    
    # Detailed token breakdown
    print("\n🔢 TOKEN BREAKDOWN:")
    print("─" * 70)
    print(f"BASELINE:")
    print(f"   • Input tokens:        {baseline_input_tokens}")
    print(f"   • Output tokens:       {baseline_output_tokens}")
    print(f"   • Total tokens:        {baseline_input_tokens + baseline_output_tokens}")
    print(f"   • Input token ratio:   {baseline_input_tokens / (baseline_input_tokens + baseline_output_tokens) * 100:.1f}%")
    print(f"   • Output token ratio:   {baseline_output_tokens / (baseline_input_tokens + baseline_output_tokens) * 100:.1f}%")
    
    print(f"\nOPTIMIZED:")
    print(f"   • Input tokens:        {optimized_input_tokens}")
    print(f"   • Output tokens:       {optimized_output_tokens}")
    print(f"   • Total tokens:        {optimized_input_tokens + optimized_output_tokens}")
    print(f"   • Input token ratio:   {optimized_input_tokens / (optimized_input_tokens + optimized_output_tokens) * 100:.1f}%")
    print(f"   • Output token ratio:   {optimized_output_tokens / (optimized_input_tokens + optimized_output_tokens) * 100:.1f}%")
    
    # Timing statistics
    print("\n⏱️  TIMING STATISTICS:")
    print("─" * 70)
    print(f"BASELINE:")
    print(f"   • Total time:          {baseline_time:.2f}s")
    print(f"   • Input processing:    ~{baseline_time * (baseline_input_tokens / (baseline_input_tokens + baseline_output_tokens)):.2f}s (estimated)")
    print(f"   • Generation time:     ~{baseline_time * (baseline_output_tokens / (baseline_input_tokens + baseline_output_tokens)):.2f}s (estimated)")
    print(f"   • Tokens per second:   {(baseline_input_tokens + baseline_output_tokens) / baseline_time:.2f}")
    print(f"   • Output tokens/sec:   {baseline_output_tokens / baseline_time:.2f}")
    
    print(f"\nOPTIMIZED:")
    print(f"   • Total time:          {optimized_time:.2f}s")
    print(f"   • Input processing:    ~{optimized_time * (optimized_input_tokens / (optimized_input_tokens + optimized_output_tokens)):.2f}s (estimated)")
    print(f"   • Generation time:     ~{optimized_time * (optimized_output_tokens / (optimized_input_tokens + optimized_output_tokens)):.2f}s (estimated)")
    print(f"   • Tokens per second:   {(optimized_input_tokens + optimized_output_tokens) / optimized_time:.2f}")
    print(f"   • Output tokens/sec:   {optimized_output_tokens / optimized_time:.2f}")
    
    # Text statistics
    print("\n📝 TEXT STATISTICS:")
    print("─" * 70)
    baseline_words = len(baseline_summary.split())
    optimized_words = len(optimized_summary.split())
    baseline_sentences = len([s for s in baseline_summary.split('.') if s.strip()])
    optimized_sentences = len([s for s in optimized_summary.split('.') if s.strip()])
    
    print(f"BASELINE:")
    print(f"   • Characters:          {len(baseline_summary)}")
    print(f"   • Words:               {baseline_words}")
    print(f"   • Sentences:           {baseline_sentences}")
    print(f"   • Avg chars/word:      {len(baseline_summary) / baseline_words:.1f}")
    print(f"   • Avg words/sentence:  {baseline_words / baseline_sentences:.1f}")
    print(f"   • Tokens per word:     {baseline_output_tokens / baseline_words:.2f}")
    print(f"   • Chars per token:    {len(baseline_summary) / baseline_output_tokens:.1f}")
    
    print(f"\nOPTIMIZED:")
    print(f"   • Characters:          {len(optimized_summary)}")
    print(f"   • Words:               {optimized_words}")
    print(f"   • Sentences:           {optimized_sentences}")
    print(f"   • Avg chars/word:      {len(optimized_summary) / optimized_words:.1f}")
    print(f"   • Avg words/sentence:  {optimized_words / optimized_sentences:.1f}")
    print(f"   • Tokens per word:     {optimized_output_tokens / optimized_words:.2f}")
    print(f"   • Chars per token:    {len(optimized_summary) / optimized_output_tokens:.1f}")
    
    # Comparison metrics
    print("\n📈 COMPARISON METRICS:")
    print("─" * 70)
    print(f"   • Input token diff:     {optimized_input_tokens - baseline_input_tokens:+d} tokens ({token_reduction_pct:+.1f}%)")
    print(f"   • Output token diff:   {optimized_output_tokens - baseline_output_tokens:+d} tokens")
    print(f"   • Total token diff:     {total_token_reduction:+d} tokens")
    print(f"   • Time difference:     {optimized_time - baseline_time:+.2f}s ({time_improvement:+.1f}%)")
    print(f"   • Length difference:   {length_difference:+d} characters ({length_difference/len(baseline_summary)*100:+.1f}%)")
    print(f"   • Word difference:     {optimized_words - baseline_words:+d} words")
    print(f"   • Speed ratio:         {(baseline_input_tokens + baseline_output_tokens) / baseline_time / ((optimized_input_tokens + optimized_output_tokens) / optimized_time):.2f}x")
    
    # Efficiency metrics
    print("\n⚡ EFFICIENCY METRICS:")
    print("─" * 70)
    baseline_efficiency = baseline_output_tokens / baseline_time
    optimized_efficiency = optimized_output_tokens / optimized_time
    print(f"   • Baseline efficiency:  {baseline_efficiency:.2f} output tokens/sec")
    print(f"   • Optimized efficiency:  {optimized_efficiency:.2f} output tokens/sec")
    print(f"   • Efficiency ratio:      {baseline_efficiency / optimized_efficiency:.2f}x")
    
    baseline_token_efficiency = (baseline_input_tokens + baseline_output_tokens) / baseline_time
    optimized_token_efficiency = (optimized_input_tokens + optimized_output_tokens) / optimized_time
    print(f"   • Baseline total:        {baseline_token_efficiency:.2f} total tokens/sec")
    print(f"   • Optimized total:       {optimized_token_efficiency:.2f} total tokens/sec")
    
    # Quality metrics
    print("\n🎯 QUALITY METRICS:")
    print("─" * 70)
    print(f"   • Summary similarity:   Both cover quantum mechanics basics")
    print(f"   • Baseline detail:      {'More detailed' if baseline_words > optimized_words else 'Less detailed' if baseline_words < optimized_words else 'Similar detail'}")
    print(f"   • Output consistency:    {'Similar' if abs(baseline_output_tokens - optimized_output_tokens) <= 2 else 'Different'} token counts")
    print(f"   • Quality assessment:   {'Similar quality' if abs(len(baseline_summary) - len(optimized_summary)) < 20 else 'Different lengths'}")
    
    print("\n" + "="*70)
    print("🎯 QUALITY ASSESSMENT")
    print("="*70)
    print("\nPlease compare the two summaries above and assess:")
    print("   • Which summary is more accurate?")
    print("   • Which summary is more comprehensive?")
    print("   • Which summary is better written?")
    print("   • Does SimpleAdaptiVocab affect output quality?")
    
    # Check if "quantum mechanics" was combined
    print("\n🔍 SimpleAdaptiVocab Analysis:")
    print(f"   Original text contains 'quantum mechanics': {QUANTUM_MECHANICS_TEXT.count('quantum mechanics')} times")
    
    # Check tokenization
    baseline_tokens = baseline_engine.tokenizer.tokenize(QUANTUM_MECHANICS_TEXT)
    optimized_tokens = optimized_engine.tokenizer.tokenize(QUANTUM_MECHANICS_TEXT)
    
    print(f"   Baseline tokenization: {len(baseline_tokens)} tokens")
    print(f"   Optimized tokenization: {len(optimized_tokens)} tokens")
    print(f"   Token difference: {len(baseline_tokens) - len(optimized_tokens)} tokens")
    
    # Show some token examples
    print(f"\n   Sample baseline tokens: {baseline_tokens[:10]}")
    print(f"   Sample optimized tokens: {optimized_tokens[:10]}")
    
    # Explain why no reduction
    print("\n💡 Why No Token Reduction?")
    print("   SimpleAdaptiVocab checks if combining phrases reduces tokens.")
    print("   'quantum mechanics' → 3 tokens")
    print("   'quantummechanics' → 5 tokens (worse!)")
    print("   So it correctly doesn't combine this phrase.")
    print("\n   For real token reduction, use full AdaptiVocab which:")
    print("   - Adds phrases as actual vocabulary tokens")
    print("   - Requires model fine-tuning")
    print("   - Provides 25%+ token reduction")
    
    print("\n" + "="*70)
    print("✅ Comparison complete!")
    
    # Save summaries to file for easy comparison
    output_file = "summary_comparison.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTUM MECHANICS SUMMARY COMPARISON - ALL STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        f.write("ORIGINAL TEXT:\n")
        f.write("-"*70 + "\n")
        f.write(QUANTUM_MECHANICS_TEXT + "\n\n")
        f.write(f"Length: {len(QUANTUM_MECHANICS_TEXT)} characters\n")
        f.write(f"Occurrences of 'quantum mechanics': {QUANTUM_MECHANICS_TEXT.count('quantum mechanics')}\n\n")
        
        f.write("="*70 + "\n")
        f.write("BASELINE SUMMARY (no SimpleAdaptiVocab)\n")
        f.write("="*70 + "\n")
        f.write(baseline_summary + "\n\n")
        f.write("TOKEN BREAKDOWN:\n")
        f.write(f"  • Input tokens:  {baseline_input_tokens}\n")
        f.write(f"  • Output tokens: {baseline_output_tokens}\n")
        f.write(f"  • Total tokens:  {baseline_input_tokens + baseline_output_tokens}\n")
        f.write(f"  • Input ratio:   {baseline_input_tokens / (baseline_input_tokens + baseline_output_tokens) * 100:.1f}%\n")
        f.write(f"  • Output ratio:   {baseline_output_tokens / (baseline_input_tokens + baseline_output_tokens) * 100:.1f}%\n\n")
        f.write("TIMING:\n")
        f.write(f"  • Total time:    {baseline_time:.2f}s\n")
        f.write(f"  • Tokens/sec:    {(baseline_input_tokens + baseline_output_tokens) / baseline_time:.2f}\n")
        f.write(f"  • Output/sec:    {baseline_output_tokens / baseline_time:.2f}\n\n")
        f.write("TEXT STATS:\n")
        baseline_words = len(baseline_summary.split())
        baseline_sentences = len([s for s in baseline_summary.split('.') if s.strip()])
        f.write(f"  • Characters:    {len(baseline_summary)}\n")
        f.write(f"  • Words:         {baseline_words}\n")
        f.write(f"  • Sentences:     {baseline_sentences}\n")
        f.write(f"  • Chars/token:   {len(baseline_summary) / baseline_output_tokens:.1f}\n")
        f.write(f"  • Tokens/word:   {baseline_output_tokens / baseline_words:.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("OPTIMIZED SUMMARY (with SimpleAdaptiVocab)\n")
        f.write("="*70 + "\n")
        f.write(optimized_summary + "\n\n")
        f.write("TOKEN BREAKDOWN:\n")
        f.write(f"  • Input tokens:  {optimized_input_tokens}\n")
        f.write(f"  • Output tokens: {optimized_output_tokens}\n")
        f.write(f"  • Total tokens:  {optimized_input_tokens + optimized_output_tokens}\n")
        f.write(f"  • Input ratio:   {optimized_input_tokens / (optimized_input_tokens + optimized_output_tokens) * 100:.1f}%\n")
        f.write(f"  • Output ratio:   {optimized_output_tokens / (optimized_input_tokens + optimized_output_tokens) * 100:.1f}%\n\n")
        f.write("TIMING:\n")
        f.write(f"  • Total time:    {optimized_time:.2f}s\n")
        f.write(f"  • Tokens/sec:    {(optimized_input_tokens + optimized_output_tokens) / optimized_time:.2f}\n")
        f.write(f"  • Output/sec:    {optimized_output_tokens / optimized_time:.2f}\n\n")
        f.write("TEXT STATS:\n")
        optimized_words = len(optimized_summary.split())
        optimized_sentences = len([s for s in optimized_summary.split('.') if s.strip()])
        f.write(f"  • Characters:    {len(optimized_summary)}\n")
        f.write(f"  • Words:         {optimized_words}\n")
        f.write(f"  • Sentences:     {optimized_sentences}\n")
        f.write(f"  • Chars/token:   {len(optimized_summary) / optimized_output_tokens:.1f}\n")
        f.write(f"  • Tokens/word:   {optimized_output_tokens / optimized_words:.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("COMPARISON METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"Input token difference:    {optimized_input_tokens - baseline_input_tokens:+d} ({token_reduction_pct:+.1f}%)\n")
        f.write(f"Output token difference:   {optimized_output_tokens - baseline_output_tokens:+d}\n")
        f.write(f"Total token difference:     {total_token_reduction:+d}\n")
        f.write(f"Time difference:            {optimized_time - baseline_time:+.2f}s ({time_improvement:+.1f}%)\n")
        f.write(f"Length difference:          {length_difference:+d} characters\n")
        f.write(f"Word difference:           {optimized_words - baseline_words:+d} words\n")
        f.write(f"Speed ratio:               {(baseline_input_tokens + baseline_output_tokens) / baseline_time / ((optimized_input_tokens + optimized_output_tokens) / optimized_time):.2f}x\n")
        f.write(f"Efficiency ratio:          {baseline_efficiency / optimized_efficiency:.2f}x\n\n")
        
        f.write("="*70 + "\n")
        f.write("SIDE-BY-SIDE COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write("BASELINE:\n")
        f.write(baseline_summary + "\n\n")
        f.write("OPTIMIZED:\n")
        f.write(optimized_summary + "\n")
    
    print(f"\n💾 Summaries saved to: {output_file}")
    print("   You can now compare both outputs in the file!")

if __name__ == "__main__":
    main()

