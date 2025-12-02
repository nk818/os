#!/usr/bin/env python3
"""
Example usage of the Standalone Fused LLM Chatbot
"""

from standalone_chat import FusedLLMEngine

def example_basic():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    engine = FusedLLMEngine(model_name="gpt2", device="cpu")
    
    prompt = "The future of artificial intelligence"
    response = engine.generate(prompt, max_new_tokens=50)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")


def example_chat():
    """Chat interface example."""
    print("=" * 60)
    print("Example 2: Chat Interface")
    print("=" * 60)
    
    engine = FusedLLMEngine(model_name="gpt2", device="cpu")
    conversation_history = []
    
    messages = [
        "Hello! How are you?",
        "Tell me about machine learning",
        "What are neural networks?",
    ]
    
    for message in messages:
        print(f"User: {message}")
        response = engine.chat(message, conversation_history)
        print(f"Assistant: {response}\n")


def example_adaptivocab():
    """AdaptiVocab example (requires PatchTokenizer)."""
    print("=" * 60)
    print("Example 3: With AdaptiVocab")
    print("=" * 60)
    
    # Replace with your actual PatchTokenizer path
    patch_tokenizer_path = None  # "path/to/patch_tokenizer.pkl"
    
    if patch_tokenizer_path:
        engine = FusedLLMEngine(
            model_name="gpt2",
            patch_tokenizer_path=patch_tokenizer_path,
            device="cpu",
        )
        
        prompt = "The future of artificial intelligence"
        response = engine.generate(prompt, max_new_tokens=50)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
    else:
        print("⚠️  PatchTokenizer path not set. Skipping AdaptiVocab example.\n")


def example_larosa():
    """LaRoSA example."""
    print("=" * 60)
    print("Example 4: With LaRoSA (40% sparsity)")
    print("=" * 60)
    
    engine = FusedLLMEngine(
        model_name="gpt2",
        larosa_sparsity=0.4,
        device="cpu",
    )
    
    prompt = "The future of artificial intelligence"
    response = engine.generate(prompt, max_new_tokens=50)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")


if __name__ == "__main__":
    print("\n🤖 Standalone Fused LLM Chatbot - Examples\n")
    
    try:
        example_basic()
        example_chat()
        example_adaptivocab()
        example_larosa()
        
        print("=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

