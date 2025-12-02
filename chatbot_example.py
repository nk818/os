#!/usr/bin/env python3
"""
Example: Using the Fused Chatbot Programmatically

This example shows how to use the FusedChatbot class programmatically
instead of using the interactive mode.
"""

from fused_chatbot import FusedChatbot, FusedChatbotConfig

def main():
    # Create configuration
    config = FusedChatbotConfig()
    
    # Configure optimizations
    config['model_name'] = "gpt2"
    config['larosa_sparsity'] = 0.4  # 40% sparsity
    
    # Try to auto-detect PatchTokenizer
    from fused_chatbot import find_patch_tokenizer
    patch_tokenizer = find_patch_tokenizer()
    if patch_tokenizer:
        config['patch_tokenizer_path'] = patch_tokenizer
        print(f"✅ Found PatchTokenizer: {patch_tokenizer}")
    else:
        print("⚠️  No PatchTokenizer found. AdaptiVocab will be disabled.")
    
    # Create chatbot
    chatbot = FusedChatbot(config)
    
    try:
        # Start chatbot
        if not chatbot.start():
            print("Failed to start chatbot")
            return
        
        # Example conversation
        questions = [
            "Hello! Can you tell me about the optimizations you're using?",
            "What is AdaptiVocab?",
            "How does vAttention work?",
        ]
        
        for question in questions:
            print(f"\n👤 User: {question}")
            response = chatbot.chat(question)
            print(f"🤖 Assistant: {response}")
        
        # Show status
        chatbot.show_status()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        chatbot.stop()

if __name__ == "__main__":
    main()



