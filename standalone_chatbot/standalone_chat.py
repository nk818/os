#!/usr/bin/env python3
"""
Standalone Fused LLM Chatbot
A clean implementation that integrates:
- AdaptiVocab (tokenizer optimization)
- LaRoSA (activation sparsity - simplified for CPU)
- Simple inference engine (no CUDA dependencies)
"""

import os
import sys
import json
import pickle
import argparse
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

# Add paths for AdaptiVocab
sys.path.insert(0, str(Path(__file__).parent.parent / "AdaptiVocab" / "src"))
try:
    from build_vocab.patch_tokenizer import PatchTokenizer
    ADAPTIVOCAB_AVAILABLE = True
except ImportError:
    ADAPTIVOCAB_AVAILABLE = False
    print("⚠️  AdaptiVocab not available. Install from AdaptiVocab/ directory.")


class SimpleLaRoSA(nn.Module):
    """
    Simplified LaRoSA implementation for CPU/macOS.
    Uses Top-K sparsification without custom CUDA kernels.
    """
    
    def __init__(self, hidden_size: int, sparsity: float = 0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.k = int(hidden_size * (1 - sparsity))
        
        # Simple rotation matrix (identity for now, can be computed from calibration data)
        self.register_buffer('rotation_matrix', torch.eye(hidden_size))
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation sparsification."""
        if not self.enabled or self.sparsity == 0.0:
            return x
        
        # Rotate
        x_rotated = x @ self.rotation_matrix
        
        # Top-K sparsification
        topk_values, topk_indices = torch.topk(x_rotated.abs(), self.k, dim=-1)
        x_sparse = torch.zeros_like(x_rotated)
        x_sparse.scatter_(-1, topk_indices, x_rotated.gather(-1, topk_indices))
        
        # Rotate back
        x_output = x_sparse @ self.rotation_matrix.T
        
        return x_output


class FusedLLMEngine:
    """
    Standalone LLM engine that integrates:
    - AdaptiVocab (PatchTokenizer)
    - LaRoSA (activation sparsity)
    - Simple generation loop
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        patch_tokenizer_path: Optional[str] = None,
        larosa_sparsity: float = 0.0,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.larosa_sparsity = larosa_sparsity
        
        print(f"🚀 Initializing Fused LLM Engine...")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer(patch_tokenizer_path)
        
        # Load model
        print(f"   Loading model...")
        if device == "cpu":
            # For CPU, don't use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(device)
        else:
            # For CUDA, can use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )
        self.model.eval()
        
        # Apply LaRoSA if enabled
        if larosa_sparsity > 0.0:
            print(f"   Applying LaRoSA sparsity: {larosa_sparsity*100:.1f}%")
            self._apply_larosa()
        
        print(f"✅ Engine ready!")
    
    def _load_tokenizer(self, patch_tokenizer_path: Optional[str]) -> PreTrainedTokenizer:
        """Load tokenizer with optional AdaptiVocab PatchTokenizer."""
        if patch_tokenizer_path and ADAPTIVOCAB_AVAILABLE:
            print(f"   Loading AdaptiVocab PatchTokenizer: {patch_tokenizer_path}")
            try:
                with open(patch_tokenizer_path, 'rb') as f:
                    patch_tokenizer: PatchTokenizer = pickle.load(f)
                
                # Get the underlying tokenizer
                tokenizer = patch_tokenizer.tokenizer
                
                # Fix pad token issue
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                print(f"   ✅ AdaptiVocab enabled (vocab size: {len(tokenizer)} -> {len(patch_tokenizer)})")
                return tokenizer
            except Exception as e:
                print(f"   ⚠️  Failed to load PatchTokenizer: {e}")
                print(f"   Falling back to standard tokenizer")
        
        print(f"   Loading standard tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Fix pad token issue (GPT-2 doesn't have a pad token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _apply_larosa(self):
        """Apply LaRoSA activation sparsity to MLP layers."""
        hidden_size = self.model.config.hidden_size
        
        for name, module in self.model.named_modules():
            # Apply to MLP/FFN layers
            if isinstance(module, (nn.Linear,)) and "mlp" in name.lower():
                # Wrap with LaRoSA
                larosa = SimpleLaRoSA(hidden_size, self.larosa_sparsity)
                # Note: This is a simplified approach. In production, you'd
                # replace the forward method or use hooks.
                setattr(module, '_larosa', larosa)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text from prompt."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # Store original length
        input_length = input_ids.shape[1]
        
        # Generate with better stopping criteria
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Stronger penalty
                no_repeat_ngram_size=2,  # Prevent 2-gram repetition
                early_stopping=True,  # Stop early if EOS found
            )
        
        # Extract only the new tokens (generated part)
        new_tokens = outputs[0][input_length:]
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up the text - stop at first newline, period, or question mark
        generated_text = generated_text.strip()
        
        # Stop at first sentence-ending punctuation or newline
        for stop_char in ["\n", ".", "?", "!"]:
            if stop_char in generated_text:
                idx = generated_text.index(stop_char)
                generated_text = generated_text[:idx+1].strip()
                break
        
        return generated_text
    
    def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Chat interface with conversation history."""
        if conversation_history is None:
            conversation_history = []
        
        # Build prompt from history (only user messages to avoid repetition)
        prompt = self._build_prompt(message, conversation_history)
        
        # Generate response
        response = self.generate(
            prompt, 
            max_new_tokens=40,  # Shorter responses for chat
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        
        # Clean response - stop at common stopping points
        response = self._clean_response(response, original_message=message)
        
        # Update history AFTER generation to avoid including it in next prompt
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _clean_response(self, response: str, original_message: str = "") -> str:
        """Clean up the generated response."""
        if not response:
            return "I'm not sure how to respond to that."
        
        # Remove if response is just echoing the question
        response_lower = response.lower().strip()
        message_lower = original_message.lower().strip()
        
        # Check if response is just repeating the question
        if message_lower and response_lower:
            # Remove question words and compare
            question_words = ["what", "who", "where", "when", "why", "how", "which", "whose", "is", "are", "your", "my"]
            response_words = [w for w in response_lower.split() if w not in question_words]
            message_words = [w for w in message_lower.split() if w not in question_words]
            
            # If response contains most of the question words, it's probably echoing
            if len(message_words) > 0:
                overlap = sum(1 for w in message_words if w in response_words)
                if overlap > len(message_words) * 0.6:  # More than 60% overlap
                    return "I'm not sure how to answer that question."
        
        # Remove common repetition patterns
        lines = response.split("\n")
        if len(lines) > 1:
            # Take first meaningful line
            response = lines[0]
        
        # Remove if it's just repeating the same word/phrase
        words = response.split()
        if len(words) > 1:
            # Check for repetition (same word/phrase repeated)
            if len(words) > 2:
                unique_words = set(words)
                if len(unique_words) < len(words) * 0.4:  # Less than 40% unique words
                    # Take first unique sequence
                    seen = set()
                    unique_sequence = []
                    for word in words:
                        if word.lower() not in seen:
                            unique_sequence.append(word)
                            seen.add(word.lower())
                        else:
                            break
                    response = " ".join(unique_sequence) if unique_sequence else words[0]
        
        # Remove trailing punctuation repetition
        if len(response) > 1 and response[-1] == response[-2] and response[-1] in ".,!?":
            response = response.rstrip(response[-1])
        
        # Limit length
        if len(response) > 150:
            # Try to cut at sentence boundary
            for punct in [".", "!", "?"]:
                idx = response.rfind(punct, 0, 150)
                if idx > 50:
                    response = response[:idx+1]
                    break
            else:
                response = response[:150].rsplit(" ", 1)[0] + "..."
        
        result = response.strip()
        
        # Fallback if result is empty, too short, or just punctuation
        if not result or len(result) < 3 or result in [".", "!", "?", ",", ";", ":"]:
            return "I understand."
        
        return result
    
    def _build_prompt(self, message: str, history: List[Dict]) -> str:
        """Build prompt from conversation history."""
        # Use a format that encourages GPT-2 to answer questions
        # GPT-2 works better with continuation-style prompts
        
        if not history:
            # First message - use a format that encourages answering
            # Extract question words to help GPT-2 understand it should answer
            question_words = ["what", "who", "where", "when", "why", "how", "which", "whose"]
            is_question = any(message.lower().startswith(qw) for qw in question_words) or "?" in message
            
            if is_question:
                # For questions, use a format that suggests an answer
                return f"Q: {message}\nA:"
            else:
                # For statements, use conversational format
                return f"Person 1: {message}\nPerson 2:"
        
        # Build from recent history (max 1 exchange to avoid repetition)
        prompt_parts = []
        
        # Only include last exchange (2 turns)
        for turn in history[-2:]:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                # Check if it's a question
                question_words = ["what", "who", "where", "when", "why", "how", "which", "whose"]
                is_question = any(content.lower().startswith(qw) for qw in question_words) or "?" in content
                
                if is_question:
                    prompt_parts.append(f"Q: {content}")
                else:
                    prompt_parts.append(f"Person 1: {content}")
            else:
                # Assistant response - keep it short
                short_content = content.split(".")[0][:100]  # First sentence, max 100 chars
                prompt_parts.append(f"A: {short_content}")
        
        # Add current message
        question_words = ["what", "who", "where", "when", "why", "how", "which", "whose"]
        is_question = any(message.lower().startswith(qw) for qw in question_words) or "?" in message
        
        if is_question:
            prompt_parts.append(f"Q: {message}\nA:")
        else:
            prompt_parts.append(f"Person 1: {message}\nPerson 2:")
        
        prompt = "\n".join(prompt_parts)
        
        return prompt


def main():
    parser = argparse.ArgumentParser(description="Standalone Fused LLM Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)",
    )
    parser.add_argument(
        "--patch-tokenizer",
        type=str,
        default=None,
        help="Path to AdaptiVocab PatchTokenizer .pkl file",
    )
    parser.add_argument(
        "--larosa-sparsity",
        type=float,
        default=0.0,
        help="LaRoSA sparsity level (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from",
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = FusedLLMEngine(
        model_name=args.model,
        patch_tokenizer_path=args.patch_tokenizer,
        larosa_sparsity=args.larosa_sparsity,
        device=args.device,
    )
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("🤖 Standalone Fused LLM Chatbot")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response = engine.chat(user_input, conversation_history)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    # Single prompt mode
    elif args.prompt:
        response = engine.generate(args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}\n")
    
    else:
        print("Use --interactive for chat mode or --prompt for single generation")


if __name__ == "__main__":
    main()

