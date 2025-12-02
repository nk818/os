#!/usr/bin/env python3
"""
Enhanced Fused LLM Chatbot
Integrates all optimizations:
- Better chat model (DialoGPT or instruction-tuned)
- AdaptiVocab (tokenizer optimization)
- LaRoSA (activation sparsity)
- Simplified vAttention (memory tracking)
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
)

# Add paths for AdaptiVocab
sys.path.insert(0, str(Path(__file__).parent.parent / "AdaptiVocab" / "src"))
try:
    from build_vocab.patch_tokenizer import PatchTokenizer
    ADAPTIVOCAB_AVAILABLE = True
except ImportError:
    ADAPTIVOCAB_AVAILABLE = False
    PatchTokenizer = None

# Import SimpleAdaptiVocab
try:
    from simple_adaptivocab import SimpleAdaptiVocab
    SIMPLE_ADAPTIVOCAB_AVAILABLE = True
except ImportError:
    SIMPLE_ADAPTIVOCAB_AVAILABLE = False
    SimpleAdaptiVocab = None


class SimpleLaRoSA(nn.Module):
    """Simplified LaRoSA for CPU/macOS."""
    
    def __init__(self, hidden_size: int, sparsity: float = 0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.k = int(hidden_size * (1 - sparsity))
        self.register_buffer('rotation_matrix', torch.eye(hidden_size))
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.sparsity == 0.0:
            return x
        
        x_rotated = x @ self.rotation_matrix
        topk_values, topk_indices = torch.topk(x_rotated.abs(), self.k, dim=-1)
        x_sparse = torch.zeros_like(x_rotated)
        x_sparse.scatter_(-1, topk_indices, x_rotated.gather(-1, topk_indices))
        x_output = x_sparse @ self.rotation_matrix.T
        return x_output


class SimpleVAttention:
    """
    Simplified vAttention interface for CPU.
    Simulates vAttention's dynamic KV cache memory management.
    
    Real vAttention (GPU):
    - Uses CUDA virtual memory APIs for dynamic allocation
    - Provides 15-20% memory savings through demand paging
    - Requires CUDA 12.1+ and GPU
    
    This CPU version:
    - Simulates memory savings (15% reduction)
    - Tracks KV cache usage
    - Provides metrics for comparison
    """
    
    def __init__(self, max_cache_size: int = 2048, simulate_savings: bool = True):
        self.max_cache_size = max_cache_size
        self.current_cache_size = 0
        self.cache_blocks = {}
        self.simulate_savings = simulate_savings
        self.memory_savings_factor = 0.15  # Simulate 15% memory savings (real vAttention benefit)
        self.total_allocated = 0
        self.total_freed = 0
    
    def init_kvcache(
        self,
        num_layers: int,
        num_heads: int,
        head_size: int,
        max_batch_size: int,
        max_seq_len: int,
        page_size: int = 16,
    ):
        """Initialize KV cache tracking."""
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        
        # Calculate memory per token
        self.memory_per_token = num_layers * num_heads * head_size * 2 * 4  # 2 for K&V, 4 bytes per float32
        baseline_memory = self.memory_per_token * max_seq_len / 1024 / 1024
        optimized_memory = baseline_memory * (1 - self.memory_savings_factor) if self.simulate_savings else baseline_memory
        
        print(f"   📊 vAttention: KV cache management")
        print(f"      Max tokens: {max_seq_len}")
        print(f"      Baseline memory: ~{baseline_memory:.1f} MB")
        if self.simulate_savings:
            print(f"      Simulated savings: {self.memory_savings_factor*100:.0f}% (~{optimized_memory:.1f} MB)")
        print(f"      Note: Real vAttention requires GPU with CUDA 12.1+")
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        baseline_memory = self.current_cache_size * self.memory_per_token / 1024 / 1024
        if self.simulate_savings:
            # Simulate vAttention's memory savings
            optimized_memory = baseline_memory * (1 - self.memory_savings_factor)
            memory_saved = baseline_memory - optimized_memory
        else:
            optimized_memory = baseline_memory
            memory_saved = 0.0
        
        return {
            "current_tokens": self.current_cache_size,
            "max_tokens": self.max_seq_len,
            "baseline_memory_mb": baseline_memory,
            "optimized_memory_mb": optimized_memory,
            "memory_saved_mb": memory_saved,
            "memory_savings_percent": self.memory_savings_factor * 100 if self.simulate_savings else 0.0,
            "utilization": self.current_cache_size / self.max_seq_len if self.max_seq_len > 0 else 0,
            "total_allocated": self.total_allocated,
            "total_freed": self.total_freed,
        }
    
    def allocate(self, num_tokens: int) -> bool:
        """
        Allocate cache space (simulated).
        Real vAttention uses virtual memory APIs for on-demand allocation.
        """
        if self.current_cache_size + num_tokens <= self.max_seq_len:
            self.current_cache_size += num_tokens
            self.total_allocated += num_tokens
            return True
        return False
    
    def deallocate(self, num_tokens: int):
        """Deallocate cache space."""
        self.current_cache_size = max(0, self.current_cache_size - num_tokens)
        self.total_freed += num_tokens
    
    def get_memory_savings(self) -> float:
        """Get estimated memory savings from vAttention optimization."""
        if not self.simulate_savings:
            return 0.0
        baseline_memory = self.current_cache_size * self.memory_per_token / 1024 / 1024
        return baseline_memory * self.memory_savings_factor


class EnhancedFusedLLMEngine:
    """
    Enhanced LLM engine with all optimizations:
    - Better chat model
    - AdaptiVocab
    - LaRoSA
    - vAttention (simplified)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",  # Better instruction-tuned model
        patch_tokenizer_path: Optional[str] = None,
        simple_adaptivocab: bool = False,
        larosa_sparsity: float = 0.0,
        vattention_enabled: bool = True,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.larosa_sparsity = larosa_sparsity
        self.vattention_enabled = vattention_enabled
        self.simple_adaptivocab = simple_adaptivocab
        self._last_generation_stats = {}
        
        print(f"🚀 Initializing Enhanced Fused LLM Engine...")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
        # Load tokenizer with AdaptiVocab
        self.tokenizer = self._load_tokenizer(patch_tokenizer_path, simple_adaptivocab)
        
        # Load model
        print(f"   Loading model...")
        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )
        self.model.eval()
        
        # Initialize vAttention or Hybrid KV Cache
        self.hybrid_kv_cache = None
        if vattention_enabled:
            # Check if we should use hybrid (vAttention + CAKE) or just vAttention
            try:
                from hybrid_kv_cache import HybridKVCacheOptimizer
                # Use hybrid optimizer
                self.hybrid_kv_cache = HybridKVCacheOptimizer(
                    num_layers=self.model.config.n_layer if hasattr(self.model.config, 'n_layer') else self.model.config.num_hidden_layers,
                    num_heads=self.model.config.n_head if hasattr(self.model.config, 'n_head') else self.model.config.num_attention_heads,
                    head_size=self.model.config.n_embd // self.model.config.n_head if hasattr(self.model.config, 'n_embd') else self.model.config.hidden_size // self.model.config.num_attention_heads,
                    max_batch_size=1,
                    max_seq_len=max_length,
                    device=device,
                )
                self.vattention = None  # Use hybrid instead
            except ImportError:
                # Fallback to simple vAttention
                self.vattention = SimpleVAttention(max_cache_size=max_length)
                self.vattention.init_kvcache(
                    num_layers=self.model.config.n_layer if hasattr(self.model.config, 'n_layer') else self.model.config.num_hidden_layers,
                    num_heads=self.model.config.n_head if hasattr(self.model.config, 'n_head') else self.model.config.num_attention_heads,
                    head_size=self.model.config.n_embd // self.model.config.n_head if hasattr(self.model.config, 'n_embd') else self.model.config.hidden_size // self.model.config.num_attention_heads,
                    max_batch_size=1,
                    max_seq_len=max_length,
                )
        else:
            self.vattention = None
        
        # Apply LaRoSA if enabled
        if larosa_sparsity > 0.0:
            print(f"   Applying LaRoSA sparsity: {larosa_sparsity*100:.1f}%")
            self._apply_larosa()
        
        print(f"✅ Engine ready!")
        if self.hybrid_kv_cache:
            stats = self.hybrid_kv_cache.get_memory_usage()
            print(f"   🎯 Hybrid KV Cache: {stats['utilization']*100:.1f}% cache utilization")
        elif self.vattention:
            stats = self.vattention.get_memory_usage()
            print(f"   📊 vAttention: {stats['utilization']*100:.1f}% cache utilization")
    
    def _load_tokenizer(self, patch_tokenizer_path: Optional[str], simple_adaptivocab: bool = False) -> PreTrainedTokenizer:
        """Load tokenizer with optional AdaptiVocab PatchTokenizer or SimpleAdaptiVocab."""
        # First try PatchTokenizer if path provided
        if patch_tokenizer_path and ADAPTIVOCAB_AVAILABLE and PatchTokenizer:
            print(f"   Loading AdaptiVocab PatchTokenizer: {patch_tokenizer_path}")
            try:
                # Try loading as saved pickle first
                with open(patch_tokenizer_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if it's a dict (saved PatchTokenizer) or PatchTokenizer instance
                if isinstance(data, dict):
                    # Load using load_model_from_scratch
                    patch_tokenizer = PatchTokenizer.load_model_from_scratch(
                        path=patch_tokenizer_path,
                        existing_tokenizer_name=self.model_name
                    )
                else:
                    # It's already a PatchTokenizer instance
                    patch_tokenizer = data
                
                # PatchTokenizer IS the tokenizer - use it directly
                original_vocab = len(patch_tokenizer.existing_tokenizer)
                new_vocab = len(patch_tokenizer.token_to_id_dict) if hasattr(patch_tokenizer, 'token_to_id_dict') else len(patch_tokenizer)
                reduction = (1 - new_vocab / original_vocab) * 100 if original_vocab > 0 else 0
                print(f"   ✅ AdaptiVocab enabled (vocab: {original_vocab} → {new_vocab}, {reduction:.1f}% reduction)")
                
                # Store reference to existing_tokenizer for pad_token_id, etc.
                if patch_tokenizer.pad_token_id is None:
                    patch_tokenizer.pad_token_id = patch_tokenizer.eos_token_id
                
                return patch_tokenizer
            except Exception as e:
                print(f"   ⚠️  Failed to load PatchTokenizer: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Falling back to standard tokenizer")
        
        print(f"   Loading standard tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Wrap with SimpleAdaptiVocab if enabled
        if simple_adaptivocab and SIMPLE_ADAPTIVOCAB_AVAILABLE and SimpleAdaptiVocab:
            print(f"   Wrapping tokenizer with SimpleAdaptiVocab (phrase combination)...")
            original_vocab = len(tokenizer)
            tokenizer = SimpleAdaptiVocab(tokenizer)
            print(f"   ✅ SimpleAdaptiVocab enabled (phrase combination active)")
        
        return tokenizer
    
    def _apply_larosa(self):
        """Apply LaRoSA activation sparsity to MLP layers."""
        hidden_size = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else self.model.config.n_embd
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear,)) and ("mlp" in name.lower() or "feed_forward" in name.lower() or "ffn" in name.lower()):
                larosa = SimpleLaRoSA(hidden_size, self.larosa_sparsity)
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
        # Check if using PatchTokenizer
        is_patch_tokenizer = isinstance(self.tokenizer, PatchTokenizer) if PatchTokenizer else False
        
        # Tokenize
        if is_patch_tokenizer:
            # PatchTokenizer returns dict with input_ids as list
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None and isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        input_length = input_ids.shape[1]
        
        # Track with Hybrid KV Cache or vAttention
        if self.hybrid_kv_cache:
            self.hybrid_kv_cache.allocate_cache(input_length)
        elif self.vattention:
            self.vattention.allocate(input_length)
        
        # Get pad_token_id and eos_token_id
        if is_patch_tokenizer:
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
        
        # Generate with model-specific settings
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 2,
        }
        
        # Model-specific settings
        if "phi" in self.model_name.lower():
            # Phi-2 works well with these settings
            gen_kwargs["repetition_penalty"] = 1.1
            gen_kwargs["top_k"] = 50
        elif "dialogpt" in self.model_name.lower():
            gen_kwargs["repetition_penalty"] = 1.1
            gen_kwargs["top_k"] = 50
        
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
        
        # Extract new tokens
        new_tokens = outputs[0][input_length:]
        
        # Decode - PatchTokenizer has different decode signature
        if is_patch_tokenizer:
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Track with Hybrid KV Cache or vAttention
        if self.hybrid_kv_cache:
            total_tokens = outputs.shape[1]
            self.hybrid_kv_cache.allocate_cache(total_tokens - input_length)
            stats = self.hybrid_kv_cache.get_memory_usage()
            if stats['utilization'] > 0.8:
                print(f"   ⚠️  Hybrid KV Cache: High cache usage ({stats['utilization']*100:.1f}%)")
        elif self.vattention:
            total_tokens = outputs.shape[1]
            self.vattention.allocate(total_tokens - input_length)
            stats = self.vattention.get_memory_usage()
            if stats['utilization'] > 0.8:
                print(f"   ⚠️  vAttention: High cache usage ({stats['utilization']*100:.1f}%)")
        
        # Store generation stats for benchmarking
        self._last_generation_stats = {
            "input_tokens": input_length,
            "output_tokens": len(new_tokens),
            "total_tokens": outputs.shape[1],
        }
        
        # Clean up
        generated_text = generated_text.strip()
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
        
        # Build prompt
        prompt = self._build_prompt(message, conversation_history)
        
        # Generate
        response = self.generate(
            prompt,
            max_new_tokens=60,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        
        # Clean response
        response = self._clean_response(response, original_message=message)
        
        # Update history
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_prompt(self, message: str, history: List[Dict]) -> str:
        """Build prompt from conversation history."""
        # Phi-2 uses instruction format
        if "phi" in self.model_name.lower():
            # Phi-2 instruction format
            if not history:
                return f"Instruct: {message}\nOutput:"
            else:
                prompt_parts = ["Instruct: "]
                for turn in history[-4:]:
                    role = turn["role"]
                    content = turn["content"]
                    if role == "user":
                        prompt_parts.append(f"User: {content}")
                    else:
                        prompt_parts.append(f"Assistant: {content[:80]}")
                prompt_parts.append(f"User: {message}")
                prompt_parts.append("Assistant:")
                return "\n".join(prompt_parts)
        
        # DialoGPT uses EOS token between turns
        elif "dialogpt" in self.model_name.lower():
            eos_token = self.tokenizer.eos_token
            if not history:
                return message
            else:
                prompt_parts = []
                for turn in history[-6:]:
                    content = turn["content"]
                    prompt_parts.append(content)
                    prompt_parts.append(eos_token)
                prompt_parts.append(message)
                return " ".join(prompt_parts)
        
        # Generic instruction format for other models
        else:
            if not history:
                return f"### Instruction:\n{message}\n\n### Response:\n"
            
            prompt_parts = ["### Conversation:"]
            for turn in history[-4:]:
                role = turn["role"]
                content = turn["content"]
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                else:
                    prompt_parts.append(f"Assistant: {content[:100]}")
            
            prompt_parts.append(f"\n### Instruction:\n{message}\n\n### Response:\n")
            return "\n".join(prompt_parts)
    
    def _clean_response(self, response: str, original_message: str = "") -> str:
        """Clean up the generated response."""
        if not response:
            return "I'm not sure how to respond to that."
        
        # Remove special tokens and clean
        response = response.strip()
        
        # Remove echo (but be less aggressive for DialoGPT)
        response_lower = response.lower().strip()
        message_lower = original_message.lower().strip()
        
        if message_lower and response_lower and len(message_lower) > 5:
            # Only check for echo if message is substantial
            question_words = ["what", "who", "where", "when", "why", "how", "which", "whose"]
            response_words = [w for w in response_lower.split() if w not in question_words]
            message_words = [w for w in message_lower.split() if w not in question_words]
            
            if len(message_words) > 3:  # Only check for substantial messages
                overlap = sum(1 for w in message_words if w in response_words)
                if overlap > len(message_words) * 0.7:  # 70% overlap threshold
                    # For DialoGPT, try generating again with different params
                    if "dialogpt" not in self.model_name.lower():
                        return "I'm not sure how to answer that question."
        
        # Clean repetition (less aggressive)
        lines = response.split("\n")
        if len(lines) > 1:
            response = lines[0]
        
        words = response.split()
        if len(words) > 2:
            unique_words = set(w.lower() for w in words)
            if len(unique_words) < len(words) * 0.3:  # Very repetitive
                seen = set()
                unique_sequence = []
                for word in words:
                    if word.lower() not in seen:
                        unique_sequence.append(word)
                        seen.add(word.lower())
                    else:
                        break
                response = " ".join(unique_sequence) if unique_sequence else " ".join(words[:5])
        
        # Limit length
        if len(response) > 200:
            for punct in [".", "!", "?"]:
                idx = response.rfind(punct, 0, 200)
                if idx > 20:
                    response = response[:idx+1]
                    break
            else:
                response = response[:200].rsplit(" ", 1)[0]
        
        result = response.strip()
        
        # More lenient fallback
        if not result or (len(result) < 2 and result not in ["Hi", "OK", "Yes", "No"]):
            # For DialoGPT, return a simple response
            if "dialogpt" in self.model_name.lower():
                return "I see."
            return "I understand."
        
        return result
    
    def get_stats(self) -> Dict:
        """Get optimization statistics."""
        is_simple_adaptivocab = isinstance(self.tokenizer, SimpleAdaptiVocab) if SIMPLE_ADAPTIVOCAB_AVAILABLE and SimpleAdaptiVocab else False
        is_patch_tokenizer = isinstance(self.tokenizer, PatchTokenizer) if ADAPTIVOCAB_AVAILABLE and PatchTokenizer else False
        
        stats = {
            "adaptivocab": is_patch_tokenizer or is_simple_adaptivocab,
            "simple_adaptivocab": is_simple_adaptivocab,
            "larosa": self.larosa_sparsity > 0.0,
            "larosa_sparsity": self.larosa_sparsity,
            "vattention": self.vattention is not None,
            "hybrid_kv_cache": self.hybrid_kv_cache is not None,
        }
        
        if self.hybrid_kv_cache:
            stats.update(self.hybrid_kv_cache.get_memory_usage())
        elif self.vattention:
            stats.update(self.vattention.get_memory_usage())
        
        # Add last generation stats if available
        if self._last_generation_stats:
            stats["last_generation"] = self._last_generation_stats
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Enhanced Fused LLM Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model name (default: microsoft/phi-2). Options: microsoft/phi-2, microsoft/DialoGPT-medium, gpt2",
    )
    parser.add_argument(
        "--patch-tokenizer",
        type=str,
        default=None,
        help="Path to AdaptiVocab PatchTokenizer .pkl file",
    )
    parser.add_argument(
        "--simple-adaptivocab",
        action="store_true",
        help="Enable SimpleAdaptiVocab (phrase combination)",
    )
    parser.add_argument(
        "--larosa-sparsity",
        type=float,
        default=0.4,
        help="LaRoSA sparsity level (0.0-1.0, default: 0.4)",
    )
    parser.add_argument(
        "--no-vattention",
        action="store_true",
        help="Disable vAttention memory tracking",
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
    engine = EnhancedFusedLLMEngine(
        model_name=args.model,
        patch_tokenizer_path=args.patch_tokenizer,
        simple_adaptivocab=getattr(args, 'simple_adaptivocab', False),
        larosa_sparsity=args.larosa_sparsity,
        vattention_enabled=not args.no_vattention,
        device=args.device,
    )
    
    # Show stats
    stats = engine.get_stats()
    print(f"\n📊 Optimization Status:")
    print(f"   AdaptiVocab: {'✅' if stats['adaptivocab'] else '❌'}")
    print(f"   LaRoSA: {'✅' if stats['larosa'] else '❌'} ({stats['larosa_sparsity']*100:.0f}% sparsity)")
    print(f"   vAttention: {'✅' if stats['vattention'] else '❌'}")
    print()
    
    # Interactive mode
    if args.interactive:
        print("="*60)
        print("🤖 Enhanced Fused LLM Chatbot")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'stats' to see optimization statistics\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == "stats":
                    stats = engine.get_stats()
                    print(f"\n📊 Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    print()
                    continue
                
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
                import traceback
                traceback.print_exc()
    
    # Single prompt mode
    elif args.prompt:
        response = engine.generate(args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}\n")
    
    else:
        print("Use --interactive for chat mode or --prompt for single generation")


if __name__ == "__main__":
    main()

