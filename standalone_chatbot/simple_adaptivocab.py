#!/usr/bin/env python3
"""
Simple AdaptiVocab Implementation
Combines common phrases into single tokens to reduce token count.
Example: "quantum mechanics" -> "quantummechanics" (1 token instead of 2)
"""

import re
from typing import List, Dict, Set, Optional
from collections import Counter
from transformers import PreTrainedTokenizer


class SimpleAdaptiVocab:
    """
    Simple phrase-based token reduction.
    Combines common word pairs/phrases into single tokens.
    """
    
    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizer,
        common_phrases: Optional[List[str]] = None,
        min_phrase_frequency: int = 3,
    ):
        """
        Initialize SimpleAdaptiVocab.
        
        Args:
            base_tokenizer: Base tokenizer to wrap
            common_phrases: List of common phrases to combine (e.g., ["quantum mechanics", "artificial intelligence"])
            min_phrase_frequency: Minimum frequency to consider a phrase common
        """
        self.base_tokenizer = base_tokenizer
        self.phrase_map: Dict[str, str] = {}  # "quantum mechanics" -> "quantummechanics"
        self.reverse_map: Dict[str, str] = {}  # "quantummechanics" -> "quantum mechanics"
        
        # Default common phrases (domain-specific)
        # Note: These are combined only if the tokenizer would benefit
        # Full AdaptiVocab adds these as actual vocabulary tokens
        default_phrases = [
            # Common abbreviations that benefit from combination
            "artificial intelligence",  # Often tokenized efficiently already
            "machine learning",
            "deep learning", 
            "neural network",
            "natural language processing",
            "computer science",
            "data science",
            "big data",
            "cloud computing",
            "software engineering",
            "web development",
            "user interface",
            "application programming interface",
            "operating system",
            "database management",
            "information technology",
            "cyber security",
            "data structure",
            "algorithm design",
            "computer vision",
            "speech recognition",
            "image processing",
            "pattern recognition",
            "reinforcement learning",
            "supervised learning",
            "unsupervised learning",
            "transfer learning",
            "feature extraction",
            "model training",
            "hyperparameter tuning",
            "gradient descent",
            "backpropagation",
            "activation function",
            "loss function",
            "optimization algorithm",
            "training data",
            "test data",
            "validation data",
            "cross validation",
            "model evaluation",
            "performance metrics",
        ]
        
        # Use provided phrases or defaults
        if common_phrases:
            phrases_to_use = common_phrases
        else:
            phrases_to_use = default_phrases
        
        # Build phrase maps
        for phrase in phrases_to_use:
            combined = phrase.replace(" ", "")
            self.phrase_map[phrase.lower()] = combined
            self.reverse_map[combined.lower()] = phrase
        
        # Sort phrases by length (longest first) for proper matching
        self.sorted_phrases = sorted(
            self.phrase_map.keys(),
            key=len,
            reverse=True
        )
    
    def _combine_phrases(self, text: str) -> str:
        """Combine common phrases in text, only if it reduces tokens."""
        text_lower = text.lower()
        result = text
        
        # Find all phrase matches
        matches = []
        for phrase in self.sorted_phrases:
            combined = self.phrase_map[phrase]
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for match in re.finditer(pattern, text_lower):
                # Check if combining actually reduces tokens
                original_tokens = len(self.base_tokenizer.encode(phrase))
                combined_tokens = len(self.base_tokenizer.encode(combined))
                
                # Only combine if it reduces tokens
                if combined_tokens < original_tokens:
                    matches.append((match.start(), match.end(), phrase, combined))
        
        # Sort matches by position (reverse order for safe replacement)
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # Replace phrases (from end to start to preserve indices)
        for start, end, phrase, combined in matches:
            # Preserve original case
            original_text = text[start:end]
            # Try to preserve capitalization
            if original_text[0].isupper():
                combined = combined.capitalize()
            result = result[:start] + combined + result[end:]
        
        return result
    
    def _split_phrases(self, text: str) -> str:
        """Split combined phrases back into words."""
        result = text
        
        # Sort reverse map by length (longest first)
        sorted_combined = sorted(
            self.reverse_map.keys(),
            key=len,
            reverse=True
        )
        
        # Replace combined tokens with original phrases
        for combined in sorted_combined:
            phrase = self.reverse_map[combined]
            # Use word boundaries
            pattern = r'\b' + re.escape(combined) + r'\b'
            result = re.sub(pattern, phrase, result, flags=re.IGNORECASE)
        
        return result
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text with phrase combination."""
        combined_text = self._combine_phrases(text)
        return self.base_tokenizer.encode(combined_text, **kwargs)
    
    def decode(self, token_ids, **kwargs) -> str:
        """Decode tokens and split phrases back."""
        decoded = self.base_tokenizer.decode(token_ids, **kwargs)
        return self._split_phrases(decoded)
    
    def __call__(self, text, return_tensors=None, **kwargs):
        """Tokenize text with phrase combination."""
        combined_text = self._combine_phrases(text)
        result = self.base_tokenizer(combined_text, return_tensors=return_tensors, **kwargs)
        return result
    
    def __getattr__(self, name):
        """Delegate other attributes to base tokenizer."""
        return getattr(self.base_tokenizer, name)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (same as base tokenizer)."""
        return len(self.base_tokenizer)
    
    def count_tokens_saved(self, text: str) -> Dict[str, int]:
        """Count how many tokens would be saved by phrase combination."""
        original_tokens = len(self.base_tokenizer.encode(text))
        combined_text = self._combine_phrases(text)
        combined_tokens = len(self.base_tokenizer.encode(combined_text))
        
        return {
            "original_tokens": original_tokens,
            "combined_tokens": combined_tokens,
            "tokens_saved": original_tokens - combined_tokens,
            "reduction_percent": ((original_tokens - combined_tokens) / original_tokens * 100) if original_tokens > 0 else 0,
        }


def learn_phrases_from_text(
    texts: List[str],
    base_tokenizer: PreTrainedTokenizer,
    min_frequency: int = 3,
    max_phrases: int = 100,
) -> List[str]:
    """
    Learn common phrases from a corpus of texts.
    
    Args:
        texts: List of text samples
        base_tokenizer: Tokenizer to use
        min_frequency: Minimum frequency for a phrase to be considered
        max_phrases: Maximum number of phrases to return
    
    Returns:
        List of common phrases
    """
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = base_tokenizer.tokenize(text)
        all_tokens.append(tokens)
    
    # Find bigrams (2-word phrases)
    bigrams = Counter()
    for tokens in all_tokens:
        for i in range(len(tokens) - 1):
            # Remove ## prefix from subword tokens
            token1 = tokens[i].replace("##", "")
            token2 = tokens[i + 1].replace("##", "")
            if token1 and token2:
                bigram = f"{token1} {token2}"
                bigrams[bigram] += 1
    
    # Find trigrams (3-word phrases)
    trigrams = Counter()
    for tokens in all_tokens:
        for i in range(len(tokens) - 2):
            token1 = tokens[i].replace("##", "")
            token2 = tokens[i + 1].replace("##", "")
            token3 = tokens[i + 2].replace("##", "")
            if token1 and token2 and token3:
                trigram = f"{token1} {token2} {token3}"
                trigrams[trigram] += 1
    
    # Combine and filter
    all_phrases = bigrams + trigrams
    common_phrases = [
        phrase for phrase, count in all_phrases.most_common(max_phrases * 2)
        if count >= min_frequency
    ][:max_phrases]
    
    return common_phrases

