"""
Adaptive Tokenization System

4-tier strategy for handling ANY vocabulary size:
1. Pretrained Tokenizer Matching: Exact vocab_size match to known tokenizers
2. Custom BPE Training: Train tokenizer on user dataset (100+ samples, 5K-100K vocab)
3. Character-Level: Fallback for any vocab size
4. User Upload: Optional user-provided tokenizer

This module automatically selects the optimal strategy based on vocab_size
and available dataset.
"""

import os
from typing import Optional, Union
from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class AdaptiveTokenizer:
    """
    Adaptive tokenization strategy selector.

    Automatically chooses the best tokenization approach based on:
    - Vocabulary size
    - Dataset availability and size
    - Known pretrained tokenizer mappings

    Example:
        >>> # Auto-detect and load tokenizer
        >>> tokenizer = AdaptiveTokenizer.load_or_create(
        ...     vocab_size=50257,
        ...     dataset=my_dataset
        ... )
        >>> # Will automatically use GPT-2 tokenizer (exact match)
    """

    # Known pretrained tokenizers mapped by exact vocab_size
    KNOWN_TOKENIZERS = {
        # GPT family
        50257: "gpt2",                              # GPT-2 (124M, 355M, 774M, 1.5B)
        50400: "EleutherAI/gpt-neo-1.3B",          # GPT-Neo
        50280: "EleutherAI/gpt-j-6B",              # GPT-J

        # LLaMA family
        32000: "meta-llama/Llama-2-7b-hf",         # LLaMA 2
        128000: "meta-llama/Meta-Llama-3-8B",      # LLaMA 3
        128256: "meta-llama/Llama-3.1-8B",         # LLaMA 3.1

        # BERT family
        30522: "bert-base-uncased",                 # BERT
        28996: "bert-base-cased",                   # BERT-cased

        # OPT family
        50265: "facebook/opt-125m",                 # OPT-125M
        50272: "facebook/opt-350m",                 # OPT-350M
        250002: "facebook/opt-2.7b",                # OPT-2.7B+

        # Phi family
        49152: "microsoft/phi-2",                   # Phi-2
        51200: "microsoft/phi-1_5",                 # Phi-1.5
        100352: "microsoft/phi-3-mini-4k-instruct", # Phi-3 Mini
        151936: "microsoft/Phi-3-medium-128k-instruct",  # Phi-3 Medium

        # Qwen family
        100277: "Qwen/Qwen-7B",                     # Qwen
        151851: "Qwen/Qwen1.5-7B",                  # Qwen 1.5
        151643: "Qwen/Qwen2-7B",                    # Qwen 2

        # Mistral/Mixtral
        32000: "mistralai/Mistral-7B-v0.1",        # Mistral (shares with LLaMA)
        32768: "mistralai/Mixtral-8x7B-v0.1",      # Mixtral

        # Gemma
        256000: "google/gemma-7b",                  # Gemma

        # Other models
        32100: "google/flan-t5-base",              # FLAN-T5
        51200: "bigscience/bloom-560m",            # BLOOM
    }

    @classmethod
    def detect_strategy(cls, vocab_size: int, dataset_size: int = 0) -> str:
        """
        Detect optimal tokenization strategy.

        Strategy selection logic:
        1. If vocab_size matches known tokenizer → use 'pretrained'
        2. If dataset_size >= 100 and 5000 <= vocab_size <= 100000 → 'train_bpe'
        3. Otherwise → 'character' (universal fallback)

        Args:
            vocab_size: Target vocabulary size
            dataset_size: Number of samples in dataset

        Returns:
            Strategy name: 'pretrained', 'train_bpe', or 'character'
        """
        # Tier 1: Exact match to known tokenizer
        if vocab_size in cls.KNOWN_TOKENIZERS:
            return 'pretrained'

        # Tier 2: Custom BPE training
        # Requirements:
        # - At least 100 samples for meaningful training
        # - Vocab size in reasonable range (5K-100K)
        if dataset_size >= 100 and 5000 <= vocab_size <= 100000:
            return 'train_bpe'

        # Tier 3: Character-level fallback (works for any vocab_size)
        return 'character'

    @classmethod
    def load_or_create(cls,
                       vocab_size: int,
                       dataset: Optional[Dataset] = None,
                       cache_dir: str = "./tokenizer_cache",
                       special_tokens: Optional[list] = None) -> Union[PreTrainedTokenizer, 'CharacterLevelTokenizer']:
        """
        Load or create tokenizer based on optimal strategy.

        Automatically selects and executes the best tokenization approach.

        Args:
            vocab_size: Target vocabulary size
            dataset: Optional dataset for BPE training
            cache_dir: Directory for caching tokenizers
            special_tokens: Optional list of special tokens (defaults to standard set)

        Returns:
            Tokenizer instance (PreTrainedTokenizer or CharacterLevelTokenizer)

        Example:
            >>> # Known vocab size - loads pretrained
            >>> tok = AdaptiveTokenizer.load_or_create(50257)
            >>> # Uses GPT-2 tokenizer automatically
            >>>
            >>> # Unknown vocab size with dataset - trains BPE
            >>> tok = AdaptiveTokenizer.load_or_create(
            ...     vocab_size=15000,
            ...     dataset=my_dataset
            ... )
            >>> # Trains custom BPE on dataset
        """
        # Default special tokens
        if special_tokens is None:
            special_tokens = ['<pad>', '<unk>', '<s>', '</s>']

        # Detect strategy
        dataset_size = len(dataset) if dataset is not None else 0
        strategy = cls.detect_strategy(vocab_size, dataset_size)

        print(f"📊 Vocab size: {vocab_size:,}")
        print(f"📊 Dataset size: {dataset_size:,} samples")
        print(f"🎯 Selected strategy: {strategy}")

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Execute strategy
        if strategy == 'pretrained':
            tokenizer = cls._load_pretrained(vocab_size, cache_dir)

        elif strategy == 'train_bpe':
            tokenizer = cls._train_bpe(vocab_size, dataset, special_tokens, cache_dir)

        else:  # character
            tokenizer = cls._create_character(vocab_size, special_tokens)

        # Validate tokenizer
        from .validator import TokenizerValidator
        TokenizerValidator.validate(tokenizer, vocab_size)

        return tokenizer

    @classmethod
    def _load_pretrained(cls, vocab_size: int, cache_dir: str) -> PreTrainedTokenizer:
        """
        Load pretrained tokenizer by vocab_size lookup.

        Args:
            vocab_size: Vocabulary size to match
            cache_dir: Cache directory

        Returns:
            Loaded pretrained tokenizer
        """
        from transformers import AutoTokenizer

        model_name = cls.KNOWN_TOKENIZERS[vocab_size]
        print(f"✓ Loading pretrained tokenizer: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True  # Some models require this
        )

        print(f"✓ Loaded tokenizer with vocab_size={len(tokenizer)}")
        
        # Handle missing pad_token for decoder-only models (GPT-2, GPT-Neo, LLaMA, etc.)
        # Industry standard: set pad_token = eos_token for autoregressive models
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"✓ Set pad_token = eos_token (id={tokenizer.eos_token_id}) for decoder-only model")
            else:
                # Fallback: add custom pad token if eos_token also missing
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"✓ Added custom pad_token '[PAD]' (id={tokenizer.pad_token_id})")
        
        # Handle missing unk_token (rarer, but validator requires it)
        if tokenizer.unk_token is None:
            if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
                # Token ID exists but string representation missing - acceptable
                pass
            else:
                tokenizer.add_special_tokens({'unk_token': '[UNK]'})
                print(f"✓ Added unk_token '[UNK]' (id={tokenizer.unk_token_id})")
        
        return tokenizer

    @classmethod
    def _train_bpe(cls,
                   vocab_size: int,
                   dataset: Dataset,
                   special_tokens: list,
                   cache_dir: str) -> PreTrainedTokenizerFast:
        """
        Train custom BPE tokenizer on dataset.

        Args:
            vocab_size: Target vocabulary size
            dataset: Dataset to train on
            special_tokens: List of special tokens
            cache_dir: Cache directory

        Returns:
            Trained BPE tokenizer
        """
        from .bpe_trainer import FastBPETrainer

        print(f"✓ Training custom BPE tokenizer...")
        print(f"  Target vocab_size: {vocab_size:,}")
        print(f"  Training samples: {len(dataset):,}")

        # Extract text from dataset
        texts = dataset['text'] if 'text' in dataset.column_names else []

        if not texts:
            raise ValueError("Dataset must have a 'text' column for BPE training")

        tokenizer = FastBPETrainer.train_on_dataset(
            texts=texts,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            cache_dir=cache_dir
        )

        print(f"✓ BPE training complete!")
        return tokenizer

    @classmethod
    def _create_character(cls, vocab_size: int, special_tokens: list) -> 'CharacterLevelTokenizer':
        """
        Create character-level tokenizer.

        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens

        Returns:
            Character-level tokenizer
        """
        from .character_tokenizer import CharacterLevelTokenizer

        print(f"✓ Creating character-level tokenizer...")
        print(f"  Target vocab_size: {vocab_size:,}")

        tokenizer = CharacterLevelTokenizer(
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

        print(f"✓ Character tokenizer created!")
        return tokenizer

    @classmethod
    def get_known_tokenizers(cls) -> dict:
        """
        Get mapping of all known pretrained tokenizers.

        Returns:
            Dictionary mapping vocab_size → model_name
        """
        return cls.KNOWN_TOKENIZERS.copy()

    @classmethod
    def is_known_vocab_size(cls, vocab_size: int) -> bool:
        """
        Check if vocab_size matches a known pretrained tokenizer.

        Args:
            vocab_size: Vocabulary size to check

        Returns:
            True if pretrained tokenizer available
        """
        return vocab_size in cls.KNOWN_TOKENIZERS

    @classmethod
    def suggest_tokenizer(cls, vocab_size: int) -> Optional[str]:
        """
        Suggest a tokenizer for given vocab_size.

        Args:
            vocab_size: Vocabulary size

        Returns:
            Suggested model name or None
        """
        if vocab_size in cls.KNOWN_TOKENIZERS:
            return cls.KNOWN_TOKENIZERS[vocab_size]

        # Find closest match
        closest = min(cls.KNOWN_TOKENIZERS.keys(), key=lambda x: abs(x - vocab_size))
        diff = abs(closest - vocab_size)

        if diff < 1000:  # Within 1K difference
            return f"{cls.KNOWN_TOKENIZERS[closest]} (closest match, diff={diff})"

        return None
