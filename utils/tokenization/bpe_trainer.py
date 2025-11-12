"""
Fast BPE Tokenizer Training

Train custom Byte-Pair Encoding (BPE) tokenizers on user datasets.
Optimized for Google Colab with streaming and memory-efficient training.

Typical training times:
- 100 samples: ~10 seconds
- 1,000 samples: ~30 seconds
- 10,000 samples: ~2 minutes
"""

import os
from typing import List, Optional, Iterator
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


class FastBPETrainer:
    """
    Train custom BPE tokenizers efficiently.

    Uses HuggingFace tokenizers library for fast training with:
    - ByteLevel pre-tokenization
    - Streaming support for large datasets
    - Progress tracking
    - Automatic caching

    Example:
        >>> texts = ["Hello world", "How are you", ...]
        >>> tokenizer = FastBPETrainer.train_on_dataset(
        ...     texts=texts,
        ...     vocab_size=10000,
        ...     special_tokens=['<pad>', '<unk>', '<s>', '</s>']
        ... )
        >>> tokenizer.save_pretrained("./my_tokenizer")
    """

    @staticmethod
    def train_on_dataset(texts: List[str],
                         vocab_size: int,
                         special_tokens: List[str],
                         cache_dir: str = "./tokenizer_cache",
                         min_frequency: int = 2,
                         show_progress: bool = True) -> PreTrainedTokenizerFast:
        """
        Train BPE tokenizer on text dataset.

        Args:
            texts: List of text strings to train on
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens (e.g., ['<pad>', '<unk>', '<s>', '</s>'])
            cache_dir: Directory to cache trained tokenizer
            min_frequency: Minimum frequency for a token to be included
            show_progress: Show progress bar during training

        Returns:
            Trained PreTrainedTokenizerFast instance

        Raises:
            ValueError: If texts is empty or vocab_size is invalid
        """
        if not texts:
            raise ValueError("Cannot train tokenizer on empty text list")

        if vocab_size < 100:
            raise ValueError(f"vocab_size must be at least 100, got {vocab_size}")

        print(f"🔧 Training BPE tokenizer...")
        print(f"   Samples: {len(texts):,}")
        print(f"   Target vocab_size: {vocab_size:,}")
        print(f"   Special tokens: {special_tokens}")

        # Initialize BPE model
        tokenizer = Tokenizer(models.BPE())

        # Set up pre-tokenizer (ByteLevel handles UTF-8, spaces, etc.)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Set up decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # Train tokenizer
        print("   Training...")
        tokenizer.train_from_iterator(
            FastBPETrainer._text_iterator(texts),
            trainer=trainer
        )

        print(f"✓ Training complete! Vocab size: {tokenizer.get_vocab_size()}")

        # Wrap in PreTrainedTokenizerFast for HuggingFace compatibility
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=special_tokens[1] if len(special_tokens) > 1 else '<unk>',
            pad_token=special_tokens[0] if len(special_tokens) > 0 else '<pad>',
            bos_token=special_tokens[2] if len(special_tokens) > 2 else '<s>',
            eos_token=special_tokens[3] if len(special_tokens) > 3 else '</s>',
        )

        # Add post-processor for proper formatting
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"bpe_vocab_{vocab_size}")
        wrapped_tokenizer.save_pretrained(cache_path)
        print(f"✓ Cached to: {cache_path}")

        return wrapped_tokenizer

    @staticmethod
    def _text_iterator(texts: List[str]) -> Iterator[str]:
        """
        Create memory-efficient iterator over texts.

        Args:
            texts: List of text strings

        Yields:
            Individual text strings
        """
        for text in texts:
            if text and isinstance(text, str):
                yield text

    @staticmethod
    def train_with_streaming(text_iterator: Iterator[str],
                            vocab_size: int,
                            special_tokens: List[str],
                            cache_dir: str = "./tokenizer_cache",
                            min_frequency: int = 2) -> PreTrainedTokenizerFast:
        """
        Train BPE tokenizer with streaming (for very large datasets).

        Use this when dataset doesn't fit in memory. Provide an iterator
        that yields text samples one at a time.

        Args:
            text_iterator: Iterator yielding text strings
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens
            cache_dir: Directory to cache trained tokenizer
            min_frequency: Minimum frequency for a token

        Returns:
            Trained PreTrainedTokenizerFast instance

        Example:
            >>> def my_iterator():
            ...     for line in open('huge_file.txt'):
            ...         yield line.strip()
            >>> tokenizer = FastBPETrainer.train_with_streaming(
            ...     my_iterator(), vocab_size=10000, special_tokens=[...]
            ... )
        """
        print(f"🔧 Training BPE tokenizer (streaming mode)...")
        print(f"   Target vocab_size: {vocab_size:,}")

        # Initialize
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # Train from iterator
        print("   Training (streaming)...")
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        print(f"✓ Training complete! Vocab size: {tokenizer.get_vocab_size()}")

        # Wrap and cache
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=special_tokens[1] if len(special_tokens) > 1 else '<unk>',
            pad_token=special_tokens[0] if len(special_tokens) > 0 else '<pad>',
            bos_token=special_tokens[2] if len(special_tokens) > 2 else '<s>',
            eos_token=special_tokens[3] if len(special_tokens) > 3 else '</s>',
        )

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Cache
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"bpe_vocab_{vocab_size}_streaming")
        wrapped_tokenizer.save_pretrained(cache_path)
        print(f"✓ Cached to: {cache_path}")

        return wrapped_tokenizer

    @staticmethod
    def estimate_training_time(num_samples: int) -> str:
        """
        Estimate training time based on dataset size.

        Args:
            num_samples: Number of text samples

        Returns:
            Estimated time as string (e.g., "~30 seconds")
        """
        if num_samples < 100:
            return "~10 seconds"
        elif num_samples < 1000:
            return "~30 seconds"
        elif num_samples < 10000:
            return "~2 minutes"
        elif num_samples < 100000:
            return "~10 minutes"
        else:
            return "~30+ minutes"

    @staticmethod
    def validate_texts(texts: List[str]) -> tuple[bool, str]:
        """
        Validate text dataset before training.

        Args:
            texts: List of text strings

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not texts:
            return False, "Text list is empty"

        if not isinstance(texts, list):
            return False, "Texts must be a list"

        # Check for non-string items
        non_strings = sum(1 for t in texts if not isinstance(t, str))
        if non_strings > 0:
            return False, f"Found {non_strings} non-string items in texts"

        # Check for empty strings
        empty_count = sum(1 for t in texts if not t.strip())
        if empty_count > len(texts) * 0.5:
            return False, f"More than 50% of texts are empty ({empty_count}/{len(texts)})"

        # Check average length
        avg_length = sum(len(t) for t in texts) / len(texts)
        if avg_length < 10:
            return False, f"Average text length too short ({avg_length:.1f} chars). Need longer samples."

        return True, "Validation passed"


class BPETrainerConfig:
    """Configuration for BPE training with common presets."""

    # Common vocab size presets
    VOCAB_SMALL = 5000      # For small domains/languages
    VOCAB_MEDIUM = 10000    # Good default
    VOCAB_LARGE = 25000     # For diverse datasets
    VOCAB_XLARGE = 50000    # For very large corpora

    # Common special token sets
    SPECIAL_TOKENS_MINIMAL = ['<pad>', '<unk>']
    SPECIAL_TOKENS_STANDARD = ['<pad>', '<unk>', '<s>', '</s>']
    SPECIAL_TOKENS_EXTENDED = ['<pad>', '<unk>', '<s>', '</s>', '<mask>', '<sep>', '<cls>']

    @staticmethod
    def get_preset(preset_name: str) -> dict:
        """
        Get training configuration preset.

        Args:
            preset_name: Name of preset ('small', 'medium', 'large', 'xlarge')

        Returns:
            Dictionary with training configuration
        """
        presets = {
            'small': {
                'vocab_size': BPETrainerConfig.VOCAB_SMALL,
                'min_frequency': 2,
                'special_tokens': BPETrainerConfig.SPECIAL_TOKENS_STANDARD,
            },
            'medium': {
                'vocab_size': BPETrainerConfig.VOCAB_MEDIUM,
                'min_frequency': 2,
                'special_tokens': BPETrainerConfig.SPECIAL_TOKENS_STANDARD,
            },
            'large': {
                'vocab_size': BPETrainerConfig.VOCAB_LARGE,
                'min_frequency': 3,
                'special_tokens': BPETrainerConfig.SPECIAL_TOKENS_EXTENDED,
            },
            'xlarge': {
                'vocab_size': BPETrainerConfig.VOCAB_XLARGE,
                'min_frequency': 5,
                'special_tokens': BPETrainerConfig.SPECIAL_TOKENS_EXTENDED,
            },
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Choose from: {list(presets.keys())}")

        return presets[preset_name]
