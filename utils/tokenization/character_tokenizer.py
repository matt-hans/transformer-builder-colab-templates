"""
Character-Level Tokenizer

Universal fallback tokenizer that works for ANY vocabulary size.
Tokenizes text at the character level with special token support.

This tokenizer:
- Always works (no training required)
- Handles any alphabet/language
- Supports arbitrary vocab sizes
- Fast and memory-efficient
"""

import string
import torch
from typing import Dict, List, Optional, Union


class CharacterLevelTokenizer:
    """
    Character-level tokenizer with HuggingFace-compatible interface.

    Tokenizes text character-by-character, treating each character as a token.
    Works with any vocabulary size and supports special tokens.

    Example:
        >>> tokenizer = CharacterLevelTokenizer(
        ...     vocab_size=1000,
        ...     special_tokens=['<pad>', '<unk>', '<s>', '</s>']
        ... )
        >>> encoded = tokenizer.encode("Hello world!", max_length=20)
        >>> print(encoded['input_ids'])
        >>> decoded = tokenizer.decode(encoded['input_ids'])
        >>> print(decoded)  # "Hello world!"
    """

    def __init__(self, vocab_size: int, special_tokens: List[str]):
        """
        Initialize character-level tokenizer.

        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens (e.g., ['<pad>', '<unk>', '<s>', '</s>'])
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build character vocabulary with special tokens."""
        # Start with special tokens
        self.special_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        current_id = len(self.special_tokens)

        # Add printable ASCII characters
        self.char_to_id = {}
        for char in string.printable:
            if current_id >= self.vocab_size:
                break
            self.char_to_id[char] = current_id
            current_id += 1

        # If we still have room, add more Unicode characters
        if current_id < self.vocab_size:
            # Add common Unicode ranges
            unicode_ranges = [
                (0x00A0, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
                (0x0370, 0x03FF),  # Greek
                (0x0400, 0x04FF),  # Cyrillic
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs (sample)
            ]

            for start, end in unicode_ranges:
                for code_point in range(start, min(end, start + 100)):  # Limit per range
                    if current_id >= self.vocab_size:
                        break
                    try:
                        char = chr(code_point)
                        if char not in self.char_to_id:
                            self.char_to_id[char] = current_id
                            current_id += 1
                    except ValueError:
                        continue
                if current_id >= self.vocab_size:
                    break

        # Fill remaining slots with placeholder characters if needed
        while current_id < self.vocab_size:
            placeholder = f"<char_{current_id}>"
            self.char_to_id[placeholder] = current_id
            current_id += 1

        # Combine vocabularies
        self.vocab = {**self.special_to_id, **self.char_to_id}

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Set special token attributes (HuggingFace compatibility)
        self.pad_token = self.special_tokens[0] if len(self.special_tokens) > 0 else '<pad>'
        self.unk_token = self.special_tokens[1] if len(self.special_tokens) > 1 else '<unk>'
        self.bos_token = self.special_tokens[2] if len(self.special_tokens) > 2 else '<s>'
        self.eos_token = self.special_tokens[3] if len(self.special_tokens) > 3 else '</s>'

        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)

    def encode(self, text: str, max_length: int = 512,
               padding: str = 'max_length', truncation: bool = True,
               add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', 'longest', or 'do_not_pad')
            truncation: Whether to truncate to max_length
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        # Convert text to token IDs
        tokens = []

        # Add BOS token if requested
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # Tokenize characters
        for char in text:
            token_id = self.char_to_id.get(char, self.unk_token_id)
            tokens.append(token_id)

        # Add EOS token if requested
        if add_special_tokens:
            tokens.append(self.eos_token_id)

        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)

        # Pad if necessary
        if padding == 'max_length':
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens.extend([self.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor],
               skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        # Convert tensor to list if necessary
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Decode tokens
        chars = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.unk_token)

            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue

            # Skip padding tokens
            if token == self.pad_token:
                continue

            chars.append(token)

        return ''.join(chars)

    def __call__(self, text: Union[str, List[str]],
                 max_length: int = 512,
                 padding: str = 'max_length',
                 truncation: bool = True,
                 return_tensors: Optional[str] = 'pt') -> Dict[str, torch.Tensor]:
        """
        Tokenize text (HuggingFace-compatible interface).

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format ('pt' for PyTorch tensors)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single text
        if isinstance(text, str):
            return self.encode(text, max_length=max_length, padding=padding, truncation=truncation)

        # Handle batch of texts
        batch_encoding = {'input_ids': [], 'attention_mask': []}
        for single_text in text:
            encoded = self.encode(single_text, max_length=max_length, padding=padding, truncation=truncation)
            batch_encoding['input_ids'].append(encoded['input_ids'])
            batch_encoding['attention_mask'].append(encoded['attention_mask'])

        # Stack into batch tensors
        batch_encoding['input_ids'] = torch.stack(batch_encoding['input_ids'])
        batch_encoding['attention_mask'] = torch.stack(batch_encoding['attention_mask'])

        return batch_encoding

    def batch_decode(self, sequences: Union[List[List[int]], torch.Tensor],
                     skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of sequences.

        Args:
            sequences: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping (HuggingFace compatibility)."""
        return self.vocab.copy()

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer configuration.

        Args:
            save_directory: Directory to save tokenizer config
        """
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)

        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'tokenizer_class': 'CharacterLevelTokenizer',
        }

        config_path = os.path.join(save_directory, 'tokenizer_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save vocabulary
        vocab_path = os.path.join(save_directory, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)

        print(f"✓ Saved character tokenizer to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'CharacterLevelTokenizer':
        """
        Load tokenizer from directory.

        Args:
            load_directory: Directory containing tokenizer config

        Returns:
            Loaded CharacterLevelTokenizer instance
        """
        import json
        import os

        config_path = os.path.join(load_directory, 'tokenizer_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(
            vocab_size=config['vocab_size'],
            special_tokens=config['special_tokens']
        )
