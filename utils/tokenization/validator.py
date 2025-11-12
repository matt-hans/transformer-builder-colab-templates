"""
Tokenizer Validator

Validates that tokenizers meet model requirements and work correctly.
Checks vocabulary size, special tokens, and encode/decode functionality.
"""

from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class TokenizerValidator:
    """
    Validate tokenizer compatibility with model configuration.

    Performs comprehensive checks:
    - Vocabulary size matches expected
    - Special tokens are present
    - Encode/decode round-trip works
    - Token IDs are in valid range

    Example:
        >>> TokenizerValidator.validate(tokenizer, expected_vocab_size=50257)
        ✓ Tokenizer validated (vocab_size=50257)
    """

    @staticmethod
    def validate(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, 'CharacterLevelTokenizer'],
                 expected_vocab_size: int,
                 strict: bool = True) -> bool:
        """
        Validate tokenizer meets requirements.

        Args:
            tokenizer: Tokenizer to validate
            expected_vocab_size: Expected vocabulary size
            strict: If True, raise exception on validation failure. If False, return bool.

        Returns:
            True if validation passes, False otherwise (when strict=False)

        Raises:
            ValueError: If validation fails and strict=True
        """
        errors = []

        # Check 1: Vocabulary size
        try:
            actual_vocab_size = TokenizerValidator._get_vocab_size(tokenizer)

            if actual_vocab_size != expected_vocab_size:
                errors.append(
                    f"Vocab size mismatch: expected {expected_vocab_size}, got {actual_vocab_size}"
                )
        except Exception as e:
            errors.append(f"Could not determine vocab size: {e}")

        # Check 2: Special tokens
        special_token_errors = TokenizerValidator._validate_special_tokens(tokenizer)
        errors.extend(special_token_errors)

        # Check 3: Encode/decode functionality
        encode_decode_errors = TokenizerValidator._validate_encode_decode(tokenizer)
        errors.extend(encode_decode_errors)

        # Check 4: Token ID range
        range_errors = TokenizerValidator._validate_token_range(tokenizer, expected_vocab_size)
        errors.extend(range_errors)

        # Report results
        if errors:
            error_message = "\n".join(f"  ❌ {error}" for error in errors)
            if strict:
                raise ValueError(f"Tokenizer validation failed:\n{error_message}")
            else:
                print(f"⚠️  Tokenizer validation warnings:\n{error_message}")
                return False
        else:
            actual_size = TokenizerValidator._get_vocab_size(tokenizer)
            print(f"✓ Tokenizer validated (vocab_size={actual_size:,})")

            # Print diagnostic info
            TokenizerValidator._print_diagnostics(tokenizer)

            return True

    @staticmethod
    def _get_vocab_size(tokenizer) -> int:
        """Get vocabulary size from tokenizer."""
        # Try different methods to get vocab size
        if hasattr(tokenizer, 'vocab_size'):
            return tokenizer.vocab_size
        elif hasattr(tokenizer, 'get_vocab'):
            return len(tokenizer.get_vocab())
        elif hasattr(tokenizer, 'vocab'):
            return len(tokenizer.vocab)
        elif hasattr(tokenizer, '__len__'):
            return len(tokenizer)
        else:
            raise AttributeError("Could not determine tokenizer vocab size")

    @staticmethod
    def _validate_special_tokens(tokenizer) -> list:
        """Validate special tokens are present."""
        errors = []

        required_tokens = {
            'pad_token': 'Padding token',
            'unk_token': 'Unknown token',
        }

        recommended_tokens = {
            'bos_token': 'Beginning-of-sequence token',
            'eos_token': 'End-of-sequence token',
        }

        # Check required tokens
        for attr_name, description in required_tokens.items():
            if not hasattr(tokenizer, attr_name) or getattr(tokenizer, attr_name) is None:
                errors.append(f"Missing required special token: {description} ({attr_name})")

        # Check recommended tokens (warnings, not errors)
        for attr_name, description in recommended_tokens.items():
            if not hasattr(tokenizer, attr_name) or getattr(tokenizer, attr_name) is None:
                # Just a warning, don't add to errors
                pass

        return errors

    @staticmethod
    def _validate_encode_decode(tokenizer) -> list:
        """Validate encode/decode functionality."""
        errors = []

        test_cases = [
            "Hello world!",
            "The quick brown fox jumps over the lazy dog.",
            "Testing 123... αβγ 中文",
            "",  # Empty string
        ]

        for test_text in test_cases:
            try:
                # Encode
                if hasattr(tokenizer, 'encode'):
                    encoded = tokenizer.encode(test_text)
                    if hasattr(encoded, 'input_ids'):
                        encoded = encoded['input_ids']
                elif callable(tokenizer):
                    result = tokenizer(test_text)
                    encoded = result['input_ids']
                else:
                    errors.append("Tokenizer has no encode method")
                    break

                # Decode
                if hasattr(tokenizer, 'decode'):
                    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                else:
                    errors.append("Tokenizer has no decode method")
                    break

                # For non-empty strings, check if decode preserves some content
                if test_text and not decoded:
                    errors.append(f"Decode returned empty for non-empty input: '{test_text}'")

            except Exception as e:
                errors.append(f"Encode/decode failed for '{test_text}': {e}")

        return errors

    @staticmethod
    def _validate_token_range(tokenizer, expected_vocab_size: int) -> list:
        """Validate token IDs are in valid range."""
        errors = []

        test_text = "Sample text for range validation"

        try:
            # Encode text
            if hasattr(tokenizer, 'encode'):
                encoded = tokenizer.encode(test_text)
                if hasattr(encoded, 'input_ids'):
                    token_ids = encoded['input_ids']
                    if hasattr(token_ids, 'tolist'):
                        token_ids = token_ids.tolist()
                else:
                    token_ids = encoded if isinstance(encoded, list) else [encoded]
            elif callable(tokenizer):
                result = tokenizer(test_text)
                token_ids = result['input_ids']
                if hasattr(token_ids, 'tolist'):
                    token_ids = token_ids.tolist()
            else:
                return []  # Skip if can't encode

            # Check all token IDs are in valid range
            for token_id in token_ids:
                if not isinstance(token_id, int):
                    continue

                if token_id < 0:
                    errors.append(f"Found negative token ID: {token_id}")
                elif token_id >= expected_vocab_size:
                    errors.append(
                        f"Token ID {token_id} exceeds vocab_size {expected_vocab_size}"
                    )

        except Exception as e:
            errors.append(f"Token range validation failed: {e}")

        return errors

    @staticmethod
    def _print_diagnostics(tokenizer):
        """Print diagnostic information about tokenizer."""
        # Get special tokens
        special_tokens = []
        for attr in ['pad_token', 'unk_token', 'bos_token', 'eos_token']:
            if hasattr(tokenizer, attr):
                token = getattr(tokenizer, attr)
                if token is not None:
                    token_id = getattr(tokenizer, f"{attr}_id", "?")
                    special_tokens.append(f"{attr}='{token}' (id={token_id})")

        if special_tokens:
            print(f"  Special tokens: {', '.join(special_tokens)}")

        # Test encode/decode
        test_text = "Hello world!"
        try:
            if hasattr(tokenizer, 'encode'):
                encoded = tokenizer.encode(test_text)
                if hasattr(encoded, 'input_ids'):
                    token_ids = encoded['input_ids']
                else:
                    token_ids = encoded
            elif callable(tokenizer):
                result = tokenizer(test_text)
                token_ids = result['input_ids']
            else:
                return

            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()

            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

            print(f"  Test encode: '{test_text}' → {len(token_ids)} tokens")
            print(f"  Test decode: '{decoded}'")

        except Exception as e:
            print(f"  ⚠️  Diagnostic test failed: {e}")

    @staticmethod
    def quick_validate(tokenizer, expected_vocab_size: int) -> bool:
        """
        Quick validation without exceptions.

        Args:
            tokenizer: Tokenizer to validate
            expected_vocab_size: Expected vocabulary size

        Returns:
            True if validation passes, False otherwise
        """
        try:
            return TokenizerValidator.validate(tokenizer, expected_vocab_size, strict=False)
        except Exception:
            return False
