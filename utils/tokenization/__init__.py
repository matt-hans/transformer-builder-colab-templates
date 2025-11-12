"""
Adaptive tokenization system supporting any vocabulary size.

Implements 4-tier strategy:
1. Pretrained tokenizer matching (for known vocab sizes)
2. Custom BPE training (for medium-sized vocabs with sufficient data)
3. Character-level tokenization (fallback for any vocab size)
4. User-provided tokenizer upload (optional)
"""

# Will be uncommented as implementations are added:
# from .adaptive_tokenizer import AdaptiveTokenizer
# from .bpe_trainer import FastBPETrainer
# from .character_tokenizer import CharacterLevelTokenizer
# from .validator import TokenizerValidator
# from .data_module import AdaptiveTokenizerDataModule

__all__ = [
    # 'AdaptiveTokenizer',
    # 'FastBPETrainer',
    # 'CharacterLevelTokenizer',
    # 'TokenizerValidator',
    # 'AdaptiveTokenizerDataModule',
]
