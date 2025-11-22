"""Shared constants for training pipeline.

This module provides a single source of truth for task-specific configuration,
eliminating duplication between data_collator.py and data_quality.py.
"""

# Task-specific minimum sequence lengths
# These define the minimum number of tokens required for each task type.
# Causal LM requires 2 tokens minimum for token shifting (input[:-1] → target[1:])
TASK_MIN_SEQ_LEN = {
    'lm': 2,                    # Causal LM (token shifting)
    'causal_lm': 2,             # Alias for causal LM
    'language_modeling': 2,      # Legacy alias (backward compatibility)
    'seq2seq': 2,               # Encoder-decoder (also needs shifting)
    'classification': 1,         # Classification (single token → single prediction)
    'text_classification': 1,    # Alias for classification
    'vision_classification': 0,  # Vision (no text sequences)
    'vision_multilabel': 0,      # Vision (no text sequences)
}

# Dataset-level validation thresholds
# These apply to the entire dataset during preprocessing, NOT individual batches
MAX_FILTER_RATE_STRICT = 0.05       # 5% - for clean production datasets
MAX_FILTER_RATE_PERMISSIVE = 0.20   # 20% - for datasets with known issues (WikiText has 15-25% empty lines)

# Batch-level thresholds (DEPRECATED)
# Note: Statistically invalid for batch_size < 10 due to high variance
# Will be removed in v5.0 - use dataset-level validation instead
BATCH_FILTER_THRESHOLD = 0.10  # DEPRECATED: Only kept for backward compatibility
