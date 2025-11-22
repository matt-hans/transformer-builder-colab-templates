"""
Universal data quality utilities for robust training.

Provides dataset-agnostic filtering and validation that works with:
- HuggingFace datasets (WikiText, C4, Common Crawl, etc.)
- PyTorch datasets
- Custom user datasets
- Any data source with tokenized sequences
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataQualityFilter:
    """
    Universal sequence filter for causal LM, classification, and other tasks.

    Handles common data quality issues across all datasets:
    - Empty sequences (0 tokens)
    - Single-token sequences (insufficient for causal LM)
    - Padding-only sequences
    - Malformed data
    """

    def __init__(self, min_seq_len: int = 2, field_name: str = 'input_ids'):
        """
        Initialize data quality filter.

        Args:
            min_seq_len: Minimum sequence length (task-specific)
                - Causal LM: 2 (token shifting requirement)
                - Classification: 1 (single token valid)
            field_name: Field containing tokenized sequence (default: 'input_ids')
        """
        self.min_seq_len = min_seq_len
        self.field_name = field_name

    def __call__(self, example: Dict[str, Any]) -> bool:
        """
        Filter function for HuggingFace datasets.filter().

        Returns:
            True if example is valid, False if should be filtered
        """
        sequence = example.get(self.field_name, [])
        return len(sequence) >= self.min_seq_len

    @staticmethod
    def filter_dataset(
        dataset: Any,
        min_seq_len: int,
        field_name: str = 'input_ids',
        desc: Optional[str] = None
    ) -> Any:
        """
        Filter dataset to remove short sequences.

        Works with:
        - HuggingFace datasets (uses .filter())
        - PyTorch datasets (returns new filtered list)
        - Any iterable dataset

        Args:
            dataset: Dataset to filter
            min_seq_len: Minimum sequence length
            field_name: Field containing sequences
            desc: Description for progress bar

        Returns:
            Filtered dataset (same type as input)
        """
        original_size = len(dataset)

        # HuggingFace Dataset
        if hasattr(dataset, 'filter'):
            filter_fn = DataQualityFilter(min_seq_len, field_name)
            filtered = dataset.filter(filter_fn, desc=desc or "Filtering sequences")

        # PyTorch Dataset or list
        elif isinstance(dataset, (list, tuple)):
            filter_fn = DataQualityFilter(min_seq_len, field_name)
            filtered = [ex for ex in dataset if filter_fn(ex)]

        # Other iterables
        else:
            filter_fn = DataQualityFilter(min_seq_len, field_name)
            filtered = [ex for ex in dataset if filter_fn(ex)]

        # Log statistics
        filtered_size = len(filtered)
        removed = original_size - filtered_size
        filter_rate = removed / original_size if original_size > 0 else 0

        logger.info(
            f"Data Quality Filtering: {original_size} → {filtered_size} "
            f"(-{removed} sequences, {filter_rate:.1%} filtered)"
        )

        # Warn if high filter rate (may indicate data issue)
        if filter_rate > 0.20:
            logger.warning(
                f"⚠️  High filter rate ({filter_rate:.1%}). "
                f"This may indicate data quality issues in the source dataset."
            )

        return filtered


def get_filter_for_task(task_type: str) -> DataQualityFilter:
    """
    Get appropriate data quality filter for a task type.

    Args:
        task_type: Task type ('lm', 'classification', etc.)

    Returns:
        DataQualityFilter configured for the task
    """
    # Map task types to minimum sequence lengths
    TASK_MIN_SEQ_LEN = {
        'lm': 2,
        'causal_lm': 2,
        'language_modeling': 2,
        'seq2seq': 2,
        'classification': 1,
        'text_classification': 1,
    }

    min_seq_len = TASK_MIN_SEQ_LEN.get(task_type, 1)
    return DataQualityFilter(min_seq_len=min_seq_len)


# Convenience function for notebook usage
def filter_short_sequences(dataset, min_length=2, field='input_ids', verbose=True):
    """
    Quick utility to filter short sequences from any dataset.

    Example:
        >>> from utils.training.data_quality import filter_short_sequences
        >>> train_data = filter_short_sequences(train_data, min_length=2)
        >>> val_data = filter_short_sequences(val_data, min_length=2)

    Args:
        dataset: HuggingFace dataset, PyTorch dataset, or list
        min_length: Minimum sequence length to keep
        field: Field name containing tokenized sequences
        verbose: Print filtering statistics

    Returns:
        Filtered dataset
    """
    return DataQualityFilter.filter_dataset(
        dataset=dataset,
        min_seq_len=min_length,
        field_name=field,
        desc="Filtering short sequences" if verbose else None
    )
