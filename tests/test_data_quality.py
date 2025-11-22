"""Tests for data quality utilities."""

from utils.training.data_quality import get_filter_for_task
from utils.training.constants import TASK_MIN_SEQ_LEN


def test_get_filter_for_task_uses_constants():
    """Verify get_filter_for_task uses centralized constants."""
    for task_type, expected_min_len in TASK_MIN_SEQ_LEN.items():
        filter_fn = get_filter_for_task(task_type)
        assert filter_fn.min_seq_len == expected_min_len, (
            f"Task '{task_type}' should have min_seq_len={expected_min_len}, "
            f"got {filter_fn.min_seq_len}"
        )
