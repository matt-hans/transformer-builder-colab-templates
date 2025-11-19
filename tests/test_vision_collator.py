"""Tests for VisionDataCollator.

Tests the vision data collator implementation including:
- Unit tests for batching and normalization
- Integration tests for auto-selection based on TaskSpec
- Edge case tests for grayscale images and variable sizes
- Performance validation against torchvision.transforms.Normalize
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any

from utils.tokenization.data_collator import VisionDataCollator
from utils.tokenization.data_module import _get_collator
from utils.training.task_spec import TaskSpec


# ============================================================================
# Unit Tests
# ============================================================================

def test_vision_collator_batching():
    """Test that VisionDataCollator correctly stacks pixel_values."""
    collator = VisionDataCollator(normalize=False)

    # Create sample batch
    batch = [
        {'pixel_values': torch.randn(3, 224, 224), 'labels': 0},
        {'pixel_values': torch.randn(3, 224, 224), 'labels': 1},
        {'pixel_values': torch.randn(3, 224, 224), 'labels': 2},
        {'pixel_values': torch.randn(3, 224, 224), 'labels': 0},
    ]

    result = collator(batch)

    # Check pixel_values shape
    assert 'pixel_values' in result
    assert result['pixel_values'].shape == (4, 3, 224, 224)

    # Check labels
    assert 'labels' in result
    assert result['labels'].shape == (4,)
    assert torch.equal(result['labels'], torch.tensor([0, 1, 2, 0]))


def test_vision_collator_normalization():
    """Test normalization matches torchvision.transforms.Normalize."""
    # Create collator with ImageNet stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    collator = VisionDataCollator(normalize=True, mean=mean, std=std)

    # Create deterministic pixel values - shape (C, H, W)
    pixel_values = torch.tensor([
        [[0.5, 0.6], [0.7, 0.8]],  # Red channel
        [[0.4, 0.5], [0.6, 0.7]],  # Green channel
        [[0.3, 0.4], [0.5, 0.6]],  # Blue channel
    ], dtype=torch.float32)

    batch = [{'pixel_values': pixel_values, 'labels': 0}]
    result = collator(batch)

    # Expected normalization: (pixel - mean) / std
    expected = torch.zeros_like(pixel_values)
    for c in range(3):
        expected[c] = (pixel_values[c] - mean[c]) / std[c]

    # Compare with tolerance
    torch.testing.assert_close(
        result['pixel_values'][0],
        expected,
        rtol=1e-5,
        atol=1e-6
    )


def test_vision_collator_normalization_vs_torchvision():
    """Compare VisionDataCollator normalization to torchvision.transforms.Normalize."""
    try:
        from torchvision import transforms
    except ImportError:
        pytest.skip("torchvision not installed")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Our collator
    collator = VisionDataCollator(normalize=True, mean=mean, std=std)

    # Torchvision normalize
    tv_normalize = transforms.Normalize(mean=mean, std=std)

    # Create random pixel values (batch of 4 images)
    pixel_values_list = [torch.rand(3, 64, 64) for _ in range(4)]
    batch = [{'pixel_values': pv, 'labels': i % 2} for i, pv in enumerate(pixel_values_list)]

    # Apply our collator
    result = collator(batch)

    # Apply torchvision normalize to each image and stack
    tv_normalized = torch.stack([tv_normalize(pv) for pv in pixel_values_list])

    # Compare results
    torch.testing.assert_close(
        result['pixel_values'],
        tv_normalized,
        rtol=1e-5,
        atol=1e-6,
        msg="VisionDataCollator normalization should match torchvision.transforms.Normalize"
    )


def test_vision_collator_without_labels():
    """Test VisionDataCollator in inference mode (no labels)."""
    collator = VisionDataCollator(normalize=True)

    # Batch without labels
    batch = [
        {'pixel_values': torch.randn(3, 224, 224)},
        {'pixel_values': torch.randn(3, 224, 224)},
    ]

    result = collator(batch)

    # Should have pixel_values but no labels
    assert 'pixel_values' in result
    assert result['pixel_values'].shape == (2, 3, 224, 224)
    assert 'labels' not in result


def test_vision_collator_grayscale_images():
    """Test VisionDataCollator with 1-channel grayscale images."""
    # Use first channel of ImageNet mean/std for grayscale
    collator = VisionDataCollator(normalize=True)

    # Grayscale images (1 channel)
    batch = [
        {'pixel_values': torch.randn(1, 28, 28), 'labels': 0},
        {'pixel_values': torch.randn(1, 28, 28), 'labels': 1},
    ]

    result = collator(batch)

    # Check shape
    assert result['pixel_values'].shape == (2, 1, 28, 28)
    assert result['labels'].shape == (2,)

    # Verify normalization applied (values should be different from input)
    input_tensor = torch.stack([item['pixel_values'] for item in batch])
    assert not torch.equal(result['pixel_values'], input_tensor)


def test_vision_collator_custom_normalization():
    """Test VisionDataCollator with custom mean/std (e.g., CIFAR-10)."""
    # CIFAR-10 normalization
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    collator = VisionDataCollator(normalize=True, mean=mean, std=std)

    pixel_values = torch.ones(3, 32, 32) * 0.5  # All pixels = 0.5
    batch = [{'pixel_values': pixel_values, 'labels': 0}]

    result = collator(batch)

    # With mean=0.5, std=0.5, normalized should be (0.5 - 0.5) / 0.5 = 0
    expected = torch.zeros(1, 3, 32, 32)
    torch.testing.assert_close(result['pixel_values'], expected, rtol=1e-5, atol=1e-6)


def test_vision_collator_disable_normalization():
    """Test VisionDataCollator with normalization disabled."""
    collator = VisionDataCollator(normalize=False)

    pixel_values = torch.randn(3, 64, 64)
    batch = [{'pixel_values': pixel_values, 'labels': 0}]

    result = collator(batch)

    # Should be unchanged
    assert torch.equal(result['pixel_values'][0], pixel_values)


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_vision_collator_variable_sizes_raises_error():
    """Test that VisionDataCollator raises error for inconsistent shapes."""
    collator = VisionDataCollator(normalize=False)

    # Batch with mismatched sizes
    batch = [
        {'pixel_values': torch.randn(3, 224, 224), 'labels': 0},
        {'pixel_values': torch.randn(3, 128, 128), 'labels': 1},  # Different size!
    ]

    with pytest.raises(ValueError, match="Inconsistent pixel_values shapes"):
        collator(batch)


def test_vision_collator_mean_std_length_mismatch():
    """Test that VisionDataCollator raises error if mean/std lengths differ."""
    with pytest.raises(ValueError, match="mean and std must have same length"):
        VisionDataCollator(
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5)  # Missing one value!
        )


def test_vision_collator_channel_mismatch():
    """Test error when number of channels doesn't match mean/std length."""
    # Collator expects 3 channels (RGB)
    collator = VisionDataCollator(
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    # Provide 4-channel image
    batch = [{'pixel_values': torch.randn(4, 64, 64), 'labels': 0}]

    with pytest.raises(ValueError, match="Number of channels.*doesn't match"):
        collator(batch)


def test_vision_collator_single_sample_batch():
    """Test VisionDataCollator with batch size of 1."""
    collator = VisionDataCollator(normalize=True)

    batch = [{'pixel_values': torch.randn(3, 64, 64), 'labels': 5}]
    result = collator(batch)

    assert result['pixel_values'].shape == (1, 3, 64, 64)
    assert result['labels'].shape == (1,)
    assert result['labels'][0] == 5


# ============================================================================
# Integration Tests
# ============================================================================

def test_data_module_auto_selects_vision_collator():
    """Test that _get_collator auto-selects VisionDataCollator for vision tasks."""
    task_spec = TaskSpec(
        name="vision_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"],
        modality="vision",  # Key field for auto-selection
        input_schema={"image_size": [3, 224, 224]},
        output_schema={"num_classes": 10},
        preprocessing_config={
            "normalize": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    )

    collator = _get_collator(
        task_spec=task_spec,
        use_dynamic_collator=True
    )

    # Should be VisionDataCollator
    assert isinstance(collator, VisionDataCollator)
    assert collator.normalize is True
    # mean/std can be list or tuple depending on source (JSON vs Python)
    assert tuple(collator.mean) == (0.485, 0.456, 0.406)
    assert tuple(collator.std) == (0.229, 0.224, 0.225)


def test_data_module_auto_selects_text_collator():
    """Test that _get_collator auto-selects LanguageModelingDataCollator for text tasks."""
    from utils.tokenization.data_collator import LanguageModelingDataCollator
    from types import SimpleNamespace

    task_spec = TaskSpec(
        name="lm_test",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids", "attention_mask"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "perplexity"],
        modality="text",  # Text modality
        input_schema={"max_seq_len": 128, "vocab_size": 50257},
        output_schema={"vocab_size": 50257}
    )

    # Mock tokenizer
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        padding_side='right',
        pad=lambda *args, **kwargs: {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
    )

    collator = _get_collator(
        task_spec=task_spec,
        tokenizer=tokenizer,
        use_dynamic_collator=True
    )

    # Should be LanguageModelingDataCollator
    assert isinstance(collator, LanguageModelingDataCollator)


def test_data_module_fallback_to_default_without_task_spec():
    """Test that _get_collator falls back to default_data_collator without task_spec."""
    from transformers import default_data_collator

    collator = _get_collator(
        task_spec=None,
        use_dynamic_collator=True
    )

    # Should be default collator
    assert collator == default_data_collator


def test_data_module_fallback_to_default_when_disabled():
    """Test that _get_collator uses default when use_dynamic_collator=False."""
    from transformers import default_data_collator

    task_spec = TaskSpec(
        name="vision_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss"],
        modality="vision"
    )

    collator = _get_collator(
        task_spec=task_spec,
        use_dynamic_collator=False  # Explicitly disabled
    )

    # Should be default collator
    assert collator == default_data_collator


def test_data_module_unsupported_modality():
    """Test that _get_collator raises error for unsupported modality."""
    task_spec = TaskSpec(
        name="audio_test",
        task_type="audio_classification",
        model_family="encoder_only",
        input_fields=["audio_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss"],
        modality="audio",  # Unsupported modality
    )

    with pytest.raises(ValueError, match="Unsupported modality: audio"):
        _get_collator(
            task_spec=task_spec,
            use_dynamic_collator=True
        )


def test_vision_collator_with_preprocessing_config_defaults():
    """Test VisionDataCollator defaults to ImageNet when preprocessing_config is empty."""
    task_spec = TaskSpec(
        name="vision_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss"],
        modality="vision",
        preprocessing_config={}  # Empty config
    )

    collator = _get_collator(
        task_spec=task_spec,
        use_dynamic_collator=True
    )

    # Should use ImageNet defaults
    assert isinstance(collator, VisionDataCollator)
    assert collator.mean == (0.485, 0.456, 0.406)
    assert collator.std == (0.229, 0.224, 0.225)


def test_vision_collator_with_custom_preprocessing_config():
    """Test VisionDataCollator with custom preprocessing config from TaskSpec."""
    task_spec = TaskSpec(
        name="cifar10_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss"],
        modality="vision",
        preprocessing_config={
            "normalize": True,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }
    )

    collator = _get_collator(
        task_spec=task_spec,
        use_dynamic_collator=True
    )

    # Should use custom values
    assert isinstance(collator, VisionDataCollator)
    # mean/std can be list or tuple depending on source (JSON vs Python)
    assert tuple(collator.mean) == (0.5, 0.5, 0.5)
    assert tuple(collator.std) == (0.5, 0.5, 0.5)


# ============================================================================
# Performance / Numerical Validation Tests
# ============================================================================

def test_vision_collator_numerical_stability():
    """Test that normalization is numerically stable with extreme values."""
    collator = VisionDataCollator(
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )

    # Very large values
    batch_large = [{'pixel_values': torch.ones(3, 64, 64) * 1000.0, 'labels': 0}]
    result_large = collator(batch_large)
    assert torch.isfinite(result_large['pixel_values']).all()

    # Very small values
    batch_small = [{'pixel_values': torch.ones(3, 64, 64) * 1e-10, 'labels': 0}]
    result_small = collator(batch_small)
    assert torch.isfinite(result_small['pixel_values']).all()


def test_vision_collator_device_agnostic():
    """Test that VisionDataCollator works on both CPU and GPU tensors."""
    collator = VisionDataCollator(normalize=True)

    # CPU tensors
    batch_cpu = [{'pixel_values': torch.randn(3, 64, 64), 'labels': 0}]
    result_cpu = collator(batch_cpu)
    assert result_cpu['pixel_values'].device.type == 'cpu'

    # GPU tensors (if available)
    if torch.cuda.is_available():
        batch_gpu = [{'pixel_values': torch.randn(3, 64, 64).cuda(), 'labels': 0}]
        result_gpu = collator(batch_gpu)
        assert result_gpu['pixel_values'].device.type == 'cuda'


def test_vision_collator_dtype_preservation():
    """Test that VisionDataCollator preserves dtype."""
    collator = VisionDataCollator(normalize=True)

    # float32
    batch_f32 = [{'pixel_values': torch.randn(3, 64, 64, dtype=torch.float32), 'labels': 0}]
    result_f32 = collator(batch_f32)
    assert result_f32['pixel_values'].dtype == torch.float32

    # float16 (if supported)
    batch_f16 = [{'pixel_values': torch.randn(3, 64, 64, dtype=torch.float16), 'labels': 0}]
    result_f16 = collator(batch_f16)
    assert result_f16['pixel_values'].dtype == torch.float16
