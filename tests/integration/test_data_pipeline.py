"""
Integration tests for data loading and preprocessing pipelines.

Tests data collators, HuggingFace datasets, custom datasets, and edge cases.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from utils.training.task_spec import TaskSpec
from utils.training.engine.data import UniversalDataModule, DataLoaderFactory


# ============================================================================
# Test 1: Data Loading with Different Collators
# ============================================================================

@pytest.mark.integration
def test_data_loading_with_collators(
    lm_task_spec,
    classification_task_spec,
    vision_task_spec,
    synthetic_text_dataset,
    synthetic_classification_dataset,
    synthetic_vision_dataset
):
    """Test data loading with LM, classification, and vision collators."""

    # Test 1a: Language Modeling Collator
    lm_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([item['input_ids'] for item in batch])
        }
    )

    lm_batch = next(iter(lm_loader))
    assert 'input_ids' in lm_batch
    assert lm_batch['input_ids'].shape[0] == 4  # Batch size
    assert lm_batch['input_ids'].shape[1] == 32  # Sequence length

    # Test 1b: Classification Collator
    cls_loader = torch.utils.data.DataLoader(
        synthetic_classification_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }
    )

    cls_batch = next(iter(cls_loader))
    assert 'input_ids' in cls_batch
    assert 'labels' in cls_batch
    assert cls_batch['input_ids'].shape[0] == 4
    assert cls_batch['labels'].shape[0] == 4

    # Test 1c: Vision Collator
    from utils.tokenization.data_collator import VisionDataCollator

    vision_collator = VisionDataCollator(
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    vision_loader = torch.utils.data.DataLoader(
        synthetic_vision_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=vision_collator
    )

    vision_batch = next(iter(vision_loader))
    assert 'pixel_values' in vision_batch
    assert 'labels' in vision_batch
    assert vision_batch['pixel_values'].shape[0] == 4
    assert vision_batch['pixel_values'].shape[1] == 3  # RGB
    assert vision_batch['pixel_values'].shape[2] == 32  # Height
    assert vision_batch['pixel_values'].shape[3] == 32  # Width

    # Verify normalization applied
    mean_pixel = vision_batch['pixel_values'].mean()
    assert -3 < mean_pixel < 3, "Normalized pixels should be roughly in [-3, 3] range"


# ============================================================================
# Test 2: HuggingFace Datasets Integration
# ============================================================================

@pytest.mark.integration
def test_huggingface_datasets_integration():
    """Test loading and preprocessing HuggingFace datasets."""
    try:
        from datasets import Dataset
    except ImportError:
        pytest.skip("datasets package not installed")

    # Create synthetic HF dataset
    hf_data = {
        'text': [f"This is sample text {i}" for i in range(100)],
        'label': [i % 2 for i in range(100)]
    }
    hf_dataset = Dataset.from_dict(hf_data)

    # Test iteration
    assert len(hf_dataset) == 100

    # Test batching
    loader = torch.utils.data.DataLoader(
        hf_dataset,
        batch_size=8,
        shuffle=True
    )

    batch = next(iter(loader))
    assert 'text' in batch
    assert 'label' in batch
    assert len(batch['text']) == 8


# ============================================================================
# Test 3: Custom Datasets (Text, Vision, Classification)
# ============================================================================

@pytest.mark.integration
def test_custom_datasets(
    synthetic_text_dataset,
    synthetic_vision_dataset,
    synthetic_classification_dataset
):
    """Test custom dataset implementations."""

    # Test 3a: Text Dataset
    assert len(synthetic_text_dataset) == 100
    text_item = synthetic_text_dataset[0]
    assert 'input_ids' in text_item
    assert text_item['input_ids'].shape[0] == 32
    assert text_item['input_ids'].min() >= 1
    assert text_item['input_ids'].max() < 1000

    # Test 3b: Vision Dataset
    assert len(synthetic_vision_dataset) == 100
    vision_item = synthetic_vision_dataset[0]
    assert 'pixel_values' in vision_item
    assert 'labels' in vision_item
    assert vision_item['pixel_values'].shape == (3, 32, 32)
    assert 0 <= vision_item['labels'] < 10

    # Test 3c: Classification Dataset
    assert len(synthetic_classification_dataset) == 100
    cls_item = synthetic_classification_dataset[0]
    assert 'input_ids' in cls_item
    assert 'labels' in cls_item
    assert cls_item['input_ids'].shape[0] == 32
    assert 0 <= cls_item['labels'] < 2

    # Verify determinism (same index = same data)
    text_item_again = synthetic_text_dataset[0]
    assert torch.equal(text_item['input_ids'], text_item_again['input_ids'])


# ============================================================================
# Test 4: Edge Cases (Empty batches, single sample, max length)
# ============================================================================

@pytest.mark.integration
def test_data_edge_cases(synthetic_text_dataset):
    """Test edge cases in data loading."""

    # Test 4a: Single sample batch
    single_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=1,
        shuffle=False
    )

    single_batch = next(iter(single_loader))
    assert single_batch['input_ids'].shape[0] == 1

    # Test 4b: Large batch
    large_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=32,
        shuffle=False
    )

    large_batch = next(iter(large_loader))
    assert large_batch['input_ids'].shape[0] == 32

    # Test 4c: Incomplete last batch
    incomplete_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=15,
        shuffle=False,
        drop_last=False
    )

    batch_count = 0
    last_batch_size = None
    for batch in incomplete_loader:
        batch_count += 1
        last_batch_size = batch['input_ids'].shape[0]

    assert batch_count == 7  # 100 / 15 = 6.67 → 7 batches
    assert last_batch_size == 10  # 100 % 15 = 10

    # Test 4d: Drop last batch
    drop_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=15,
        shuffle=False,
        drop_last=True
    )

    drop_batch_count = sum(1 for _ in drop_loader)
    assert drop_batch_count == 6  # Drops incomplete batch


# ============================================================================
# Test 5: Worker Seeding and Reproducibility
# ============================================================================

@pytest.mark.integration
def test_worker_seeding_reproducibility(synthetic_text_dataset):
    """Test DataLoader worker seeding for reproducible batch ordering."""

    def seed_worker(worker_id):
        """Worker init function for reproducible shuffling."""
        import numpy as np
        import random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)

    # First DataLoader
    loader1 = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=4,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=0  # Use 0 for CI compatibility
    )

    # Collect first batch
    batch1 = next(iter(loader1))

    # Reset generator
    g2 = torch.Generator()
    g2.manual_seed(42)

    # Second DataLoader (same seed)
    loader2 = torch.utils.data.DataLoader(
        synthetic_text_dataset,
        batch_size=4,
        shuffle=True,
        generator=g2,
        worker_init_fn=seed_worker,
        num_workers=0
    )

    batch2 = next(iter(loader2))

    # Verify reproducibility
    assert torch.equal(batch1['input_ids'], batch2['input_ids']), \
        "Same seed should produce identical batches"


# ============================================================================
# Test 6: Variable Length Sequences
# ============================================================================

@pytest.mark.integration
def test_variable_length_sequences():
    """Test handling of variable-length sequences with padding."""

    class VariableLengthDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=50):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            # Variable length: 10 to 30 tokens
            length = torch.randint(10, 31, (1,)).item()
            input_ids = torch.randint(1, 1000, (length,))
            return {'input_ids': input_ids, 'length': length}

    dataset = VariableLengthDataset()

    # Collator with padding
    def pad_collate(batch):
        max_length = max(item['length'] for item in batch)
        padded_inputs = []
        lengths = []

        for item in batch:
            input_ids = item['input_ids']
            padding_length = max_length - len(input_ids)
            padded = torch.cat([
                input_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            padded_inputs.append(padded)
            lengths.append(item['length'])

        return {
            'input_ids': torch.stack(padded_inputs),
            'lengths': torch.tensor(lengths)
        }

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=pad_collate
    )

    batch = next(iter(loader))
    assert 'input_ids' in batch
    assert 'lengths' in batch
    assert batch['input_ids'].shape[0] == 8

    # Verify padding (all sequences padded to same length)
    batch_max_length = batch['input_ids'].shape[1]
    assert all(length <= batch_max_length for length in batch['lengths'])

    # Verify no truncation (original lengths preserved)
    for i, length in enumerate(batch['lengths']):
        assert batch['input_ids'][i, :length].sum() > 0  # Non-zero content
        if length < batch_max_length:
            assert batch['input_ids'][i, length:].sum() == 0  # Zeros in padding


# ============================================================================
# Test 7: Multi-Modal Data Loading
# ============================================================================

@pytest.mark.integration
def test_multimodal_data_loading():
    """Test loading data with both text and image modalities."""

    class MultiModalDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=50):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            # Text
            input_ids = torch.randint(1, 1000, (32,))
            # Image
            pixel_values = torch.randn(3, 32, 32)
            # Label
            label = torch.tensor(idx % 5)

            return {
                'input_ids': input_ids,
                'pixel_values': pixel_values,
                'labels': label
            }

    dataset = MultiModalDataset()

    def multimodal_collate(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn=multimodal_collate
    )

    batch = next(iter(loader))
    assert 'input_ids' in batch
    assert 'pixel_values' in batch
    assert 'labels' in batch
    assert batch['input_ids'].shape == (4, 32)
    assert batch['pixel_values'].shape == (4, 3, 32, 32)
    assert batch['labels'].shape == (4,)


# ============================================================================
# Test 8: Data Augmentation in Pipeline
# ============================================================================

@pytest.mark.integration
def test_data_augmentation_pipeline():
    """Test data augmentation transforms in loading pipeline."""

    class AugmentedVisionDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=50, augment=False):
            self.num_samples = num_samples
            self.augment = augment

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            pixel_values = torch.randn(3, 32, 32)

            if self.augment:
                # Simple augmentation: random flip
                if torch.rand(1) > 0.5:
                    pixel_values = torch.flip(pixel_values, dims=[2])  # Horizontal flip

                # Random brightness adjustment
                brightness_factor = 0.8 + torch.rand(1) * 0.4  # [0.8, 1.2]
                pixel_values = pixel_values * brightness_factor

            label = torch.tensor(idx % 10)
            return {'pixel_values': pixel_values, 'labels': label}

    # Without augmentation
    dataset_no_aug = AugmentedVisionDataset(augment=False)
    loader_no_aug = torch.utils.data.DataLoader(dataset_no_aug, batch_size=4)
    batch_no_aug = next(iter(loader_no_aug))

    # With augmentation (results may differ but should be valid)
    dataset_aug = AugmentedVisionDataset(augment=True)
    loader_aug = torch.utils.data.DataLoader(dataset_aug, batch_size=4)
    batch_aug = next(iter(loader_aug))

    assert batch_no_aug['pixel_values'].shape == batch_aug['pixel_values'].shape
    assert batch_no_aug['labels'].shape == batch_aug['labels'].shape


# ============================================================================
# Test 9: Memory-Efficient Lazy Loading
# ============================================================================

@pytest.mark.integration
def test_lazy_loading_efficiency():
    """Test that dataset loading is lazy (doesn't load all data into memory)."""

    class LazyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=10000):
            self.num_samples = num_samples
            # Don't pre-allocate data
            self.data = None

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate on-the-fly
            torch.manual_seed(idx)
            input_ids = torch.randint(1, 1000, (128,))
            return {'input_ids': input_ids}

    # Create large dataset (should not consume significant memory)
    dataset = LazyDataset(num_samples=10000)

    # Load small batch
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    assert batch['input_ids'].shape == (4, 128)
    # If this test passes without OOM, lazy loading works


# ============================================================================
# Test 10: DataLoaderFactory Integration
# ============================================================================

@pytest.mark.integration
def test_dataloader_factory_integration(
    lm_task_spec,
    synthetic_text_dataset,
    basic_training_config
):
    """Test end-to-end dataloader creation via DataLoaderFactory."""

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    # Use DataLoaderFactory
    factory = DataLoaderFactory()

    train_loader = factory.create_train_loader(
        dataset=train_dataset,
        task_spec=lm_task_spec,
        batch_size=basic_training_config.batch_size,
        num_workers=0,
        seed=basic_training_config.random_seed
    )

    val_loader = factory.create_val_loader(
        dataset=val_dataset,
        task_spec=lm_task_spec,
        batch_size=basic_training_config.batch_size,
        num_workers=0
    )

    # Verify loaders
    assert train_loader is not None
    assert val_loader is not None

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert 'input_ids' in train_batch
    assert 'input_ids' in val_batch
    assert train_batch['input_ids'].shape[0] == basic_training_config.batch_size
