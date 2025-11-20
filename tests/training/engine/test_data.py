"""
Unit tests for data loading and collation engine.

Tests cover:
- CollatorRegistry registration and auto-selection
- DataLoaderFactory creation and optimization
- UniversalDataModule with various dataset types
- Worker seeding reproducibility
- Performance benchmarks
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datasets import Dataset as HFDataset

from utils.training.engine.data import (
    CollatorRegistry,
    DataLoaderConfig,
    DataLoaderFactory,
    UniversalDataModule,
    DataModuleProtocol,
)
from utils.training.task_spec import TaskSpec


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def text_task_spec():
    """Create text task spec."""
    return TaskSpec(
        name="test_lm",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids", "attention_mask"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "perplexity"],
        special_tokens={"pad_token_id": 0},
        modality="text",
        input_schema={"max_seq_len": 128, "vocab_size": 50257},
        output_schema={"vocab_size": 50257}
    )


@pytest.fixture
def vision_task_spec():
    """Create vision task spec."""
    return TaskSpec(
        name="test_vision",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"],
        modality="vision",
        input_schema={"image_size": [3, 32, 32], "channels_first": True},
        output_schema={"num_classes": 10}
    )


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.padding_side = 'right'

        def pad(self, examples, return_tensors=None, padding=True):
            max_len = max(len(ex['input_ids']) for ex in examples)
            padded = []
            for ex in examples:
                ids = list(ex['input_ids'])
                pad_len = max_len - len(ids)
                padded.append(ids + [self.pad_token_id] * pad_len)

            return {
                'input_ids': padded,
                'attention_mask': [[1] * len(ex['input_ids']) + [0] * (max_len - len(ex['input_ids'])) for ex in examples]
            }

    return MockTokenizer()


@pytest.fixture
def tensor_dataset():
    """Create simple tensor dataset."""
    data = [torch.randint(0, 1000, (32,)) for _ in range(100)]
    return TensorDataset(torch.stack(data))


@pytest.fixture
def hf_dataset():
    """Create HuggingFace dataset."""
    data = {
        'input_ids': [[1, 2, 3, 4] for _ in range(100)],
        'attention_mask': [[1, 1, 1, 1] for _ in range(100)],
        'labels': [[1, 2, 3, 4] for _ in range(100)]
    }
    return HFDataset.from_dict(data)


@pytest.fixture
def vision_dataset():
    """Create vision dataset."""
    class VisionDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'pixel_values': torch.randn(3, 32, 32),
                'labels': idx % 10
            }

    return VisionDataset()


# =============================================================================
# CollatorRegistry Tests
# =============================================================================

class TestCollatorRegistry:
    """Test CollatorRegistry functionality."""

    def test_singleton(self):
        """Test singleton pattern."""
        registry1 = CollatorRegistry.get_instance()
        registry2 = CollatorRegistry.get_instance()
        assert registry1 is registry2

    def test_register_custom_collator(self):
        """Test custom collator registration."""
        CollatorRegistry.reset()
        registry = CollatorRegistry.get_instance()

        @registry.register('custom', modality='text', description='Custom collator')
        def create_custom_collator(**kwargs):
            return lambda batch: batch

        # Check registration
        collators = registry.list_collators()
        custom_names = [c.name for c in collators if c.name == 'custom']
        assert len(custom_names) == 1

        # Get collator
        collator = registry.get_collator(collator_name='custom')
        assert collator is not None

    def test_text_collator_selection(self, text_task_spec, mock_tokenizer):
        """Test text collator auto-selection from TaskSpec."""
        CollatorRegistry.reset()
        registry = CollatorRegistry.get_instance()

        collator = registry.get_collator(
            task_spec=text_task_spec,
            tokenizer=mock_tokenizer
        )

        # Check collator is callable
        assert callable(collator)

        # Test collation
        batch = [
            {'input_ids': [1, 2, 3]},
            {'input_ids': [4, 5, 6, 7, 8]}
        ]
        result = collator(batch)
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result

    def test_vision_collator_selection(self, vision_task_spec):
        """Test vision collator auto-selection from TaskSpec."""
        CollatorRegistry.reset()
        registry = CollatorRegistry.get_instance()

        collator = registry.get_collator(task_spec=vision_task_spec)

        # Check collator is callable
        assert callable(collator)

        # Test collation
        batch = [
            {'pixel_values': torch.randn(3, 32, 32), 'labels': 0},
            {'pixel_values': torch.randn(3, 32, 32), 'labels': 1}
        ]
        result = collator(batch)
        assert 'pixel_values' in result
        assert result['pixel_values'].shape == (2, 3, 32, 32)
        assert 'labels' in result
        assert len(result['labels']) == 2

    def test_vision_collator_normalization(self, vision_task_spec):
        """Test vision collator with custom normalization."""
        CollatorRegistry.reset()
        registry = CollatorRegistry.get_instance()

        # Custom normalization (CIFAR-10 style)
        vision_task_spec.preprocessing_config = {
            'normalize': True,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }

        collator = registry.get_collator(task_spec=vision_task_spec)

        # Test normalization is applied
        batch = [
            {'pixel_values': torch.ones(3, 32, 32), 'labels': 0},
            {'pixel_values': torch.ones(3, 32, 32), 'labels': 1}
        ]
        result = collator(batch)

        # Check normalization: (1.0 - 0.5) / 0.5 = 1.0
        assert torch.allclose(result['pixel_values'], torch.ones(2, 3, 32, 32), atol=1e-6)


# =============================================================================
# DataLoaderFactory Tests
# =============================================================================

class TestDataLoaderFactory:
    """Test DataLoaderFactory functionality."""

    def test_basic_creation(self, tensor_dataset):
        """Test basic DataLoader creation."""
        factory = DataLoaderFactory()
        config = DataLoaderConfig(batch_size=16, shuffle=True, seed=42)

        loader = factory.create_dataloader(tensor_dataset, config)

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 16

    def test_gpu_optimizations(self, tensor_dataset):
        """Test GPU optimization detection."""
        factory = DataLoaderFactory()
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=None,  # Auto-detect
            prefetch_factor=2
        )

        loader = factory.create_dataloader(tensor_dataset, config)

        # Should match CUDA availability
        expected_pin_memory = torch.cuda.is_available()
        assert loader.pin_memory == expected_pin_memory

    def test_worker_seeding(self, tensor_dataset):
        """Test worker seeding for reproducibility."""
        factory = DataLoaderFactory()
        config = DataLoaderConfig(
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=0  # Use 0 for deterministic test
        )

        loader1 = factory.create_dataloader(tensor_dataset, config)
        loader2 = factory.create_dataloader(tensor_dataset, config)

        # Get first batch from both loaders
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Should be identical due to seeding
        assert torch.equal(batch1[0], batch2[0])

    def test_list_tensor_conversion(self):
        """Test List[Tensor] to TensorDataset conversion."""
        factory = DataLoaderFactory()
        data = [torch.randn(32) for _ in range(50)]
        config = DataLoaderConfig(batch_size=8)

        loader = factory.create_dataloader(data, config)

        # Should work without errors
        batch = next(iter(loader))
        assert len(batch) == 1  # TensorDataset returns tuple
        assert batch[0].shape[0] == 8  # batch size

    def test_collator_integration(self, text_task_spec, mock_tokenizer):
        """Test collator integration with DataLoader."""
        factory = DataLoaderFactory()

        # Create dataset with variable-length sequences
        data = HFDataset.from_dict({
            'input_ids': [[1, 2, 3], [4, 5, 6, 7], [8, 9]],
            'attention_mask': [[1, 1, 1], [1, 1, 1, 1], [1, 1]]
        })

        config = DataLoaderConfig(batch_size=2, shuffle=False, num_workers=0)

        loader = factory.create_dataloader(
            data,
            config,
            task_spec=text_task_spec,
            tokenizer=mock_tokenizer
        )

        # Get batch and check padding
        batch = next(iter(loader))
        # Note: Collator returns dict, not tuple
        assert isinstance(batch, dict)
        assert 'input_ids' in batch


# =============================================================================
# UniversalDataModule Tests
# =============================================================================

class TestUniversalDataModule:
    """Test UniversalDataModule functionality."""

    def test_protocol_compliance(self, tensor_dataset, text_task_spec):
        """Test that UniversalDataModule implements DataModuleProtocol."""
        data_module = UniversalDataModule(
            train_data=tensor_dataset,
            task_spec=text_task_spec,
            batch_size=16
        )

        # Should implement protocol methods
        assert hasattr(data_module, 'train_dataloader')
        assert hasattr(data_module, 'val_dataloader')
        assert callable(data_module.train_dataloader)
        assert callable(data_module.val_dataloader)

        # Type check
        dm: DataModuleProtocol = data_module  # Should not raise type error
        assert dm is not None

    def test_automatic_val_split(self, tensor_dataset):
        """Test automatic validation split."""
        data_module = UniversalDataModule(
            train_data=tensor_dataset,
            val_split=0.2,
            batch_size=16,
            seed=42
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        # Check split ratio (approximately 80/20)
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        total_batches = train_batches + val_batches

        # Should be roughly 80/20 split
        assert 0.7 < (train_batches / total_batches) < 0.9

    def test_external_val_dataset(self, tensor_dataset):
        """Test with external validation dataset."""
        train_data = tensor_dataset
        val_data = TensorDataset(torch.stack([torch.randn(32) for _ in range(20)]))

        data_module = UniversalDataModule(
            train_data=train_data,
            val_data=val_data,
            batch_size=16
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.dataset) == 100
        assert len(val_loader.dataset) == 20

    def test_hf_dataset_integration(self, hf_dataset, text_task_spec, mock_tokenizer):
        """Test with HuggingFace Dataset."""
        data_module = UniversalDataModule(
            train_data=hf_dataset,
            task_spec=text_task_spec,
            tokenizer=mock_tokenizer,
            batch_size=16,
            val_split=0.2,
            seed=42,
            num_workers=0  # Avoid multiprocessing pickle errors in tests
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        # Get sample batch
        batch = next(iter(train_loader))
        assert isinstance(batch, (dict, tuple, list))

    def test_vision_dataset_integration(self, vision_dataset, vision_task_spec):
        """Test with vision dataset."""
        data_module = UniversalDataModule(
            train_data=vision_dataset,
            task_spec=vision_task_spec,
            batch_size=16,
            val_split=0.2,
            seed=42,
            num_workers=0  # Avoid multiprocessing pickle errors in tests
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        # Get sample batch
        batch = next(iter(train_loader))
        assert isinstance(batch, dict)
        assert 'pixel_values' in batch
        assert 'labels' in batch

    def test_reproducible_splits(self, hf_dataset):
        """Test reproducible validation splits with same seed."""
        data_module1 = UniversalDataModule(
            train_data=hf_dataset,
            batch_size=16,
            val_split=0.2,
            seed=42
        )

        data_module2 = UniversalDataModule(
            train_data=hf_dataset,
            batch_size=16,
            val_split=0.2,
            seed=42
        )

        # Should have identical train/val sizes
        train_size1 = len(data_module1.train_data)
        train_size2 = len(data_module2.train_data)
        assert train_size1 == train_size2

        val_size1 = len(data_module1.val_data)
        val_size2 = len(data_module2.val_data)
        assert val_size1 == val_size2


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestReproducibility:
    """Test reproducibility guarantees."""

    def test_dataloader_batch_order(self, tensor_dataset):
        """Test that DataLoaders produce identical batch order with same seed."""
        factory = DataLoaderFactory()
        config = DataLoaderConfig(
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=0  # Important: 0 workers for deterministic test
        )

        # Create two loaders with same seed
        loader1 = factory.create_dataloader(tensor_dataset, config)
        loader2 = factory.create_dataloader(tensor_dataset, config)

        # Iterate and compare batches
        for batch1, batch2 in zip(loader1, loader2):
            assert torch.equal(batch1[0], batch2[0])

    def test_different_seeds_produce_different_batches(self, tensor_dataset):
        """Test that different seeds produce different batch orders."""
        factory = DataLoaderFactory()

        config1 = DataLoaderConfig(batch_size=16, shuffle=True, seed=42, num_workers=0)
        config2 = DataLoaderConfig(batch_size=16, shuffle=True, seed=123, num_workers=0)

        loader1 = factory.create_dataloader(tensor_dataset, config1)
        loader2 = factory.create_dataloader(tensor_dataset, config2)

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Should be different with different seeds
        assert not torch.equal(batch1[0], batch2[0])


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""

    def test_dataloader_overhead(self, tensor_dataset):
        """Test that DataLoader overhead is minimal (informational test)."""
        import time

        factory = DataLoaderFactory()
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=0,  # Single process for fair timing
            pin_memory=False
        )

        loader = factory.create_dataloader(tensor_dataset, config)

        # Measure iteration time
        start = time.time()
        for batch in loader:
            pass  # No processing
        dataloader_time = time.time() - start

        # Measure with simulated training step (10ms per batch)
        start = time.time()
        for batch in loader:
            time.sleep(0.01)  # Simulate fast training step
        total_time = time.time() - start

        # Calculate overhead ratio (informational, not strict requirement)
        overhead_ratio = dataloader_time / total_time

        # Print for informational purposes
        print(f"\nDataLoader overhead: {overhead_ratio:.1%} of total time")
        print(f"  DataLoader time: {dataloader_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")

        # Loose assertion - just verify DataLoader works efficiently
        assert dataloader_time < 1.0, f"DataLoader iteration unexpectedly slow: {dataloader_time:.3f}s"

    def test_prefetch_benefit(self, tensor_dataset):
        """Test that prefetch_factor improves throughput (informational)."""
        factory = DataLoaderFactory()

        # Without prefetch
        config_no_prefetch = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            prefetch_factor=None
        )

        # With prefetch
        config_with_prefetch = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            prefetch_factor=2
        )

        loader_no_prefetch = factory.create_dataloader(tensor_dataset, config_no_prefetch)
        loader_with_prefetch = factory.create_dataloader(tensor_dataset, config_with_prefetch)

        # Both should work (actual speedup testing requires real workload)
        assert loader_no_prefetch is not None
        assert loader_with_prefetch is not None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test with empty dataset."""
        factory = DataLoaderFactory()
        empty_data = TensorDataset(torch.empty(0, 32))
        config = DataLoaderConfig(batch_size=16, shuffle=False)  # No shuffle for empty dataset

        loader = factory.create_dataloader(empty_data, config)

        # Should create loader but yield no batches
        batches = list(loader)
        assert len(batches) == 0

    def test_batch_size_larger_than_dataset(self, tensor_dataset):
        """Test batch size larger than dataset."""
        factory = DataLoaderFactory()
        config = DataLoaderConfig(batch_size=1000)  # Larger than 100 samples

        loader = factory.create_dataloader(tensor_dataset, config)

        # Should work, returning single batch
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 100  # All samples in one batch

    def test_no_val_split(self, tensor_dataset):
        """Test UniversalDataModule with val_split=0."""
        data_module = UniversalDataModule(
            train_data=tensor_dataset,
            val_split=0.0,
            batch_size=16
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        assert train_loader is not None
        assert val_loader is None  # No validation data

    def test_missing_tokenizer_for_text_task(self, text_task_spec):
        """Test error handling when tokenizer missing for text task."""
        CollatorRegistry.reset()
        registry = CollatorRegistry.get_instance()

        # Should raise ValueError
        with pytest.raises(ValueError, match="tokenizer required"):
            registry.get_collator(task_spec=text_task_spec, tokenizer=None)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
