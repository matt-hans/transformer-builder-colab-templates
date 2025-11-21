"""
Type System Tests

Tests for type inference, Protocol conformance, and TypedDict validation.
These tests verify the type system works correctly at runtime.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import cast, Any, Dict, List, Optional

# Test Protocol runtime checking
from utils.training.engine.data import DataModuleProtocol, UniversalDataModule
from utils.training.engine.loss import (
    LossStrategy,
    LossInputs,
    LanguageModelingLoss,
    ClassificationLoss,
    VisionLoss,
    ModelOutput
)
from utils.training.task_spec import TaskSpec


class TestProtocolConformance:
    """Test Protocol implementations with isinstance() checks."""

    def test_data_module_protocol_runtime_check(self):
        """Test DataModuleProtocol runtime checking with isinstance()."""
        # Create minimal data module
        train_data = TensorDataset(torch.randn(10, 8))
        data_module = UniversalDataModule(
            train_data=train_data,
            batch_size=4
        )

        # Runtime check should work (DataModuleProtocol is @runtime_checkable)
        assert isinstance(data_module, DataModuleProtocol)

        # Verify protocol methods exist and work
        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, DataLoader)

        val_loader = data_module.val_dataloader()
        assert val_loader is None or isinstance(val_loader, DataLoader)

    def test_loss_strategy_implementations(self):
        """Test all LossStrategy implementations conform to the protocol."""
        # Note: LossStrategy is NOT @runtime_checkable, so we test structurally

        strategies: List[LossStrategy] = [
            LanguageModelingLoss(),
            ClassificationLoss(),
            VisionLoss(task_type='vision_classification')
        ]

        for strategy in strategies:
            # Verify protocol methods exist
            assert hasattr(strategy, 'compute')
            assert hasattr(strategy, 'name')
            assert callable(strategy.compute)
            assert isinstance(strategy.name, str)


class TestTypedDictValidation:
    """Test TypedDict usage (LossInputs)."""

    def test_loss_inputs_required_fields(self):
        """Test LossInputs with required fields."""
        logits = torch.randn(4, 10, 1000)
        labels = torch.randint(0, 1000, (4, 10))

        # Create LossInputs (all fields are optional via total=False)
        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        assert inputs['logits'].shape == (4, 10, 1000)
        assert inputs['labels'].shape == (4, 10)

    def test_loss_inputs_optional_fields(self):
        """Test LossInputs with optional fields."""
        logits = torch.randn(4, 10, 1000)
        labels = torch.randint(0, 1000, (4, 10))
        attention_mask = torch.ones(4, 10)

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels,
            'attention_mask': attention_mask,
            'pad_token_id': 0,
            'task_type': 'lm'
        }

        assert 'attention_mask' in inputs
        assert inputs['pad_token_id'] == 0
        assert inputs['task_type'] == 'lm'


class TestTypeInference:
    """Test type inference with reveal_type() style checks."""

    def test_generic_dict_inference(self):
        """Test Dict[str, float] type inference."""
        metrics: Dict[str, float] = {
            'loss': 0.5,
            'accuracy': 0.95
        }

        # Type should be inferred correctly
        assert isinstance(metrics, dict)
        assert all(isinstance(k, str) for k in metrics.keys())
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_optional_type_narrowing(self):
        """Test Optional type narrowing with isinstance()."""
        def get_optional_value() -> Optional[int]:
            return 42

        value = get_optional_value()

        # Before narrowing, value is Optional[int]
        if value is not None:
            # After narrowing, value is int
            squared: int = value * value
            assert squared == 1764

    def test_union_type_narrowing(self):
        """Test Union type narrowing."""
        from typing import Union

        data: Union[int, str, List[int]] = [1, 2, 3]

        if isinstance(data, list):
            # Type narrowed to List[int]
            total: int = sum(data)
            assert total == 6
        elif isinstance(data, int):
            # Type narrowed to int
            doubled = data * 2
        else:
            # Type narrowed to str
            upper = data.upper()


class TestModelOutputHandling:
    """Test ModelOutput type handling."""

    def test_model_output_creation(self):
        """Test creating ModelOutput instances."""
        logits = torch.randn(4, 10, 1000)

        # Tensor only
        output1 = logits
        assert isinstance(output1, torch.Tensor)

        # Tuple format
        output2 = (logits,)
        assert isinstance(output2, tuple)
        assert len(output2) == 1

        # Dict format
        output3 = {'logits': logits}
        assert isinstance(output3, dict)
        assert 'logits' in output3


class TestCallableTypes:
    """Test Callable type annotations."""

    def test_callable_with_parameters(self):
        """Test Callable[..., Any] for variable arguments."""
        from typing import Callable

        def apply_transform(
            data: torch.Tensor,
            transform: Callable[[torch.Tensor], torch.Tensor]
        ) -> torch.Tensor:
            return transform(data)

        # Define a simple transform
        def normalize(x: torch.Tensor) -> torch.Tensor:
            return (x - x.mean()) / x.std()

        data = torch.randn(10, 5)
        result = apply_transform(data, normalize)
        assert result.shape == data.shape

    def test_collate_fn_callable(self):
        """Test collate_fn as Callable[..., Any]."""
        from typing import Callable, Any

        def custom_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            return {
                'input_ids': torch.stack([b['input_ids'] for b in batch]),
                'labels': torch.stack([b['labels'] for b in batch])
            }

        # Type should be Callable[..., Any]
        collate_fn: Callable[..., Any] = custom_collator

        batch = [
            {'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([3, 4])},
            {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([7, 8])}
        ]

        result = collate_fn(batch)
        assert result['input_ids'].shape == (2, 2)


class TestTypeAliases:
    """Test type alias usage."""

    def test_type_alias_dict(self):
        """Test Dict type alias."""
        from typing import Dict

        BatchDict = Dict[str, torch.Tensor]

        batch: BatchDict = {
            'input_ids': torch.randint(0, 1000, (4, 10)),
            'labels': torch.randint(0, 1000, (4, 10))
        }

        assert 'input_ids' in batch
        assert isinstance(batch['input_ids'], torch.Tensor)

    def test_type_alias_union(self):
        """Test Union type alias."""
        from typing import Union, List
        from torch.utils.data import Dataset

        DatasetUnion = Union[Dataset, List[torch.Tensor]]

        # List variant
        data1: DatasetUnion = [torch.randn(10, 5), torch.randn(10, 5)]
        assert isinstance(data1, list)

        # Dataset variant
        data2: DatasetUnion = TensorDataset(torch.randn(10, 5))
        assert isinstance(data2, Dataset)


def test_mypy_compatibility():
    """
    Test that imports work correctly for mypy.

    This test ensures all type imports are available and don't cause
    circular dependency issues.
    """
    # Protocol imports
    from utils.training.engine.data import DataModuleProtocol
    from utils.training.engine.loss import LossStrategy

    # TypedDict imports
    from utils.training.engine.loss import LossInputs, ModelOutput

    # Dataclass imports
    from utils.training.engine.data import DataLoaderConfig
    from utils.training.engine.gradient_accumulator import AccumulationStats

    # All imports should succeed
    assert DataModuleProtocol is not None
    assert LossStrategy is not None
    assert LossInputs is not None
    assert DataLoaderConfig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
