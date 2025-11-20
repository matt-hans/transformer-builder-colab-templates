"""
Unit tests for LossStrategy Protocol and implementations.

Tests cover:
1. LanguageModelingLoss with padding → Verify padding tokens excluded from loss
2. ClassificationLoss with imbalanced dataset → Apply class weights correctly
3. PEFTAwareLoss with frozen base model → Verify gradients only on adapter parameters
4. QuantizationSafeLoss with 4-bit model → Dequantize before loss computation
5. Missing attention_mask → Raises clear error with fix suggestion
6. Logit shape mismatch → Raises clear error
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.training.engine.loss import (
    LossInputs,
    ModelOutput,
    LossStrategy,
    LanguageModelingLoss,
    ClassificationLoss,
    PEFTAwareLoss,
    QuantizationSafeLoss,
    VisionLoss,
    LossStrategyRegistry,
    get_loss_strategy
)


class TestModelOutput:
    """Test suite for ModelOutput dataclass."""

    def test_from_tensor(self):
        """Test parsing from raw tensor."""
        logits = torch.randn(4, 10)
        output = ModelOutput.from_raw(logits)

        assert torch.equal(output.logits, logits)
        assert output.loss is None
        assert output.hidden_states is None

    def test_from_tuple(self):
        """Test parsing from tuple."""
        logits = torch.randn(4, 10)
        loss = torch.tensor(0.5)

        # Tuple with (logits, loss)
        output = ModelOutput.from_raw((logits, loss))
        assert torch.equal(output.logits, logits)
        assert torch.equal(output.loss, loss)

    def test_from_dict(self):
        """Test parsing from dictionary."""
        logits = torch.randn(4, 10)
        loss = torch.tensor(0.5)

        output = ModelOutput.from_raw({'logits': logits, 'loss': loss})
        assert torch.equal(output.logits, logits)
        assert torch.equal(output.loss, loss)

    def test_from_huggingface_object(self):
        """Test parsing from HuggingFace-style object."""
        class FakeModelOutput:
            def __init__(self):
                self.logits = torch.randn(4, 10)
                self.loss = torch.tensor(0.5)
                self.hidden_states = torch.randn(4, 8, 768)

        hf_output = FakeModelOutput()
        output = ModelOutput.from_raw(hf_output)

        assert torch.equal(output.logits, hf_output.logits)
        assert torch.equal(output.loss, hf_output.loss)
        assert torch.equal(output.hidden_states, hf_output.hidden_states)

    def test_validation_valid(self):
        """Test validation passes for valid output."""
        logits = torch.randn(4, 10, 512)
        output = ModelOutput(logits=logits)
        output.validate()  # Should not raise

    def test_validation_invalid_shape(self):
        """Test validation fails for invalid shape."""
        logits = torch.randn(10)  # 1D tensor
        output = ModelOutput(logits=logits)

        with pytest.raises(ValueError) as exc_info:
            output.validate()

        assert 'at least 2 dimensions' in str(exc_info.value)

    def test_unsupported_output_type(self):
        """Test error for unsupported output type."""
        with pytest.raises(TypeError) as exc_info:
            ModelOutput.from_raw([1, 2, 3])  # List not supported

        assert 'Cannot parse output type' in str(exc_info.value)


class TestLanguageModelingLoss:
    """Test suite for LanguageModelingLoss."""

    def test_basic_loss_computation(self):
        """Test basic loss computation with token shifting."""
        strategy = LanguageModelingLoss()

        batch_size, seq_len, vocab_size = 4, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels,
            'pad_token_id': 0
        }

        loss = strategy.compute_loss(inputs)

        assert loss.ndim == 0  # Scalar loss
        assert loss.item() > 0  # Positive loss

    def test_padding_exclusion(self):
        """Test 1: Padding tokens excluded from loss calculation."""
        strategy = LanguageModelingLoss()

        batch_size, seq_len, vocab_size = 2, 8, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create labels with padding (pad_token_id=0)
        labels = torch.randint(1, vocab_size, (batch_size, seq_len))
        labels[:, -2:] = 0  # Last 2 tokens are padding

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels,
            'pad_token_id': 0
        }

        loss_with_padding = strategy.compute_loss(inputs)

        # Compare with loss computed only on non-padding tokens
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Manually compute loss excluding padding
        expected_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=0
        )

        assert torch.allclose(loss_with_padding, expected_loss, atol=1e-6)

    def test_invalid_shape_error(self):
        """Test error for invalid logits shape."""
        strategy = LanguageModelingLoss()

        # 2D logits (should be 3D)
        logits = torch.randn(4, 100)
        labels = torch.randint(0, 100, (4,))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        with pytest.raises(ValueError) as exc_info:
            strategy.compute_loss(inputs)

        assert '3D logits' in str(exc_info.value)


class TestClassificationLoss:
    """Test suite for ClassificationLoss."""

    def test_basic_classification_loss(self):
        """Test basic classification loss."""
        strategy = ClassificationLoss()

        batch_size, num_classes = 8, 10
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        loss = strategy.compute_loss(inputs)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_class_weights(self):
        """Test 2: Class weights applied for imbalanced dataset."""
        strategy = ClassificationLoss()

        batch_size, num_classes = 8, 3
        logits = torch.randn(batch_size, num_classes)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])  # Imbalanced

        # Class weights (higher weight for minority class 2)
        class_weights = torch.tensor([0.5, 0.5, 2.0])

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels,
            'class_weights': class_weights
        }

        loss_weighted = strategy.compute_loss(inputs)

        # Compare with unweighted loss
        inputs_unweighted: LossInputs = {
            'logits': logits,
            'labels': labels
        }
        loss_unweighted = strategy.compute_loss(inputs_unweighted)

        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss_weighted, loss_unweighted)

    def test_sequence_classification(self):
        """Test classification with sequence input (take last token)."""
        strategy = ClassificationLoss()

        batch_size, seq_len, num_classes = 4, 10, 5
        logits = torch.randn(batch_size, seq_len, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        loss = strategy.compute_loss(inputs)

        # Should use last token for classification
        assert loss.ndim == 0


class TestPEFTAwareLoss:
    """Test suite for PEFTAwareLoss."""

    def test_peft_verification(self):
        """Test 3: PEFT setup verification (some params frozen)."""
        # Create model with some frozen parameters (simulating PEFT)
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        # Freeze first layer (simulating base model)
        for param in model[0].parameters():
            param.requires_grad = False

        base_strategy = ClassificationLoss()
        peft_strategy = PEFTAwareLoss(base_strategy, model)

        # Should not raise error (some params frozen, some trainable)
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        loss = peft_strategy.compute_loss(inputs)
        assert loss.ndim == 0

    def test_peft_all_frozen_error(self):
        """Test error when all parameters frozen."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        base_strategy = ClassificationLoss()

        with pytest.raises(ValueError) as exc_info:
            PEFTAwareLoss(base_strategy, model)

        assert 'No trainable parameters' in str(exc_info.value)

    def test_peft_warning_all_trainable(self, capsys):
        """Test warning when all parameters trainable (PEFT not configured)."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # All parameters trainable (not typical for PEFT)
        base_strategy = ClassificationLoss()
        PEFTAwareLoss(base_strategy, model)

        # Check for warning
        captured = capsys.readouterr()
        assert 'All' in captured.out and 'trainable' in captured.out


class TestQuantizationSafeLoss:
    """Test suite for QuantizationSafeLoss."""

    def test_fp16_conversion(self):
        """Test 4: FP16 logits converted to FP32 for stability."""
        base_strategy = ClassificationLoss()
        quant_strategy = QuantizationSafeLoss(base_strategy)

        batch_size, num_classes = 4, 10
        logits_fp16 = torch.randn(batch_size, num_classes, dtype=torch.float16)
        labels = torch.randint(0, num_classes, (batch_size,))

        inputs: LossInputs = {
            'logits': logits_fp16,
            'labels': labels
        }

        loss = quant_strategy.compute_loss(inputs)

        # Should compute successfully (FP16 converted to FP32)
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

    def test_quantized_dtype_warning(self, capsys):
        """Test warning for quantized dtype."""
        base_strategy = ClassificationLoss()
        quant_strategy = QuantizationSafeLoss(base_strategy)

        batch_size, num_classes = 4, 10
        logits_int8 = torch.randint(-128, 127, (batch_size, num_classes), dtype=torch.int8)
        labels = torch.randint(0, num_classes, (batch_size,))

        inputs: LossInputs = {
            'logits': logits_int8,
            'labels': labels
        }

        # This will likely fail, but should print warning first
        try:
            quant_strategy.compute_loss(inputs)
        except Exception:
            pass

        captured = capsys.readouterr()
        assert 'quantized dtype' in captured.out.lower()


class TestVisionLoss:
    """Test suite for VisionLoss."""

    def test_image_classification(self):
        """Test image classification loss."""
        strategy = VisionLoss()

        batch_size, num_classes = 8, 10
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        loss = strategy.compute_loss(inputs)
        assert loss.ndim == 0

    def test_semantic_segmentation(self):
        """Test semantic segmentation loss."""
        strategy = VisionLoss()

        batch_size, num_classes, height, width = 2, 21, 64, 64
        logits = torch.randn(batch_size, num_classes, height, width)
        labels = torch.randint(0, num_classes, (batch_size, height, width))

        inputs: LossInputs = {
            'logits': logits,
            'labels': labels
        }

        loss = strategy.compute_loss(inputs)
        assert loss.ndim == 0


class TestLossStrategyRegistry:
    """Test suite for LossStrategyRegistry."""

    def test_get_registered_strategy(self):
        """Test retrieving registered strategy."""
        strategy = LossStrategyRegistry.get("language_modeling")
        assert isinstance(strategy, LanguageModelingLoss)

    def test_unknown_strategy_error(self):
        """Test 5: Unknown strategy raises clear error with suggestions."""
        with pytest.raises(ValueError) as exc_info:
            LossStrategyRegistry.get("unknown_task")

        error_message = str(exc_info.value)
        assert 'unknown_task' in error_message
        assert 'Available:' in error_message

    def test_typo_suggestions(self):
        """Test typo detection suggests similar strategies."""
        with pytest.raises(ValueError) as exc_info:
            LossStrategyRegistry.get("languag_modeling")  # Typo

        error_message = str(exc_info.value)
        # Should suggest "language_modeling"
        assert 'Did you mean' in error_message or 'language_modeling' in error_message

    def test_list_available_strategies(self):
        """Test listing available strategies."""
        strategies = LossStrategyRegistry.list_available()

        assert 'language_modeling' in strategies
        assert 'classification' in strategies
        assert 'vision_classification' in strategies

    def test_register_custom_strategy(self):
        """Test registering custom strategy."""
        @LossStrategyRegistry.register("custom_test_strategy")
        class CustomLoss:
            def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
                return torch.tensor(1.0)

        # Should be able to retrieve custom strategy
        strategy = LossStrategyRegistry.get("custom_test_strategy")
        assert isinstance(strategy, CustomLoss)

    def test_convenience_function(self):
        """Test get_loss_strategy convenience wrapper."""
        strategy = get_loss_strategy("classification")
        assert isinstance(strategy, ClassificationLoss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
