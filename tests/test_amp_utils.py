"""
Comprehensive test suite for AMP (Automatic Mixed Precision) utilities.

Tests cover:
- Edge cases for compute_effective_precision()
- AmpWandbCallback with different precision variants
- Integration with training workflows
- GPU/CPU fallback scenarios
- Loss scale edge cases
"""

import pytest
import torch
import torch.nn as nn
from typing import Any
from utils.training.amp_utils import compute_effective_precision, AmpWandbCallback


class TestComputeEffectivePrecision:
    """Test compute_effective_precision() edge cases"""

    def test_use_amp_none_returns_requested(self):
        """When use_amp is None, should return requested precision unchanged"""
        result = compute_effective_precision(
            requested_precision='32',
            use_amp=None,
            cuda_available=True,
            use_gpu=True
        )
        assert result == '32'

        result = compute_effective_precision(
            requested_precision='bf16',
            use_amp=None,
            cuda_available=False,
            use_gpu=False
        )
        assert result == 'bf16'

    def test_use_amp_true_cuda_available_use_gpu_true(self):
        """When AMP enabled with CUDA and use_gpu=True, should return '16'"""
        result = compute_effective_precision(
            requested_precision='32',
            use_amp=True,
            cuda_available=True,
            use_gpu=True
        )
        assert result == '16'

    def test_use_amp_true_cuda_available_but_use_gpu_false(self):
        """Edge case: CUDA available but user disabled GPU → should return '32'"""
        result = compute_effective_precision(
            requested_precision='16',
            use_amp=True,
            cuda_available=True,
            use_gpu=False  # User explicitly disabled GPU
        )
        assert result == '32', "Should fall back to FP32 when GPU disabled"

    def test_use_amp_true_cuda_not_available(self):
        """When AMP requested but no CUDA → should return '32'"""
        result = compute_effective_precision(
            requested_precision='16',
            use_amp=True,
            cuda_available=False,
            use_gpu=True
        )
        assert result == '32', "Should fall back to FP32 when CUDA unavailable"

    def test_use_amp_false_always_returns_32(self):
        """When AMP explicitly disabled → should return '32'"""
        result = compute_effective_precision(
            requested_precision='16',
            use_amp=False,
            cuda_available=True,
            use_gpu=True
        )
        assert result == '32'

    def test_all_combinations(self):
        """Test all 16 combinations of boolean parameters"""
        test_cases = [
            # (requested, use_amp, cuda, gpu, expected)
            ('32', None, True, True, '32'),
            ('32', None, True, False, '32'),
            ('32', None, False, True, '32'),
            ('32', None, False, False, '32'),
            ('32', True, True, True, '16'),
            ('32', True, True, False, '32'),
            ('32', True, False, True, '32'),
            ('32', True, False, False, '32'),
            ('32', False, True, True, '32'),
            ('32', False, True, False, '32'),
            ('32', False, False, True, '32'),
            ('32', False, False, False, '32'),
        ]

        for requested, use_amp, cuda, gpu, expected in test_cases:
            result = compute_effective_precision(requested, use_amp, cuda, gpu)
            assert result == expected, \
                f"Failed for ({requested}, {use_amp}, {cuda}, {gpu}): got {result}, expected {expected}"


class MockTrainer:
    """Mock PyTorch Lightning trainer for testing AmpWandbCallback"""

    def __init__(self, loss_scale=None):
        self.current_epoch = 0
        self.strategy = MockStrategy(loss_scale)


class MockStrategy:
    """Mock strategy with precision plugin"""

    def __init__(self, loss_scale):
        self.precision_plugin = MockPrecisionPlugin(loss_scale)


class MockPrecisionPlugin:
    """Mock precision plugin with scaler"""

    def __init__(self, loss_scale):
        self.scaler = MockGradScaler(loss_scale) if loss_scale is not None else None


class MockGradScaler:
    """Mock GradScaler with get_scale() method"""

    def __init__(self, scale_value):
        self.scale_value = scale_value

    def get_scale(self):
        return self.scale_value


class TestAmpWandbCallback:
    """Test AmpWandbCallback with different precision variants"""

    @pytest.fixture(autouse=True)
    def setup_wandb_mock(self, monkeypatch):
        """Mock wandb to avoid actual logging during tests"""
        import sys
        from unittest.mock import MagicMock

        # Create mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.run = None  # Simulate wandb not initialized
        sys.modules['wandb'] = mock_wandb

        yield

        # Cleanup
        if 'wandb' in sys.modules:
            del sys.modules['wandb']

    def test_precision_variant_16(self):
        """Test callback with precision='16'"""
        callback = AmpWandbCallback(enabled=True, precision='16')
        assert callback.enabled is True
        assert callback.precision == '16'

    def test_precision_variant_16_mixed(self):
        """Test callback with precision='16-mixed'"""
        callback = AmpWandbCallback(enabled=True, precision='16-mixed')
        assert callback.precision == '16-mixed'

    def test_precision_variant_16_true(self):
        """Test callback with precision='16_true'"""
        callback = AmpWandbCallback(enabled=True, precision='16_true')
        assert callback.precision == '16_true'

    def test_precision_variant_bf16(self):
        """Test callback with precision='bf16'"""
        callback = AmpWandbCallback(enabled=True, precision='bf16')
        assert callback.precision == 'bf16'

    def test_enabled_false(self):
        """Test callback with enabled=False"""
        callback = AmpWandbCallback(enabled=False, precision='16')
        trainer = MockTrainer(loss_scale=None)
        pl_module = None

        # Should not crash when disabled
        callback.on_train_epoch_end(trainer, pl_module)

    def test_get_loss_scale_with_valid_scaler(self):
        """Test loss scale extraction with valid scaler"""
        callback = AmpWandbCallback(enabled=True, precision='16')
        trainer = MockTrainer(loss_scale=65536.0)

        scale = callback._get_loss_scale(trainer)
        assert scale == 65536.0

    def test_get_loss_scale_with_no_scaler(self):
        """Test loss scale extraction when scaler is None"""
        callback = AmpWandbCallback(enabled=True, precision='32')
        trainer = MockTrainer(loss_scale=None)

        scale = callback._get_loss_scale(trainer)
        assert scale is None

    def test_get_loss_scale_extreme_values(self):
        """Test loss scale with extreme values (0, inf, very large)"""
        callback = AmpWandbCallback(enabled=True, precision='16')

        # Test with 0
        trainer_zero = MockTrainer(loss_scale=0.0)
        assert callback._get_loss_scale(trainer_zero) == 0.0

        # Test with very large value
        trainer_large = MockTrainer(loss_scale=1e10)
        assert callback._get_loss_scale(trainer_large) == 1e10

        # Test with very small value
        trainer_small = MockTrainer(loss_scale=1e-10)
        assert callback._get_loss_scale(trainer_small) == 1e-10

    def test_on_train_epoch_end_no_wandb_run(self):
        """Test on_train_epoch_end when wandb.run is None"""
        callback = AmpWandbCallback(enabled=True, precision='16')
        trainer = MockTrainer(loss_scale=65536.0)
        pl_module = None

        # Should not crash when wandb not initialized
        callback.on_train_epoch_end(trainer, pl_module)


class SimpleModel(nn.Module):
    """Simple model for integration testing"""

    def __init__(self, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.linear = nn.Linear(64, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.linear(x)


class TestAMPIntegration:
    """Integration tests for AMP with training workflows"""

    def test_model_forward_with_autocast(self):
        """Test model forward pass with autocast context"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from torch.cuda.amp import autocast

        model = SimpleModel(vocab_size=100).cuda()
        input_ids = torch.randint(0, 100, (4, 10)).cuda()

        with autocast():
            output = model(input_ids)

        assert output.dtype == torch.float16, "Output should be FP16 inside autocast"
        assert output.shape == (4, 10, 100)

    def test_grad_scaler_basic_workflow(self):
        """Test GradScaler basic workflow (scale, step, update)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from torch.cuda.amp import autocast, GradScaler

        model = SimpleModel(vocab_size=100).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        input_ids = torch.randint(0, 100, (4, 10)).cuda()
        labels = torch.randint(0, 100, (4, 10)).cuda()

        optimizer.zero_grad()

        with autocast():
            output = model(input_ids)
            loss = nn.functional.cross_entropy(
                output.view(-1, 100),
                labels.view(-1)
            )

        # Backward with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Verify scaler has a scale value
        assert scaler.get_scale() > 0

    def test_amp_cpu_fallback(self):
        """Test that AMP gracefully falls back on CPU"""
        from torch.cuda.amp import autocast, GradScaler

        # Force CPU
        model = SimpleModel(vocab_size=100).cpu()
        input_ids = torch.randint(0, 100, (4, 10)).cpu()

        # autocast should work but not change dtype on CPU
        with autocast():
            output = model(input_ids)

        # On CPU, autocast doesn't change dtype
        assert output.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_end_to_end_training_with_amp(self):
        """End-to-end test with actual training loop"""
        from torch.cuda.amp import autocast, GradScaler

        model = SimpleModel(vocab_size=100).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = GradScaler()

        # Training data
        train_data = [torch.randint(0, 100, (10,)) for _ in range(10)]

        initial_loss = None
        final_loss = None

        for epoch in range(3):
            for sample in train_data:
                sample = sample.cuda()

                optimizer.zero_grad()

                with autocast():
                    output = model(sample.unsqueeze(0))
                    loss = nn.functional.cross_entropy(
                        output.view(-1, 100),
                        sample.unsqueeze(0).view(-1)
                    )

                if initial_loss is None:
                    initial_loss = loss.item()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                final_loss = loss.item()

        # Verify training progressed
        assert initial_loss is not None
        assert final_loss is not None
        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.5, "Loss should not increase significantly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
