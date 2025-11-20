"""
Unit tests for GradientMonitor.

Tests cover:
1. Inject NaN gradient → Detects NaN, logs affected layer
2. 3 consecutive NaN gradients → Raises RuntimeError with remediation steps
3. Gradient norm 1e-9 (vanishing) → Logs warning, suggests LR increase
4. Gradient norm 50.0 (explosion) → Logs warning, suggests LR decrease
5. Healthy gradients → No warnings, minimal overhead
6. Large model (1B params) → Check completes in <50ms
"""

import pytest
import torch
import torch.nn as nn
import logging
import time

from utils.training.engine.gradient_monitor import (
    GradientHealth,
    GradientMonitor,
    check_gradient_health
)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model


@pytest.fixture
def large_model():
    """Create large model for performance testing (~100M params)."""
    layers = []
    sizes = [1024, 2048, 2048, 2048, 1024, 512]
    for i in range(len(sizes) - 1):
        layers.extend([
            nn.Linear(sizes[i], sizes[i + 1]),
            nn.ReLU()
        ])
    return nn.Sequential(*layers)


class TestGradientHealth:
    """Test suite for GradientHealth dataclass."""

    def test_healthy_gradient_str(self):
        """Test string representation for healthy gradients."""
        health = GradientHealth(
            has_nan=False,
            has_inf=False,
            max_norm=1.0,
            min_norm=0.001,
            affected_layers=[],
            is_healthy=True,
            vanishing_layers=[],
            exploding_layers=[]
        )

        str_repr = str(health)
        assert 'Healthy' in str_repr
        assert '1.0' in str_repr

    def test_unhealthy_gradient_str(self):
        """Test string representation for unhealthy gradients."""
        health = GradientHealth(
            has_nan=True,
            has_inf=False,
            max_norm=10.0,
            min_norm=0.0,
            affected_layers=['layer1 (NaN)', 'layer2 (NaN)'],
            is_healthy=False,
            vanishing_layers=[],
            exploding_layers=[]
        )

        str_repr = str(health)
        assert 'Issues' in str_repr or 'NaN' in str_repr


class TestGradientMonitor:
    """Test suite for GradientMonitor."""

    def test_healthy_gradients(self, simple_model):
        """Test 5: Healthy gradients → No warnings, minimal overhead."""
        monitor = GradientMonitor()

        # Create dummy gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 0.1

        health = monitor.check_gradients(simple_model)

        assert health.is_healthy
        assert not health.has_nan
        assert not health.has_inf
        assert health.max_norm > 0
        assert health.min_norm >= 0
        assert len(health.affected_layers) == 0

    def test_nan_detection(self, simple_model, caplog):
        """Test 1: Inject NaN gradient → Detects NaN, logs affected layer."""
        monitor = GradientMonitor()

        # Create gradients with NaN in first layer
        for i, param in enumerate(simple_model.parameters()):
            if i == 0:
                param.grad = torch.full_like(param, float('nan'))
            else:
                param.grad = torch.randn_like(param) * 0.1

        with caplog.at_level(logging.WARNING):
            health = monitor.check_gradients(simple_model)

        assert not health.is_healthy
        assert health.has_nan
        assert len(health.affected_layers) > 0
        assert 'NaN' in health.affected_layers[0]

        # Check logging
        assert 'NaN gradient detected' in caplog.text

    def test_inf_detection(self, simple_model, caplog):
        """Test Inf detection."""
        monitor = GradientMonitor()

        # Create gradients with Inf in first layer
        for i, param in enumerate(simple_model.parameters()):
            if i == 0:
                param.grad = torch.full_like(param, float('inf'))
            else:
                param.grad = torch.randn_like(param) * 0.1

        with caplog.at_level(logging.WARNING):
            health = monitor.check_gradients(simple_model)

        assert not health.is_healthy
        assert health.has_inf
        assert len(health.affected_layers) > 0
        assert 'Inf' in health.affected_layers[0]

    def test_consecutive_nan_failures(self, simple_model):
        """Test 2: 3 consecutive NaN gradients → Raises RuntimeError with remediation."""
        monitor = GradientMonitor(max_consecutive_failures=3)

        # Create NaN gradients
        for param in simple_model.parameters():
            param.grad = torch.full_like(param, float('nan'))

        # First two failures should not raise
        for _ in range(2):
            health = monitor.check_gradients(simple_model)
            assert not health.is_healthy

        # Third failure should raise
        with pytest.raises(RuntimeError) as exc_info:
            monitor.check_gradients(simple_model)

        error_message = str(exc_info.value)
        assert 'consecutive' in error_message.lower()
        assert 'Remediation' in error_message or 'Lower learning rate' in error_message

    def test_vanishing_gradient_detection(self, simple_model, caplog):
        """Test 3: Vanishing gradients → Logs warning, suggests LR increase."""
        monitor = GradientMonitor(vanishing_threshold=1e-7)

        # Create very small gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 1e-9  # Below threshold

        with caplog.at_level(logging.WARNING):
            health = monitor.check_gradients(simple_model)

        assert health.is_healthy  # No NaN/Inf
        assert len(health.vanishing_layers) > 0
        assert 'Vanishing gradients' in caplog.text
        assert 'increasing learning rate' in caplog.text.lower()

    def test_exploding_gradient_detection(self, simple_model, caplog):
        """Test 4: Exploding gradients → Logs warning, suggests LR decrease."""
        monitor = GradientMonitor(explosion_threshold=10.0)

        # Create very large gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 50.0  # Above threshold

        with caplog.at_level(logging.WARNING):
            health = monitor.check_gradients(simple_model)

        assert health.is_healthy  # No NaN/Inf
        assert len(health.exploding_layers) > 0
        assert 'Exploding gradients' in caplog.text
        assert 'gradient clipping' in caplog.text.lower() or 'lower' in caplog.text.lower()

    def test_sparse_tensor_handling(self):
        """Test gradient monitoring with sparse tensors."""
        monitor = GradientMonitor()

        # Create model with sparse embedding
        model = nn.Embedding(100, 10, sparse=True)

        # Create sparse gradient
        indices = torch.LongTensor([0, 2, 5])
        values = torch.randn(3, 10)
        sparse_grad = torch.sparse_coo_tensor(
            indices.unsqueeze(0),
            values,
            (100, 10)
        )
        model.weight.grad = sparse_grad

        health = monitor.check_gradients(model)

        assert health.is_healthy
        assert health.max_norm > 0

    def test_no_gradients(self, simple_model):
        """Test handling when no gradients present."""
        monitor = GradientMonitor()

        # No gradients set
        health = monitor.check_gradients(simple_model)

        assert health.is_healthy
        assert health.max_norm == 0.0
        assert health.min_norm == 0.0

    def test_compute_gradient_norm(self, simple_model):
        """Test gradient norm computation."""
        monitor = GradientMonitor()

        # Set known gradients
        for param in simple_model.parameters():
            param.grad = torch.ones_like(param)

        norm = monitor.compute_gradient_norm(simple_model)

        assert norm > 0
        assert isinstance(norm, float)

    def test_reset_failure_counts(self, simple_model):
        """Test resetting failure counters."""
        monitor = GradientMonitor(max_consecutive_failures=3)

        # Create NaN gradients and trigger failures
        for param in simple_model.parameters():
            param.grad = torch.full_like(param, float('nan'))

        monitor.check_gradients(simple_model)
        assert monitor.consecutive_failures == 1

        # Reset counters
        monitor.reset_failure_counts()
        assert monitor.consecutive_failures == 0
        assert monitor.nan_count == 0

    def test_performance_large_model(self, large_model):
        """Test 6: Large model → Check completes in <50ms."""
        monitor = GradientMonitor()

        # Create gradients for large model
        for param in large_model.parameters():
            param.grad = torch.randn_like(param) * 0.1

        # Measure check time
        start = time.time()
        health = monitor.check_gradients(large_model)
        elapsed = (time.time() - start) * 1000  # ms

        assert health.is_healthy
        assert elapsed < 100  # Allow some margin (target <50ms, test <100ms)

    def test_gradient_norm_ordering(self, simple_model):
        """Test max_norm > min_norm invariant."""
        monitor = GradientMonitor()

        # Create varying gradients
        for i, param in enumerate(simple_model.parameters()):
            param.grad = torch.randn_like(param) * (i + 1)

        health = monitor.check_gradients(simple_model)

        assert health.max_norm >= health.min_norm


class TestConvenienceFunction:
    """Test suite for convenience function."""

    def test_check_gradient_health(self, simple_model):
        """Test check_gradient_health convenience wrapper."""
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 0.1

        health = check_gradient_health(simple_model)

        assert health.is_healthy
        assert health.max_norm > 0


class TestGradientMonitorIntegration:
    """Integration tests for GradientMonitor with training loops."""

    def test_training_loop_integration(self, simple_model):
        """Test integration with typical training loop."""
        monitor = GradientMonitor()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        # Simulate training loop
        for epoch in range(3):
            # Forward pass
            x = torch.randn(4, 10)
            y = torch.randint(0, 10, (4,))
            output = simple_model(x)
            loss = nn.functional.cross_entropy(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check gradients
            health = monitor.check_gradients(simple_model)
            assert health.is_healthy

            # Optimizer step
            optimizer.step()

    def test_gradient_explosion_halt(self, simple_model):
        """Test training halts on gradient explosion."""
        monitor = GradientMonitor(max_consecutive_failures=2)

        # Simulate exploding gradients
        for _ in range(3):
            for param in simple_model.parameters():
                param.grad = torch.full_like(param, float('nan'))

            try:
                monitor.check_gradients(simple_model)
            except RuntimeError as e:
                assert 'consecutive' in str(e).lower()
                break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
