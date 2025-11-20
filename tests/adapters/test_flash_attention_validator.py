"""
Test suite for FlashAttentionValidator (P2-3).

Comprehensive tests for:
- FlashAttentionValidator compatibility validation
- FlashAttentionReport serialization and loading
- Numerical accuracy validation
- Performance benchmarking
- Integration with torch.compile and AMP
- Edge cases and error handling

Follows pytest conventions with clear fixture setup and parameterized tests.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch

from utils.adapters.flash_attention_validator import (
    FlashAttentionValidator,
    FlashAttentionReport
)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

class SimpleTransformer(nn.Module):
    """Minimal transformer for testing validator."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            attn_output, _ = layer(x, x, x)
            x = x + attn_output
        return self.output(x)


@pytest.fixture
def simple_model():
    """Create simple transformer model."""
    return SimpleTransformer(vocab_size=1000, d_model=256, num_layers=2)


@pytest.fixture
def model_config():
    """Create model configuration."""
    return SimpleNamespace(
        vocab_size=1000,
        max_seq_len=32,
        d_model=256,
        num_layers=2
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# TEST FLASH ATTENTION REPORT
# ==============================================================================

class TestFlashAttentionReport:
    """Test FlashAttentionReport dataclass."""

    def test_report_creation(self):
        """Test creating a report with default values."""
        report = FlashAttentionReport()

        assert isinstance(report.compatibility_status, dict)
        assert isinstance(report.accuracy_metrics, dict)
        assert isinstance(report.performance_metrics, dict)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.model_info, dict)

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = FlashAttentionReport(
            compatibility_status={'sdpa_available': True},
            accuracy_metrics={'max_error': 1e-6},
            recommendations=['Test recommendation']
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict['compatibility_status']['sdpa_available'] is True
        assert report_dict['accuracy_metrics']['max_error'] == 1e-6
        assert 'Test recommendation' in report_dict['recommendations']

    def test_report_save_and_load(self, temp_output_dir):
        """Test saving and loading report."""
        report = FlashAttentionReport(
            compatibility_status={'test': 'value'},
            timestamp='2025-01-01T00:00:00'
        )

        # Save
        output_path = temp_output_dir / 'test_report.json'
        saved_path = report.save(str(output_path))

        assert saved_path.exists()
        assert saved_path == output_path

        # Load
        loaded_report = FlashAttentionReport.load(str(output_path))

        assert loaded_report.compatibility_status == report.compatibility_status
        assert loaded_report.timestamp == report.timestamp

    def test_report_print_summary(self, capfd):
        """Test printing report summary."""
        report = FlashAttentionReport(
            compatibility_status={
                'pytorch_version': '2.1.0',
                'pytorch_compatible': True,
                'cuda_available': True,
                'sdpa_function_available': True,
                'num_attention_layers': 2,
                'num_compatible_layers': 2
            },
            accuracy_metrics={
                'max_absolute_error': 1e-6,
                'mean_absolute_error': 1e-7,
                'max_relative_error': 1e-5,
                'gradient_check_passed': True
            },
            performance_metrics={
                'flash_latency_ms': 10.5,
                'throughput_samples_per_sec': 95.2,
                'flash_peak_memory_mb': 512.0
            },
            recommendations=['Recommendation 1', 'Recommendation 2']
        )

        report.print_summary()

        captured = capfd.readouterr()
        output = captured.out

        assert 'FLASH ATTENTION VALIDATION REPORT' in output
        assert 'PyTorch Version: 2.1.0' in output
        assert 'CUDA Available: ✅ Yes' in output
        assert 'Max Absolute Error: 1.00e-06' in output
        assert 'Flash Attention Latency: 10.50 ms' in output
        assert 'Recommendation 1' in output


# ==============================================================================
# TEST FLASH ATTENTION VALIDATOR
# ==============================================================================

class TestFlashAttentionValidator:
    """Test FlashAttentionValidator functionality."""

    def test_validator_initialization(self, simple_model, model_config):
        """Test validator initialization."""
        validator = FlashAttentionValidator(simple_model, model_config)

        assert validator.model == simple_model
        assert validator.config == model_config
        assert isinstance(validator.report, FlashAttentionReport)
        assert validator.report.model_info['model_class'] == 'SimpleTransformer'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_validator_device_selection_cuda(self, simple_model, model_config):
        """Test validator uses CUDA when available."""
        validator = FlashAttentionValidator(simple_model, model_config)

        assert validator.device.type == 'cuda'

    def test_validator_device_selection_cpu(self, simple_model, model_config):
        """Test validator fallback to CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            validator = FlashAttentionValidator(simple_model, model_config)
            assert validator.device.type == 'cpu'

    def test_validate_compatibility(self, simple_model, model_config):
        """Test compatibility validation."""
        validator = FlashAttentionValidator(simple_model, model_config)
        validator._validate_compatibility()

        compat = validator.report.compatibility_status

        assert 'pytorch_version' in compat
        assert 'pytorch_compatible' in compat
        assert 'cuda_available' in compat
        assert 'sdpa_function_available' in compat
        assert 'num_attention_layers' in compat
        assert 'num_compatible_layers' in compat
        assert 'sdpa_available' in compat

        # Should detect 2 attention layers
        assert compat['num_attention_layers'] == 2

    @patch('torch.__version__', '1.13.0')
    def test_compatibility_pytorch_1x(self, simple_model, model_config):
        """Test compatibility check fails with PyTorch 1.x."""
        validator = FlashAttentionValidator(simple_model, model_config)
        validator._validate_compatibility()

        compat = validator.report.compatibility_status

        assert compat['pytorch_version'] == '1.13.0'
        assert compat['pytorch_compatible'] is False
        assert compat['sdpa_available'] is False

    def test_validate_accuracy(self, simple_model, model_config):
        """Test numerical accuracy validation."""
        validator = FlashAttentionValidator(simple_model, model_config)
        validator._validate_accuracy(batch_size=2, seq_len=16, tolerance=1e-5)

        acc = validator.report.accuracy_metrics

        assert 'max_absolute_error' in acc
        assert 'mean_absolute_error' in acc
        assert 'max_relative_error' in acc
        assert 'gradient_check_passed' in acc
        assert 'accuracy_passed' in acc
        assert 'test_batch_size' in acc
        assert 'test_seq_len' in acc

        # Gradient check should pass
        assert acc['gradient_check_passed'] is True

    def test_validate_gradients_success(self, simple_model, model_config):
        """Test gradient validation succeeds."""
        validator = FlashAttentionValidator(simple_model, model_config)

        input_ids = torch.randint(0, 1000, (2, 16)).to(validator.device)
        result = validator._validate_gradients(input_ids)

        assert result is True

    def test_validate_gradients_failure(self, simple_model, model_config):
        """Test gradient validation handles errors gracefully."""
        validator = FlashAttentionValidator(simple_model, model_config)

        # Mock model.forward to raise exception
        original_forward = simple_model.forward

        def failing_forward(*args, **kwargs):
            raise RuntimeError("Intentional failure")

        simple_model.forward = failing_forward

        input_ids = torch.randint(0, 1000, (2, 16)).to(validator.device)
        result = validator._validate_gradients(input_ids)

        # Should return False on failure
        assert result is False

        # Restore original forward
        simple_model.forward = original_forward

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_benchmark_performance(self, simple_model, model_config):
        """Test performance benchmarking."""
        validator = FlashAttentionValidator(simple_model, model_config)
        validator._benchmark_performance(
            num_warmup=2,
            num_iterations=5,
            batch_size=2,
            seq_len=16
        )

        perf = validator.report.performance_metrics

        assert 'flash_latency_ms' in perf
        assert 'flash_latency_std_ms' in perf
        assert 'flash_latency_p50_ms' in perf
        assert 'flash_latency_p95_ms' in perf
        assert 'flash_latency_p99_ms' in perf
        assert 'throughput_samples_per_sec' in perf
        assert 'flash_peak_memory_mb' in perf

        # Latency should be positive
        assert perf['flash_latency_ms'] > 0
        assert perf['throughput_samples_per_sec'] > 0
        assert perf['flash_peak_memory_mb'] > 0

    def test_benchmark_performance_cpu_skip(self, simple_model, model_config):
        """Test performance benchmark skipped on CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            validator = FlashAttentionValidator(simple_model, model_config)
            validator._benchmark_performance()

            # Performance metrics should be empty
            assert not validator.report.performance_metrics

    def test_generate_recommendations_upgrade_pytorch(self, simple_model, model_config):
        """Test recommendation to upgrade PyTorch."""
        with patch('torch.__version__', '1.13.0'):
            validator = FlashAttentionValidator(simple_model, model_config)
            validator._validate_compatibility()
            validator._generate_recommendations()

            recommendations = validator.report.recommendations

            # Should recommend upgrading PyTorch
            assert any('Upgrade to PyTorch 2.0+' in rec for rec in recommendations)

    def test_generate_recommendations_cuda_required(self, simple_model, model_config):
        """Test recommendation to enable CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            validator = FlashAttentionValidator(simple_model, model_config)
            validator._validate_compatibility()
            validator._generate_recommendations()

            recommendations = validator.report.recommendations

            # Should recommend CUDA
            assert any('requires CUDA' in rec for rec in recommendations)

    def test_generate_recommendations_success(self, simple_model, model_config):
        """Test recommendations when everything is configured correctly."""
        # Mock successful compatibility
        validator = FlashAttentionValidator(simple_model, model_config)
        validator.report.compatibility_status = {
            'pytorch_compatible': True,
            'cuda_available': True,
            'sdpa_function_available': True,
            'num_attention_layers': 2,
            'num_compatible_layers': 2,
            'sdpa_available': True
        }
        validator._generate_recommendations()

        recommendations = validator.report.recommendations

        # Should have success message
        assert any('properly configured' in rec for rec in recommendations)

    def test_validate_all_full_suite(self, simple_model, model_config):
        """Test running full validation suite."""
        validator = FlashAttentionValidator(simple_model, model_config)

        report = validator.validate_all(
            run_performance_tests=False,  # Skip on CPU
            run_accuracy_tests=True
        )

        # Should have compatibility status
        assert report.compatibility_status
        # Should have accuracy metrics
        assert report.accuracy_metrics
        # Should have recommendations
        assert report.recommendations

    def test_validate_all_skip_accuracy(self, simple_model, model_config):
        """Test validation with accuracy tests skipped."""
        validator = FlashAttentionValidator(simple_model, model_config)

        report = validator.validate_all(
            run_performance_tests=False,
            run_accuracy_tests=False
        )

        # Should have compatibility but no accuracy
        assert report.compatibility_status
        assert not report.accuracy_metrics

    def test_validate_all_save_report(self, simple_model, model_config, temp_output_dir):
        """Test validation and save report."""
        validator = FlashAttentionValidator(simple_model, model_config)

        report = validator.validate_all(
            run_performance_tests=False,
            run_accuracy_tests=True
        )

        # Save report
        output_path = temp_output_dir / 'validation_report.json'
        saved_path = report.save(str(output_path))

        assert saved_path.exists()

        # Verify JSON structure
        with open(saved_path, 'r') as f:
            data = json.load(f)

        assert 'compatibility_status' in data
        assert 'accuracy_metrics' in data
        assert 'recommendations' in data


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestFlashAttentionIntegration:
    """Integration tests with torch.compile and AMP."""

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="Requires PyTorch 2.0+")
    def test_validator_with_compiled_model(self, simple_model, model_config):
        """Test validator with torch.compile."""
        # Compile model
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(simple_model, mode='default')
        else:
            compiled_model = simple_model

        validator = FlashAttentionValidator(compiled_model, model_config)

        # Should work with compiled model
        report = validator.validate_all(
            run_performance_tests=False,
            run_accuracy_tests=True
        )

        assert report.compatibility_status
        assert report.accuracy_metrics

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for AMP")
    def test_validator_with_amp(self, simple_model, model_config):
        """Test validator with automatic mixed precision."""
        validator = FlashAttentionValidator(simple_model, model_config)

        # Run validation in AMP context
        with torch.cuda.amp.autocast():
            validator._validate_accuracy(batch_size=2, seq_len=16)

        # Should complete without errors
        assert validator.report.accuracy_metrics


# ==============================================================================
# EDGE CASES AND ERROR HANDLING
# ==============================================================================

class TestFlashAttentionEdgeCases:
    """Test edge cases and error handling."""

    def test_validator_with_zero_attention_layers(self, model_config):
        """Test validator with model that has no attention layers."""
        # Model with no attention
        class NoAttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 1000)

            def forward(self, input_ids):
                return self.linear(torch.randn(input_ids.shape[0], 256, device=input_ids.device))

        model = NoAttentionModel()
        validator = FlashAttentionValidator(model, model_config)
        validator._validate_compatibility()

        compat = validator.report.compatibility_status

        assert compat['num_attention_layers'] == 0
        assert compat['num_compatible_layers'] == 0

    def test_validator_with_large_model(self, model_config):
        """Test validator with large model (many layers)."""
        large_model = SimpleTransformer(vocab_size=1000, d_model=256, num_layers=24)
        validator = FlashAttentionValidator(large_model, model_config)
        validator._validate_compatibility()

        compat = validator.report.compatibility_status

        # Should detect 24 attention layers
        assert compat['num_attention_layers'] == 24
        assert compat['num_compatible_layers'] == 24

    def test_validator_with_invalid_input_shape(self, simple_model, model_config):
        """Test validator handles invalid input shapes gracefully."""
        validator = FlashAttentionValidator(simple_model, model_config)

        # This should not crash, even with unusual shapes
        try:
            validator._validate_accuracy(batch_size=1, seq_len=1)
            # Should complete without raising
            assert True
        except Exception as e:
            # If it does fail, it should be a clear error
            assert isinstance(e, (RuntimeError, ValueError))

    def test_report_save_creates_parent_dirs(self, temp_output_dir):
        """Test report save creates parent directories."""
        report = FlashAttentionReport()

        # Save to nested path that doesn't exist
        nested_path = temp_output_dir / 'nested' / 'dir' / 'report.json'
        saved_path = report.save(str(nested_path))

        assert saved_path.exists()
        assert saved_path.parent.exists()
