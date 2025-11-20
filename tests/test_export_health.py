"""
Comprehensive test suite for export health check system.

Tests cover:
- ExportHealthReport creation and serialization
- CheckResult validation
- ExportHealthChecker pre-export checks
- Format-specific validation (ONNX, TorchScript, PyTorch)
- Post-export verification (numerical consistency, performance)
- Integration with create_export_bundle
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from utils.training.export_health import (
    CheckResult,
    ExportHealthChecker,
    ExportHealthReport,
)


# Fixtures
@pytest.fixture
def simple_model():
    """Create a simple transformer-like model for testing."""

    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=100, d_model=64, num_heads=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.linear = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x, _ = self.attention(x, x, x)
            return self.linear(x)

    return SimpleTransformer()


@pytest.fixture
def model_config():
    """Create model configuration."""
    return SimpleNamespace(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        max_seq_len=32,
    )


@pytest.fixture
def task_spec():
    """Create task specification."""
    return SimpleNamespace(
        name="test-task",
        modality="text",
        task_type="language_modeling",
        input_schema={"vocab_size": 100, "max_seq_len": 32},
        output_schema={"num_classes": 100},
    )


@pytest.fixture
def vision_task_spec():
    """Create vision task specification."""
    return SimpleNamespace(
        name="vision-test",
        modality="vision",
        task_type="vision_classification",
        input_schema={"image_size": [3, 64, 64]},
        output_schema={"num_classes": 10},
    )


# Test CheckResult
class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test CheckResult creation."""
        result = CheckResult(
            check_name="test_check",
            status="passed",
            message="Test passed successfully",
            details={"metric": 0.95},
            duration_seconds=0.123,
        )

        assert result.check_name == "test_check"
        assert result.status == "passed"
        assert result.message == "Test passed successfully"
        assert result.details["metric"] == 0.95
        assert result.duration_seconds == 0.123

    def test_check_result_to_dict(self):
        """Test CheckResult serialization."""
        result = CheckResult(
            check_name="test_check",
            status="passed",
            message="Test message",
            details={"key": "value"},
            duration_seconds=0.5,
        )

        result_dict = result.to_dict()
        assert result_dict["check_name"] == "test_check"
        assert result_dict["status"] == "passed"
        assert result_dict["message"] == "Test message"
        assert result_dict["details"]["key"] == "value"
        assert result_dict["duration_seconds"] == 0.5

    def test_check_result_str(self):
        """Test CheckResult string representation."""
        result_passed = CheckResult("test", "passed", "OK")
        result_warning = CheckResult("test", "warning", "Warning")
        result_failed = CheckResult("test", "failed", "Failed")

        assert "✅" in str(result_passed)
        assert "⚠️" in str(result_warning)
        assert "❌" in str(result_failed)


# Test ExportHealthReport
class TestExportHealthReport:
    """Test ExportHealthReport dataclass."""

    def test_health_report_creation(self):
        """Test health report creation."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        assert report.timestamp == "2025-01-20T10:00:00"
        assert report.model_name == "test-model"
        assert len(report.checks) == 0
        assert report.summary["total"] == 0

    def test_add_check(self):
        """Test adding checks to report."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        check1 = CheckResult("check1", "passed", "OK")
        check2 = CheckResult("check2", "failed", "Error")

        report.add_check(check1)
        assert len(report.checks) == 1
        assert report.summary["passed"] == 1

        report.add_check(check2)
        assert len(report.checks) == 2
        assert report.summary["failed"] == 1

    def test_all_passed(self):
        """Test all_passed property."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        # No checks - should pass
        assert report.all_passed

        # Add passing check
        report.add_check(CheckResult("check1", "passed", "OK"))
        assert report.all_passed

        # Add warning - should still pass
        report.add_check(CheckResult("check2", "warning", "Warning"))
        assert report.all_passed

        # Add failure - should fail
        report.add_check(CheckResult("check3", "failed", "Error"))
        assert not report.all_passed

    def test_health_score(self):
        """Test health score calculation."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        # All passed: 100%
        report.add_check(CheckResult("check1", "passed", "OK"))
        report.add_check(CheckResult("check2", "passed", "OK"))
        assert report.health_score == 100.0

        # 1 warning (50% credit): 75%
        report.add_check(CheckResult("check3", "warning", "Warning"))
        assert report.health_score == pytest.approx(83.3, rel=0.1)

        # 1 failure (0% credit)
        report.add_check(CheckResult("check4", "failed", "Error"))
        assert report.health_score == pytest.approx(62.5, rel=0.1)

    def test_get_failed_checks(self):
        """Test retrieving failed checks."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        report.add_check(CheckResult("check1", "passed", "OK"))
        report.add_check(CheckResult("check2", "failed", "Error1"))
        report.add_check(CheckResult("check3", "failed", "Error2"))

        failed = report.get_failed_checks()
        assert len(failed) == 2
        assert failed[0].status == "failed"
        assert failed[1].status == "failed"

    def test_get_warnings(self):
        """Test retrieving warnings."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        report.add_check(CheckResult("check1", "passed", "OK"))
        report.add_check(CheckResult("check2", "warning", "Warning1"))
        report.add_check(CheckResult("check3", "warning", "Warning2"))

        warnings = report.get_warnings()
        assert len(warnings) == 2
        assert warnings[0].status == "warning"
        assert warnings[1].status == "warning"

    def test_to_dict(self):
        """Test report serialization."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )
        report.add_check(CheckResult("check1", "passed", "OK"))

        report_dict = report.to_dict()
        assert report_dict["timestamp"] == "2025-01-20T10:00:00"
        assert report_dict["model_name"] == "test-model"
        assert report_dict["health_score"] == 100.0
        assert report_dict["all_passed"] is True
        assert len(report_dict["checks"]) == 1

    def test_save_json(self, tmp_path):
        """Test JSON report saving."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )
        report.add_check(CheckResult("check1", "passed", "OK"))

        output_path = tmp_path / "health_report.json"
        saved_path = report.save_json(output_path)

        assert saved_path.exists()
        with open(saved_path) as f:
            data = json.load(f)
        assert data["model_name"] == "test-model"

    def test_save_markdown(self, tmp_path):
        """Test Markdown report saving."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )
        report.add_check(CheckResult("check1", "passed", "OK"))
        report.add_check(CheckResult("check2", "failed", "Error"))

        output_path = tmp_path / "health_report.md"
        saved_path = report.save_markdown(output_path)

        assert saved_path.exists()
        content = saved_path.read_text()
        assert "# Export Health Report" in content
        assert "test-model" in content
        assert "check1" in content
        assert "check2" in content


# Test ExportHealthChecker
class TestExportHealthChecker:
    """Test ExportHealthChecker functionality."""

    def test_checker_initialization(self, simple_model, model_config, task_spec):
        """Test checker initialization."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)

        assert checker.model is simple_model
        assert checker.config is model_config
        assert checker.task_spec is task_spec
        assert checker.model_name == "test-task"

    def test_check_architecture(self, simple_model, model_config, task_spec):
        """Test architecture validation."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_architecture()

        assert result.check_name == "architecture_validation"
        assert result.status == "passed"
        assert "total_modules" in result.details
        assert result.details["total_modules"] > 0

    def test_check_parameters_valid(self, simple_model, model_config, task_spec):
        """Test parameter validation with valid parameters."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_parameters()

        assert result.check_name == "parameter_validation"
        assert result.status == "passed"
        assert result.details["nan_params"] == 0
        assert result.details["inf_params"] == 0
        assert result.details["total_params"] > 0

    def test_check_parameters_nan(self, simple_model, model_config, task_spec):
        """Test parameter validation with NaN parameters."""
        # Inject NaN into model
        with torch.no_grad():
            simple_model.linear.weight[0, 0] = float("nan")

        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_parameters()

        assert result.check_name == "parameter_validation"
        assert result.status == "failed"
        assert result.details["nan_params"] > 0

    def test_check_parameters_inf(self, simple_model, model_config, task_spec):
        """Test parameter validation with Inf parameters."""
        # Inject Inf into model
        with torch.no_grad():
            simple_model.linear.weight[0, 0] = float("inf")

        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_parameters()

        assert result.check_name == "parameter_validation"
        assert result.status == "failed"
        assert result.details["inf_params"] > 0

    def test_check_input_output_shapes(self, simple_model, model_config, task_spec):
        """Test input/output shape validation."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_input_output_shapes()

        assert result.check_name == "shape_validation"
        assert result.status == "passed"
        assert "input_shape" in result.details
        assert "output_shape" in result.details

    def test_check_memory_requirements(self, simple_model, model_config, task_spec):
        """Test memory requirements estimation."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_memory_requirements()

        assert result.check_name == "memory_requirements"
        assert result.status in ["passed", "warning"]
        assert "parameter_memory_mb" in result.details
        assert "peak_memory_mb" in result.details

    def test_check_forward_pass(self, simple_model, model_config, task_spec):
        """Test forward pass validation."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_forward_pass()

        assert result.check_name == "forward_pass_validation"
        assert result.status == "passed"
        assert "test_results" in result.details
        assert len(result.details["test_results"]) > 0

    def test_generate_dummy_input_text(self, simple_model, model_config, task_spec):
        """Test dummy input generation for text tasks."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        dummy_input = checker._generate_dummy_input(batch_size=2)

        assert dummy_input.shape[0] == 2  # batch size
        assert dummy_input.shape[1] == 32  # sequence length
        assert dummy_input.dtype == torch.long

    def test_generate_dummy_input_vision(self, simple_model, model_config, vision_task_spec):
        """Test dummy input generation for vision tasks."""
        checker = ExportHealthChecker(simple_model, model_config, vision_task_spec)
        dummy_input = checker._generate_dummy_input(batch_size=2)

        assert dummy_input.shape[0] == 2  # batch size
        assert dummy_input.shape[1] == 3  # channels
        assert dummy_input.shape[2] == 64  # height
        assert dummy_input.shape[3] == 64  # width

    def test_run_all_checks(self, simple_model, model_config, task_spec):
        """Test running all pre-export checks."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        report = checker.run_all_checks()

        assert report.model_name == "test-task"
        assert len(report.checks) > 0
        assert all(check.status in ["passed", "warning", "failed"] for check in report.checks)

    def test_recommendations_generation(self, simple_model, model_config, task_spec):
        """Test recommendation generation."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)

        # Create report with failed check
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )
        report.add_check(CheckResult("test_check", "failed", "NaN detected"))

        recommendations = checker._generate_recommendations(report)
        assert len(recommendations) > 0
        assert any("failed" in rec.lower() for rec in recommendations)


# Integration tests
class TestExportHealthIntegration:
    """Integration tests for export health checks."""

    def test_check_pytorch_format(self, simple_model, model_config, task_spec, tmp_path):
        """Test PyTorch format validation."""
        # Save model
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        torch.save(simple_model.state_dict(), artifacts_dir / "model.pytorch.pt")

        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_pytorch_format(tmp_path)

        assert result.check_name == "pytorch_validation"
        assert result.status in ["passed", "warning"]

    def test_check_torchscript_format(self, simple_model, model_config, task_spec, tmp_path):
        """Test TorchScript format validation."""
        # Export to TorchScript
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        dummy_input = torch.randint(0, 100, (1, 32))
        traced = torch.jit.trace(simple_model, dummy_input)
        torch.jit.save(traced, artifacts_dir / "model.torchscript.pt")

        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_torchscript_format(tmp_path)

        assert result.check_name == "torchscript_validation"
        assert result.status in ["passed", "warning", "failed"]

    def test_check_onnx_format_requires_onnx(self, simple_model, model_config, task_spec, tmp_path):
        """Test ONNX format validation (requires onnx package)."""
        # Export to ONNX
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        try:
            dummy_input = torch.randint(0, 100, (1, 32))
            torch.onnx.export(
                simple_model,
                dummy_input,
                artifacts_dir / "model.onnx",
                input_names=["input_ids"],
                output_names=["logits"],
                opset_version=14,
            )
        except (ImportError, ModuleNotFoundError):
            # Skip test if onnxscript/onnx not available
            pytest.skip("ONNX export dependencies not available")

        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_onnx_format(tmp_path)

        assert result.check_name == "onnx_validation"
        # Status may be warning if onnx package not installed
        assert result.status in ["passed", "warning", "failed"]

    def test_numerical_consistency_torchscript(self, simple_model, model_config, task_spec, tmp_path):
        """Test TorchScript numerical consistency check."""
        # Export to TorchScript
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        dummy_input = torch.randint(0, 100, (1, 32))
        traced = torch.jit.trace(simple_model, dummy_input)
        torch.jit.save(traced, artifacts_dir / "model.torchscript.pt")

        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_numerical_consistency_torchscript(tmp_path)

        assert result.check_name == "torchscript_numerical_consistency"
        # Should pass for simple model
        assert result.status in ["passed", "warning"]

    def test_performance_benchmark(self, simple_model, model_config, task_spec, tmp_path):
        """Test performance benchmarking."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)
        result = checker._check_performance_benchmark(tmp_path, ["pytorch"])

        assert result.check_name == "performance_benchmark"
        assert result.status in ["passed", "warning"]
        assert "pytorch_ms" in result.details


# Test edge cases
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_checker_with_invalid_model(self, model_config, task_spec):
        """Test checker with model that fails forward pass."""

        class BrokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Add a dummy parameter so next(model.parameters()) works
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                raise RuntimeError("Intentional error")

        model = BrokenModel()
        checker = ExportHealthChecker(model, model_config, task_spec)
        result = checker._check_forward_pass()

        assert result.status == "failed"
        assert "error" in result.details or not result.details.get("test_results", [{}])[0].get(
            "success", True
        )

    def test_empty_health_report(self):
        """Test health report with no checks."""
        report = ExportHealthReport(
            timestamp="2025-01-20T10:00:00",
            model_name="test-model",
        )

        assert report.all_passed is True
        assert report.health_score == 100.0
        assert len(report.checks) == 0

    def test_health_report_with_missing_files(self, simple_model, model_config, task_spec, tmp_path):
        """Test health checks with missing export files."""
        checker = ExportHealthChecker(simple_model, model_config, task_spec)

        # Try to check non-existent ONNX file
        result = checker._check_onnx_format(tmp_path)
        assert result.status == "warning"

        # Try to check non-existent TorchScript file
        result = checker._check_torchscript_format(tmp_path)
        assert result.status == "warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
