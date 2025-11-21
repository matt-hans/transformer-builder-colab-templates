"""
Export Health Check System.

Comprehensive validation and verification for model exports to ensure production readiness.

Features:
- Pre-export health checks (architecture, parameters, input/output validation)
- Format-specific validation (ONNX, TorchScript, PyTorch)
- Post-export verification (numerical consistency, performance benchmarking)
- Detailed health reports (JSON and Markdown formats)

Architecture:
    ExportHealthChecker: Main validation orchestrator
    ExportHealthReport: Structured health check results
    CheckResult: Individual check outcome

Example:
    >>> checker = ExportHealthChecker(model, config, task_spec)
    >>> health_report = checker.run_all_checks(export_dir)
    >>> health_report.save_markdown(export_dir / "health_report.md")
    >>> assert health_report.all_passed, "Health checks failed"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """
    Result of a single health check.

    Attributes:
        check_name: Unique identifier for the check
        status: Check outcome ("passed", "warning", or "failed")
        message: Human-readable description of the result
        details: Additional structured information (metrics, errors, etc.)
        duration_seconds: Time taken to run the check
    """

    check_name: str
    status: Literal["passed", "warning", "failed"]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_name": self.check_name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    def __str__(self) -> str:
        """String representation with emoji status indicator."""
        status_emoji = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌",
        }
        emoji = status_emoji.get(self.status, "❓")
        return f"{emoji} {self.check_name}: {self.message}"


@dataclass
class ExportHealthReport:
    """
    Comprehensive export health report.

    Contains results from all health checks organized by category:
    - Pre-export checks (architecture, parameters)
    - Format-specific checks (ONNX, TorchScript, PyTorch)
    - Post-export checks (numerical consistency, performance)

    Attributes:
        timestamp: When the health check was performed
        model_name: Model identifier
        checks: List of all check results
        summary: High-level summary statistics
        recommendations: List of actionable recommendations
    """

    timestamp: str
    model_name: str
    checks: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute summary statistics after initialization."""
        if not self.summary:
            self.summary = self._compute_summary()

    def _compute_summary(self) -> Dict[str, int]:
        """Compute summary statistics from check results."""
        return {
            "total": len(self.checks),
            "passed": sum(1 for c in self.checks if c.status == "passed"),
            "warnings": sum(1 for c in self.checks if c.status == "warning"),
            "failed": sum(1 for c in self.checks if c.status == "failed"),
        }

    @property
    def all_passed(self) -> bool:
        """Check if all critical checks passed (warnings allowed)."""
        return self.summary.get("failed", 0) == 0

    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.summary["total"] == 0:
            return 100.0
        passed = self.summary["passed"]
        warnings = self.summary["warnings"]
        total = self.summary["total"]
        # Passed checks: full credit, warnings: partial credit
        score = (passed + 0.5 * warnings) / total * 100
        return round(score, 1)

    def add_check(self, result: CheckResult) -> None:
        """Add a check result and update summary."""
        self.checks.append(result)
        self.summary = self._compute_summary()

    def get_failed_checks(self) -> List[CheckResult]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == "failed"]

    def get_warnings(self) -> List[CheckResult]:
        """Get all warning checks."""
        return [c for c in self.checks if c.status == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "summary": self.summary,
            "health_score": self.health_score,
            "all_passed": self.all_passed,
            "checks": [c.to_dict() for c in self.checks],
            "recommendations": self.recommendations,
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """
        Save health report as JSON.

        Args:
            output_path: Path to save JSON file

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        with output_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Health report saved to {output_path}")
        return output_path

    def save_markdown(self, output_path: Union[str, Path]) -> Path:
        """
        Save health report as Markdown.

        Args:
            output_path: Path to save Markdown file

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        md_content = self._generate_markdown()
        output_path.write_text(md_content)
        logger.info(f"Health report (Markdown) saved to {output_path}")
        return output_path

    def _generate_markdown(self) -> str:
        """Generate Markdown report content."""
        lines = [
            f"# Export Health Report: {self.model_name}",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Health Score:** {self.health_score}/100",
            "",
            "## Summary",
            "",
            f"- **Total Checks:** {self.summary['total']}",
            f"- **✅ Passed:** {self.summary['passed']}",
            f"- **⚠️  Warnings:** {self.summary['warnings']}",
            f"- **❌ Failed:** {self.summary['failed']}",
            "",
        ]

        # Overall status
        if self.all_passed:
            lines.extend([
                "**Status:** ✅ All critical checks passed - ready for production",
                "",
            ])
        else:
            lines.extend([
                "**Status:** ❌ Some checks failed - review required before production deployment",
                "",
            ])

        # Failed checks
        failed = self.get_failed_checks()
        if failed:
            lines.extend([
                "## ❌ Failed Checks",
                "",
            ])
            for check in failed:
                lines.append(f"### {check.check_name}")
                lines.append(f"**Message:** {check.message}")
                if check.details:
                    lines.append("**Details:**")
                    lines.append("```json")
                    lines.append(json.dumps(check.details, indent=2))
                    lines.append("```")
                lines.append("")

        # Warnings
        warnings = self.get_warnings()
        if warnings:
            lines.extend([
                "## ⚠️  Warnings",
                "",
            ])
            for check in warnings:
                lines.append(f"### {check.check_name}")
                lines.append(f"**Message:** {check.message}")
                if check.details:
                    lines.append("**Details:**")
                    lines.append("```json")
                    lines.append(json.dumps(check.details, indent=2))
                    lines.append("```")
                lines.append("")

        # All checks
        lines.extend([
            "## All Checks",
            "",
            "| Check | Status | Duration (s) |",
            "|-------|--------|--------------|",
        ])

        for check in self.checks:
            status_emoji = {
                "passed": "✅",
                "warning": "⚠️",
                "failed": "❌",
            }
            emoji = status_emoji.get(check.status, "❓")
            lines.append(
                f"| {check.check_name} | {emoji} {check.status} | {check.duration_seconds:.3f} |"
            )

        lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## 💡 Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a concise summary to stdout."""
        print("\n" + "=" * 80)
        print(f"Export Health Report: {self.model_name}")
        print("=" * 80)
        print(f"Health Score: {self.health_score}/100")
        print(f"Total Checks: {self.summary['total']}")
        print(f"  ✅ Passed: {self.summary['passed']}")
        print(f"  ⚠️  Warnings: {self.summary['warnings']}")
        print(f"  ❌ Failed: {self.summary['failed']}")
        print("=" * 80)

        if not self.all_passed:
            print("\n❌ Critical issues detected:")
            for check in self.get_failed_checks():
                print(f"  - {check.check_name}: {check.message}")
            print()


class ExportHealthChecker:
    """
    Comprehensive health checker for model exports.

    Performs three stages of validation:
    1. Pre-export: Model architecture, parameters, memory requirements
    2. Format-specific: ONNX/TorchScript/PyTorch validation
    3. Post-export: Numerical consistency, performance benchmarks

    Example:
        >>> checker = ExportHealthChecker(model, config, task_spec)
        >>> report = checker.run_all_checks(export_dir)
        >>> if report.all_passed:
        ...     print("Export is production-ready!")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        task_spec: Any,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize health checker.

        Args:
            model: PyTorch model to validate
            config: Model configuration (SimpleNamespace or dict)
            task_spec: TaskSpec with task information
            device: Device for validation (defaults to model's device)
        """
        self.model = model
        self.config = config
        self.task_spec = task_spec
        self.device = device or next(model.parameters()).device
        self.model_name = getattr(task_spec, "name", "unknown-model")

    def run_all_checks(
        self,
        export_dir: Optional[Path] = None,
        formats: Optional[List[str]] = None,
    ) -> ExportHealthReport:
        """
        Run all health checks and generate comprehensive report.

        Args:
            export_dir: Directory containing exported artifacts (for post-export checks)
            formats: List of export formats to validate (e.g., ["onnx", "torchscript"])

        Returns:
            ExportHealthReport with all check results
        """
        report = ExportHealthReport(
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
        )

        logger.info("Running pre-export health checks...")
        report.checks.extend(self._run_pre_export_checks())

        if export_dir and formats:
            logger.info("Running format-specific validation...")
            report.checks.extend(self._run_format_checks(export_dir, formats))

            logger.info("Running post-export verification...")
            report.checks.extend(self._run_post_export_checks(export_dir, formats))

        # Generate recommendations based on results
        report.recommendations = self._generate_recommendations(report)

        return report

    def _run_pre_export_checks(self) -> List[CheckResult]:
        """Run pre-export health checks."""
        checks = []

        # Architecture validation
        checks.append(self._check_architecture())

        # Parameter checks
        checks.append(self._check_parameters())

        # Input/output shape validation
        checks.append(self._check_input_output_shapes())

        # Memory requirements
        checks.append(self._check_memory_requirements())

        # Forward pass validation
        checks.append(self._check_forward_pass())

        return checks

    def _check_architecture(self) -> CheckResult:
        """Validate model architecture."""
        start = time.time()
        try:
            # Count layers by type
            layer_counts: Dict[str, int] = {}
            for name, module in self.model.named_modules():
                module_type = type(module).__name__
                layer_counts[module_type] = layer_counts.get(module_type, 0) + 1

            # Check for common issues
            has_batchnorm = any("BatchNorm" in k for k in layer_counts)
            has_dropout = any("Dropout" in k for k in layer_counts)

            details = {
                "total_modules": len(list(self.model.modules())),
                "layer_counts": layer_counts,
                "has_batchnorm": has_batchnorm,
                "has_dropout": has_dropout,
            }

            duration = time.time() - start
            return CheckResult(
                check_name="architecture_validation",
                status="passed",
                message=f"Architecture validated: {details['total_modules']} modules",
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="architecture_validation",
                status="failed",
                message=f"Architecture validation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_parameters(self) -> CheckResult:
        """Check model parameters for NaN/Inf and compute statistics."""
        start = time.time()
        try:
            total_params = 0
            trainable_params = 0
            nan_params = 0
            inf_params = 0
            param_stats: Dict[str, Any] = {}

            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

                # Check for NaN/Inf
                if torch.isnan(param.data).any():
                    nan_params += 1
                    logger.warning(f"NaN detected in parameter: {name}")
                if torch.isinf(param.data).any():
                    inf_params += 1
                    logger.warning(f"Inf detected in parameter: {name}")

                # Compute statistics for first few parameters
                if len(param_stats) < 5:
                    param_stats[name] = {
                        "shape": list(param.shape),
                        "mean": float(param.data.mean()),
                        "std": float(param.data.std()),
                        "min": float(param.data.min()),
                        "max": float(param.data.max()),
                    }

            details = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "nan_params": nan_params,
                "inf_params": inf_params,
                "sample_stats": param_stats,
            }

            # Determine status
            if nan_params > 0 or inf_params > 0:
                status = "failed"
                message = f"Found {nan_params} NaN and {inf_params} Inf parameters"
            else:
                status = "passed"
                message = f"All {total_params:,} parameters are valid"

            duration = time.time() - start
            return CheckResult(
                check_name="parameter_validation",
                status=status,
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="parameter_validation",
                status="failed",
                message=f"Parameter check failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_input_output_shapes(self) -> CheckResult:
        """Validate input/output shapes with dummy data."""
        start = time.time()
        try:
            # Generate dummy input based on task spec
            dummy_input = self._generate_dummy_input()

            # Run forward pass
            self.model.eval()
            with torch.no_grad():
                output = self.model(dummy_input)

            # Extract output tensor
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            elif isinstance(output, tuple):
                output_shape = list(output[0].shape)
            elif isinstance(output, dict):
                logits = output.get("logits", output.get("last_hidden_state"))
                output_shape = list(logits.shape) if logits is not None else []
            else:
                output_shape = []

            details = {
                "input_shape": list(dummy_input.shape),
                "output_shape": output_shape,
                "input_dtype": str(dummy_input.dtype),
                "output_dtype": str(output.dtype) if isinstance(output, torch.Tensor) else "unknown",
            }

            duration = time.time() - start
            return CheckResult(
                check_name="shape_validation",
                status="passed",
                message=f"Input {details['input_shape']} → Output {details['output_shape']}",
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="shape_validation",
                status="failed",
                message=f"Shape validation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_memory_requirements(self) -> CheckResult:
        """Estimate memory requirements for inference."""
        start = time.time()
        try:
            # Parameter memory
            param_memory_mb = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            ) / (1024 ** 2)

            # Estimate activation memory with dummy forward pass
            dummy_input = self._generate_dummy_input()

            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize()

                self.model.eval()
                with torch.no_grad():
                    _ = self.model(dummy_input)

                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            else:
                # Rough estimate for CPU
                peak_memory_mb = param_memory_mb * 2

            details = {
                "parameter_memory_mb": round(param_memory_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "estimated_batch_memory_mb": round(peak_memory_mb * 8, 2),  # Estimate for batch size 8
            }

            # Warning if memory is very large
            if peak_memory_mb > 1000:
                status = "warning"
                message = f"Large memory footprint: {peak_memory_mb:.1f} MB"
            else:
                status = "passed"
                message = f"Memory requirements: {peak_memory_mb:.1f} MB"

            duration = time.time() - start
            return CheckResult(
                check_name="memory_requirements",
                status=status,
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="memory_requirements",
                status="warning",
                message=f"Could not estimate memory: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_forward_pass(self) -> CheckResult:
        """Validate forward pass with multiple inputs."""
        start = time.time()
        try:
            self.model.eval()
            test_cases = [
                ("batch_size_1", 1),
                ("batch_size_4", 4),
            ]

            results = []
            for name, batch_size in test_cases:
                try:
                    dummy_input = self._generate_dummy_input(batch_size=batch_size)
                    with torch.no_grad():
                        output = self.model(dummy_input)

                    # Check output is valid
                    if isinstance(output, torch.Tensor):
                        has_nan = torch.isnan(output).any().item()
                        has_inf = torch.isinf(output).any().item()
                    else:
                        has_nan = False
                        has_inf = False

                    results.append({
                        "test_case": name,
                        "batch_size": batch_size,
                        "success": True,
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                    })
                except Exception as e:
                    results.append({
                        "test_case": name,
                        "batch_size": batch_size,
                        "success": False,
                        "error": str(e),
                    })

            # Check if all passed
            all_success = all(r["success"] for r in results)
            any_nan_inf = any(r.get("has_nan", False) or r.get("has_inf", False) for r in results)

            if not all_success:
                status = "failed"
                message = "Forward pass failed for some inputs"
            elif any_nan_inf:
                status = "failed"
                message = "Forward pass produced NaN/Inf outputs"
            else:
                status = "passed"
                message = f"Forward pass validated for {len(results)} test cases"

            duration = time.time() - start
            return CheckResult(
                check_name="forward_pass_validation",
                status=status,
                message=message,
                details={"test_results": results},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="forward_pass_validation",
                status="failed",
                message=f"Forward pass validation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _run_format_checks(self, export_dir: Path, formats: List[str]) -> List[CheckResult]:
        """Run format-specific validation checks."""
        checks = []

        if "onnx" in formats:
            checks.append(self._check_onnx_format(export_dir))

        if "torchscript" in formats:
            checks.append(self._check_torchscript_format(export_dir))

        if "pytorch" in formats:
            checks.append(self._check_pytorch_format(export_dir))

        return checks

    def _check_onnx_format(self, export_dir: Path) -> CheckResult:
        """Validate ONNX export."""
        start = time.time()
        onnx_path = export_dir / "artifacts" / "model.onnx"

        if not onnx_path.exists():
            onnx_path = export_dir / "model.onnx"

        if not onnx_path.exists():
            duration = time.time() - start
            return CheckResult(
                check_name="onnx_validation",
                status="warning",
                message="ONNX file not found",
                details={"expected_path": str(onnx_path)},
                duration_seconds=duration,
            )

        try:
            import onnx
            import onnx.checker

            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            # Get model info
            details = {
                "file_size_mb": round(onnx_path.stat().st_size / (1024 ** 2), 2),
                "ir_version": onnx_model.ir_version,
                "opset_version": (
                    onnx_model.opset_import[0].version if onnx_model.opset_import else None
                ),
                "num_nodes": len(onnx_model.graph.node),
            }

            duration = time.time() - start
            return CheckResult(
                check_name="onnx_validation",
                status="passed",
                message="ONNX model is valid",
                details=details,
                duration_seconds=duration,
            )

        except ImportError:
            duration = time.time() - start
            return CheckResult(
                check_name="onnx_validation",
                status="warning",
                message="onnx package not available for validation",
                details={},
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="onnx_validation",
                status="failed",
                message=f"ONNX validation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_torchscript_format(self, export_dir: Path) -> CheckResult:
        """Validate TorchScript export."""
        start = time.time()
        ts_path = export_dir / "artifacts" / "model.torchscript.pt"

        if not ts_path.exists():
            ts_path = export_dir / "model.torchscript.pt"

        if not ts_path.exists():
            duration = time.time() - start
            return CheckResult(
                check_name="torchscript_validation",
                status="warning",
                message="TorchScript file not found",
                details={"expected_path": str(ts_path)},
                duration_seconds=duration,
            )

        try:
            # Load TorchScript model
            scripted_model = torch.jit.load(str(ts_path), map_location=self.device)
            scripted_model.eval()

            # Test with dummy input
            dummy_input = self._generate_dummy_input(batch_size=1)
            with torch.no_grad():
                output = scripted_model(dummy_input)

            # Check output validity
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
            else:
                has_nan = False
                has_inf = False

            details = {
                "file_size_mb": round(ts_path.stat().st_size / (1024 ** 2), 2),
                "test_forward_pass": "success",
                "has_nan": has_nan,
                "has_inf": has_inf,
            }

            if has_nan or has_inf:
                status = "failed"
                message = "TorchScript model produces NaN/Inf outputs"
            else:
                status = "passed"
                message = "TorchScript model is valid"

            duration = time.time() - start
            return CheckResult(
                check_name="torchscript_validation",
                status=status,
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="torchscript_validation",
                status="failed",
                message=f"TorchScript validation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_pytorch_format(self, export_dir: Path) -> CheckResult:
        """Validate PyTorch state dict export."""
        start = time.time()
        pt_path = export_dir / "artifacts" / "model.pytorch.pt"

        if not pt_path.exists():
            pt_path = export_dir / "pytorch" / "pytorch_model.bin"

        if not pt_path.exists():
            duration = time.time() - start
            return CheckResult(
                check_name="pytorch_validation",
                status="warning",
                message="PyTorch state dict not found",
                details={"expected_path": str(pt_path)},
                duration_seconds=duration,
            )

        try:
            # Load state dict
            state_dict = torch.load(str(pt_path), map_location="cpu")

            # Validate keys and shapes
            num_keys = len(state_dict)
            total_params = sum(v.numel() for v in state_dict.values())

            # Check for NaN/Inf in state dict
            nan_tensors = sum(1 for v in state_dict.values() if torch.isnan(v).any())
            inf_tensors = sum(1 for v in state_dict.values() if torch.isinf(v).any())

            details = {
                "file_size_mb": round(pt_path.stat().st_size / (1024 ** 2), 2),
                "num_keys": num_keys,
                "total_params": total_params,
                "nan_tensors": nan_tensors,
                "inf_tensors": inf_tensors,
            }

            if nan_tensors > 0 or inf_tensors > 0:
                status = "failed"
                message = f"State dict contains {nan_tensors} NaN and {inf_tensors} Inf tensors"
            else:
                status = "passed"
                message = f"PyTorch state dict is valid ({num_keys} keys, {total_params:,} params)"

            duration = time.time() - start
            return CheckResult(
                check_name="pytorch_validation",
                status=status,
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="pytorch_validation",
                status="failed",
                message=f"PyTorch validation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _run_post_export_checks(self, export_dir: Path, formats: List[str]) -> List[CheckResult]:
        """Run post-export verification checks."""
        checks = []

        # Numerical consistency checks
        if "onnx" in formats:
            checks.append(self._check_numerical_consistency_onnx(export_dir))

        if "torchscript" in formats:
            checks.append(self._check_numerical_consistency_torchscript(export_dir))

        # Performance benchmark
        checks.append(self._check_performance_benchmark(export_dir, formats))

        return checks

    def _check_numerical_consistency_onnx(self, export_dir: Path) -> CheckResult:
        """Compare ONNX outputs with PyTorch."""
        start = time.time()
        onnx_path = export_dir / "artifacts" / "model.onnx"

        if not onnx_path.exists():
            onnx_path = export_dir / "model.onnx"

        if not onnx_path.exists():
            duration = time.time() - start
            return CheckResult(
                check_name="onnx_numerical_consistency",
                status="warning",
                message="ONNX file not found for consistency check",
                details={},
                duration_seconds=duration,
            )

        try:
            import onnxruntime as ort
            import numpy as np

            # Generate test input
            dummy_input = self._generate_dummy_input(batch_size=1)

            # PyTorch inference
            self.model.eval()
            with torch.no_grad():
                pytorch_output = self.model(dummy_input)
                if isinstance(pytorch_output, torch.Tensor):
                    pytorch_output = pytorch_output
                elif isinstance(pytorch_output, tuple):
                    pytorch_output = pytorch_output[0]
                elif isinstance(pytorch_output, dict):
                    pytorch_output = pytorch_output.get("logits", pytorch_output.get("last_hidden_state"))
                pytorch_output = pytorch_output.cpu().numpy()

            # ONNX inference
            ort_session = ort.InferenceSession(str(onnx_path))
            onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]

            # Compare outputs
            max_error = float(np.abs(pytorch_output - onnx_output).max())
            mean_error = float(np.abs(pytorch_output - onnx_output).mean())
            relative_error = float(np.abs((pytorch_output - onnx_output) / (pytorch_output + 1e-8)).mean())

            details = {
                "max_absolute_error": max_error,
                "mean_absolute_error": mean_error,
                "relative_error": relative_error,
            }

            # Determine status based on error thresholds
            if max_error > 1e-2:
                status = "failed"
                message = f"Large numerical difference: max error {max_error:.6f}"
            elif max_error > 1e-4:
                status = "warning"
                message = f"Moderate numerical difference: max error {max_error:.6f}"
            else:
                status = "passed"
                message = f"Numerical consistency validated: max error {max_error:.6f}"

            duration = time.time() - start
            return CheckResult(
                check_name="onnx_numerical_consistency",
                status=status,
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except ImportError:
            duration = time.time() - start
            return CheckResult(
                check_name="onnx_numerical_consistency",
                status="warning",
                message="onnxruntime not available for consistency check",
                details={},
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="onnx_numerical_consistency",
                status="failed",
                message=f"Numerical consistency check failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_numerical_consistency_torchscript(self, export_dir: Path) -> CheckResult:
        """Compare TorchScript outputs with PyTorch."""
        start = time.time()
        ts_path = export_dir / "artifacts" / "model.torchscript.pt"

        if not ts_path.exists():
            ts_path = export_dir / "model.torchscript.pt"

        if not ts_path.exists():
            duration = time.time() - start
            return CheckResult(
                check_name="torchscript_numerical_consistency",
                status="warning",
                message="TorchScript file not found for consistency check",
                details={},
                duration_seconds=duration,
            )

        try:
            # Load TorchScript model
            scripted_model = torch.jit.load(str(ts_path), map_location=self.device)
            scripted_model.eval()

            # Generate test input
            dummy_input = self._generate_dummy_input(batch_size=1)

            # PyTorch inference
            self.model.eval()
            with torch.no_grad():
                pytorch_output = self.model(dummy_input)
                if isinstance(pytorch_output, tuple):
                    pytorch_output = pytorch_output[0]
                elif isinstance(pytorch_output, dict):
                    pytorch_output = pytorch_output.get("logits", pytorch_output.get("last_hidden_state"))

                # TorchScript inference
                scripted_output = scripted_model(dummy_input)
                if isinstance(scripted_output, tuple):
                    scripted_output = scripted_output[0]

            # Compare
            max_error = (pytorch_output - scripted_output).abs().max().item()
            mean_error = (pytorch_output - scripted_output).abs().mean().item()

            details = {
                "max_absolute_error": max_error,
                "mean_absolute_error": mean_error,
            }

            if max_error > 1e-4:
                status = "warning"
                message = f"Numerical difference detected: max error {max_error:.6f}"
            else:
                status = "passed"
                message = f"Numerical consistency validated: max error {max_error:.6f}"

            duration = time.time() - start
            return CheckResult(
                check_name="torchscript_numerical_consistency",
                status=status,
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="torchscript_numerical_consistency",
                status="failed",
                message=f"Numerical consistency check failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _check_performance_benchmark(self, export_dir: Path, formats: List[str]) -> CheckResult:
        """Benchmark inference performance."""
        start = time.time()

        try:
            results = {}
            num_runs = 50

            # Benchmark PyTorch
            dummy_input = self._generate_dummy_input(batch_size=1)
            self.model.eval()

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = self.model(dummy_input)

            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms

            results["pytorch_ms"] = round(pytorch_time, 3)

            # Benchmark exports if available
            if "onnx" in formats:
                onnx_path = export_dir / "artifacts" / "model.onnx"
                if not onnx_path.exists():
                    onnx_path = export_dir / "model.onnx"

                if onnx_path.exists():
                    try:
                        import onnxruntime as ort
                        ort_session = ort.InferenceSession(str(onnx_path))
                        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}

                        # Warmup
                        for _ in range(5):
                            _ = ort_session.run(None, onnx_input)

                        start_time = time.time()
                        for _ in range(num_runs):
                            _ = ort_session.run(None, onnx_input)
                        onnx_time = (time.time() - start_time) / num_runs * 1000

                        results["onnx_ms"] = round(onnx_time, 3)
                        results["onnx_speedup"] = round(pytorch_time / onnx_time, 2)
                    except Exception:
                        pass

            details = results
            message = f"PyTorch: {pytorch_time:.2f}ms"
            if "onnx_ms" in results:
                message += f", ONNX: {results['onnx_ms']:.2f}ms ({results['onnx_speedup']:.2f}x)"

            duration = time.time() - start
            return CheckResult(
                check_name="performance_benchmark",
                status="passed",
                message=message,
                details=details,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            return CheckResult(
                check_name="performance_benchmark",
                status="warning",
                message=f"Performance benchmark failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=duration,
            )

    def _generate_dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """Generate dummy input based on task spec."""
        modality = getattr(self.task_spec, "modality", "text")

        if modality == "vision":
            image_size = self.task_spec.input_schema.get("image_size", [3, 224, 224])
            c, h, w = image_size
            return torch.rand(batch_size, c, h, w, device=self.device)
        else:
            vocab_size = self.task_spec.input_schema.get("vocab_size", 50257)
            max_seq_len = self.task_spec.input_schema.get("max_seq_len", 128)
            return torch.randint(0, vocab_size, (batch_size, max_seq_len), device=self.device)

    def _generate_recommendations(self, report: ExportHealthReport) -> List[str]:
        """Generate actionable recommendations based on check results."""
        recommendations = []

        # Check for failed checks
        failed = report.get_failed_checks()
        if failed:
            recommendations.append(
                "🚨 Address all failed checks before production deployment"
            )

            for check in failed:
                if "nan" in check.check_name.lower() or "nan" in check.message.lower():
                    recommendations.append(
                        "Fix NaN parameters: Check training stability, learning rate, and gradient clipping"
                    )
                elif "consistency" in check.check_name.lower():
                    recommendations.append(
                        "Numerical inconsistency detected: Verify export settings and precision"
                    )

        # Check for warnings
        warnings = report.get_warnings()
        if warnings:
            for check in warnings:
                if "memory" in check.check_name.lower():
                    recommendations.append(
                        "Large memory footprint: Consider quantization or model compression"
                    )

        # General recommendations
        if report.health_score < 100:
            recommendations.append(
                "Review all warnings and consider optimizations before deployment"
            )

        if not recommendations:
            recommendations.append(
                "✅ All checks passed! Model is ready for production deployment"
            )

        return recommendations
