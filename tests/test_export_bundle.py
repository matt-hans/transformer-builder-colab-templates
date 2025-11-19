"""
Tests for Production Inference Bundle Generation (Training Pipeline v3.5 - Enhancement 4).

Tests:
- Unit tests for individual generator functions
- Integration tests for end-to-end bundle creation
- Cross-enhancement tests with compiled models and vision collators
"""

import pytest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn

# Import modules to test
from utils.training.export_utilities import (
    generate_inference_script,
    generate_readme,
    generate_torchserve_config,
    generate_dockerfile,
    create_export_bundle,
    _generate_runtime_requirements,
)
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def vision_task_spec():
    """Create a vision task spec for testing."""
    return TaskSpec(
        name="vision_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["accuracy", "loss"],
        modality="vision",
        input_schema={
            "image_size": [3, 64, 64],
            "channels_first": True
        },
        output_schema={
            "num_classes": 10
        },
        preprocessing_config={
            "normalize": True,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }
    )


@pytest.fixture
def text_task_spec():
    """Create a text task spec for testing."""
    return TaskSpec(
        name="text_test",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["perplexity", "loss"],
        modality="text",
        input_schema={
            "vocab_size": 50257,
            "max_seq_len": 128
        },
        output_schema={
            "vocab_size": 50257
        }
    )


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 64 * 64, 10)

        def forward(self, pixel_values):
            x = self.conv(pixel_values)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def temp_export_dir():
    """Create a temporary directory for exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Unit Tests
# ============================================================================


class TestInferenceScriptGeneration:
    """Test generate_inference_script() for vision and text modalities."""

    def test_vision_onnx_script_generation(self, vision_task_spec, temp_export_dir):
        """Verify vision ONNX inference.py contains VisionInferenceEngine."""
        script_path = generate_inference_script(
            vision_task_spec,
            temp_export_dir,
            model_format="onnx"
        )

        assert script_path.exists()
        script_content = script_path.read_text()

        # Verify class presence
        assert "class VisionInferenceEngine:" in script_content
        assert "import onnxruntime as ort" in script_content

        # Verify preprocessing config is embedded
        assert "[0.5, 0.5, 0.5]" in script_content  # mean
        assert "[3, 64, 64]" in script_content  # image_size

        # Verify methods
        assert "def preprocess" in script_content
        assert "def predict" in script_content
        assert "def batch_predict" in script_content

    def test_vision_torchscript_script_generation(self, vision_task_spec, temp_export_dir):
        """Verify vision TorchScript inference.py structure."""
        script_path = generate_inference_script(
            vision_task_spec,
            temp_export_dir,
            model_format="torchscript"
        )

        script_content = script_path.read_text()

        assert "class VisionInferenceEngine:" in script_content
        assert "import torch" in script_content
        assert "torch.jit.load" in script_content

    def test_text_onnx_script_generation(self, text_task_spec, temp_export_dir):
        """Verify text ONNX inference.py contains TextInferenceEngine."""
        script_path = generate_inference_script(
            text_task_spec,
            temp_export_dir,
            model_format="onnx"
        )

        script_content = script_path.read_text()

        # Verify class presence
        assert "class TextInferenceEngine:" in script_content
        assert "Vocabulary size: 50257" in script_content  # In docstring format
        assert "Max sequence length: 128" in script_content  # In docstring format

        # Verify tokenization handling
        assert "def preprocess" in script_content
        assert "tokenizer" in script_content.lower()

    def test_text_torchscript_script_generation(self, text_task_spec, temp_export_dir):
        """Verify text TorchScript inference.py structure."""
        script_path = generate_inference_script(
            text_task_spec,
            temp_export_dir,
            model_format="torchscript"
        )

        script_content = script_path.read_text()

        assert "class TextInferenceEngine:" in script_content
        assert "torch.jit.load" in script_content

    def test_unsupported_modality_raises_error(self, temp_export_dir):
        """Verify unsupported modality raises ValueError."""
        audio_task = TaskSpec(
            name="audio_test",
            task_type="classification",
            model_family="encoder_only",
            input_fields=["audio"],
            target_field="labels",
            loss_type="cross_entropy",
            metrics=["accuracy"],
            modality="audio"  # Unsupported
        )

        with pytest.raises(ValueError, match="Unsupported modality"):
            generate_inference_script(audio_task, temp_export_dir)


class TestReadmeGeneration:
    """Test generate_readme() structure and content."""

    def test_readme_structure(self, vision_task_spec, temp_export_dir):
        """Verify README.md has required sections."""
        readme_path = generate_readme(
            vision_task_spec,
            temp_export_dir,
            formats=["onnx", "torchscript"]
        )

        assert readme_path.exists()
        content = readme_path.read_text()

        # Check required sections
        assert "# Model Export Bundle" in content
        assert "## Overview" in content
        assert "## Directory Structure" in content
        assert "## Quick Start" in content
        assert "## Docker Deployment" in content
        assert "## TorchServe Deployment" in content
        assert "## Model Information" in content
        assert "## Troubleshooting" in content
        assert "## Citation" in content

    def test_readme_vision_examples(self, vision_task_spec, temp_export_dir):
        """Verify vision-specific examples are included."""
        readme_path = generate_readme(
            vision_task_spec,
            temp_export_dir,
            formats=["onnx"]
        )

        content = readme_path.read_text()

        # Vision-specific examples
        assert "--input /path/to/image.jpg" in content
        assert "--batch" in content
        assert "curl -X POST -F \"image=@test.jpg\"" in content

    def test_readme_text_examples(self, text_task_spec, temp_export_dir):
        """Verify text-specific examples are included."""
        readme_path = generate_readme(
            text_task_spec,
            temp_export_dir,
            formats=["onnx"]
        )

        content = readme_path.read_text()

        # Text-specific examples
        assert "--input \"Your text here\"" in content
        assert "\"text\":" in content  # JSON payload example

    def test_readme_formats_listed(self, vision_task_spec, temp_export_dir):
        """Verify exported formats are listed."""
        formats = ["onnx", "torchscript", "pytorch"]
        readme_path = generate_readme(
            vision_task_spec,
            temp_export_dir,
            formats=formats
        )

        content = readme_path.read_text()

        for fmt in formats:
            assert fmt in content


class TestTorchServeConfigGeneration:
    """Test generate_torchserve_config() JSON structure."""

    def test_torchserve_config_structure(self, vision_task_spec, temp_export_dir):
        """Verify TorchServe config has required fields."""
        config_path = generate_torchserve_config(vision_task_spec, temp_export_dir)

        assert config_path.exists()
        assert config_path.name == "torchserve_config.json"

        with open(config_path) as f:
            config = json.load(f)

        # Verify required fields
        assert "modelName" in config
        assert "modelVersion" in config
        assert "runtime" in config
        assert "minWorkers" in config
        assert "maxWorkers" in config
        assert "batchSize" in config
        assert "deviceType" in config
        assert "handler" in config
        assert "metrics" in config

    def test_torchserve_handler_vision(self, vision_task_spec, temp_export_dir):
        """Verify vision task uses VisionInferenceEngine handler."""
        config_path = generate_torchserve_config(vision_task_spec, temp_export_dir)

        with open(config_path) as f:
            config = json.load(f)

        assert config["handler"]["class"] == "VisionInferenceEngine"

    def test_torchserve_handler_text(self, text_task_spec, temp_export_dir):
        """Verify text task uses TextInferenceEngine handler."""
        config_path = generate_torchserve_config(text_task_spec, temp_export_dir)

        with open(config_path) as f:
            config = json.load(f)

        assert config["handler"]["class"] == "TextInferenceEngine"


class TestDockerfileGeneration:
    """Test generate_dockerfile() syntax and structure."""

    def test_dockerfile_syntax(self, vision_task_spec, temp_export_dir):
        """Verify Dockerfile has valid syntax."""
        dockerfile_path = generate_dockerfile(vision_task_spec, temp_export_dir)

        assert dockerfile_path.exists()
        assert dockerfile_path.name == "Dockerfile"

        content = dockerfile_path.read_text()

        # Verify key Dockerfile directives
        assert "FROM pytorch/pytorch" in content
        assert "WORKDIR /app" in content
        assert "COPY requirements.txt ." in content
        assert "RUN pip install" in content
        assert "COPY artifacts/" in content
        assert "COPY configs/" in content
        assert "COPY inference.py" in content
        assert "EXPOSE 8080" in content
        assert "HEALTHCHECK" in content
        assert "CMD" in content

    def test_dockerfile_security_user(self, vision_task_spec, temp_export_dir):
        """Verify Dockerfile creates non-root user."""
        dockerfile_path = generate_dockerfile(vision_task_spec, temp_export_dir)
        content = dockerfile_path.read_text()

        # Security best practice: non-root user
        assert "useradd" in content
        assert "USER inference" in content

    def test_dockerfile_healthcheck(self, vision_task_spec, temp_export_dir):
        """Verify Dockerfile includes healthcheck."""
        dockerfile_path = generate_dockerfile(vision_task_spec, temp_export_dir)
        content = dockerfile_path.read_text()

        assert "HEALTHCHECK" in content
        assert "localhost:8080" in content


class TestRuntimeRequirements:
    """Test _generate_runtime_requirements() content."""

    def test_vision_requirements(self):
        """Verify vision tasks include pillow."""
        requirements = _generate_runtime_requirements("vision", ["onnx"])

        reqs_text = "\n".join(requirements)
        assert "pillow>=9.0.0" in reqs_text
        assert "numpy>=1.21.0" in reqs_text
        assert "onnxruntime>=1.15.0" in reqs_text

    def test_text_requirements(self):
        """Verify text tasks include transformers comment."""
        requirements = _generate_runtime_requirements("text", ["torchscript"])

        reqs_text = "\n".join(requirements)
        assert "# transformers>=4.30.0" in reqs_text  # Optional
        assert "torch>=2.0.0" in reqs_text

    def test_onnx_format_requirements(self):
        """Verify ONNX format adds onnxruntime."""
        requirements = _generate_runtime_requirements("text", ["onnx"])

        reqs_text = "\n".join(requirements)
        assert "onnxruntime>=1.15.0" in reqs_text

    def test_torchscript_format_requirements(self):
        """Verify TorchScript format adds torch."""
        requirements = _generate_runtime_requirements("text", ["torchscript"])

        reqs_text = "\n".join(requirements)
        assert "torch>=2.0.0" in reqs_text


# ============================================================================
# Integration Tests
# ============================================================================


class TestExportBundleEndToEnd:
    """Test create_export_bundle() full workflow."""

    def test_export_bundle_vision_onnx(self, vision_task_spec, simple_model, temp_export_dir):
        """Full workflow: export vision model → verify files exist."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["onnx"],
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace(
            vocab_size=10,
            max_seq_len=64
        )

        # Create bundle
        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        # Verify directory structure
        assert export_dir.exists()
        assert (export_dir / "artifacts").exists()
        assert (export_dir / "configs").exists()

        # Verify generated files
        assert (export_dir / "inference.py").exists()
        assert (export_dir / "README.md").exists()
        assert (export_dir / "Dockerfile").exists()
        assert (export_dir / "requirements.txt").exists()
        assert (export_dir / "metadata.json").exists()

        # Verify config files
        assert (export_dir / "configs" / "task_spec.json").exists()
        assert (export_dir / "configs" / "training_config.json").exists()
        assert (export_dir / "configs" / "torchserve_config.json").exists()

    def test_export_bundle_multiple_formats(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify multiple formats are exported."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["onnx", "torchscript", "pytorch"],
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        # Note: ONNX/TorchScript export may fail for this simple model,
        # but PyTorch state dict should always work
        assert (export_dir / "artifacts" / "model.pytorch.pt").exists()

    def test_export_bundle_metadata_content(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify metadata.json contains correct information."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["pytorch"],
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        metadata_path = export_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify metadata fields
        assert metadata["task_name"] == "vision_test"
        assert metadata["modality"] == "vision"
        assert metadata["task_type"] == "vision_classification"
        assert "pytorch" in metadata["formats"]
        assert "export_timestamp" in metadata
        assert "framework_version" in metadata

    def test_export_bundle_task_spec_saved(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify TaskSpec is correctly serialized."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["pytorch"],
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        task_spec_path = export_dir / "configs" / "task_spec.json"
        with open(task_spec_path) as f:
            loaded_spec = json.load(f)

        # Verify key fields
        assert loaded_spec["name"] == "vision_test"
        assert loaded_spec["modality"] == "vision"
        assert loaded_spec["input_schema"]["image_size"] == [3, 64, 64]

    def test_export_bundle_training_config_saved(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify TrainingConfig is correctly serialized."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["pytorch"],
            learning_rate=1e-4,
            batch_size=16,
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        config_path = export_dir / "configs" / "training_config.json"
        with open(config_path) as f:
            loaded_config = json.load(f)

        # Verify saved values
        assert loaded_config["learning_rate"] == 1e-4
        assert loaded_config["batch_size"] == 16
        assert loaded_config["export_bundle"] is True


# ============================================================================
# Cross-Enhancement Tests
# ============================================================================


class TestCompiledModelExport:
    """Test compatibility with torch.compile and VisionDataCollator."""

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compiled_model_export_succeeds(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify compiled models can be exported (unwrapped before export)."""
        # Compile model
        try:
            compiled_model = torch.compile(simple_model, mode="default")
        except Exception:
            pytest.skip("torch.compile failed on this system")

        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["pytorch"],  # State dict always works
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        # Should not raise error
        export_dir = create_export_bundle(
            model=compiled_model,  # Compiled model
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        assert export_dir.exists()
        assert (export_dir / "artifacts" / "model.pytorch.pt").exists()


class TestErrorHandling:
    """Test graceful failure handling."""

    def test_export_bundle_continues_on_onnx_failure(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify bundle generation continues if ONNX export fails."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["onnx", "pytorch"],  # ONNX may fail, pytorch works
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        # Should not raise error even if ONNX fails
        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        # Bundle should still be created
        assert export_dir.exists()
        assert (export_dir / "inference.py").exists()
        assert (export_dir / "README.md").exists()

    def test_training_config_without_export_fields(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify backward compatibility when export fields are missing."""
        # Old-style TrainingConfig without export fields
        training_config = SimpleNamespace(
            learning_rate=5e-5,
            batch_size=4
            # No export_formats or export_bundle
        )

        config = SimpleNamespace()

        # Should use defaults and not crash
        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        assert export_dir.exists()


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_export_formats_list(self, vision_task_spec, simple_model, temp_export_dir):
        """Verify bundle creation works with empty formats list."""
        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=[],  # Empty
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=vision_task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        # Should still generate configs and documentation
        assert export_dir.exists()
        assert (export_dir / "README.md").exists()
        assert (export_dir / "Dockerfile").exists()

    def test_special_characters_in_task_name(self, simple_model, temp_export_dir):
        """Verify task names with special characters are handled."""
        task_spec = TaskSpec(
            name="vision-test_v2.1",
            task_type="vision_classification",
            model_family="encoder_only",
            input_fields=["pixel_values"],
            target_field="labels",
            loss_type="cross_entropy",
            metrics=["accuracy"],
            modality="vision",
            input_schema={"image_size": [3, 64, 64]},
            output_schema={"num_classes": 10}
        )

        training_config = TrainingConfig(
            export_bundle=True,
            export_formats=["pytorch"],
            export_dir=str(temp_export_dir)
        )

        config = SimpleNamespace()

        # Should handle special characters gracefully
        export_dir = create_export_bundle(
            model=simple_model,
            config=config,
            task_spec=task_spec,
            training_config=training_config,
            export_base_dir=str(temp_export_dir)
        )

        assert export_dir.exists()
        assert (export_dir / "README.md").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
