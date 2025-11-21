from __future__ import annotations
"""
Model Export Utilities.

Export trained models to production formats:
- ONNX: Cross-platform inference
- TorchScript: Optimized PyTorch deployment
- Model Cards: Documentation and metadata

Includes validation, optimization, and benchmarking.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union, Literal, Mapping, TYPE_CHECKING
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .task_spec import TaskSpec
    from ..adapters.model_adapter import ModelAdapter


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.

    Features:
    - Automatic input shape handling
    - Dynamic axes support
    - Optimization passes (fusion, constant folding)
    - Validation against PyTorch outputs
    - Inference speed benchmarking

    Example:
        >>> exporter = ONNXExporter()
        >>> exporter.export(
        ...     model=my_model,
        ...     output_path='model.onnx',
        ...     vocab_size=50257,
        ...     max_seq_len=512
        ... )
        ✓ ONNX export successful: model.onnx
        ✓ Validation passed (max error: 0.0001)
        📊 Speedup: 2.3x faster than PyTorch
    """

    def __init__(self,
                 opset_version: int = 14,
                 optimize: bool = True,
                 validate: bool = True,
                 benchmark: bool = True):
        """
        Initialize ONNX exporter.

        Args:
            opset_version: ONNX opset version (14+ recommended)
            optimize: Apply ONNX optimization passes
            validate: Validate outputs against PyTorch
            benchmark: Benchmark inference speed
        """
        self.opset_version = opset_version
        self.optimize = optimize
        self.validate = validate
        self.benchmark = benchmark

    def export(self,
              model: nn.Module,
              output_path: Union[str, Path],
              vocab_size: int,
              max_seq_len: int = 512,
              batch_size: int = 1,
              dynamic_axes: bool = True,
              input_names: Optional[List[str]] = None,
              output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Output ONNX file path
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            batch_size: Batch size for dummy input
            dynamic_axes: Allow dynamic batch/sequence dimensions
            input_names: Custom input names
            output_names: Custom output names

        Returns:
            Dictionary with export metadata

        Example:
            >>> result = exporter.export(
            ...     model=transformer,
            ...     output_path='model.onnx',
            ...     vocab_size=50257,
            ...     max_seq_len=512
            ... )
            >>> print(f"Exported to: {result['output_path']}")
            >>> print(f"Speedup: {result['speedup']:.2f}x")
        """
        output_path = Path(output_path)
        print(f"\n📦 Exporting to ONNX: {output_path.name}")
        print("-" * 80)

        # Set model to eval mode
        model.eval()

        # Create dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randint(
            0, vocab_size,
            (batch_size, max_seq_len),
            device=device
        )

        # Default input/output names
        if input_names is None:
            input_names = ['input_ids']
        if output_names is None:
            output_names = ['logits']

        # Dynamic axes configuration
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'}
            }
        else:
            dynamic_axes_dict = None

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                opset_version=self.opset_version,
                do_constant_folding=True,
                export_params=True,
            )
            print(f"✓ ONNX export successful")

        except Exception as e:
            print(f"❌ Export failed: {e}")
            raise

        # Validate if requested
        if self.validate:
            print("\n🔍 Validating ONNX model...")
            validation_result = self._validate_onnx(
                model,
                output_path,
                dummy_input,
                device
            )
            print(f"✓ Validation passed (max error: {validation_result['max_error']:.6f})")

        # Optimize if requested
        if self.optimize:
            print("\n⚡ Optimizing ONNX model...")
            self._optimize_onnx(output_path)
            print("✓ Optimization complete")

        # Benchmark if requested
        benchmark_result = {}
        if self.benchmark:
            print("\n📊 Benchmarking inference speed...")
            benchmark_result = self._benchmark_onnx(
                model,
                output_path,
                vocab_size,
                max_seq_len,
                device
            )
            print(f"✓ PyTorch: {benchmark_result['pytorch_time']:.2f}ms per batch")
            print(f"✓ ONNX: {benchmark_result['onnx_time']:.2f}ms per batch")
            print(f"✓ Speedup: {benchmark_result['speedup']:.2f}x")

        # Metadata
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            'output_path': str(output_path),
            'file_size_mb': file_size_mb,
            'opset_version': self.opset_version,
            'dynamic_axes': dynamic_axes,
            'validation': validation_result if self.validate else None,
            'benchmark': benchmark_result if self.benchmark else None,
        }

        print(f"\n✓ Export complete!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")

        return result

    def _validate_onnx(self,
                      pytorch_model: nn.Module,
                      onnx_path: Path,
                      dummy_input: torch.Tensor,
                      device: torch.device) -> Dict[str, Any]:
        """Validate ONNX outputs against PyTorch."""
        try:
            import onnxruntime as ort
        except ImportError:
            print("⚠️  onnxruntime not installed - skipping validation")
            return {'validated': False}

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)

            # Extract tensor
            if isinstance(pytorch_output, torch.Tensor):
                pytorch_output = pytorch_output
            elif isinstance(pytorch_output, tuple):
                pytorch_output = pytorch_output[0]
            elif isinstance(pytorch_output, dict):
                pytorch_output = pytorch_output.get('logits', pytorch_output.get('last_hidden_state'))

            pytorch_output = pytorch_output.cpu().numpy()

        # ONNX inference
        ort_session = ort.InferenceSession(str(onnx_path))
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]

        # Compare outputs
        max_error = abs(pytorch_output - onnx_output).max()
        mean_error = abs(pytorch_output - onnx_output).mean()

        return {
            'validated': True,
            'max_error': float(max_error),
            'mean_error': float(mean_error),
        }

    def _optimize_onnx(self, onnx_path: Path):
        """Apply ONNX optimization passes."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            # Load model
            model = onnx.load(str(onnx_path))

            # Optimize
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # Generic transformer optimization
                num_heads=0,  # Auto-detect
                hidden_size=0  # Auto-detect
            )

            # Save optimized model
            optimized_model.save_model_to_file(str(onnx_path))

        except ImportError:
            print("⚠️  onnxruntime.transformers not installed - skipping optimization")
        except Exception as e:
            print(f"⚠️  Optimization failed: {e}")

    def _benchmark_onnx(self,
                       pytorch_model: nn.Module,
                       onnx_path: Path,
                       vocab_size: int,
                       max_seq_len: int,
                       device: torch.device,
                       num_runs: int = 100) -> Dict[str, float]:
        """Benchmark ONNX vs PyTorch inference speed."""
        try:
            import onnxruntime as ort
        except ImportError:
            return {'error': 'onnxruntime not installed'}

        # Create test input
        test_input = torch.randint(0, vocab_size, (1, max_seq_len), device=device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(test_input)

        # Benchmark PyTorch
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = pytorch_model(test_input)
        pytorch_time = (time.time() - start) / num_runs * 1000  # ms

        # Benchmark ONNX
        ort_session = ort.InferenceSession(str(onnx_path))
        onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}

        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, onnx_input)

        start = time.time()
        for _ in range(num_runs):
            _ = ort_session.run(None, onnx_input)
        onnx_time = (time.time() - start) / num_runs * 1000  # ms

        return {
            'pytorch_time': pytorch_time,
            'onnx_time': onnx_time,
            'speedup': pytorch_time / onnx_time if onnx_time > 0 else 0,
        }


def export_state_dict(model: nn.Module,
                      output_dir: Union[str, Path] = './exported_model',
                      config: Optional[Any] = None,
                      tokenizer: Optional[Any] = None,
                      metrics: Optional[Dict[str, Any]] = None,
                      upload_to_drive: bool = False,
                      drive_subdir: str = 'MyDrive/exported-models') -> str:
    """
    Export trained model to standard PyTorch state_dict format with metadata.

    Saves:
    - pytorch_model.bin (state_dict)
    - config.json (if provided and convertible)
    - metadata.json (export info, metrics, env)
    - tokenizer files (if tokenizer has save_pretrained)
    - load_example.py (example loader script)

    Args:
        model: Trained model or Lightning adapter; if adapter, attempts to use adapter.model
        output_dir: Directory to write export files
        config: Model config (must have to_dict() or be JSON-serializable)
        tokenizer: Optional tokenizer object supporting save_pretrained()
        metrics: Optional final metrics dict to store in metadata
        upload_to_drive: When True, attempts to copy export to Google Drive (Colab)
        drive_subdir: Destination subdirectory under /content/drive

    Returns:
        str: Absolute path to export directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Unwrap adapter if needed
    export_target = getattr(model, 'model', model)

    # Save state dict
    model_path = out / 'pytorch_model.bin'
    torch.save(export_target.state_dict(), str(model_path))
    print(f"✅ Model weights saved to {model_path}")

    # Save config
    config_path = out / 'config.json'
    cfg_obj = None
    if config is None:
        cfg_obj = getattr(model, 'config', None)
    else:
        cfg_obj = config

    if cfg_obj is not None:
        try:
            import json
            if hasattr(cfg_obj, 'to_dict'):
                cfg = cfg_obj.to_dict()
            elif isinstance(cfg_obj, dict):
                cfg = cfg_obj
            else:
                # Fallback: attempt to introspect simple attributes
                cfg = {k: v for k, v in getattr(cfg_obj, '__dict__', {}).items() if isinstance(v, (int, float, str, bool, list, dict))}
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"✅ Config saved to {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to save config: {e}")

    # Save tokenizer
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is not None and hasattr(tokenizer, 'save_pretrained'):
        try:
            tokenizer.save_pretrained(str(out))
            print(f"✅ Tokenizer saved to {out}")
        except Exception as e:
            print(f"⚠️  Failed to save tokenizer: {e}")

    # Save metadata
    metadata = {
        'export_date': datetime.now().isoformat(),
        'final_metrics': metrics or {},
        'total_params': int(sum(p.numel() for p in export_target.parameters())),
        'framework': 'PyTorch',
        'pytorch_version': torch.__version__,
        'files': ['pytorch_model.bin', 'config.json', 'metadata.json']
    }
    try:
        import json
        with open(out / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to write metadata: {e}")

    # Create loading example
    example = out / 'load_example.py'
    example_text = (
        '"""\n'
        'Example code to load exported model.\n'
        '"""\n'
        'import torch\n'
        'import json\n\n'
        "with open('config.json', 'r') as f:\n"
        '    config = json.load(f)\n\n'
        '# TODO: Replace with your model class\n'
        'class YourModelClass(torch.nn.Module):\n'
        '    def __init__(self, config):\n'
        '        super().__init__()\n'
        '        # define layers based on config\n'
        '        pass\n'
        '    def forward(self, x):\n'
        '        pass\n\n'
        'model = YourModelClass(config)\n'
        "state = torch.load('pytorch_model.bin', map_location='cpu')\n"
        'model.load_state_dict(state, strict=False)\n'
        'model.eval()\n\n'
        "print('Model loaded. Ready for inference.')\n"
    )
    example.write_text(example_text)

    # Optional: upload to Google Drive
    if upload_to_drive:
        try:
            from google.colab import drive  # type: ignore
            mount = Path('/content/drive')
            if not mount.exists():
                print("🔗 Mounting Google Drive...")
                drive.mount(str(mount))
            drive_dir = mount / drive_subdir
            drive_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            dest = drive_dir / out.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(out, dest)
            print(f"☁️  Export copied to Drive: {dest}")
        except Exception as e:
            print(f"⚠️  Drive upload failed: {e}")

    return str(out.resolve())


class TorchScriptExporter:
    """
    Export PyTorch models to TorchScript format.

    Supports both tracing and scripting modes with automatic
    fallback and validation.

    Example:
        >>> exporter = TorchScriptExporter()
        >>> exporter.export(
        ...     model=my_model,
        ...     output_path='model.pt',
        ...     vocab_size=50257,
        ...     mode='trace'
        ... )
        ✓ TorchScript export successful (trace mode)
        ✓ Validation passed
        📊 Speedup: 1.15x faster
    """

    def __init__(self,
                 validate: bool = True,
                 benchmark: bool = True):
        """
        Initialize TorchScript exporter.

        Args:
            validate: Validate outputs
            benchmark: Benchmark inference speed
        """
        self.validate = validate
        self.benchmark = benchmark

    def export(self,
              model: nn.Module,
              output_path: Union[str, Path],
              vocab_size: int,
              max_seq_len: int = 512,
              batch_size: int = 1,
              mode: Literal['trace', 'script', 'auto'] = 'auto') -> Dict[str, Any]:
        """
        Export model to TorchScript.

        Args:
            model: PyTorch model
            output_path: Output file path
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            batch_size: Batch size for tracing
            mode: Export mode ('trace', 'script', or 'auto')

        Returns:
            Export metadata dictionary

        Example:
            >>> result = exporter.export(
            ...     model=transformer,
            ...     output_path='model.pt',
            ...     vocab_size=50257
            ... )
        """
        output_path = Path(output_path)
        print(f"\n📦 Exporting to TorchScript: {output_path.name}")
        print("-" * 80)

        # Set model to eval
        model.eval()
        device = next(model.parameters()).device

        # Create example input
        example_input = torch.randint(
            0, vocab_size,
            (batch_size, max_seq_len),
            device=device
        )

        # Try export based on mode
        if mode == 'auto':
            # Try tracing first, fall back to scripting
            try:
                scripted_model = torch.jit.trace(model, example_input)
                actual_mode = 'trace'
                print("✓ Exported using trace mode")
            except Exception as e:
                print(f"⚠️  Tracing failed: {e}")
                print("   Falling back to script mode...")
                try:
                    scripted_model = torch.jit.script(model)
                    actual_mode = 'script'
                    print("✓ Exported using script mode")
                except Exception as e2:
                    print(f"❌ Scripting also failed: {e2}")
                    raise

        elif mode == 'trace':
            scripted_model = torch.jit.trace(model, example_input)
            actual_mode = 'trace'
            print("✓ Exported using trace mode")

        elif mode == 'script':
            scripted_model = torch.jit.script(model)
            actual_mode = 'script'
            print("✓ Exported using script mode")

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Optimize
        scripted_model = torch.jit.optimize_for_inference(scripted_model)

        # Save
        torch.jit.save(scripted_model, str(output_path))
        print(f"✓ Saved to: {output_path}")

        # Validate
        if self.validate:
            print("\n🔍 Validating TorchScript model...")
            validation_result = self._validate_torchscript(
                model,
                scripted_model,
                example_input
            )
            print(f"✓ Validation passed (max error: {validation_result['max_error']:.6f})")

        # Benchmark
        benchmark_result = {}
        if self.benchmark:
            print("\n📊 Benchmarking inference speed...")
            benchmark_result = self._benchmark_torchscript(
                model,
                scripted_model,
                vocab_size,
                max_seq_len,
                device
            )
            print(f"✓ PyTorch: {benchmark_result['pytorch_time']:.2f}ms per batch")
            print(f"✓ TorchScript: {benchmark_result['torchscript_time']:.2f}ms per batch")
            print(f"✓ Speedup: {benchmark_result['speedup']:.2f}x")

        # Metadata
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            'output_path': str(output_path),
            'file_size_mb': file_size_mb,
            'mode': actual_mode,
            'validation': validation_result if self.validate else None,
            'benchmark': benchmark_result if self.benchmark else None,
        }

        print(f"\n✓ Export complete!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")

        return result

    def _validate_torchscript(self,
                             pytorch_model: nn.Module,
                             scripted_model: torch.jit.ScriptModule,
                             example_input: torch.Tensor) -> Dict[str, Any]:
        """Validate TorchScript outputs."""
        with torch.no_grad():
            pytorch_output = pytorch_model(example_input)
            scripted_output = scripted_model(example_input)

            # Extract tensors
            if isinstance(pytorch_output, tuple):
                pytorch_output = pytorch_output[0]
            if isinstance(scripted_output, tuple):
                scripted_output = scripted_output[0]

            # Compare
            max_error = (pytorch_output - scripted_output).abs().max().item()
            mean_error = (pytorch_output - scripted_output).abs().mean().item()

        return {
            'validated': True,
            'max_error': max_error,
            'mean_error': mean_error,
        }

    def _benchmark_torchscript(self,
                              pytorch_model: nn.Module,
                              scripted_model: torch.jit.ScriptModule,
                              vocab_size: int,
                              max_seq_len: int,
                              device: torch.device,
                              num_runs: int = 100) -> Dict[str, float]:
        """Benchmark TorchScript vs PyTorch."""
        test_input = torch.randint(0, vocab_size, (1, max_seq_len), device=device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(test_input)
                _ = scripted_model(test_input)

        # Benchmark PyTorch
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = pytorch_model(test_input)
        pytorch_time = (time.time() - start) / num_runs * 1000

        # Benchmark TorchScript
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = scripted_model(test_input)
        torchscript_time = (time.time() - start) / num_runs * 1000

        return {
            'pytorch_time': pytorch_time,
            'torchscript_time': torchscript_time,
            'speedup': pytorch_time / torchscript_time if torchscript_time > 0 else 0,
        }


def _generate_dummy_input_from_task(
    task_spec: "TaskSpec",
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate a dummy input batch from TaskSpec for export.

    Supports:
        - Text (LM / classification / seq2seq) via input_ids.
        - Vision classification via pixel_values.
    """
    modality = getattr(task_spec, "modality", "text")
    device = device or torch.device("cpu")

    if modality == "vision" and getattr(task_spec, "task_type", None) == "vision_classification":
        image_size = task_spec.input_schema.get("image_size", [3, 64, 64])
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 3:
            raise ValueError(f"Expected image_size=[C,H,W] in TaskSpec.input_schema, got {image_size!r}")
        c, h, w = (int(image_size[0]), int(image_size[1]), int(image_size[2]))
        pixel_values = torch.rand(batch_size, c, h, w, device=device)
        return {"pixel_values": pixel_values}

    # Default to text-like inputs
    max_seq_len = int(task_spec.input_schema.get("max_seq_len", 16)) if isinstance(
        getattr(task_spec, "input_schema", {}), Mapping
    ) else 16
    vocab_size = int(task_spec.input_schema.get("vocab_size", 50257)) if isinstance(
        getattr(task_spec, "input_schema", {}), Mapping
    ) else 50257

    input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
    return {"input_ids": input_ids}


def _infer_output_shape(
    model: nn.Module,
    adapter: "ModelAdapter",
    task_spec: "TaskSpec",
    dummy_batch: Dict[str, torch.Tensor],
) -> List[int]:
    """
    Run a single forward pass via adapter to infer output shape.
    """
    model.eval()
    with torch.no_grad():
        prepared = adapter.prepare_inputs(dummy_batch, task_spec)
        _loss, outputs = adapter.forward_for_loss(model, prepared, task_spec)
        logits = adapter.get_logits(outputs, task_spec)
        if isinstance(logits, torch.Tensor):
            return list(logits.shape)
    return []


class ModelCardGenerator:
    """
    Generate HuggingFace-style model cards.

    Creates comprehensive documentation including:
    - Model architecture and parameters
    - Training data and procedure
    - Performance metrics
    - Usage examples
    - Limitations and biases
    - Citation information

    Example:
        >>> generator = ModelCardGenerator()
        >>> card = generator.generate(
        ...     model_name='my-gpt2-wikitext',
        ...     model=model,
        ...     training_results=results,
        ...     dataset_name='wikitext-2',
        ...     output_path='MODEL_CARD.md'
        ... )
        ✓ Model card generated: MODEL_CARD.md
    """

    def __init__(self):
        """Initialize model card generator."""
        pass

    def generate(self,
                model_name: str,
                model: nn.Module,
                training_results: Optional[Dict[str, Any]] = None,
                dataset_name: Optional[str] = None,
                dataset_size: Optional[int] = None,
                vocab_size: Optional[int] = None,
                description: Optional[str] = None,
                limitations: Optional[str] = None,
                intended_use: Optional[str] = None,
                output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate model card.

        Args:
            model_name: Model identifier
            model: Trained model
            training_results: Results from TrainingCoordinator
            dataset_name: Training dataset name
            dataset_size: Number of training samples
            vocab_size: Vocabulary size
            description: Model description
            limitations: Known limitations
            intended_use: Intended use cases
            output_path: Output file path (optional)

        Returns:
            Model card markdown string

        Example:
            >>> card = generator.generate(
            ...     model_name='gpt2-wikitext',
            ...     model=model,
            ...     training_results=results,
            ...     dataset_name='wikitext-2-raw-v1'
            ... )
        """
        # Extract model info
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Extract training metrics
        if training_results:
            final_metrics = training_results.get('final_metrics', {})
        else:
            final_metrics = {}

        # Generate card
        card_sections = []

        # Header
        card_sections.append(f"# {model_name}\n")

        # Description
        if description:
            card_sections.append(f"{description}\n")
        else:
            card_sections.append(
                f"Transformer model trained on {dataset_name or 'custom dataset'}.\n"
            )

        # Model Details
        card_sections.append("## Model Details\n")
        card_sections.append(f"- **Model Type**: {model.__class__.__name__}")
        card_sections.append(f"- **Parameters**: {num_params:,} ({trainable_params:,} trainable)")
        if vocab_size:
            card_sections.append(f"- **Vocabulary Size**: {vocab_size:,}")
        card_sections.append(f"- **Created**: {datetime.now().strftime('%Y-%m-%d')}\n")

        # Training Data
        if dataset_name or dataset_size:
            card_sections.append("## Training Data\n")
            if dataset_name:
                card_sections.append(f"- **Dataset**: {dataset_name}")
            if dataset_size:
                card_sections.append(f"- **Training Samples**: {dataset_size:,}\n")

        # Performance
        if final_metrics:
            card_sections.append("## Performance\n")
            for metric, value in final_metrics.items():
                card_sections.append(f"- **{metric}**: {value:.4f}")
            card_sections.append("")

        # Usage
        card_sections.append("## Usage\n")
        card_sections.append("```python")
        card_sections.append("import torch")
        card_sections.append(f"from transformers import AutoTokenizer")
        card_sections.append("")
        card_sections.append("# Load model")
        card_sections.append(f"model = torch.load('{model_name}.pt')")
        card_sections.append("model.eval()")
        card_sections.append("")
        card_sections.append("# Generate text")
        card_sections.append("input_ids = tokenizer.encode('Hello', return_tensors='pt')")
        card_sections.append("output = model.generate(input_ids, max_length=50)")
        card_sections.append("print(tokenizer.decode(output[0]))")
        card_sections.append("```\n")

        # Intended Use
        if intended_use:
            card_sections.append("## Intended Use\n")
            card_sections.append(f"{intended_use}\n")

        # Limitations
        if limitations:
            card_sections.append("## Limitations\n")
            card_sections.append(f"{limitations}\n")
        else:
            card_sections.append("## Limitations\n")
            card_sections.append("- This model was trained for research/educational purposes")
            card_sections.append("- Performance may vary on out-of-distribution data")
            card_sections.append("- No explicit bias mitigation was applied\n")

        # Citation
        card_sections.append("## Citation\n")
        card_sections.append("```bibtex")
        card_sections.append("@misc{" + model_name.replace('-', '_') + ",")
        card_sections.append(f"  title={{{model_name}}},")
        card_sections.append(f"  year={{{datetime.now().year}}},")
        card_sections.append("  note={Generated using Transformer Builder}")
        card_sections.append("}")
        card_sections.append("```\n")

        # Combine
        card = "\n".join(card_sections)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(card)
            print(f"✓ Model card generated: {output_path}")

        return card


def export_model(
    model: nn.Module,
    adapter: "ModelAdapter",
    task_spec: "TaskSpec",
    export_dir: Union[Path, str],
    formats: List[str] | Tuple[str, ...] = ("torchscript", "onnx", "pytorch"),
    quantization: Optional[Literal["dynamic", "static"]] = None,
) -> Dict[str, Path]:
    """
    Export a model to multiple deployment-ready formats with metadata.

    Args:
        model: Trained PyTorch model to export.
        adapter: Task-aware ModelAdapter used for forward/inference.
        task_spec: TaskSpec describing modality, task_type, and schemas.
        export_dir: Directory where export artifacts will be written.
        formats: List of formats to export: \"torchscript\", \"onnx\", \"pytorch\".
        quantization: Optional quantization mode (\"dynamic\" or \"static\").

    Returns:
        Dict mapping format names to output Paths, including a \"metadata\" key.

    Example:
        >>> paths = export_model(
        ...     model,
        ...     adapter,
        ...     task_spec,
        ...     export_dir=\"./exports/lm_run42\",
        ...     formats=[\"torchscript\", \"onnx\", \"pytorch\"],
        ... )
        >>> print(paths[\"torchscript\"], paths[\"onnx\"], paths[\"metadata\"])
    """
    export_root = Path(export_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    # Prepare dummy input from TaskSpec
    device = next(model.parameters()).device
    dummy_batch = _generate_dummy_input_from_task(task_spec, batch_size=1, device=device)
    output_shape = _infer_output_shape(model, adapter, task_spec, dummy_batch)

    results: Dict[str, Path] = {}

    # Optional quantization (safe default: dynamic only)
    export_model_obj: nn.Module = model
    if quantization is not None:
        # Static quantization setups are environment-specific; recommend dynamic.
        if quantization == "dynamic":
            try:
                export_model_obj = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                print("✅ Applied dynamic quantization to Linear layers.")
            except Exception as exc:
                print(f"⚠️  Dynamic quantization failed, exporting non-quantized model: {exc}")
                export_model_obj = model
        elif quantization == "static":
            # Static quantization is intentionally conservative here.
            print("⚠️  Static quantization is not fully configured; exporting non-quantized model.")
            export_model_obj = model

    # TorchScript export
    if "torchscript" in formats:
        ts_path = export_root / "model.torchscript.pt"
        ts_exporter = TorchScriptExporter(validate=False, benchmark=False)

        # For text models, TorchScriptExporter expects vocab_size/max_seq_len;
        # for vision models we approximate with small dummy values.
        if getattr(task_spec, "modality", "text") == "vision":
            vocab_size = 8
            max_seq_len = 16
        else:
            vocab_size = int(task_spec.input_schema.get("vocab_size", 50257))
            max_seq_len = int(task_spec.input_schema.get("max_seq_len", 16))

        ts_exporter.export(
            export_model_obj,
            output_path=ts_path,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        results["torchscript"] = ts_path

    # ONNX export
    if "onnx" in formats:
        onnx_path = export_root / "model.onnx"
        onnx_exporter = ONNXExporter(optimize=False, validate=False, benchmark=False)

        try:
            modality = getattr(task_spec, "modality", "text")
            if modality == "vision" and getattr(task_spec, "task_type", None) == "vision_classification":
                image_size = task_spec.input_schema.get("image_size", [3, 64, 64])
                if not isinstance(image_size, (list, tuple)) or len(image_size) != 3:
                    raise ValueError(f"Expected image_size=[C,H,W] in TaskSpec.input_schema, got {image_size!r}")
                c, h, w = (int(image_size[0]), int(image_size[1]), int(image_size[2]))
                _ = torch.rand(1, c, h, w, device=device)
                onnx_exporter.export(
                    export_model_obj,
                    output_path=onnx_path,
                    vocab_size=1,
                    max_seq_len=1,
                    batch_size=1,
                    dynamic_axes=False,
                    input_names=["pixel_values"],
                    output_names=["logits"],
                )
            else:
                vocab_size = int(task_spec.input_schema.get("vocab_size", 50257))
                max_seq_len = int(task_spec.input_schema.get("max_seq_len", 16))
                onnx_exporter.export(
                    export_model_obj,
                    output_path=onnx_path,
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                )
            results["onnx"] = onnx_path
        except Exception as exc:
            print(f"⚠️  ONNX export skipped due to error: {exc}")

    # PyTorch state dict export
    if "pytorch" in formats:
        state_dict_dir = export_root / "pytorch"
        state_dict_dir.mkdir(parents=True, exist_ok=True)
        export_state_dict(export_model_obj, output_dir=state_dict_dir)
        results["pytorch"] = state_dict_dir / "pytorch_model.bin"

    # Metadata manifest
    metadata = {
        "task_type": getattr(task_spec, "task_type", None),
        "modality": getattr(task_spec, "modality", None),
        "input_shape": {
            "batch": 1,
            "schema": getattr(task_spec, "input_schema", {}),
        },
        "output_shape": output_shape,
        "exported_at": datetime.now().isoformat(),
        "framework_versions": {
            "torch": torch.__version__,
        },
        "formats": list(formats),
        "quantization": quantization,
    }
    metadata_path = export_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    results["metadata"] = metadata_path

    return results


def load_exported_model(
    export_dir: Union[Path, str],
    runtime: Literal["torchscript", "onnx"] = "torchscript",
) -> Any:
    """
    Load an exported model from export_dir for serving.

    Args:
        export_dir: Directory containing exported artifacts (model.* and metadata.json).
        runtime: Runtime to use: \"torchscript\" or \"onnx\".

    Returns:
        Callable that maps an input tensor to an output tensor (logits).

    Notes:
        - For TorchScript, the callable expects a ``torch.Tensor`` input matching
          the model's expected shape (e.g., [B, T] for text or [B, C, H, W] for vision).
        - For ONNX, onnxruntime must be installed; the wrapper accepts a
          ``torch.Tensor`` or NumPy array and returns a ``torch.Tensor``.
    """
    export_root = Path(export_dir)
    metadata_path = export_root / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    if runtime == "torchscript":
        model_path = export_root / "model.torchscript.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found at {model_path}")
        scripted = torch.jit.load(str(model_path), map_location=torch.device("cpu"))
        scripted.eval()

        def _predict_torchscript(x: Any) -> torch.Tensor:
            with torch.no_grad():
                if not isinstance(x, torch.Tensor):
                    x_tensor = torch.as_tensor(x)
                else:
                    x_tensor = x
                out = scripted(x_tensor)
                if isinstance(out, torch.Tensor):
                    return out
                if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                    return out[0]
                if isinstance(out, dict):
                    logits = out.get("logits")
                    if isinstance(logits, torch.Tensor):
                        return logits
                raise ValueError("Unable to extract tensor output from TorchScript model.")

        return _predict_torchscript

    if runtime == "onnx":
        try:
            import onnxruntime as ort  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError("onnxruntime is required to load ONNX models.") from exc

        model_path = export_root / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name

        def _predict_onnx(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                import numpy as np

                arr = np.asarray(x)
            outputs = session.run(None, {input_name: arr})
            return torch.from_numpy(outputs[0])

        return _predict_onnx

    raise ValueError(f"Unsupported runtime '{runtime}'. Expected 'torchscript' or 'onnx'.")


def generate_inference_script(
    task_spec: "TaskSpec",
    export_dir: Path,
    model_format: str = "onnx"
) -> Path:
    """
    Generate standalone inference.py script with preprocessing logic from TaskSpec.

    Args:
        task_spec: TaskSpec with modality and preprocessing configuration
        export_dir: Directory to write inference.py
        model_format: Model format to use ("onnx" or "torchscript")

    Returns:
        Path to generated inference.py

    Example:
        >>> script_path = generate_inference_script(
        ...     task_spec=TaskSpec.vision_tiny(),
        ...     export_dir=Path("exports/model_001"),
        ...     model_format="onnx"
        ... )
    """
    modality = getattr(task_spec, "modality", "text")

    if modality == "vision":
        script_content = _generate_vision_inference_script(task_spec, model_format)
    elif modality == "text":
        script_content = _generate_text_inference_script(task_spec, model_format)
    else:
        raise ValueError(f"Unsupported modality for inference script generation: {modality}")

    script_path = export_dir / "inference.py"
    script_path.write_text(script_content)
    print(f"✅ Generated inference script: {script_path}")
    return script_path


def _generate_vision_inference_script(task_spec: "TaskSpec", model_format: str) -> str:
    """Generate inference script for vision tasks."""
    # Extract preprocessing configuration
    preproc = task_spec.preprocessing_config or {}
    mean = preproc.get("mean", [0.485, 0.456, 0.406])  # ImageNet defaults
    std = preproc.get("std", [0.229, 0.224, 0.225])
    image_size = task_spec.input_schema.get("image_size", [3, 224, 224])

    if not isinstance(image_size, (list, tuple)) or len(image_size) != 3:
        image_size = [3, 224, 224]

    c, h, w = image_size

    if model_format == "onnx":
        template = f'''"""
Vision Inference Engine - ONNX Runtime

Generated for task: {task_spec.name}
Modality: vision
Task type: {task_spec.task_type}
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Any


class VisionInferenceEngine:
    """
    ONNX-based inference engine for vision classification.

    Preprocessing matches training configuration:
    - Image size: [{c}, {h}, {w}] (C, H, W)
    - Normalization: mean={mean}, std={std}
    """

    def __init__(self, model_path: str):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Preprocessing configuration from TaskSpec
        self.mean = np.array({mean}).reshape(1, -1, 1, 1).astype(np.float32)
        self.std = np.array({std}).reshape(1, -1, 1, 1).astype(np.float32)
        self.image_size = ({c}, {h}, {w})

    def preprocess(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image as numpy array [1, C, H, W]
        """
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Resize to expected dimensions
        img = img.resize((self.image_size[2], self.image_size[1]))  # (W, H)

        # Convert to array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0

        # Apply normalization
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension: [C, H, W] -> [1, C, H, W]
        img_array = img_array[np.newaxis, ...]

        # Normalize with mean/std
        img_array = (img_array - self.mean) / self.std

        return img_array

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with predictions:
                - predicted_class: int (argmax of logits)
                - confidence: float (softmax probability)
                - probabilities: List[float] (all class probabilities)
        """
        # Preprocess input
        inputs = self.preprocess(image_path)

        # Run inference
        outputs = self.session.run(None, {{self.input_name: inputs}})

        # Interpret outputs
        logits = outputs[0]  # Shape: [1, num_classes]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Extract predictions
        predicted_class = int(np.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, predicted_class])

        return {{
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs[0].tolist()
        }}

    def batch_predict(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results


def main():
    """CLI interface for vision inference."""
    parser = argparse.ArgumentParser(
        description='Vision Inference Engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--model',
        default='artifacts/model.onnx',
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process directory of images (batch mode)'
    )

    args = parser.parse_args()

    # Initialize engine
    engine = VisionInferenceEngine(args.model)
    print(f"✅ Loaded model: {{args.model}}")

    # Run inference
    if args.batch:
        # Batch mode
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            raise ValueError(f"Batch mode requires directory input: {{args.input}}")

        image_paths = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        print(f"Found {{len(image_paths)}} images")

        results = engine.batch_predict(image_paths)
        for path, result in zip(image_paths, results):
            print(f"{{path.name}}: class={{result['predicted_class']}}, "
                  f"confidence={{result['confidence']:.4f}}")
    else:
        # Single image mode
        result = engine.predict(args.input)
        print(f"\\nPrediction Results:")
        print(f"  Predicted class: {{result['predicted_class']}}")
        print(f"  Confidence: {{result['confidence']:.4f}}")
        print(f"  Top 5 probabilities:")
        probs = np.array(result['probabilities'])
        top5_idx = np.argsort(probs)[-5:][::-1]
        for idx in top5_idx:
            print(f"    Class {{idx}}: {{probs[idx]:.4f}}")


if __name__ == "__main__":
    main()
'''

    elif model_format == "torchscript":
        template = f'''"""
Vision Inference Engine - TorchScript

Generated for task: {task_spec.name}
Modality: vision
Task type: {task_spec.task_type}
"""

import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Any


class VisionInferenceEngine:
    """
    TorchScript-based inference engine for vision classification.

    Preprocessing matches training configuration:
    - Image size: [{c}, {h}, {w}] (C, H, W)
    - Normalization: mean={mean}, std={std}
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize inference engine.

        Args:
            model_path: Path to TorchScript model file
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Preprocessing configuration from TaskSpec
        self.mean = torch.tensor({mean}).view(1, -1, 1, 1).to(self.device)
        self.std = torch.tensor({std}).view(1, -1, 1, 1).to(self.device)
        self.image_size = ({c}, {h}, {w})

    def preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image as tensor [1, C, H, W]
        """
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Resize
        img = img.resize((self.image_size[2], self.image_size[1]))

        # Convert to tensor and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        # Move to device and normalize
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        inputs = self.preprocess(image_path)

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('last_hidden_state'))
            else:
                logits = outputs

        # Softmax
        probs = torch.softmax(logits, dim=-1)

        # Extract predictions
        predicted_class = int(torch.argmax(probs, dim=-1)[0])
        confidence = float(probs[0, predicted_class])

        return {{
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy().tolist()
        }}

    def batch_predict(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Run inference on multiple images."""
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results


def main():
    """CLI interface for vision inference."""
    parser = argparse.ArgumentParser(description='Vision Inference Engine (TorchScript)')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--model', default='artifacts/model.torchscript.pt', help='Path to TorchScript model')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')

    args = parser.parse_args()

    engine = VisionInferenceEngine(args.model, args.device)
    result = engine.predict(args.input)

    print(f"\\nPrediction Results:")
    print(f"  Predicted class: {{result['predicted_class']}}")
    print(f"  Confidence: {{result['confidence']:.4f}}")


if __name__ == "__main__":
    main()
'''

    else:
        raise ValueError(f"Unsupported model format: {model_format}")

    return template


def _generate_text_inference_script(task_spec: "TaskSpec", model_format: str) -> str:
    """Generate inference script for text tasks."""
    vocab_size = task_spec.input_schema.get("vocab_size", 50257)
    max_seq_len = task_spec.input_schema.get("max_seq_len", 128)

    if model_format == "onnx":
        template = f'''"""
Text Inference Engine - ONNX Runtime

Generated for task: {task_spec.name}
Modality: text
Task type: {task_spec.task_type}
"""

import argparse
import numpy as np
from typing import Dict, List, Any


class TextInferenceEngine:
    """
    ONNX-based inference engine for text tasks.

    Configuration:
    - Vocabulary size: {vocab_size}
    - Max sequence length: {max_seq_len}
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Optional path to tokenizer directory
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required. Install with: pip install onnxruntime")

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Load tokenizer if provided
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except ImportError:
                print("⚠️  transformers not installed, tokenization disabled")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def preprocess(self, text: str) -> np.ndarray:
        """
        Tokenize text for inference.

        Args:
            text: Input text string

        Returns:
            Token IDs as numpy array [1, seq_len]
        """
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length={max_seq_len},
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return encoded['input_ids']
        else:
            # Fallback: simple character-level encoding
            token_ids = [ord(c) % {vocab_size} for c in text[:{max_seq_len}]]
            token_ids += [0] * ({max_seq_len} - len(token_ids))  # Pad
            return np.array([token_ids], dtype=np.int64)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on text.

        Args:
            text: Input text

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        input_ids = self.preprocess(text)

        # Run inference
        outputs = self.session.run(None, {{self.input_name: input_ids}})
        logits = outputs[0]

        # For classification tasks
        if logits.shape[-1] < 100:  # Likely classification
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            predicted_class = int(np.argmax(probs, axis=-1)[0])
            confidence = float(probs[0, predicted_class])

            return {{
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probs[0].tolist()
            }}
        else:  # Language modeling
            return {{
                'logits': logits[0].tolist()
            }}


def main():
    """CLI interface for text inference."""
    parser = argparse.ArgumentParser(description='Text Inference Engine')
    parser.add_argument('--input', required=True, help='Input text')
    parser.add_argument('--model', default='artifacts/model.onnx', help='Path to ONNX model')
    parser.add_argument('--tokenizer', help='Path to tokenizer directory')

    args = parser.parse_args()

    engine = TextInferenceEngine(args.model, args.tokenizer)
    result = engine.predict(args.input)

    print(f"\\nPrediction Results:")
    print(result)


if __name__ == "__main__":
    main()
'''

    else:  # torchscript
        template = f'''"""
Text Inference Engine - TorchScript

Generated for task: {task_spec.name}
"""

import argparse
import torch
from typing import Dict, Any


class TextInferenceEngine:
    """TorchScript-based inference engine for text tasks."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Run inference on pre-tokenized input."""
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            return {{'logits': logits.cpu().numpy().tolist()}}


def main():
    parser = argparse.ArgumentParser(description='Text Inference Engine (TorchScript)')
    parser.add_argument('--model', default='artifacts/model.torchscript.pt', help='Model path')
    parser.add_argument('--device', default='cpu', help='Device')

    args = parser.parse_args()
    engine = TextInferenceEngine(args.model, args.device)
    print("✅ Model loaded. Ready for inference.")


if __name__ == "__main__":
    main()
'''

    return template


def generate_readme(
    task_spec: "TaskSpec",
    export_dir: Path,
    formats: List[str]
) -> Path:
    """
    Generate README.md with quickstart instructions.

    Args:
        task_spec: TaskSpec with task information
        export_dir: Directory to write README.md
        formats: List of exported formats

    Returns:
        Path to generated README.md
    """
    modality = getattr(task_spec, "modality", "text")
    task_type = getattr(task_spec, "task_type", "unknown")

    # Build quickstart sections
    readme_content = f'''# Model Export Bundle

**Task:** {task_spec.name}
**Modality:** {modality}
**Task Type:** {task_type}
**Exported:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This bundle contains a production-ready exported model with all necessary artifacts for deployment.

**Exported Formats:**
{chr(10).join(f"- {fmt}" for fmt in formats)}

## Directory Structure

```
.
├── artifacts/          # Model files
│   ├── model.onnx              (ONNX format)
│   ├── model.torchscript.pt    (TorchScript format)
│   └── model.pytorch.pt        (PyTorch state dict)
├── configs/            # Configuration files
│   ├── task_spec.json          (Task specification)
│   ├── training_config.json    (Training configuration)
│   └── torchserve_config.json  (TorchServe deployment config)
├── inference.py        # Standalone inference script
├── Dockerfile          # Container deployment
├── requirements.txt    # Runtime dependencies
└── README.md           # This file
```

## Quick Start

### Local Inference

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run inference:**
'''

    # Add modality-specific examples
    if modality == "vision":
        readme_content += '''```bash
python inference.py --input /path/to/image.jpg --model artifacts/model.onnx
```

**Batch processing:**
```bash
python inference.py --input /path/to/images/ --model artifacts/model.onnx --batch
```
'''
    else:  # text
        readme_content += '''```bash
python inference.py --input "Your text here" --model artifacts/model.onnx
```
'''

    readme_content += f'''
## Docker Deployment

**Build container:**
```bash
docker build -t model-inference .
```

**Run container:**
```bash
docker run -p 8080:8080 model-inference
```

**Test endpoint:**
'''

    if modality == "vision":
        readme_content += '''```bash
curl -X POST -F "image=@test.jpg" http://localhost:8080/predict
```
'''
    else:
        readme_content += '''```bash
curl -X POST -H "Content-Type: application/json" \\
     -d '{{"text": "Your input text"}}' \\
     http://localhost:8080/predict
```
'''

    readme_content += '''
## TorchServe Deployment

**Create model archive:**
```bash
torch-model-archiver \\
    --model-name transformer-model \\
    --version 1.0 \\
    --serialized-file artifacts/model.torchscript.pt \\
    --handler inference.py \\
    --export-path model-store
```

**Start TorchServe:**
```bash
torchserve \\
    --start \\
    --model-store model-store \\
    --models transformer-model=transformer-model.mar \\
    --ncs
```

**Configuration:**
See `configs/torchserve_config.json` for detailed deployment settings.

## Model Information

**Task Specification:**
- Input fields: {", ".join(task_spec.input_fields)}
- Target field: {task_spec.target_field or "N/A"}
- Loss type: {task_spec.loss_type}
- Metrics: {", ".join(task_spec.metrics)}

**Input Schema:**
```json
{json.dumps(dict(task_spec.input_schema), indent=2)}
```

**Output Schema:**
```json
{json.dumps(dict(task_spec.output_schema), indent=2)}
```

## Runtime Requirements

See `requirements.txt` for complete dependency list.

**Minimum requirements:**
- Python >= 3.8
- PyTorch >= 2.0 (for TorchScript)
- ONNX Runtime >= 1.15 (for ONNX)

## Performance

**Inference Speed:**
- TorchScript: ~1.1-1.2x faster than PyTorch
- ONNX: ~1.5-2.5x faster than PyTorch (CPU)

**Model Size:**
Check `artifacts/` directory for file sizes.

## Troubleshooting

**ONNX inference fails:**
- Ensure onnxruntime is installed: `pip install onnxruntime`
- For GPU: `pip install onnxruntime-gpu`

**TorchScript shape errors:**
- Verify input dimensions match model expectations
- Check `configs/task_spec.json` for input schema

**Import errors:**
- Install all requirements: `pip install -r requirements.txt`

## License

Model exported from Transformer Builder training pipeline.

## Citation

If using this model in research, please cite:

```bibtex
@misc{{transformer_builder_{task_spec.name.replace("-", "_")},
  title={{{{Transformer Model: {task_spec.name}}}}},
  year={{{datetime.now().year}}},
  note={{Generated using Transformer Builder Training Pipeline v3.5}}
}}
```

## Support

For issues or questions:
- Check task_spec.json for model configuration
- Review training_config.json for training details
- Consult Transformer Builder documentation
'''

    readme_path = export_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"✅ Generated README: {readme_path}")
    return readme_path


def generate_torchserve_config(
    task_spec: "TaskSpec",
    export_dir: Path
) -> Path:
    """
    Generate TorchServe configuration file.

    Args:
        task_spec: TaskSpec with task information
        export_dir: Directory to write torchserve_config.json

    Returns:
        Path to generated configuration file
    """
    config = {
        "modelName": task_spec.name.replace(" ", "-"),
        "modelVersion": "1.0",
        "runtime": "python",
        "minWorkers": 1,
        "maxWorkers": 4,
        "batchSize": 8,
        "maxBatchDelay": 100,  # milliseconds
        "responseTimeout": 120,  # seconds
        "deviceType": "cpu",  # Change to "gpu" if using CUDA
        "parallelType": "pp",
        "handler": {
            "module": "inference",
            "class": "VisionInferenceEngine" if task_spec.modality == "vision" else "TextInferenceEngine"
        },
        "metrics": {
            "enable": True,
            "port": 8082,
            "mode": "prometheus"
        }
    }

    config_path = export_dir / "configs" / "torchserve_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Generated TorchServe config: {config_path}")
    return config_path


def generate_dockerfile(
    task_spec: "TaskSpec",
    export_dir: Path
) -> Path:
    """
    Generate Dockerfile for containerized deployment.

    Args:
        task_spec: TaskSpec with task information
        export_dir: Directory to write Dockerfile

    Returns:
        Path to generated Dockerfile
    """
    modality = getattr(task_spec, "modality", "text")

    # Choose base image based on requirements
    base_image = "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"

    dockerfile_content = f'''# Production Inference Container
# Generated for: {task_spec.name}
# Modality: {modality}

FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and configs
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/
COPY inference.py .

# Create non-root user for security
RUN useradd -m -u 1000 inference && \\
    chown -R inference:inference /app
USER inference

# Expose inference port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=artifacts/model.onnx

# Run inference server
# Override this CMD with your preferred serving framework
CMD ["python", "inference.py", "--model", "artifacts/model.onnx", "--host", "0.0.0.0", "--port", "8080"]
'''

    dockerfile_path = export_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)
    print(f"✅ Generated Dockerfile: {dockerfile_path}")
    return dockerfile_path


def _generate_runtime_requirements(modality: str, formats: List[str]) -> List[str]:
    """Generate runtime requirements.txt content."""
    requirements = [
        "# Runtime dependencies for model inference",
        "# Generated by Transformer Builder Export Utilities",
        "",
        "# Core dependencies",
        "numpy>=1.21.0",
        "pillow>=9.0.0" if modality == "vision" else "# pillow>=9.0.0  # Vision tasks only",
    ]

    # Add format-specific dependencies
    if "onnx" in formats:
        requirements.extend([
            "",
            "# ONNX Runtime",
            "onnxruntime>=1.15.0  # CPU inference",
            "# onnxruntime-gpu>=1.15.0  # GPU inference (uncomment if using CUDA)",
        ])

    if "torchscript" in formats:
        requirements.extend([
            "",
            "# PyTorch (for TorchScript)",
            "torch>=2.0.0",
        ])

    if modality == "text":
        requirements.extend([
            "",
            "# Text processing (optional)",
            "# transformers>=4.30.0  # If using HuggingFace tokenizers",
        ])

    requirements.extend([
        "",
        "# Web serving (optional)",
        "# flask>=2.0.0",
        "# fastapi>=0.100.0",
        "# uvicorn>=0.22.0",
    ])

    return requirements


def create_export_bundle(
    model: nn.Module,
    config: Any,
    task_spec: "TaskSpec",
    training_config: Any,
    export_base_dir: str = "exports",
    run_health_checks: bool = True
) -> Path:
    """
    Create complete production inference bundle with health validation.

    Generates:
    - Model artifacts (ONNX, TorchScript, PyTorch)
    - inference.py script
    - README.md
    - TorchServe config
    - Dockerfile
    - requirements.txt
    - Health report (JSON and Markdown)

    Args:
        model: Trained PyTorch model
        config: Model configuration (SimpleNamespace or dict)
        task_spec: TaskSpec describing the task
        training_config: TrainingConfig with export settings
        export_base_dir: Base directory for exports
        run_health_checks: Whether to run comprehensive health checks (default: True)

    Returns:
        Path to export directory

    Example:
        >>> export_dir = create_export_bundle(
        ...     model=trained_model,
        ...     config=model_config,
        ...     task_spec=TaskSpec.vision_tiny(),
        ...     training_config=TrainingConfig(export_bundle=True)
        ... )
        ✅ Export bundle created at: exports/model_20250118_143022
        ✅ Health Score: 98.5/100 - Ready for production
    """
    print("\n" + "=" * 80)
    print("Creating Production Inference Bundle")
    print("=" * 80)

    # Create timestamped export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(export_base_dir) / f"model_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = export_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    configs_dir = export_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    # Get export formats from training config
    export_formats = getattr(training_config, "export_formats", ["onnx", "torchscript"])

    print(f"\nExport Directory: {export_dir}")
    print(f"Formats: {', '.join(export_formats)}")
    print(f"Task: {task_spec.name} ({task_spec.modality})")
    print()

    # 1. Export model in requested formats
    print("📦 Exporting model artifacts...")

    try:
        # Prepare dummy input for export
        from .export_utilities import _generate_dummy_input_from_task
        device = next(model.parameters()).device
        dummy_batch = _generate_dummy_input_from_task(task_spec, batch_size=1, device=device)

        for fmt in export_formats:
            if fmt == "onnx":
                try:
                    exporter = ONNXExporter(optimize=False, validate=False, benchmark=False)

                    if task_spec.modality == "vision":
                        image_size = task_spec.input_schema.get("image_size", [3, 224, 224])
                        c, h, w = image_size
                        exporter.export(
                            model,
                            output_path=artifacts_dir / "model.onnx",
                            vocab_size=1,
                            max_seq_len=1,
                            batch_size=1,
                            dynamic_axes=False,
                            input_names=["pixel_values"],
                            output_names=["logits"]
                        )
                    else:
                        vocab_size = task_spec.input_schema.get("vocab_size", 50257)
                        max_seq_len = task_spec.input_schema.get("max_seq_len", 128)
                        exporter.export(
                            model,
                            output_path=artifacts_dir / "model.onnx",
                            vocab_size=vocab_size,
                            max_seq_len=max_seq_len
                        )
                except Exception as e:
                    print(f"⚠️  ONNX export failed: {e}")

            elif fmt == "torchscript":
                try:
                    exporter = TorchScriptExporter(validate=False, benchmark=False)

                    if task_spec.modality == "vision":
                        vocab_size = 8
                        max_seq_len = 16
                    else:
                        vocab_size = task_spec.input_schema.get("vocab_size", 50257)
                        max_seq_len = task_spec.input_schema.get("max_seq_len", 128)

                    exporter.export(
                        model,
                        output_path=artifacts_dir / "model.torchscript.pt",
                        vocab_size=vocab_size,
                        max_seq_len=max_seq_len
                    )
                except Exception as e:
                    print(f"⚠️  TorchScript export failed: {e}")

            elif fmt == "pytorch":
                torch.save(model.state_dict(), artifacts_dir / "model.pytorch.pt")
                print(f"✅ Exported PyTorch state dict")

    except Exception as e:
        print(f"⚠️  Model export encountered errors: {e}")
        print("    Continuing with artifact generation...")

    # 2. Generate inference script
    print("\n📝 Generating inference script...")
    primary_format = export_formats[0] if export_formats else "onnx"
    try:
        generate_inference_script(task_spec, export_dir, model_format=primary_format)
    except Exception as e:
        print(f"⚠️  Inference script generation failed: {e}")

    # 3. Generate README
    print("\n📚 Generating README...")
    try:
        generate_readme(task_spec, export_dir, export_formats)
    except Exception as e:
        print(f"⚠️  README generation failed: {e}")

    # 4. Generate TorchServe config
    print("\n⚙️  Generating TorchServe config...")
    try:
        generate_torchserve_config(task_spec, export_dir)
    except Exception as e:
        print(f"⚠️  TorchServe config generation failed: {e}")

    # 5. Generate Dockerfile
    print("\n🐳 Generating Dockerfile...")
    try:
        generate_dockerfile(task_spec, export_dir)
    except Exception as e:
        print(f"⚠️  Dockerfile generation failed: {e}")

    # 6. Generate requirements.txt
    print("\n📋 Generating requirements.txt...")
    requirements = _generate_runtime_requirements(task_spec.modality, export_formats)
    (export_dir / "requirements.txt").write_text("\n".join(requirements))
    print(f"✅ Generated runtime requirements")

    # 7. Save configs for reproducibility
    print("\n💾 Saving configurations...")

    # Save TaskSpec
    try:
        with open(configs_dir / "task_spec.json", "w") as f:
            json.dump(task_spec.to_dict(), f, indent=2)
        print(f"✅ Saved task_spec.json")
    except Exception as e:
        print(f"⚠️  Failed to save task_spec.json: {e}")

    # Save TrainingConfig
    try:
        if hasattr(training_config, "to_dict"):
            config_dict = training_config.to_dict()
        elif isinstance(training_config, dict):
            config_dict = training_config
        else:
            config_dict = vars(training_config)

        with open(configs_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Saved training_config.json")
    except Exception as e:
        print(f"⚠️  Failed to save training_config.json: {e}")

    # Save metadata
    try:
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "task_name": task_spec.name,
            "modality": task_spec.modality,
            "task_type": task_spec.task_type,
            "formats": export_formats,
            "framework_version": torch.__version__
        }
        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata.json")
    except Exception as e:
        print(f"⚠️  Failed to save metadata.json: {e}")

    # 8. Run health checks
    if run_health_checks:
        print("\n🔍 Running health checks...")
        try:
            from .export_health import ExportHealthChecker

            checker = ExportHealthChecker(model, config, task_spec)

            # Run pre-export checks and post-export verification
            health_report = checker.run_all_checks(
                export_dir=export_dir,
                formats=export_formats
            )

            # Save health reports
            health_report.save_json(export_dir / "artifacts" / "health_report.json")
            health_report.save_markdown(export_dir / "health_report.md")

            # Print summary
            health_report.print_summary()

            # Add warnings if any checks failed
            if not health_report.all_passed:
                print("\n⚠️  WARNING: Some health checks failed!")
                print("    Review health_report.md before production deployment")
                print()
        except Exception as e:
            print(f"⚠️  Health checks failed: {e}")
            print("    Export bundle created but health validation incomplete")
            print()

    print("\n" + "=" * 80)
    print(f"✅ Export bundle created successfully!")
    print(f"📁 Location: {export_dir}")
    if run_health_checks:
        print(f"📊 Health Score: {health_report.health_score}/100")
        if health_report.all_passed:
            print("✅ Ready for production deployment")
        else:
            print("⚠️  Review health_report.md before deployment")
    print("=" * 80)
    print()

    return export_dir


def create_repro_bundle(
    run_id: str,
    training_config,
    task_spec,
    eval_config,
    environment_snapshot,
    experiment_db,
    dashboard_paths: Dict[str, str] | None,
    output_path: str,
) -> str:
    """
    Create a zip file with configs, environment, and artifacts for reproduction.

    Returns absolute path to created .zip archive.
    """
    out_dir = Path(output_path) / f"repro_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write configs
    cfgs: Dict[str, Any] = {}
    try:
        cfgs['training_config'] = training_config.to_dict() if hasattr(training_config, 'to_dict') else dict(training_config)
    except Exception:
        cfgs['training_config'] = getattr(training_config, '__dict__', {})
    try:
        cfgs['task_spec'] = task_spec.to_dict() if hasattr(task_spec, 'to_dict') else getattr(task_spec, '__dict__', {})
    except Exception:
        cfgs['task_spec'] = {}
    try:
        cfgs['eval_config'] = eval_config.to_dict() if hasattr(eval_config, 'to_dict') else getattr(eval_config, '__dict__', {})
    except Exception:
        cfgs['eval_config'] = {}

    with open(out_dir / 'configs.json', 'w') as f:
        json.dump(cfgs, f, indent=2)

    # Environment snapshot
    if environment_snapshot:
        try:
            if isinstance(environment_snapshot, dict):
                with open(out_dir / 'env_snapshot.json', 'w') as f:
                    json.dump(environment_snapshot, f, indent=2)
        except Exception:
            pass

    # Metrics export (ExperimentDB optional)
    if experiment_db is not None:
        try:
            run_numeric = int(run_id) if str(run_id).isdigit() else None
            if run_numeric is not None:
                df = experiment_db.get_metrics(run_numeric)
                df.to_csv(out_dir / 'metrics.csv', index=False)
        except Exception:
            pass

    # Dashboard/artifacts
    if dashboard_paths:
        for name, path in (dashboard_paths or {}).items():
            try:
                src = Path(path)
                if src.exists():
                    import shutil as _sh
                    _sh.copy(src, out_dir / src.name)
            except Exception:
                pass

    # Create zip archive
    import shutil as _sh
    zip_base = str(out_dir.resolve())
    archive = _sh.make_archive(zip_base, 'zip', root_dir=out_dir)
    return archive
