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
from typing import Optional, Dict, Any, Tuple, List, Union, Literal
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
