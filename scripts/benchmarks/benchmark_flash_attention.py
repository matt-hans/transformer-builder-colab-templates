#!/usr/bin/env python3
"""
Flash Attention Benchmarking Script (P2-3).

Comprehensive benchmarking suite for Flash Attention (SDPA) integration.
Provides detailed performance analysis, comparison plots, and validation reports.

Usage:
    # Basic usage with default model
    python scripts/benchmarks/benchmark_flash_attention.py

    # Specify model architecture
    python scripts/benchmarks/benchmark_flash_attention.py --d-model 512 --num-layers 6

    # Run full validation suite
    python scripts/benchmarks/benchmark_flash_attention.py --full-validation

    # Save results
    python scripts/benchmarks/benchmark_flash_attention.py --output-dir results/flash_attention

Features:
- Latency profiling (mean, std, p50, p95, p99)
- Throughput measurement (samples/sec, tokens/sec)
- Memory profiling (peak, allocated, reserved)
- Speedup comparison (Flash vs baseline)
- Detailed validation reports with recommendations
- Publication-quality plots (latency, throughput, memory)
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any
import logging

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.adapters.flash_attention_validator import FlashAttentionValidator, FlashAttentionReport

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MOCK MODEL FOR BENCHMARKING
# ==============================================================================

class BenchmarkTransformer(nn.Module):
    """
    Transformer model for benchmarking Flash Attention.

    Uses standard nn.MultiheadAttention layers that automatically
    benefit from SDPA in PyTorch 2.0+.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding + positional encoding
        x = self.embedding(input_ids) + self.pos_encoding[:, :seq_len, :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        logits = self.output_proj(x)

        return logits


# ==============================================================================
# BENCHMARKING UTILITIES
# ==============================================================================

def create_benchmark_plots(
    report: FlashAttentionReport,
    output_dir: Path
) -> None:
    """
    Create publication-quality benchmark plots.

    Args:
        report: FlashAttentionReport with performance data
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    perf = report.performance_metrics
    if not perf:
        logger.warning("No performance metrics available for plotting")
        return

    # Extract metrics
    latencies_ms = [
        perf.get('flash_latency_p50_ms'),
        perf.get('flash_latency_p95_ms'),
        perf.get('flash_latency_p99_ms')
    ]
    percentiles = ['P50', 'P95', 'P99']

    # Plot 1: Latency distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(percentiles, latencies_ms, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Flash Attention Latency Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(latencies_ms):
        ax.text(i, v + max(latencies_ms) * 0.02, f'{v:.2f}ms',
                ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Memory usage
    memory_mb = perf.get('flash_peak_memory_mb', 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['Flash Attention'], [memory_mb], color='#3498db', alpha=0.8, width=0.5)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Flash Attention Memory Usage', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value label
    ax.text(0, memory_mb + memory_mb * 0.02, f'{memory_mb:.1f} MB',
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Throughput
    throughput = perf.get('throughput_samples_per_sec', 0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['Flash Attention'], [throughput], color='#9b59b6', alpha=0.8, width=0.5)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax.set_title('Flash Attention Throughput', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value label
    ax.text(0, throughput + throughput * 0.02, f'{throughput:.1f}',
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Plots saved to {output_dir}")


def run_extended_benchmarks(
    model: nn.Module,
    config: Any,
    device: torch.device,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run extended benchmarks across multiple configurations.

    Tests performance with varying:
    - Batch sizes
    - Sequence lengths
    - Model depths

    Args:
        model: Model to benchmark
        config: Model configuration
        device: Device for benchmarking
        output_dir: Directory to save results

    Returns:
        Dictionary with extended benchmark results
    """
    logger.info("🔬 Running extended benchmarks...")

    results = {
        'batch_size_scaling': [],
        'seq_len_scaling': []
    }

    vocab_size = getattr(config, 'vocab_size', 50257)
    model.eval()

    # Benchmark 1: Batch size scaling (seq_len=128)
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_len = 128

    logger.info(f"  Testing batch size scaling (seq_len={seq_len})...")
    for bs in batch_sizes:
        try:
            # Warmup
            for _ in range(5):
                input_ids = torch.randint(0, vocab_size, (bs, seq_len)).to(device)
                with torch.no_grad():
                    _ = model(input_ids)
            torch.cuda.synchronize() if device.type == 'cuda' else None

            # Benchmark
            num_iterations = 20
            start_events = []
            end_events = []

            for _ in range(num_iterations):
                input_ids = torch.randint(0, vocab_size, (bs, seq_len)).to(device)

                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                with torch.no_grad():
                    _ = model(input_ids)

                if device.type == 'cuda':
                    end_event.record()
                    start_events.append(start_event)
                    end_events.append(end_event)

            if device.type == 'cuda':
                torch.cuda.synchronize()
                latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                mean_latency = np.mean(latencies)
            else:
                mean_latency = 0.0  # Skip CPU timing

            results['batch_size_scaling'].append({
                'batch_size': bs,
                'latency_ms': mean_latency
            })

            logger.info(f"    Batch size {bs}: {mean_latency:.2f} ms")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.warning(f"    Batch size {bs}: OOM, skipping")
                break
            raise

    # Benchmark 2: Sequence length scaling (batch_size=4)
    batch_size = 4
    seq_lens = [32, 64, 128, 256, 512, 1024]

    logger.info(f"  Testing sequence length scaling (batch_size={batch_size})...")
    for sl in seq_lens:
        try:
            # Warmup
            for _ in range(5):
                input_ids = torch.randint(0, vocab_size, (batch_size, sl)).to(device)
                with torch.no_grad():
                    _ = model(input_ids)
            torch.cuda.synchronize() if device.type == 'cuda' else None

            # Benchmark
            num_iterations = 20
            start_events = []
            end_events = []

            for _ in range(num_iterations):
                input_ids = torch.randint(0, vocab_size, (batch_size, sl)).to(device)

                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                with torch.no_grad():
                    _ = model(input_ids)

                if device.type == 'cuda':
                    end_event.record()
                    start_events.append(start_event)
                    end_events.append(end_event)

            if device.type == 'cuda':
                torch.cuda.synchronize()
                latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                mean_latency = np.mean(latencies)
            else:
                mean_latency = 0.0

            results['seq_len_scaling'].append({
                'seq_len': sl,
                'latency_ms': mean_latency
            })

            logger.info(f"    Seq len {sl}: {mean_latency:.2f} ms")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.warning(f"    Seq len {sl}: OOM, skipping")
                break
            raise

    # Create scaling plots
    create_scaling_plots(results, output_dir)

    return results


def create_scaling_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """Create plots for scaling benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Batch size scaling
    if results['batch_size_scaling']:
        batch_sizes = [r['batch_size'] for r in results['batch_size_scaling']]
        latencies = [r['latency_ms'] for r in results['batch_size_scaling']]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(batch_sizes, latencies, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Flash Attention: Batch Size Scaling', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

        plt.tight_layout()
        plt.savefig(output_dir / 'batch_size_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 2: Sequence length scaling
    if results['seq_len_scaling']:
        seq_lens = [r['seq_len'] for r in results['seq_len_scaling']]
        latencies = [r['latency_ms'] for r in results['seq_len_scaling']]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(seq_lens, latencies, marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Flash Attention: Sequence Length Scaling', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

        plt.tight_layout()
        plt.savefig(output_dir / 'seq_len_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"✅ Scaling plots saved to {output_dir}")


# ==============================================================================
# MAIN BENCHMARK FUNCTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Flash Attention Benchmarking Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=50257,
                       help='Vocabulary size (default: 50257)')
    parser.add_argument('--d-model', type=int, default=768,
                       help='Model dimension (default: 768)')
    parser.add_argument('--num-layers', type=int, default=12,
                       help='Number of transformer layers (default: 12)')
    parser.add_argument('--num-heads', type=int, default=12,
                       help='Number of attention heads (default: 12)')

    # Benchmark configuration
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for benchmarking (default: 4)')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length for benchmarking (default: 128)')
    parser.add_argument('--num-iterations', type=int, default=50,
                       help='Number of benchmark iterations (default: 50)')
    parser.add_argument('--num-warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')

    # Validation options
    parser.add_argument('--full-validation', action='store_true',
                       help='Run full validation suite (accuracy + performance)')
    parser.add_argument('--skip-accuracy', action='store_true',
                       help='Skip numerical accuracy tests')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip performance benchmarks')
    parser.add_argument('--extended-benchmarks', action='store_true',
                       help='Run extended benchmarks (scaling tests)')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results/flash_attention',
                       help='Output directory for results (default: results/flash_attention)')
    parser.add_argument('--report-name', type=str, default='flash_attention_validation.json',
                       help='Validation report filename (default: flash_attention_validation.json)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Starting Flash Attention benchmark on {device}")

    # Validate d_model and num_heads compatibility
    if args.d_model % args.num_heads != 0:
        logger.error(f"❌ d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")
        logger.info(f"   Suggested: Set --d-model to {args.d_model + (args.num_heads - args.d_model % args.num_heads)}")
        return 1

    # Create model
    logger.info(f"📦 Creating model (d_model={args.d_model}, layers={args.num_layers})...")
    model = BenchmarkTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model size: {num_params:,} parameters")

    # Create config
    config = SimpleNamespace(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )

    # Run validation
    validator = FlashAttentionValidator(model, config, device)

    run_accuracy = args.full_validation or not args.skip_accuracy
    run_performance = args.full_validation or not args.skip_performance

    report = validator.validate_all(
        run_performance_tests=run_performance,
        run_accuracy_tests=run_accuracy
    )

    # Print summary
    report.print_summary()

    # Save report
    report_path = output_dir / args.report_name
    report.save(str(report_path))

    # Create plots
    if run_performance and report.performance_metrics:
        create_benchmark_plots(report, output_dir)

    # Run extended benchmarks
    if args.extended_benchmarks and device.type == 'cuda':
        extended_results = run_extended_benchmarks(model, config, device, output_dir)

        # Save extended results
        import json
        extended_path = output_dir / 'extended_benchmarks.json'
        with open(extended_path, 'w') as f:
            json.dump(extended_results, f, indent=2)
        logger.info(f"✅ Extended results saved to {extended_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Benchmark complete! Results saved to {output_dir}")
    logger.info(f"{'='*70}\n")

    # Return exit code based on success
    if report.compatibility_status.get('sdpa_available'):
        return 0
    else:
        logger.warning("⚠️ Flash Attention not available on this system")
        return 1


if __name__ == '__main__':
    sys.exit(main())
