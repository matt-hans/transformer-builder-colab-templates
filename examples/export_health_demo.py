"""
Export Health Check Demo

This script demonstrates the export health check system with a simple transformer model.
It shows:
1. Model creation
2. Export bundle generation with health checks
3. Health report analysis
4. Remediation examples
"""

import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace

# Import health check utilities
from utils.training.export_health import ExportHealthChecker
from utils.training.export_utilities import create_export_bundle
from utils.training.training_config import TrainingConfig


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstration."""

    def __init__(self, vocab_size=1000, d_model=128, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.attention(x, x, x)
        return self.linear(x)


def main():
    """Run export health check demonstration."""
    print("=" * 80)
    print("Export Health Check Demonstration")
    print("=" * 80)

    # 1. Create model and configuration
    print("\n1. Creating model and configuration...")
    model = SimpleTransformer(vocab_size=1000, d_model=128, num_heads=4)

    config = SimpleNamespace(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        max_seq_len=64,
    )

    task_spec = SimpleNamespace(
        name="demo-task",
        modality="text",
        task_type="language_modeling",
        input_schema={"vocab_size": 1000, "max_seq_len": 64},
        output_schema={"num_classes": 1000},
    )

    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # 2. Run standalone health checks (pre-export)
    print("\n2. Running pre-export health checks...")
    checker = ExportHealthChecker(model, config, task_spec)
    report = checker.run_all_checks()

    # Print summary
    print(f"\n✓ Health checks completed:")
    print(f"  - Total checks: {report.summary['total']}")
    print(f"  - Passed: {report.summary['passed']}")
    print(f"  - Warnings: {report.summary['warnings']}")
    print(f"  - Failed: {report.summary['failed']}")
    print(f"  - Health Score: {report.health_score}/100")

    # Show individual check results
    print("\n3. Individual check results:")
    for check in report.checks[:5]:  # Show first 5
        status_emoji = {"passed": "✅", "warning": "⚠️", "failed": "❌"}
        emoji = status_emoji.get(check.status, "❓")
        print(f"  {emoji} {check.check_name}: {check.message}")

    # 4. Create full export bundle with health checks
    print("\n4. Creating export bundle with health checks...")

    training_config = TrainingConfig(
        export_bundle=True,
        export_formats=["pytorch"],  # Use PyTorch only for quick demo
        random_seed=42,
    )

    export_dir = create_export_bundle(
        model=model,
        config=config,
        task_spec=task_spec,
        training_config=training_config,
        export_base_dir="examples/demo_exports",
        run_health_checks=True,
    )

    print(f"\n✓ Export bundle created at: {export_dir}")

    # 5. Analyze health report
    print("\n5. Analyzing health report...")

    # Load JSON report
    import json

    health_json = export_dir / "artifacts" / "health_report.json"
    if health_json.exists():
        with open(health_json) as f:
            health_data = json.load(f)

        print(f"✓ Health report loaded:")
        print(f"  - Timestamp: {health_data['timestamp']}")
        print(f"  - Model: {health_data['model_name']}")
        print(f"  - Health Score: {health_data['health_score']}/100")
        print(f"  - All Passed: {health_data['all_passed']}")

        # Show recommendations
        if health_data.get("recommendations"):
            print(f"\n  Recommendations:")
            for i, rec in enumerate(health_data["recommendations"], 1):
                print(f"    {i}. {rec}")

    # Load Markdown report
    health_md = export_dir / "health_report.md"
    if health_md.exists():
        print(f"\n✓ Markdown report available at: {health_md}")
        print(f"  (Open in text editor for human-readable format)")

    # 6. Demonstrate failed check scenario
    print("\n6. Demonstrating failed check scenario...")
    print("   (Injecting NaN parameter for demonstration)")

    # Create model with NaN
    bad_model = SimpleTransformer(vocab_size=1000, d_model=128, num_heads=4)
    with torch.no_grad():
        bad_model.linear.weight[0, 0] = float("nan")

    bad_checker = ExportHealthChecker(bad_model, config, task_spec)
    bad_report = bad_checker.run_all_checks()

    print(f"\n✓ Health checks completed for bad model:")
    print(f"  - Health Score: {bad_report.health_score}/100")
    print(f"  - All Passed: {bad_report.all_passed}")

    # Show failed checks
    failed = bad_report.get_failed_checks()
    if failed:
        print(f"\n  Failed Checks:")
        for check in failed:
            print(f"    ❌ {check.check_name}: {check.message}")

    # Show recommendations
    if bad_report.recommendations:
        print(f"\n  Recommendations:")
        for i, rec in enumerate(bad_report.recommendations, 1):
            print(f"    {i}. {rec}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print(f"\nExplore the generated files:")
    print(f"  - Export bundle: {export_dir}")
    print(f"  - Health report (JSON): {health_json}")
    print(f"  - Health report (MD): {health_md}")
    print()


if __name__ == "__main__":
    main()
