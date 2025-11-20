"""
Comprehensive demo of the automated retraining trigger system.

This example demonstrates:
1. Basic trigger usage (individual triggers)
2. Composite triggers (AND/OR logic)
3. Trigger manager with multiple triggers
4. Integration with MetricsEngine and ExperimentDB
5. Report generation and saving
6. Example configurations (conservative, aggressive, balanced)

Run with:
    python examples/retraining_trigger_demo.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.training.retraining_triggers import (
    # Core classes
    RetrainingTriggerManager,

    # Trigger implementations
    DriftTrigger,
    PerformanceTrigger,
    TimeTrigger,
    DataVolumeTrigger,
    CompositeTrigger,

    # Example configs
    get_conservative_config,
    get_aggressive_config,
    get_balanced_config,
)

from utils.training.experiment_db import ExperimentDB


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def demo_basic_triggers():
    """Demonstrate basic trigger usage."""
    print_section("1. Basic Trigger Usage")

    # -------------------------------------------------------------------------
    # Drift Trigger
    # -------------------------------------------------------------------------
    print("1.1 Drift Trigger")
    print("-" * 40)

    drift_trigger = DriftTrigger(
        threshold=0.15,
        metric_name='js_divergence',
        severity='warning'
    )

    # Simulate high drift scenario
    drift_metrics_high = {'js_divergence': 0.22}
    result = drift_trigger.evaluate(drift_metrics=drift_metrics_high)

    print(f"Drift value: {drift_metrics_high['js_divergence']:.2f}")
    print(f"Threshold: {drift_trigger.threshold:.2f}")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()

    # Simulate low drift scenario
    drift_metrics_low = {'js_divergence': 0.08}
    result = drift_trigger.evaluate(drift_metrics=drift_metrics_low)

    print(f"Drift value: {drift_metrics_low['js_divergence']:.2f}")
    print(f"Threshold: {drift_trigger.threshold:.2f}")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()

    # -------------------------------------------------------------------------
    # Performance Trigger
    # -------------------------------------------------------------------------
    print("1.2 Performance Trigger")
    print("-" * 40)

    perf_trigger = PerformanceTrigger(
        threshold=0.05,  # 5% degradation
        metric_name='val_loss',
        mode='min'
    )

    # Simulate performance degradation
    current_metrics = {'val_loss': 0.45, 'val_accuracy': 0.82}
    baseline_metrics = {'val_loss': 0.40, 'val_accuracy': 0.85}

    result = perf_trigger.evaluate(
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics
    )

    print(f"Baseline val_loss: {baseline_metrics['val_loss']:.4f}")
    print(f"Current val_loss: {current_metrics['val_loss']:.4f}")
    print(f"Change: {((current_metrics['val_loss'] - baseline_metrics['val_loss']) / baseline_metrics['val_loss'] * 100):+.1f}%")
    print(f"Threshold: {perf_trigger.threshold * 100:.1f}%")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()

    # -------------------------------------------------------------------------
    # Time Trigger
    # -------------------------------------------------------------------------
    print("1.3 Time Trigger")
    print("-" * 40)

    time_trigger = TimeTrigger(
        interval_hours=168,  # 1 week
        severity='info'
    )

    # Simulate old training
    last_training = datetime.now() - timedelta(days=10)
    metadata = {'last_training_time': last_training.isoformat()}

    result = time_trigger.evaluate(metadata=metadata)

    print(f"Last training: {last_training.strftime('%Y-%m-%d %H:%M')}")
    print(f"Interval: {time_trigger.interval_hours / 24:.1f} days")
    print(f"Elapsed: {result.actual_value / 24:.1f} days")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()

    # -------------------------------------------------------------------------
    # Data Volume Trigger
    # -------------------------------------------------------------------------
    print("1.4 Data Volume Trigger")
    print("-" * 40)

    volume_trigger = DataVolumeTrigger(
        threshold_samples=1000,
        threshold_percentage=0.20,
        severity='info'
    )

    # Simulate data growth
    metadata = {
        'current_count': 6500,
        'baseline_count': 5000
    }

    result = volume_trigger.evaluate(metadata=metadata)

    new_samples = metadata['current_count'] - metadata['baseline_count']
    percentage = (new_samples / metadata['baseline_count']) * 100

    print(f"Baseline count: {metadata['baseline_count']}")
    print(f"Current count: {metadata['current_count']}")
    print(f"New samples: {new_samples} ({percentage:.1f}% increase)")
    print(f"Thresholds: {volume_trigger.threshold_samples} samples OR {volume_trigger.threshold_percentage * 100:.0f}%")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()


def demo_composite_triggers():
    """Demonstrate composite trigger usage."""
    print_section("2. Composite Triggers (AND/OR Logic)")

    # -------------------------------------------------------------------------
    # OR Logic: Trigger if drift OR performance degrades
    # -------------------------------------------------------------------------
    print("2.1 OR Logic: Drift OR Performance")
    print("-" * 40)

    drift_trigger = DriftTrigger(threshold=0.15, name='drift_check')
    perf_trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss', name='perf_check')

    composite_or = CompositeTrigger(
        triggers=[drift_trigger, perf_trigger],
        logic='OR',
        name='drift_or_perf'
    )

    # Scenario: High drift, stable performance
    result = composite_or.evaluate(
        drift_metrics={'js_divergence': 0.22},
        current_metrics={'val_loss': 0.40},
        baseline_metrics={'val_loss': 0.39}
    )

    print(f"Scenario: High drift (0.22), stable performance")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()

    # -------------------------------------------------------------------------
    # AND Logic: Trigger if drift AND performance degrades
    # -------------------------------------------------------------------------
    print("2.2 AND Logic: Drift AND Performance")
    print("-" * 40)

    composite_and = CompositeTrigger(
        triggers=[drift_trigger, perf_trigger],
        logic='AND',
        name='drift_and_perf'
    )

    # Scenario: High drift, stable performance (should NOT trigger)
    result = composite_and.evaluate(
        drift_metrics={'js_divergence': 0.22},
        current_metrics={'val_loss': 0.40},
        baseline_metrics={'val_loss': 0.39}
    )

    print(f"Scenario: High drift (0.22), stable performance")
    print(f"Triggered: {result.triggered} (AND requires both)")
    print(f"Reason: {result.reason}")
    print()

    # Scenario: High drift AND performance degrades (should trigger)
    result = composite_and.evaluate(
        drift_metrics={'js_divergence': 0.22},
        current_metrics={'val_loss': 0.45},
        baseline_metrics={'val_loss': 0.40}
    )

    print(f"Scenario: High drift (0.22), performance degrades (0.40 -> 0.45)")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: {result.reason}")
    print()

    # -------------------------------------------------------------------------
    # Nested Logic: (Drift OR Performance) AND Time
    # -------------------------------------------------------------------------
    print("2.3 Nested Logic: (Drift OR Perf) AND Time")
    print("-" * 40)

    time_trigger = TimeTrigger(interval_hours=168, name='time_check')

    # Inner OR: drift OR performance
    inner_or = CompositeTrigger(
        triggers=[drift_trigger, perf_trigger],
        logic='OR',
        name='inner_or'
    )

    # Outer AND: (drift OR performance) AND time
    outer_and = CompositeTrigger(
        triggers=[inner_or, time_trigger],
        logic='AND',
        name='complex_policy'
    )

    last_training = datetime.now() - timedelta(days=10)

    result = outer_and.evaluate(
        drift_metrics={'js_divergence': 0.22},
        current_metrics={'val_loss': 0.40},
        baseline_metrics={'val_loss': 0.39},
        metadata={'last_training_time': last_training.isoformat()}
    )

    print(f"Scenario: High drift (0.22), stable perf, old training (10 days)")
    print(f"Triggered: {result.triggered}")
    print(f"Reason: Inner OR fires (drift), Outer AND requires time (also fires)")
    print()


def demo_trigger_manager():
    """Demonstrate trigger manager with multiple triggers."""
    print_section("3. Trigger Manager")

    # -------------------------------------------------------------------------
    # Setup manager with multiple triggers
    # -------------------------------------------------------------------------
    print("3.1 Register Multiple Triggers")
    print("-" * 40)

    manager = RetrainingTriggerManager()

    # Register triggers using convenience methods
    manager.register_drift_trigger(threshold=0.15, severity='warning')
    manager.register_performance_trigger(
        threshold=0.05,
        metric_name='val_loss',
        mode='min',
        severity='warning'
    )
    manager.register_time_trigger(interval_hours=168, severity='info')
    manager.register_data_volume_trigger(
        threshold_samples=1000,
        threshold_percentage=0.20,
        severity='info'
    )

    print(f"Registered {len(manager.triggers)} triggers:")
    for name in manager.triggers:
        print(f"  - {name}")
    print()

    # -------------------------------------------------------------------------
    # Evaluate all triggers
    # -------------------------------------------------------------------------
    print("3.2 Evaluate All Triggers")
    print("-" * 40)

    # Simulate comprehensive scenario
    last_training = datetime.now() - timedelta(days=10)

    report = manager.evaluate(
        drift_metrics={'js_divergence': 0.18, 'seq_length_js': 0.12},
        current_metrics={'val_loss': 0.45, 'val_accuracy': 0.82},
        baseline_metrics={'val_loss': 0.40, 'val_accuracy': 0.85},
        metadata={
            'last_training_time': last_training.isoformat(),
            'current_count': 6500,
            'baseline_count': 5000
        }
    )

    print(f"Overall Status: {'🔴 TRIGGERED' if report.triggered else '🟢 OK'}")
    print(f"Severity: {report.severity.upper()}")
    print(f"Timestamp: {report.timestamp}")
    print()

    print("Individual Trigger Results:")
    print("-" * 40)
    for detail in report.trigger_details:
        status = '✅' if detail.triggered else '⬜'
        print(f"{status} {detail.trigger_name}")
        print(f"   Severity: {detail.severity}")
        print(f"   Reason: {detail.reason}")
        if detail.threshold is not None and detail.actual_value is not None:
            print(f"   Threshold: {detail.threshold:.4f}, Actual: {detail.actual_value:.4f}")
        print()

    print("Recommendations:")
    print("-" * 40)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    print()

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------
    print("3.3 Save Report")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_json_path = Path(f.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        report_md_path = Path(f.name)

    try:
        # Save as JSON
        report.to_json(report_json_path)
        print(f"✓ Saved JSON report: {report_json_path}")

        # Save as Markdown
        markdown = report.to_markdown()
        with open(report_md_path, 'w') as f:
            f.write(markdown)
        print(f"✓ Saved Markdown report: {report_md_path}")

        # Display Markdown excerpt
        print("\nMarkdown Report Preview:")
        print("-" * 40)
        lines = markdown.split('\n')
        for line in lines[:20]:  # Show first 20 lines
            print(line)
        print("...")

    finally:
        # Cleanup
        report_json_path.unlink()
        report_md_path.unlink()

    print()


def demo_integration_with_experimentdb():
    """Demonstrate integration with ExperimentDB."""
    print_section("4. Integration with ExperimentDB")

    # -------------------------------------------------------------------------
    # Setup ExperimentDB
    # -------------------------------------------------------------------------
    print("4.1 Initialize ExperimentDB")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = Path(f.name)

    try:
        db = ExperimentDB(db_path)
        print(f"✓ Created database: {db_path}")

        # Create a training run
        run_id = db.log_run(
            run_name='baseline_model_v1',
            config={
                'learning_rate': 5e-5,
                'batch_size': 8,
                'epochs': 10
            },
            notes='Initial baseline model'
        )
        print(f"✓ Created training run: {run_id}")

        # Log metrics
        db.log_metric(run_id, 'val_loss', 0.40, epoch=0)
        db.log_metric(run_id, 'val_accuracy', 0.85, epoch=0)
        print(f"✓ Logged initial metrics")
        print()

        # -------------------------------------------------------------------------
        # Create manager with ExperimentDB
        # -------------------------------------------------------------------------
        print("4.2 Create Manager with ExperimentDB")
        print("-" * 40)

        manager = RetrainingTriggerManager(experiment_db=db)

        manager.register_drift_trigger(threshold=0.15)
        manager.register_performance_trigger(threshold=0.05, metric_name='val_loss')

        print(f"✓ Registered {len(manager.triggers)} triggers with ExperimentDB")
        print()

        # -------------------------------------------------------------------------
        # Evaluate and auto-log to database
        # -------------------------------------------------------------------------
        print("4.3 Evaluate and Auto-Log Trigger Event")
        print("-" * 40)

        last_training = datetime.now() - timedelta(days=10)

        report = manager.evaluate(
            drift_metrics={'js_divergence': 0.18},
            current_metrics={'val_loss': 0.45, 'val_accuracy': 0.82},
            baseline_metrics={'val_loss': 0.40, 'val_accuracy': 0.85},
            metadata={'last_training_time': last_training.isoformat()}
        )

        print(f"Triggered: {report.triggered}")
        print(f"Severity: {report.severity}")

        # Check if trigger event was logged
        runs = db.list_runs(limit=10)
        trigger_runs = runs[runs['run_name'].str.contains('trigger_event', na=False)]

        if len(trigger_runs) > 0:
            print(f"✓ Trigger event logged to ExperimentDB")
            print(f"  Run ID: {trigger_runs.iloc[0]['run_id']}")
            print(f"  Run Name: {trigger_runs.iloc[0]['run_name']}")
        else:
            print(f"  No trigger event logged (trigger didn't fire)")

        print()

        # -------------------------------------------------------------------------
        # Query trigger history
        # -------------------------------------------------------------------------
        print("4.4 Query Trigger History")
        print("-" * 40)

        # Perform multiple evaluations
        for i in range(3):
            manager.evaluate(
                drift_metrics={'js_divergence': 0.10 + i * 0.03},
                current_metrics={'val_loss': 0.40 + i * 0.02},
                baseline_metrics={'val_loss': 0.40}
            )

        history = manager.get_trigger_history(limit=5)
        print(f"Trigger history (last 5 evaluations):")
        for i, hist_report in enumerate(history, 1):
            print(f"  {i}. Triggered: {hist_report.triggered}, Severity: {hist_report.severity}")

        print()

    finally:
        # Cleanup
        db_path.unlink()
        print(f"✓ Cleaned up database: {db_path}")

    print()


def demo_example_configurations():
    """Demonstrate example configuration presets."""
    print_section("5. Example Configuration Presets")

    # -------------------------------------------------------------------------
    # Conservative configuration
    # -------------------------------------------------------------------------
    print("5.1 Conservative Configuration (Production)")
    print("-" * 40)

    conservative = get_conservative_config()

    print("Conservative thresholds (high thresholds, infrequent retraining):")
    print(f"  - Drift threshold: {conservative['drift'].threshold:.2f}")
    print(f"  - Performance threshold: {conservative['performance'].threshold:.2%}")
    print(f"  - Time interval: {conservative['time'].interval_hours / 24:.0f} days")
    print(f"  - Data volume: {conservative['data_volume'].threshold_percentage:.0%} increase")
    print()

    # -------------------------------------------------------------------------
    # Aggressive configuration
    # -------------------------------------------------------------------------
    print("5.2 Aggressive Configuration (Rapid Iteration)")
    print("-" * 40)

    aggressive = get_aggressive_config()

    print("Aggressive thresholds (low thresholds, frequent retraining):")
    print(f"  - Drift threshold: {aggressive['drift'].threshold:.2f}")
    print(f"  - Performance threshold: {aggressive['performance'].threshold:.2%}")
    print(f"  - Time interval: {aggressive['time'].interval_hours / 24:.0f} days")
    print(f"  - Data volume: {aggressive['data_volume'].threshold_percentage:.0%} increase")
    print()

    # -------------------------------------------------------------------------
    # Balanced configuration
    # -------------------------------------------------------------------------
    print("5.3 Balanced Configuration (Recommended Default)")
    print("-" * 40)

    balanced = get_balanced_config()

    print("Balanced thresholds (medium thresholds, weekly retraining):")
    print(f"  - Drift threshold: {balanced['drift'].threshold:.2f}")
    print(f"  - Performance threshold: {balanced['performance'].threshold:.2%}")
    print(f"  - Time interval: {balanced['time'].interval_hours / 24:.0f} days")
    print(f"  - Data volume: {balanced['data_volume'].threshold_percentage:.0%} increase")
    print()

    # -------------------------------------------------------------------------
    # Apply configuration to manager
    # -------------------------------------------------------------------------
    print("5.4 Apply Configuration to Manager")
    print("-" * 40)

    manager = RetrainingTriggerManager()

    # Apply balanced config
    balanced_config = get_balanced_config()

    manager.register_drift_trigger(
        threshold=balanced_config['drift'].threshold,
        severity=balanced_config['drift'].severity
    )
    manager.register_performance_trigger(
        threshold=balanced_config['performance'].threshold,
        metric_name=balanced_config['performance'].metric_name,
        mode=balanced_config['performance'].mode,
        severity=balanced_config['performance'].severity
    )
    manager.register_time_trigger(
        interval_hours=balanced_config['time'].interval_hours,
        severity=balanced_config['time'].severity
    )

    print(f"✓ Applied balanced configuration")
    print(f"  Registered {len(manager.triggers)} triggers")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("  Automated Retraining Trigger System - Comprehensive Demo")
    print("=" * 80)

    try:
        demo_basic_triggers()
        demo_composite_triggers()
        demo_trigger_manager()
        demo_integration_with_experimentdb()
        demo_example_configurations()

        print_section("Demo Complete!")
        print("Key Takeaways:")
        print("  1. Individual triggers (Drift, Performance, Time, DataVolume) monitor specific conditions")
        print("  2. Composite triggers combine multiple triggers with AND/OR logic")
        print("  3. RetrainingTriggerManager provides centralized trigger management")
        print("  4. Integration with ExperimentDB enables automatic logging of trigger events")
        print("  5. Predefined configurations (conservative, aggressive, balanced) for common scenarios")
        print()
        print("Next Steps:")
        print("  - Run tests: pytest tests/training/test_retraining_triggers.py -v")
        print("  - Customize triggers for your use case")
        print("  - Integrate with production ML pipeline")
        print()

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
