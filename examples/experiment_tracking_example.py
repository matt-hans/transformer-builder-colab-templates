"""
Example demonstrating ExperimentDB for local experiment tracking.

This example shows how to use ExperimentDB alongside (or instead of) W&B
for tracking training experiments locally with SQLite.

Usage:
    python examples/experiment_tracking_example.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.training.experiment_db import ExperimentDB
from utils.training.training_config import TrainingConfig


def example_basic_tracking():
    """Basic experiment tracking workflow."""
    print("\n=== Basic Experiment Tracking ===\n")

    # Initialize database
    db = ExperimentDB('experiments.db')

    # Create training configuration
    config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=4,
        epochs=10,
        random_seed=42,
        wandb_project="transformer-training",
        run_name="baseline-v1"
    )

    # Log new run
    run_id = db.log_run(
        run_name='baseline-v1',
        config=config.to_dict(),
        notes='Initial baseline with default hyperparameters'
    )
    print(f"Created run {run_id}: baseline-v1")

    # Simulate training loop
    print("\nSimulating training loop...")
    for epoch in range(3):
        # Simulate epoch metrics
        train_loss = 0.5 - epoch * 0.1
        val_loss = 0.45 - epoch * 0.08
        val_accuracy = 0.75 + epoch * 0.05

        # Log epoch metrics
        db.log_metric(run_id, 'train/loss', train_loss, epoch=epoch)
        db.log_metric(run_id, 'val/loss', val_loss, epoch=epoch)
        db.log_metric(run_id, 'val/accuracy', val_accuracy, epoch=epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Simulate step-level metrics (10 batches per epoch)
        for step in range(10):
            global_step = epoch * 10 + step
            batch_loss = train_loss + (step * 0.01)
            db.log_metric(run_id, 'train/batch_loss', batch_loss, step=global_step, epoch=epoch)

    # Log artifacts
    db.log_artifact(run_id, 'checkpoint', 'checkpoints/epoch_2.pt',
                   metadata={'epoch': 2, 'val_loss': 0.29})
    print(f"\nLogged checkpoint artifact")

    # Mark run as completed
    db.update_run_status(run_id, 'completed')
    print(f"Marked run {run_id} as completed")


def example_compare_runs():
    """Compare multiple experiment runs."""
    print("\n=== Comparing Multiple Runs ===\n")

    db = ExperimentDB('experiments.db')

    # Create 3 different runs with different hyperparameters
    configs = [
        {'learning_rate': 1e-4, 'batch_size': 4, 'name': 'lr-1e4'},
        {'learning_rate': 5e-5, 'batch_size': 4, 'name': 'lr-5e5'},
        {'learning_rate': 1e-5, 'batch_size': 8, 'name': 'lr-1e5-bs8'},
    ]

    run_ids = []
    for config in configs:
        run_id = db.log_run(config['name'], config, notes=f"Testing {config['name']}")
        run_ids.append(run_id)

        # Simulate different final losses
        final_loss = 0.5 + (hash(config['name']) % 100) / 1000
        for epoch in range(5):
            val_loss = final_loss - epoch * 0.02
            db.log_metric(run_id, 'val/loss', val_loss, epoch=epoch)

        db.update_run_status(run_id, 'completed')

    # Compare runs
    comparison = db.compare_runs(run_ids)
    print("Run Comparison:")
    print(comparison[['run_id', 'run_name', 'final_val_loss', 'best_val_loss', 'best_epoch']])


def example_find_best_run():
    """Find best run by metric."""
    print("\n=== Finding Best Run ===\n")

    db = ExperimentDB('experiments.db')

    # Find best run by validation loss
    try:
        best_run = db.get_best_run('val/loss', mode='min')
        print(f"Best run by val/loss:")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Run Name: {best_run['run_name']}")
        print(f"  Best Val Loss: {best_run['best_value']:.4f}")
        print(f"  Best Epoch: {best_run['best_epoch']}")
        print(f"  Config: {best_run['config']}")
    except ValueError as e:
        print(f"Error: {e}")


def example_query_metrics():
    """Query and analyze metrics."""
    print("\n=== Querying Metrics ===\n")

    db = ExperimentDB('experiments.db')

    # List recent runs
    recent_runs = db.list_runs(limit=5)
    print("Recent runs:")
    print(recent_runs[['run_id', 'run_name', 'status', 'created_at']])

    if len(recent_runs) > 0:
        # Get metrics for first run
        run_id = recent_runs.iloc[0]['run_id']
        print(f"\nMetrics for run {run_id}:")

        # Get all metrics
        all_metrics = db.get_metrics(run_id)
        print(f"  Total metrics logged: {len(all_metrics)}")

        # Get specific metric
        train_loss = db.get_metrics(run_id, 'train/loss')
        if not train_loss.empty:
            print(f"\nTrain loss history:")
            print(train_loss[['epoch', 'value', 'timestamp']])


def cleanup_example_db():
    """Clean up example database."""
    db_path = Path('experiments.db')
    if db_path.exists():
        db_path.unlink()
        print("\nCleaned up example database")


if __name__ == '__main__':
    print("ExperimentDB - Local Experiment Tracking Example")
    print("=" * 60)

    # Run examples
    example_basic_tracking()
    example_compare_runs()
    example_find_best_run()
    example_query_metrics()

    # Optionally cleanup (comment out to keep database)
    # cleanup_example_db()

    print("\n" + "=" * 60)
    print("Example complete! Check 'experiments.db' for stored data.")
    print("\nTo explore the database:")
    print("  sqlite3 experiments.db")
    print("  SELECT * FROM runs;")
    print("  SELECT * FROM metrics;")
