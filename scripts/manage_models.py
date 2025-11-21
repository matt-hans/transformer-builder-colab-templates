#!/usr/bin/env python3
"""
CLI tool for managing model registry.

Provides command-line interface for:
- Registering models
- Listing and querying models
- Promoting models to tags
- Retiring and deleting models
- Comparing model metrics
- Viewing model lineage

Usage:
    # Register a model
    python scripts/manage_models.py register \\
        --name gpt-small \\
        --version 1.0.0 \\
        --checkpoint checkpoints/epoch_10.pt \\
        --task-type language_modeling \\
        --metrics '{"val_loss": 0.38, "perplexity": 1.46}'

    # List all models
    python scripts/manage_models.py list

    # List production models
    python scripts/manage_models.py list --tag production

    # Promote model to production
    python scripts/manage_models.py promote --model-id 5 --tag production

    # Compare models
    python scripts/manage_models.py compare --model-ids 1 2 3

    # View model lineage
    python scripts/manage_models.py lineage --model-id 5

    # Retire model
    python scripts/manage_models.py retire --model-id 3

    # Delete model
    python scripts/manage_models.py delete --model-id 7 --force

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'utils' / 'training'))

# Import directly to avoid torch dependency
from model_registry import ModelRegistry


def register_model(args: argparse.Namespace) -> None:
    """Register a new model."""
    registry = ModelRegistry(args.db)

    # Parse metrics JSON
    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in --metrics: {e}")
        sys.exit(1)

    # Parse optional fields
    export_formats = json.loads(args.export_formats) if args.export_formats else None
    metadata = json.loads(args.metadata) if args.metadata else None
    tags = args.tags.split(',') if args.tags else None

    # Compute config hash if config provided
    if args.config:
        try:
            config = json.loads(args.config)
            config_hash = ModelRegistry.compute_config_hash(config)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --config: {e}")
            sys.exit(1)
    else:
        config_hash = args.config_hash

    try:
        model_id = registry.register_model(
            name=args.name,
            version=args.version,
            checkpoint_path=args.checkpoint,
            task_type=args.task_type,
            config_hash=config_hash,
            metrics=metrics,
            export_formats=export_formats,
            model_size_mb=args.model_size_mb,
            memory_req_gb=args.memory_req_gb,
            training_run_id=args.training_run_id,
            parent_model_id=args.parent_model_id,
            metadata=metadata,
            tags=tags
        )
        print(f"✅ Registered model {model_id}: '{args.name}' v{args.version}")
        print(f"   Checkpoint: {args.checkpoint}")
        if tags:
            print(f"   Tags: {', '.join(tags)}")
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        sys.exit(1)


def list_models(args: argparse.Namespace) -> None:
    """List models with optional filtering."""
    registry = ModelRegistry(args.db)

    models = registry.list_models(
        task_type=args.task_type,
        tag=args.tag,
        status=args.status,
        limit=args.limit
    )

    if models.empty:
        print("No models found matching criteria.")
        return

    # Format output
    print(f"\n{'='*80}")
    print(f"{'Model ID':<10} {'Name':<25} {'Version':<10} {'Task Type':<20}")
    print(f"{'='*80}")

    for _, model in models.iterrows():
        print(f"{model['model_id']:<10} {model['name']:<25} "
              f"{model['version']:<10} {model['task_type']:<20}")

    print(f"{'='*80}")
    print(f"Total: {len(models)} model(s)\n")

    if args.verbose:
        print("\nDetailed view:")
        for _, model in models.iterrows():
            print(f"\n--- Model {model['model_id']} ---")
            print(f"Name: {model['name']}")
            print(f"Version: {model['version']}")
            print(f"Task: {model['task_type']}")
            print(f"Status: {model['status']}")
            print(f"Checkpoint: {model['checkpoint_path']}")
            print(f"Created: {model['created_at']}")

            metrics = json.loads(model['metrics']) if model['metrics'] else {}
            if metrics:
                print("Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")


def get_model(args: argparse.Namespace) -> None:
    """Get model details."""
    registry = ModelRegistry(args.db)

    if args.model_id:
        model = registry.get_model(model_id=args.model_id)
    elif args.name and args.version:
        model = registry.get_model(name=args.name, version=args.version)
    elif args.tag:
        model = registry.get_model(tag=args.tag)
    else:
        print("Error: Must provide --model-id, (--name and --version), or --tag")
        sys.exit(1)

    if model is None:
        print("Model not found.")
        sys.exit(1)

    # Print formatted output
    print(f"\n{'='*80}")
    print(f"Model {model['model_id']}: {model['name']} v{model['version']}")
    print(f"{'='*80}")
    print(f"Task Type: {model['task_type']}")
    print(f"Status: {model['status']}")
    print(f"Checkpoint: {model['checkpoint_path']}")
    print(f"Config Hash: {model['config_hash']}")
    print(f"Created: {model['created_at']}")
    print(f"Model Size: {model['model_size_mb']:.2f} MB")
    if model['memory_req_gb']:
        print(f"Memory Required: {model['memory_req_gb']:.2f} GB")
    if model['training_run_id']:
        print(f"Training Run ID: {model['training_run_id']}")
    if model['parent_model_id']:
        print(f"Parent Model ID: {model['parent_model_id']}")

    if model['metrics']:
        print("\nMetrics:")
        for key, value in model['metrics'].items():
            print(f"  {key}: {value}")

    if model['export_formats']:
        print(f"\nExport Formats: {', '.join(model['export_formats'])}")

    if model['metadata']:
        print("\nMetadata:")
        print(json.dumps(model['metadata'], indent=2))

    print(f"{'='*80}\n")


def promote_model(args: argparse.Namespace) -> None:
    """Promote model to tag."""
    registry = ModelRegistry(args.db)

    try:
        registry.promote_model(
            model_id=args.model_id,
            tag=args.tag,
            remove_from_others=not args.keep_others
        )
        print(f"✅ Promoted model {args.model_id} to tag '{args.tag}'")
        if not args.keep_others:
            print(f"   (Removed '{args.tag}' from other models)")
    except Exception as e:
        print(f"❌ Error promoting model: {e}")
        sys.exit(1)


def retire_model(args: argparse.Namespace) -> None:
    """Retire model."""
    registry = ModelRegistry(args.db)

    try:
        registry.retire_model(args.model_id)
        print(f"✅ Retired model {args.model_id}")
    except Exception as e:
        print(f"❌ Error retiring model: {e}")
        sys.exit(1)


def delete_model(args: argparse.Namespace) -> None:
    """Delete model."""
    registry = ModelRegistry(args.db)

    if not args.force:
        response = input(f"Are you sure you want to delete model {args.model_id}? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    try:
        registry.delete_model(args.model_id, force=args.force)
        print(f"✅ Deleted model {args.model_id}")
    except Exception as e:
        print(f"❌ Error deleting model: {e}")
        sys.exit(1)


def compare_models(args: argparse.Namespace) -> None:
    """Compare multiple models."""
    registry = ModelRegistry(args.db)

    model_ids = [int(id) for id in args.model_ids.split(',')]
    metrics = args.metrics.split(',') if args.metrics else None

    try:
        comparison = registry.compare_models(model_ids, metrics=metrics)

        if comparison.empty:
            print("No models found for comparison.")
            return

        # Print comparison table
        print(f"\n{'='*100}")
        print("Model Comparison")
        print(f"{'='*100}")
        print(comparison.to_string(index=False))
        print(f"{'='*100}\n")

    except Exception as e:
        print(f"❌ Error comparing models: {e}")
        sys.exit(1)


def show_lineage(args: argparse.Namespace) -> None:
    """Show model lineage."""
    registry = ModelRegistry(args.db)

    try:
        lineage = registry.get_model_lineage(args.model_id)

        if not lineage:
            print(f"No lineage found for model {args.model_id}")
            return

        print(f"\n{'='*80}")
        print(f"Model Lineage for Model {args.model_id}")
        print(f"{'='*80}")

        for i, model in enumerate(lineage):
            indent = "  " * i
            arrow = "└─> " if i > 0 else ""
            print(f"{indent}{arrow}Model {model['model_id']}: {model['name']} v{model['version']}")
            print(f"{indent}    Task: {model['task_type']}")
            print(f"{indent}    Created: {model['created_at']}")
            if model['metrics']:
                key_metrics = list(model['metrics'].items())[:3]
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in key_metrics])
                print(f"{indent}    Metrics: {metrics_str}")
            print()

        print(f"{'='*80}\n")

    except Exception as e:
        print(f"❌ Error retrieving lineage: {e}")
        sys.exit(1)


def add_export(args: argparse.Namespace) -> None:
    """Add export format to model."""
    registry = ModelRegistry(args.db)

    metadata = json.loads(args.metadata) if args.metadata else None

    try:
        registry.add_export_format(
            model_id=args.model_id,
            export_format=args.format,
            export_path=args.path,
            metadata=metadata
        )
        print(f"✅ Added {args.format} export for model {args.model_id}")
        print(f"   Path: {args.path}")
    except Exception as e:
        print(f"❌ Error adding export: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a model
  %(prog)s register --name gpt-small --version 1.0.0 \\
      --checkpoint checkpoints/epoch_10.pt --task-type language_modeling \\
      --metrics '{"val_loss": 0.38}' --config-hash abc123

  # List all models
  %(prog)s list

  # Promote to production
  %(prog)s promote --model-id 5 --tag production

  # Compare models
  %(prog)s compare --model-ids 1,2,3
        """
    )

    parser.add_argument(
        '--db',
        default='model_registry.db',
        help='Path to registry database (default: model_registry.db)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new model')
    register_parser.add_argument('--name', required=True, help='Model name')
    register_parser.add_argument('--version', required=True, help='Model version (e.g., 1.0.0)')
    register_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    register_parser.add_argument('--task-type', required=True, help='Task type')
    register_parser.add_argument('--metrics', required=True, help='Metrics JSON string')
    register_parser.add_argument('--config', help='Config JSON string (alternative to --config-hash)')
    register_parser.add_argument('--config-hash', help='Config hash (alternative to --config)')
    register_parser.add_argument('--export-formats', help='Export formats JSON list')
    register_parser.add_argument('--model-size-mb', type=float, help='Model size in MB')
    register_parser.add_argument('--memory-req-gb', type=float, help='Memory requirement in GB')
    register_parser.add_argument('--training-run-id', type=int, help='Training run ID')
    register_parser.add_argument('--parent-model-id', type=int, help='Parent model ID')
    register_parser.add_argument('--metadata', help='Metadata JSON string')
    register_parser.add_argument('--tags', help='Comma-separated tags')

    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--task-type', help='Filter by task type')
    list_parser.add_argument('--tag', help='Filter by tag')
    list_parser.add_argument('--status', default='active', help='Filter by status (default: active)')
    list_parser.add_argument('--limit', type=int, default=50, help='Max number of results')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed view')

    # Get command
    get_parser = subparsers.add_parser('get', help='Get model details')
    get_parser.add_argument('--model-id', type=int, help='Model ID')
    get_parser.add_argument('--name', help='Model name')
    get_parser.add_argument('--version', help='Model version')
    get_parser.add_argument('--tag', help='Tag name')

    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote model to tag')
    promote_parser.add_argument('--model-id', type=int, required=True, help='Model ID')
    promote_parser.add_argument('--tag', required=True, help='Tag name')
    promote_parser.add_argument('--keep-others', action='store_true',
                                help='Keep tag on other models')

    # Retire command
    retire_parser = subparsers.add_parser('retire', help='Retire model')
    retire_parser.add_argument('--model-id', type=int, required=True, help='Model ID')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete model')
    delete_parser.add_argument('--model-id', type=int, required=True, help='Model ID')
    delete_parser.add_argument('--force', action='store_true',
                              help='Force delete even if tagged')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('--model-ids', required=True,
                                help='Comma-separated model IDs')
    compare_parser.add_argument('--metrics', help='Comma-separated metrics to compare')

    # Lineage command
    lineage_parser = subparsers.add_parser('lineage', help='Show model lineage')
    lineage_parser.add_argument('--model-id', type=int, required=True, help='Model ID')

    # Add export command
    export_parser = subparsers.add_parser('add-export', help='Add export format')
    export_parser.add_argument('--model-id', type=int, required=True, help='Model ID')
    export_parser.add_argument('--format', required=True, help='Export format')
    export_parser.add_argument('--path', required=True, help='Path to export file')
    export_parser.add_argument('--metadata', help='Metadata JSON string')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate handler
    handlers = {
        'register': register_model,
        'list': list_models,
        'get': get_model,
        'promote': promote_model,
        'retire': retire_model,
        'delete': delete_model,
        'compare': compare_models,
        'lineage': show_lineage,
        'add-export': add_export
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
