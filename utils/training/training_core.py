"""
Training Coordinator - High-level training API.

Simplifies the entire training workflow:
1. Load dataset → 2. Create tokenizer → 3. Train model → 4. Evaluate → 5. Save

One function to rule them all with smart defaults and automatic configuration.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, Literal, List
import torch
# Optional dependency - only needed for Tier 3
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    pl = None
    HAS_LIGHTNING = False

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import Dataset

from ..adapters.model_adapter import UniversalModelAdapter
from ..tokenization.adaptive_tokenizer import AdaptiveTokenizer
from ..tokenization.data_module import AdaptiveTokenizerDataModule
from .checkpoint_manager import CheckpointManager
from .dataset_utilities import DatasetLoader


class TrainingCoordinator:
    """
    High-level training orchestrator.

    Handles the complete training pipeline:
    - Dataset loading and preprocessing
    - Tokenizer creation/loading
    - Model adapter setup
    - Training with best practices
    - Checkpointing and early stopping
    - Metrics logging and visualization

    Example:
        >>> # Simple training
        >>> coordinator = TrainingCoordinator()
        >>> results = coordinator.train(
        ...     model=my_model,
        ...     dataset='wikitext',
        ...     config_name='wikitext-2-raw-v1',
        ...     vocab_size=50257,
        ...     max_epochs=3
        ... )
        >>>
        >>> # Advanced training with custom settings
        >>> results = coordinator.train(
        ...     model=my_model,
        ...     dataset_path='my_data.txt',
        ...     vocab_size=25000,
        ...     batch_size=32,
        ...     learning_rate=5e-4,
        ...     max_epochs=10,
        ...     early_stopping_patience=3
        ... )
    """

    def __init__(self,
                 output_dir: str = './training_output',
                 use_gpu: bool = True,
                 precision: Literal['32', '16', 'bf16'] = '16',
                 gradient_clip_val: float = 1.0):
        """
        Initialize training coordinator.

        Args:
            output_dir: Base directory for outputs
            use_gpu: Use GPU if available
            precision: Training precision ('32', '16', 'bf16')
            gradient_clip_val: Gradient clipping value
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_gpu = use_gpu
        self.precision = precision
        self.gradient_clip_val = gradient_clip_val

        # Subdirectories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

    def train(self,
              model: torch.nn.Module,
              dataset: Optional[Union[str, Dataset]] = None,
              dataset_path: Optional[str] = None,
              config_name: Optional[str] = None,
              val_dataset: Optional[Union[str, Dataset]] = None,
              val_config_name: Optional[str] = None,
              vocab_size: int = 50257,
              batch_size: int = 16,
              max_length: int = 512,
              learning_rate: float = 1e-4,
              max_epochs: int = 3,
              val_split: float = 0.1,
              accumulate_grad_batches: int = 1,
              early_stopping_patience: Optional[int] = 5,
              early_stopping_min_delta: float = 0.0,
              save_top_k: int = 3,
              save_every_n_epochs: int = 1,
              num_workers: int = 2,
              dataset_cache_dir: Optional[str] = None,
              tokenizer: Optional[Any] = None,
              datamodule: Optional[Any] = None,
              resume_from_checkpoint: Optional[str] = None,
              seed: int = 42,
              deterministic: bool = False,
              use_amp: Optional[bool] = None,
              drive_backup: bool = False,
              drive_base_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model end-to-end.

        Args:
            model: PyTorch model to train
            dataset: HuggingFace dataset name OR Dataset object
            dataset_path: Path to local dataset file
            config_name: HuggingFace dataset config (e.g., 'wikitext-2-raw-v1')
            val_dataset: Optional separate validation dataset (HF name or Dataset)
            val_config_name: Optional config name for validation dataset
            vocab_size: Vocabulary size for tokenizer
            batch_size: Training batch size
            max_length: Maximum sequence length
            learning_rate: Learning rate
            max_epochs: Maximum epochs to train
            val_split: Validation split fraction (0.0-1.0)
            accumulate_grad_batches: Gradient accumulation steps
            early_stopping_patience: Early stopping patience (None to disable)
            save_top_k: Number of best checkpoints to keep
            num_workers: DataLoader workers
            tokenizer: Pre-created tokenizer (optional)
            datamodule: Pre-created datamodule (optional)
            resume_from_checkpoint: Path to checkpoint to resume from
            seed: Random seed

        Returns:
            Dictionary with training results:
            - best_model_path: Path to best checkpoint
            - final_metrics: Final validation metrics
            - trainer: Lightning Trainer instance
            - model: Trained model

        Example:
            >>> results = coordinator.train(
            ...     model=transformer,
            ...     dataset='wikitext',
            ...     config_name='wikitext-2-raw-v1',
            ...     vocab_size=50257,
            ...     max_epochs=5
            ... )
            >>> print(f"Best model: {results['best_model_path']}")
            >>> print(f"Final loss: {results['final_metrics']['val_loss']:.4f}")
        """
        print("=" * 80)
        print("🚀 Training Coordinator")
        print("=" * 80)

        # Set seed for reproducibility
        # Use our comprehensive seed management instead of pl.seed_everything()
        # to ensure DataLoader workers and all randomness sources are seeded
        from .seed_manager import set_random_seed
        set_random_seed(seed, deterministic=deterministic)

        # Step 1: Load dataset (if not using pre-created datamodule)
        if datamodule is None:
            print("\n📊 Step 1: Loading Dataset")
            print("-" * 80)

            if dataset is not None:
                if isinstance(dataset, str):
                    # Load from HuggingFace
                    loader = DatasetLoader(cache_dir=dataset_cache_dir)
                    dataset_obj = loader.load_huggingface(
                        dataset,
                        config_name=config_name,
                        split='train'
                    )
                    loader.print_statistics(dataset_obj)
                else:
                    # Already a Dataset object
                    dataset_obj = dataset
            elif dataset_path is not None:
                # Load from local file
                loader = DatasetLoader()
                dataset_obj = loader.load_local_file(dataset_path)
                loader.print_statistics(dataset_obj)
            else:
                raise ValueError("Must provide either 'dataset' or 'dataset_path'")

            # Step 2: Create tokenizer (if not provided)
            if tokenizer is None:
                print("\n🔤 Step 2: Creating Tokenizer")
                print("-" * 80)

                tokenizer = AdaptiveTokenizer.load_or_create(
                    vocab_size=vocab_size,
                    dataset=dataset_obj
                )

            # Optional: separate validation dataset
            val_dataset_obj = None
            if val_dataset is not None:
                if isinstance(val_dataset, str):
                    loader = DatasetLoader()
                    val_dataset_obj = loader.load_huggingface(
                        val_dataset,
                        config_name=val_config_name,
                        split='validation'
                    )
                else:
                    val_dataset_obj = val_dataset

            # Step 3: Create DataModule
            print("\n📦 Step 3: Preparing DataModule")
            print("-" * 80)

            datamodule = AdaptiveTokenizerDataModule(
                dataset=dataset_obj,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                val_split=val_split,
                num_workers=num_workers,
                seed=seed,
                external_val_dataset=val_dataset_obj
            )

        # Step 4: Wrap model with adapter
        print("\n🔧 Step 4: Configuring Model Adapter")
        print("-" * 80)

        adapter = UniversalModelAdapter(
            model=model,
            learning_rate=learning_rate,
            vocab_size=vocab_size
        )

        print(f"  Model: {model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Vocab size: {vocab_size}")

        # Step 5: Setup callbacks
        print("\n⚙️  Step 5: Configuring Training")
        print("-" * 80)

        callbacks = []

        # Checkpoint callback
        from datetime import datetime
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_top_k=save_top_k,
            monitor='val_loss',
            mode='min',
            save_last=True,
            save_every_n_epochs=save_every_n_epochs,
            drive_backup=drive_backup,
            drive_backup_path=(
                (drive_base_dir or 'MyDrive/transformer-checkpoints') +
                f"/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ) if drive_backup else None
        )
        checkpoint_callback = checkpoint_manager.get_callback()
        callbacks.append(checkpoint_callback)

        # Best state_dict saver (saves best.pt on metric improvement)
        try:
            from .checkpoint_manager import BestStateDictCallback
            callbacks.append(BestStateDictCallback(
                checkpoint_dir=self.checkpoint_dir,
                metric_name='val_loss',
                mode='min'
            ))
        except Exception:
            pass

        # Optional Google Drive backup
        backup_cb = checkpoint_manager.get_backup_callback()
        if backup_cb is not None:
            callbacks.append(backup_cb)
            print(f"  Drive backup: enabled → {checkpoint_manager.drive_backup_path}")
        else:
            if drive_backup:
                print("  Drive backup: requested but not available (non-Colab env)")

        # Early stopping
        if early_stopping_patience is not None:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                mode='min',
                min_delta=early_stopping_min_delta,
                verbose=True
            )
            callbacks.append(early_stop)
            print(f"  Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

            # Add W&B logger for early stopping event/status (non-blocking)
            try:
                from .early_stopping import EarlyStoppingWandbCallback
                callbacks.append(EarlyStoppingWandbCallback(
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    mode='min'
                ))
            except Exception:
                pass

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        # Logger
        logger = TensorBoardLogger(
            save_dir=str(self.log_dir),
            name='training'
        )

        print(f"  Max epochs: {max_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {accumulate_grad_batches}")
        # Determine precision based on use_amp and environment
        from .amp_utils import compute_effective_precision
        effective_precision = compute_effective_precision(
            requested_precision=self.precision,
            use_amp=use_amp,
            cuda_available=torch.cuda.is_available(),
            use_gpu=self.use_gpu,
        )

        print(f"  Precision: {effective_precision}")
        print(f"  Gradient clip: {self.gradient_clip_val}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

        # Step 6: Create trainer
        print("\n🏃 Step 6: Starting Training")
        print("-" * 80)

        # Determine accelerator
        accelerator = 'auto' if self.use_gpu else 'cpu'
        devices = 'auto' if self.use_gpu else 1

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=effective_precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            val_check_interval=1.0,
        )

        # AMP monitoring (W&B): log enabled flag, precision, and loss scale when available
        try:
            from .amp_utils import AmpWandbCallback
            amp_cb = AmpWandbCallback(enabled=(effective_precision in ('16', '16-mixed', '16_true')),
                                      precision=effective_precision)
            callbacks.append(amp_cb)
            # Also update W&B config if active
            try:
                import wandb  # type: ignore
                if getattr(wandb, 'run', None):
                    wandb.config.update({'amp_enabled': amp_cb.enabled, 'amp_precision': amp_cb.precision}, allow_val_change=True)
            except Exception:
                pass
        except Exception:
            pass

        # Train
        trainer.fit(
            adapter,
            datamodule=datamodule,
            ckpt_path=resume_from_checkpoint
        )

        # Step 7: Training complete
        print("\n" + "=" * 80)
        print("✓ Training Complete!")
        print("=" * 80)

        # Get results
        best_model_path = checkpoint_callback.best_model_path
        final_metrics = trainer.callback_metrics

        print(f"\n📊 Final Results:")
        print(f"  Best checkpoint: {Path(best_model_path).name}")

        # Print metrics
        for key, value in final_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"  {key}: {value:.4f}")

        # Best validation perplexity (from best val_loss)
        best_score = getattr(checkpoint_callback, 'best_model_score', None)
        try:
            if best_score is not None:
                bs = best_score.item() if hasattr(best_score, 'item') else float(best_score)
                import math
                best_ppl = math.exp(min(bs, 20.0))
                print(f"  Best val perplexity (from best val_loss): {best_ppl:.2f}")
                # Log to W&B summary if active
                try:
                    import wandb
                    if getattr(wandb, 'run', None):
                        wandb.run.summary['best_val_perplexity'] = best_ppl
                except Exception:
                    pass
        except Exception:
            pass

        # Baseline references (approximate)
        print("\n📎 Perplexity Baselines (approx.):")
        print("  • GPT-2 small (WikiText-103): ~26")
        print("  • GPT-2 medium: ~19 | GPT-2 large: ~17")

        # TensorBoard info
        print(f"\n📈 View training progress:")
        print(f"  tensorboard --logdir {self.log_dir}")

        results = {
            'best_model_path': best_model_path,
            'final_metrics': {k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in final_metrics.items()},
            'trainer': trainer,
            'model': adapter,
            'checkpoint_manager': checkpoint_manager,
            'tokenizer': tokenizer if 'tokenizer' in locals() else None,
        }

        return results

    def export_state_dict(self,
                          results: Dict[str, Any],
                          output_dir: str = './exported_model',
                          upload_to_drive: bool = False,
                          drive_subdir: str = 'MyDrive/exported-models') -> str:
        """
        Convenience wrapper to export a trained model to PyTorch state_dict.

        Args:
            results: Training results dict returned by train()
            output_dir: Local export directory
            upload_to_drive: Copy export to Google Drive when running in Colab
            drive_subdir: Drive subdirectory to copy into

        Returns:
            Path to export directory
        """
        from .export_utilities import export_state_dict

        model = results.get('model')
        tokenizer = results.get('tokenizer')
        final_metrics = results.get('final_metrics', {})

        # Attempt to retrieve config from adapter or model
        cfg = None
        if hasattr(model, 'config'):
            cfg = getattr(model, 'config')

        export_path = export_state_dict(
            model=model,
            output_dir=output_dir,
            config=cfg,
            tokenizer=tokenizer,
            metrics=final_metrics,
            upload_to_drive=upload_to_drive,
            drive_subdir=drive_subdir
        )
        print(f"📦 Export complete: {export_path}")
        return export_path

    def publish_to_hub(self,
                       results: Dict[str, Any],
                       repo_name: str,
                       private: bool = False,
                       commit_message: str = 'Upload trained model') -> Optional[str]:
        """
        Convenience wrapper to push the trained model to HuggingFace Hub.
        Requires huggingface_hub; degrades gracefully if unavailable.
        """
        from .hf_hub import push_model_to_hub
        model = results.get('model')
        tokenizer = results.get('tokenizer')
        final_metrics = results.get('final_metrics', {})
        cfg = getattr(model, 'config', None)
        return push_model_to_hub(
            model=model,
            config=cfg,
            training_results=final_metrics,
            repo_name=repo_name,
            private=private,
            commit_message=commit_message
        )

    def quick_train(self,
                   model: torch.nn.Module,
                   dataset: str = 'wikitext',
                   config_name: str = 'wikitext-2-raw-v1',
                   vocab_size: int = 50257,
                   max_epochs: int = 3,
                   **kwargs) -> Dict[str, Any]:
        """
        Quick training with minimal configuration.

        Uses sensible defaults for common scenarios.

        Args:
            model: PyTorch model
            dataset: HuggingFace dataset name
            config_name: Dataset configuration
            vocab_size: Vocabulary size
            max_epochs: Number of epochs
            **kwargs: Additional arguments passed to train()

        Returns:
            Training results dictionary

        Example:
            >>> # Train GPT-2 on WikiText-2
            >>> results = coordinator.quick_train(
            ...     model=gpt2_model,
            ...     dataset='wikitext',
            ...     config_name='wikitext-2-raw-v1',
            ...     max_epochs=3
            ... )
        """
        return self.train(
            model=model,
            dataset=dataset,
            config_name=config_name,
            vocab_size=vocab_size,
            max_epochs=max_epochs,
            **kwargs
        )

    def resume_training(self,
                       checkpoint_path: str,
                       model_class: type,
                       max_epochs: int = 10,
                       **kwargs) -> Dict[str, Any]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class (UniversalModelAdapter)
            max_epochs: New max epochs
            **kwargs: Additional arguments

        Returns:
            Training results

        Example:
            >>> results = coordinator.resume_training(
            ...     checkpoint_path='checkpoints/best.ckpt',
            ...     model_class=UniversalModelAdapter,
            ...     max_epochs=10
            ... )
        """
        print(f"📂 Resuming from: {checkpoint_path}")

        return self.train(
            resume_from_checkpoint=checkpoint_path,
            max_epochs=max_epochs,
            **kwargs
        )


def train_model(model: torch.nn.Module,
                dataset: Union[str, Dataset],
                vocab_size: int,
                max_epochs: int = 3,
                batch_size: int = 16,
                learning_rate: float = 1e-4,
                **kwargs) -> Dict[str, Any]:
    """
    Convenient function for simple training workflows.

    This is a simplified wrapper around TrainingCoordinator.train()
    for quick experimentation.

    Args:
        model: PyTorch model
        dataset: HuggingFace dataset name or Dataset object
        vocab_size: Vocabulary size
        max_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional arguments

    Returns:
        Training results dictionary

    Example:
        >>> from utils.training import train_model
        >>> results = train_model(
        ...     model=my_transformer,
        ...     dataset='wikitext',
        ...     vocab_size=50257,
        ...     max_epochs=5
        ... )
        >>> print(f"Training complete! Best model: {results['best_model_path']}")
    """
    coordinator = TrainingCoordinator()

    return coordinator.train(
        model=model,
        dataset=dataset,
        vocab_size=vocab_size,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )
