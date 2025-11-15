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
              vocab_size: int = 50257,
              batch_size: int = 16,
              max_length: int = 512,
              learning_rate: float = 1e-4,
              max_epochs: int = 3,
              val_split: float = 0.1,
              accumulate_grad_batches: int = 1,
              early_stopping_patience: Optional[int] = None,
              save_top_k: int = 3,
              num_workers: int = 2,
              tokenizer: Optional[Any] = None,
              datamodule: Optional[Any] = None,
              resume_from_checkpoint: Optional[str] = None,
              seed: int = 42) -> Dict[str, Any]:
        """
        Train model end-to-end.

        Args:
            model: PyTorch model to train
            dataset: HuggingFace dataset name OR Dataset object
            dataset_path: Path to local dataset file
            config_name: HuggingFace dataset config (e.g., 'wikitext-2-raw-v1')
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

        # Set seed
        pl.seed_everything(seed)

        # Step 1: Load dataset (if not using pre-created datamodule)
        if datamodule is None:
            print("\n📊 Step 1: Loading Dataset")
            print("-" * 80)

            if dataset is not None:
                if isinstance(dataset, str):
                    # Load from HuggingFace
                    loader = DatasetLoader()
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

            # Step 3: Create DataModule
            print("\n📦 Step 3: Preparing DataModule")
            print("-" * 80)

            datamodule = AdaptiveTokenizerDataModule(
                dataset=dataset_obj,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                val_split=val_split,
                num_workers=num_workers
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
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_top_k=save_top_k,
            monitor='val_loss',
            mode='min',
            save_last=True
        )
        checkpoint_callback = checkpoint_manager.get_callback()
        callbacks.append(checkpoint_callback)

        # Early stopping
        if early_stopping_patience is not None:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                mode='min',
                verbose=True
            )
            callbacks.append(early_stop)
            print(f"  Early stopping: patience={early_stopping_patience}")

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
        print(f"  Precision: {self.precision}")
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
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            val_check_interval=1.0,
        )

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
