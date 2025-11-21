"""
DataLoader and collation engine for training pipeline.

Provides architecture-agnostic data loading with:
- Protocol-based interfaces for data modules
- Registry-based collator system (text, vision, multimodal)
- Worker seeding for reproducibility
- Performance optimizations (pin_memory, prefetch)
- Automatic collator selection from TaskSpec

Phase 1 of training engine refactoring - extracted from tier3_training_utilities.py.
Integrates with SeedManager, TaskSpec, and CheckpointManager.
"""

from __future__ import annotations

import logging
from typing import Protocol, Any, Callable, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from datasets import Dataset as HFDataset

# Import existing utilities
from utils.training.seed_manager import seed_worker, create_seeded_generator
from utils.training.task_spec import TaskSpec

# Optional imports
try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    PreTrainedTokenizer = None
    PreTrainedTokenizerFast = None
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definitions
# =============================================================================

class DataModuleProtocol(Protocol):
    """
    Protocol for data modules providing train/val dataloaders.

    Inspired by PyTorch Lightning's LightningDataModule but framework-agnostic.
    Any class implementing train_dataloader() and val_dataloader() methods
    satisfies this protocol and can be used with the training engine.

    Example:
        >>> class MyDataModule:
        ...     def train_dataloader(self) -> DataLoader:
        ...         return DataLoader(train_dataset, batch_size=32)
        ...
        ...     def val_dataloader(self) -> DataLoader:
        ...         return DataLoader(val_dataset, batch_size=32)
        >>>
        >>> # Automatically satisfies DataModuleProtocol
        >>> dm: DataModuleProtocol = MyDataModule()
    """

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        ...

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return validation DataLoader (optional)."""
        ...


# =============================================================================
# Collator Registry
# =============================================================================

@dataclass
class CollatorInfo:
    """Metadata about a registered collator."""
    name: str
    factory: Callable[..., Any]
    modality: str
    description: str


class CollatorRegistry:
    """
    Registry for task-specific data collators.

    Enables automatic collator selection based on TaskSpec modality
    and supports custom collator registration.

    Features:
    - Decorator-based registration (@register_collator)
    - Auto-selection from TaskSpec.modality
    - Support for custom collators
    - Type-safe factory pattern

    Example:
        >>> registry = CollatorRegistry.get_instance()
        >>>
        >>> # Register custom collator
        >>> @registry.register('custom_text', modality='text')
        >>> def create_custom_collator(tokenizer):
        ...     return CustomCollator(tokenizer)
        >>>
        >>> # Auto-select from TaskSpec
        >>> task_spec = TaskSpec.vision_tiny()
        >>> collator = registry.get_collator(task_spec)
        >>> # Returns VisionDataCollator
    """

    _instance: Optional['CollatorRegistry'] = None

    def __init__(self) -> None:
        self._collators: Dict[str, CollatorInfo] = {}
        self._register_builtin_collators()

    @classmethod
    def get_instance(cls) -> 'CollatorRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(
        self,
        name: str,
        modality: str,
        description: str = ""
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a collator factory.

        Args:
            name: Unique collator name (e.g., 'text', 'vision')
            modality: Data modality ('text', 'vision', 'audio', 'tabular')
            description: Human-readable description

        Example:
            >>> registry = CollatorRegistry.get_instance()
            >>> @registry.register('my_collator', modality='text')
            >>> def create_my_collator(tokenizer):
            ...     return MyCollator(tokenizer)
        """
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._collators:
                logger.warning(f"Overwriting existing collator: {name}")

            self._collators[name] = CollatorInfo(
                name=name,
                factory=factory,
                modality=modality,
                description=description or f"{modality} collator"
            )
            logger.debug(f"Registered collator: {name} (modality={modality})")
            return factory

        return decorator

    def get_collator(
        self,
        task_spec: Optional[TaskSpec] = None,
        collator_name: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Get collator instance based on TaskSpec or explicit name.

        Args:
            task_spec: Optional TaskSpec for auto-selection by modality
            collator_name: Explicit collator name (overrides task_spec)
            **kwargs: Arguments passed to collator factory

        Returns:
            Collator instance (callable)

        Raises:
            ValueError: If collator not found or modality unsupported

        Example:
            >>> # Auto-select from TaskSpec
            >>> task_spec = TaskSpec.vision_tiny()
            >>> collator = registry.get_collator(task_spec)
            >>>
            >>> # Explicit collator name
            >>> collator = registry.get_collator(collator_name='text', tokenizer=tokenizer)
        """
        # Explicit name takes priority
        if collator_name:
            if collator_name not in self._collators:
                raise ValueError(
                    f"Collator '{collator_name}' not found. "
                    f"Available: {list(self._collators.keys())}"
                )
            return self._collators[collator_name].factory(**kwargs)

        # Auto-select from TaskSpec
        if task_spec:
            modality = task_spec.modality

            # Find collator for modality
            for info in self._collators.values():
                if info.modality == modality:
                    logger.debug(f"Auto-selected collator '{info.name}' for modality '{modality}'")
                    return info.factory(task_spec=task_spec, **kwargs)

            # Fallback to default_data_collator for unsupported modalities
            logger.warning(
                f"No collator found for modality '{modality}', "
                f"falling back to default_data_collator"
            )
            from transformers import default_data_collator
            return default_data_collator

        # No selection criteria provided
        raise ValueError(
            "Must provide either task_spec or collator_name for collator selection"
        )

    def list_collators(self) -> List[CollatorInfo]:
        """List all registered collators."""
        return list(self._collators.values())

    def _register_builtin_collators(self) -> None:
        """Register built-in collators (text, vision)."""

        # Text collator factory
        @self.register('text', modality='text', description='Text collator with dynamic padding')
        def create_text_collator(
            task_spec: Optional[TaskSpec] = None,
            tokenizer: Optional[Any] = None,
            padding_side: str = 'right',
            **kwargs: Any
        ) -> Any:
            """Create text collator (LanguageModelingDataCollator)."""
            from utils.tokenization.data_collator import LanguageModelingDataCollator

            if tokenizer is None:
                raise ValueError("tokenizer required for text collator")

            # Determine if masked LM from task_spec
            # Note: 'masked_lm' is not in current TaskType Literal, so mlm is always False
            # This is kept for future compatibility
            mlm = False
            # if task_spec and hasattr(task_spec, 'task_type'):
            #     mlm = (task_spec.task_type == 'masked_lm')

            return LanguageModelingDataCollator(
                tokenizer=tokenizer,
                mlm=mlm,
                padding_side=padding_side
            )

        # Vision collator factory
        @self.register('vision', modality='vision', description='Vision collator with normalization')
        def create_vision_collator(
            task_spec: Optional[TaskSpec] = None,
            normalize: bool = True,
            mean: Optional[tuple[float, ...]] = None,
            std: Optional[tuple[float, ...]] = None,
            **kwargs: Any
        ) -> Any:
            """Create vision collator (VisionDataCollator)."""
            from utils.tokenization.data_collator import VisionDataCollator

            # Extract normalization params from task_spec if provided
            if task_spec and task_spec.preprocessing_config:
                preproc = task_spec.preprocessing_config
                normalize = preproc.get('normalize', normalize)
                mean = preproc.get('mean', mean)
                std = preproc.get('std', std)

            return VisionDataCollator(
                normalize=normalize,
                mean=mean,
                std=std
            )

        # Default fallback collator
        @self.register('default', modality='any', description='HuggingFace default collator')
        def create_default_collator(**kwargs: Any) -> Any:
            """Create default collator."""
            if HAS_TRANSFORMERS:
                from transformers import default_data_collator
                return default_data_collator
            else:
                # Simple fallback that returns batch as-is
                return lambda batch: batch


# =============================================================================
# DataLoader Factory
# =============================================================================

@dataclass
class DataLoaderConfig:
    """
    Configuration for DataLoader creation.

    Attributes:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data (typically True for train, False for val)
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster CPU->GPU transfer (auto-detect if None)
        prefetch_factor: Batches to prefetch per worker (None for default)
        persistent_workers: Keep workers alive between epochs (auto-detect if None)
        drop_last: Drop incomplete last batch (useful for batch norm)
        seed: Random seed for reproducible shuffling
        collate_fn: Custom collate function (None for auto-selection)
    """
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: Optional[bool] = None  # Auto-detect based on CUDA
    prefetch_factor: Optional[int] = 2
    persistent_workers: Optional[bool] = None  # Auto-detect based on num_workers
    drop_last: bool = False
    seed: int = 42
    collate_fn: Optional[Callable[..., Any]] = None


class DataLoaderFactory:
    """
    Factory for creating optimized DataLoaders with reproducibility guarantees.

    Features:
    - Automatic GPU optimization (pin_memory, prefetch_factor)
    - Worker seeding for reproducibility
    - Collator auto-selection from TaskSpec
    - Performance optimizations configurable per environment

    Example:
        >>> factory = DataLoaderFactory()
        >>>
        >>> # Create train DataLoader with auto-optimizations
        >>> train_loader = factory.create_dataloader(
        ...     dataset=train_dataset,
        ...     config=DataLoaderConfig(batch_size=32, shuffle=True),
        ...     task_spec=task_spec,
        ...     tokenizer=tokenizer
        ... )
        >>>
        >>> # Create val DataLoader
        >>> val_loader = factory.create_dataloader(
        ...     dataset=val_dataset,
        ...     config=DataLoaderConfig(batch_size=32, shuffle=False),
        ...     task_spec=task_spec,
        ...     tokenizer=tokenizer
        ... )
    """

    def __init__(self, collator_registry: Optional[CollatorRegistry] = None):
        """
        Initialize factory.

        Args:
            collator_registry: Optional custom registry (uses singleton if None)
        """
        self.collator_registry = collator_registry or CollatorRegistry.get_instance()

    def create_dataloader(
        self,
        dataset: Union[Dataset, HFDataset, List[torch.Tensor]],
        config: DataLoaderConfig,
        task_spec: Optional[TaskSpec] = None,
        tokenizer: Optional[Any] = None
    ) -> DataLoader:
        """
        Create DataLoader with optimizations and reproducibility guarantees.

        Args:
            dataset: PyTorch Dataset, HuggingFace Dataset, or List[Tensor]
            config: DataLoader configuration
            task_spec: Optional TaskSpec for auto-collator selection
            tokenizer: Optional tokenizer (required for text tasks)

        Returns:
            Configured DataLoader

        Example:
            >>> config = DataLoaderConfig(batch_size=32, shuffle=True, seed=42)
            >>> loader = factory.create_dataloader(dataset, config, task_spec)
        """
        # Auto-detect optimizations
        cuda_available = torch.cuda.is_available()
        pin_memory = config.pin_memory if config.pin_memory is not None else cuda_available

        # Auto-detect persistent_workers (only if num_workers > 0)
        use_workers = config.num_workers > 0
        persistent_workers = config.persistent_workers
        if persistent_workers is None:
            persistent_workers = use_workers

        # Handle prefetch_factor (only valid if num_workers > 0)
        prefetch_factor = config.prefetch_factor if use_workers else None

        # Convert List[Tensor] to TensorDataset for backward compatibility
        if isinstance(dataset, list) and all(isinstance(x, torch.Tensor) for x in dataset):
            logger.debug(f"Converting List[Tensor] ({len(dataset)} samples) to TensorDataset")
            dataset = TensorDataset(torch.stack(dataset))

        # Select collator
        collate_fn = self._get_collate_fn(config, task_spec, tokenizer)

        # Create seeded generator for reproducible shuffling
        generator = create_seeded_generator(config.seed) if config.shuffle else None

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            drop_last=config.drop_last,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,  # Reproducibility: seed each worker
            generator=generator  # Reproducibility: seeded shuffling
        )

        logger.debug(
            f"Created DataLoader: batch_size={config.batch_size}, "
            f"num_workers={config.num_workers}, pin_memory={pin_memory}, "
            f"prefetch_factor={prefetch_factor}, shuffle={config.shuffle}"
        )

        return loader

    def _get_collate_fn(
        self,
        config: DataLoaderConfig,
        task_spec: Optional[TaskSpec],
        tokenizer: Optional[Any]
    ) -> Optional[Callable[..., Any]]:
        """Get collate function from config or auto-select from task_spec."""
        # Explicit collate_fn takes priority
        if config.collate_fn is not None:
            return config.collate_fn

        # Auto-select from task_spec
        if task_spec:
            try:
                collator = self.collator_registry.get_collator(
                    task_spec=task_spec,
                    tokenizer=tokenizer
                )
                # get_collator returns Any (could be various collator classes)
                # All collators are callable, so this is safe
                return collator  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning(f"Failed to auto-select collator: {e}")
                return None

        # No collator specified
        return None


# =============================================================================
# Universal Data Module
# =============================================================================

class UniversalDataModule:
    """
    Universal data module for training pipeline.

    Provides a unified interface for data loading across different data sources:
    - HuggingFace Datasets (lazy loading, production-ready)
    - PyTorch Datasets (standard interface)
    - List[Tensor] (legacy, backward compatibility)

    Features:
    - Automatic train/val split or external val dataset
    - Collator auto-selection from TaskSpec
    - Worker seeding for reproducibility
    - GPU optimizations (pin_memory, prefetch)
    - Integration with CheckpointManager (via DataLoader state)

    Example:
        >>> # From HuggingFace Dataset
        >>> from datasets import load_dataset
        >>> dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        >>>
        >>> data_module = UniversalDataModule(
        ...     train_data=dataset,
        ...     task_spec=TaskSpec.lm_tiny(),
        ...     tokenizer=tokenizer,
        ...     batch_size=32,
        ...     val_split=0.1,
        ...     seed=42
        ... )
        >>>
        >>> train_loader = data_module.train_dataloader()
        >>> val_loader = data_module.val_dataloader()
    """

    def __init__(
        self,
        train_data: Union[Dataset, HFDataset, List[torch.Tensor]],
        val_data: Optional[Union[Dataset, HFDataset, List[torch.Tensor]]] = None,
        task_spec: Optional[TaskSpec] = None,
        tokenizer: Optional[Any] = None,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 2,
        seed: int = 42,
        collator_registry: Optional[CollatorRegistry] = None
    ):
        """
        Initialize data module.

        Args:
            train_data: Training dataset (HF Dataset, PyTorch Dataset, or List[Tensor])
            val_data: Optional validation dataset (None for auto-split)
            task_spec: TaskSpec for collator auto-selection
            tokenizer: Optional tokenizer for text tasks
            batch_size: Batch size for training and validation
            val_split: Fraction of train_data for validation (if val_data=None)
            num_workers: Number of data loading workers
            seed: Random seed for reproducibility
            collator_registry: Optional custom collator registry
        """
        self.train_data = train_data
        self.val_data = val_data
        self.task_spec = task_spec
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed

        # Initialize factory
        self.factory = DataLoaderFactory(collator_registry)

        # Create validation split if not provided
        if self.val_data is None and self.val_split > 0:
            self._create_val_split()

        logger.info(
            f"UniversalDataModule initialized: "
            f"train_samples={self._get_dataset_length(self.train_data)}, "
            f"val_samples={self._get_dataset_length(self.val_data) if self.val_data else 0}"
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader.

        Returns:
            Training DataLoader with shuffling and optimizations
        """
        config = DataLoaderConfig(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            seed=self.seed
        )

        return self.factory.create_dataloader(
            dataset=self.train_data,
            config=config,
            task_spec=self.task_spec,
            tokenizer=self.tokenizer
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Create validation DataLoader.

        Returns:
            Validation DataLoader (no shuffling) or None if no val data
        """
        if self.val_data is None:
            return None

        config = DataLoaderConfig(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            seed=self.seed
        )

        return self.factory.create_dataloader(
            dataset=self.val_data,
            config=config,
            task_spec=self.task_spec,
            tokenizer=self.tokenizer
        )

    def _create_val_split(self) -> None:
        """Create validation split from train_data."""
        if isinstance(self.train_data, list):
            # List[Tensor] - simple split
            split_idx = int((1 - self.val_split) * len(self.train_data))
            self.val_data = self.train_data[split_idx:]
            self.train_data = self.train_data[:split_idx]
            logger.debug(
                f"Created val split from List[Tensor]: "
                f"train={len(self.train_data)}, val={len(self.val_data)}"
            )
        elif hasattr(self.train_data, 'train_test_split'):
            # HuggingFace Dataset - use built-in split
            split = self.train_data.train_test_split(
                test_size=self.val_split,
                seed=self.seed
            )
            self.train_data = split['train']
            self.val_data = split['test']
            logger.debug(
                f"Created val split from HF Dataset: "
                f"train={len(self.train_data)}, val={len(self.val_data)}"
            )
        else:
            # PyTorch Dataset - use random split
            from torch.utils.data import random_split
            train_size = int((1 - self.val_split) * len(self.train_data))
            val_size = len(self.train_data) - train_size

            # Use seeded generator for reproducible split
            generator = create_seeded_generator(self.seed)
            self.train_data, self.val_data = random_split(
                self.train_data,
                [train_size, val_size],
                generator=generator
            )
            logger.debug(
                f"Created val split from PyTorch Dataset: "
                f"train={train_size}, val={val_size}"
            )

    @staticmethod
    def _get_dataset_length(dataset: Optional[Union[Dataset, HFDataset, List[Any]]]) -> int:
        """Get dataset length (handles various types)."""
        if dataset is None:
            return 0
        if isinstance(dataset, list):
            return len(dataset)
        if hasattr(dataset, '__len__'):
            return len(dataset)
        return 0


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'DataModuleProtocol',
    'CollatorRegistry',
    'CollatorInfo',
    'DataLoaderConfig',
    'DataLoaderFactory',
    'UniversalDataModule',
]
