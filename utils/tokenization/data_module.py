"""
Adaptive Tokenizer DataModule for PyTorch Lightning

Integrates adaptive tokenization with Lightning's data loading system.
Handles dataset tokenization, train/val splits, and batch preparation.
"""

# Optional dependency - only needed for Tier 3
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    pl = None
    HAS_LIGHTNING = False

from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Optional, Union, Any
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, default_data_collator
from ..training.seed_manager import seed_worker, create_seeded_generator


def _get_collator(
    task_spec: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_dynamic_collator: bool = False,
    padding_side: str = 'right'
) -> Any:
    """Auto-select appropriate collator based on task modality.

    Args:
        task_spec: Optional TaskSpec object with modality information
        tokenizer: Tokenizer for text tasks (required if modality is 'text')
        use_dynamic_collator: Whether to use custom collators (vs default_data_collator)
        padding_side: Padding side for text collators ('left' or 'right')

    Returns:
        Appropriate collator function/object

    Raises:
        ValueError: If modality is unsupported or required params are missing
    """
    # If no task_spec or dynamic collator not requested, use default
    if task_spec is None or not use_dynamic_collator:
        return default_data_collator

    # Get modality from task_spec
    modality = getattr(task_spec, 'modality', 'text')

    if modality == "vision":
        # Import VisionDataCollator
        try:
            from .data_collator import VisionDataCollator
        except ImportError:
            # Fallback to default if import fails
            return default_data_collator

        # Extract normalization params from preprocessing_config
        preproc = getattr(task_spec, 'preprocessing_config', None) or {}
        return VisionDataCollator(
            normalize=preproc.get('normalize', True),
            mean=preproc.get('mean', None),  # None defaults to ImageNet
            std=preproc.get('std', None)
        )

    elif modality == "text":
        # Import LanguageModelingDataCollator
        try:
            from .data_collator import LanguageModelingDataCollator
        except ImportError:
            # Fallback to default if import fails
            return default_data_collator

        if tokenizer is None:
            raise ValueError("tokenizer is required for text modality tasks")

        # Determine if masked LM based on task_type
        task_type = getattr(task_spec, 'task_type', 'lm')
        mlm = (task_type == 'masked_lm')

        return LanguageModelingDataCollator(
            tokenizer=tokenizer,
            mlm=mlm,
            padding_side=padding_side
        )

    else:
        raise ValueError(
            f"Unsupported modality: {modality}. "
            f"Supported modalities are: 'text', 'vision'"
        )


if HAS_LIGHTNING:
    class AdaptiveTokenizerDataModule(pl.LightningDataModule):
        """
        Lightning DataModule with adaptive tokenization.
    
        Automatically tokenizes datasets and creates train/val dataloaders
        compatible with UniversalModelAdapter.
    
        Example:
            >>> from datasets import load_dataset
            >>> dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            >>> tokenizer = AdaptiveTokenizer.load_or_create(50257, dataset)
            >>>
            >>> datamodule = AdaptiveTokenizerDataModule(
            ...     dataset=dataset,
            ...     tokenizer=tokenizer,
            ...     batch_size=16,
            ...     max_length=512
            ... )
            >>>
            >>> trainer = pl.Trainer()
            >>> trainer.fit(model, datamodule)
        """
    
        def __init__(self,
                     dataset: Dataset,
                     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, 'CharacterLevelTokenizer'],
                     batch_size: int = 16,
                     max_length: int = 512,
                     val_split: float = 0.1,
                     num_workers: int = 2,
                     seed: int = 42,
                     text_column: str = 'text',
                     external_val_dataset: Optional[Dataset] = None,
                     use_dynamic_collator: bool = False,
                     padding_side: str = 'right'):
            """
            Initialize DataModule.
    
            Args:
                dataset: HuggingFace Dataset with text samples
                tokenizer: Tokenizer to use for encoding
                batch_size: Batch size for training
                max_length: Maximum sequence length
                val_split: Fraction of data to use for validation (0.0-1.0)
                num_workers: Number of workers for data loading
                text_column: Name of text column in dataset
            """
            super().__init__()
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_length = max_length
            self.val_split = val_split
            self.num_workers = num_workers
            self.text_column = text_column
            self.seed = seed
            self.external_val_dataset = external_val_dataset
            self.use_dynamic_collator = use_dynamic_collator
            self.padding_side = padding_side
    
            # Will be set in setup()
            self.train_dataset = None
            self.val_dataset = None
    
        def setup(self, stage: Optional[str] = None):
            """
            Setup datasets for training/validation.
    
            Args:
                stage: 'fit', 'validate', 'test', or None
            """
            if stage == 'fit' or stage is None:
                print(f"📊 Tokenizing dataset...")
                print(f"   Samples: {len(self.dataset):,}")
                print(f"   Max length: {self.max_length}")
                print(f"   Val split: {self.val_split:.1%}")
    
                # Tokenize dataset
                tokenized_dataset = self._tokenize_dataset()
    
                # Split into train/val unless external val dataset is provided
                if self.external_val_dataset is not None:
                    # Tokenize external validation dataset
                    ext_val = self.external_val_dataset

                    def _tok_one(examples):
                        if hasattr(self.tokenizer, '__call__'):
                            return self.tokenizer(
                                examples[self.text_column],
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors=None
                            )
                        else:
                            tok = {'input_ids': [], 'attention_mask': []}
                            for text in examples[self.text_column]:
                                encoded = self.tokenizer.encode(
                                    text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True
                                )
                                tok['input_ids'].append(encoded['input_ids'].tolist())
                                tok['attention_mask'].append(encoded['attention_mask'].tolist())
                            return tok

                    tokenized_val = ext_val.map(
                        _tok_one,
                        batched=True,
                        remove_columns=ext_val.column_names,
                        desc="Tokenizing (val)"
                    )
                    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

                    self.train_dataset = tokenized_dataset
                    self.val_dataset = tokenized_val
                else:
                    if self.val_split > 0:
                        split = tokenized_dataset.train_test_split(
                            test_size=self.val_split,
                            seed=self.seed
                        )
                        self.train_dataset = split['train']
                        self.val_dataset = split['test']
                    else:
                        # No validation split
                        self.train_dataset = tokenized_dataset
                        self.val_dataset = None
    
                print(f"✓ Dataset prepared:")
                print(f"  Training samples: {len(self.train_dataset):,}")
                if self.val_dataset:
                    print(f"  Validation samples: {len(self.val_dataset):,}")
    
        def _tokenize_dataset(self) -> Dataset:
            """
            Tokenize the dataset.
    
            Returns:
                Tokenized dataset with 'input_ids', 'attention_mask', 'labels'
            """
            def tokenize_function(examples):
                """Tokenize a batch of examples."""
                # Check if tokenizer is HuggingFace or custom
                if hasattr(self.tokenizer, '__call__'):
                    # HuggingFace tokenizer
                    tokenized = self.tokenizer(
                        examples[self.text_column],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors=None  # Return lists, not tensors (for datasets library)
                    )
                else:
                    # Custom tokenizer (e.g., CharacterLevelTokenizer)
                    # Process one at a time
                    tokenized = {'input_ids': [], 'attention_mask': []}
                    for text in examples[self.text_column]:
                        encoded = self.tokenizer.encode(
                            text,
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True
                        )
                        tokenized['input_ids'].append(encoded['input_ids'].tolist())
                        tokenized['attention_mask'].append(encoded['attention_mask'].tolist())
    
                # Create labels (same as input_ids for language modeling)
                tokenized['labels'] = tokenized['input_ids'].copy()
    
                return tokenized
    
            # Apply tokenization
            tokenized_dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.dataset.column_names,
                desc="Tokenizing"
            )
    
            # Set format for PyTorch
            tokenized_dataset.set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'labels']
            )
    
            return tokenized_dataset
    
        def train_dataloader(self) -> DataLoader:
            """
            Create training dataloader.
    
            Returns:
                Training DataLoader
            """
            # Create seeded generator for reproducible shuffling
            generator = create_seeded_generator(self.seed)

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator
            )
    
        def val_dataloader(self) -> Optional[DataLoader]:
            """
            Create validation dataloader.
    
            Returns:
                Validation DataLoader or None if no validation split
            """
            if self.val_dataset is None:
                return None
    
            # Create seeded generator (not used when shuffle=False but harmless)
            generator = create_seeded_generator(self.seed)

            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator
            )
    
        def get_sample_batch(self, split: str = 'train', num_samples: int = 1) -> dict:
            """
            Get a sample batch for testing.
    
            Args:
                split: 'train' or 'val'
                num_samples: Number of samples to return
    
            Returns:
                Dictionary with sample batch
            """
            dataset = self.train_dataset if split == 'train' else self.val_dataset
    
            if dataset is None:
                raise ValueError(f"No {split} dataset available")
    
            # Get first num_samples
            samples = dataset[:num_samples]
    
            return samples
    
    
if HAS_LIGHTNING:
    class SimpleDataModule(pl.LightningDataModule):
        """
        Simplified DataModule for already-tokenized datasets.
    
        Use this when you have pre-tokenized data or want more control.
    
        Example:
            >>> datamodule = SimpleDataModule(
            ...     train_dataset=tokenized_train,
            ...     val_dataset=tokenized_val,
            ...     batch_size=16
            ... )
        """
    
        def __init__(self,
                     train_dataset: Dataset,
                     val_dataset: Optional[Dataset] = None,
                     batch_size: int = 16,
                     num_workers: int = 2,
                     task_spec: Optional[Any] = None,
                     tokenizer: Optional[Any] = None,
                     use_dynamic_collator: bool = False,
                     padding_side: str = 'right'):
            """
            Initialize with pre-tokenized datasets.

            Args:
                train_dataset: Tokenized training dataset
                val_dataset: Optional tokenized validation dataset
                batch_size: Batch size
                num_workers: Number of data loading workers
                task_spec: Optional TaskSpec for auto-selecting collator
                tokenizer: Optional tokenizer for text tasks
                use_dynamic_collator: Whether to use task-specific collators
                padding_side: Padding side for text collators
            """
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.task_spec = task_spec
            self.tokenizer = tokenizer
            self.use_dynamic_collator = use_dynamic_collator
            self.padding_side = padding_side
    
        def train_dataloader(self) -> DataLoader:
            """Create training dataloader."""
            # Auto-select collator based on task_spec modality
            collate_fn = _get_collator(
                task_spec=self.task_spec,
                tokenizer=self.tokenizer,
                use_dynamic_collator=self.use_dynamic_collator,
                padding_side=self.padding_side
            )

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
    
        def val_dataloader(self) -> Optional[DataLoader]:
            """Create validation dataloader."""
            if self.val_dataset is None:
                return None

            # Auto-select collator based on task_spec modality
            collate_fn = _get_collator(
                task_spec=self.task_spec,
                tokenizer=self.tokenizer,
                use_dynamic_collator=self.use_dynamic_collator,
                padding_side=self.padding_side
            )

            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
else:
    class SimpleDataModule:
        """Stub - requires pytorch_lightning for Tier 3"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Install pytorch_lightning for Tier 3 tests")
