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
from typing import Optional, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, default_data_collator


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
                     text_column: str = 'text'):
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
    
                # Split into train/val
                if self.val_split > 0:
                    split = tokenized_dataset.train_test_split(
                        test_size=self.val_split,
                        seed=42
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
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
                pin_memory=True
            )
    
        def val_dataloader(self) -> Optional[DataLoader]:
            """
            Create validation dataloader.
    
            Returns:
                Validation DataLoader or None if no validation split
            """
            if self.val_dataset is None:
                return None
    
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
                pin_memory=True
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
                     num_workers: int = 2):
            """
            Initialize with pre-tokenized datasets.
    
            Args:
                train_dataset: Tokenized training dataset
                val_dataset: Optional tokenized validation dataset
                batch_size: Batch size
                num_workers: Number of data loading workers
            """
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.batch_size = batch_size
            self.num_workers = num_workers
    
        def train_dataloader(self) -> DataLoader:
            """Create training dataloader."""
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
                pin_memory=True
            )
    
        def val_dataloader(self) -> Optional[DataLoader]:
            """Create validation dataloader."""
            if self.val_dataset is None:
                return None
    
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=default_data_collator,
                pin_memory=True
            )
else:
    class SimpleDataModule:
        """Stub - requires pytorch_lightning for Tier 3"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Install pytorch_lightning for Tier 3 tests")
