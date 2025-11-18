"""
Dataset loading and preprocessing utilities.

Supports multiple data sources:
- HuggingFace datasets (WikiText, TinyStories, etc.)
- Local files (TXT, JSON, CSV)
- Google Drive integration
- User uploads (Colab)

Includes automatic preprocessing, validation, and statistics.
"""

import os
import time
import json
import re
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Literal, Callable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset, load_dataset, DatasetDict
from tqdm.auto import tqdm


class DatasetLoader:
    """
    Universal dataset loader with support for multiple sources.

    Automatically handles:
    - HuggingFace dataset loading
    - Local file reading (TXT, JSON, CSV)
    - Google Drive integration
    - Text preprocessing and cleaning
    - Dataset validation and statistics

    Example:
        >>> # Load from HuggingFace
        >>> loader = DatasetLoader()
        >>> dataset = loader.load_huggingface('wikitext', 'wikitext-2-raw-v1', split='train')
        >>>
        >>> # Load from local file
        >>> dataset = loader.load_local_file('data.txt', text_column='text')
        >>>
        >>> # Load from Google Drive (in Colab)
        >>> dataset = loader.load_from_drive('/content/drive/MyDrive/data.txt')
        >>>
        >>> # Get statistics
        >>> stats = loader.get_statistics(dataset)
        >>> print(stats)
    """

    def __init__(self,
                 preprocessing: bool = True,
                 min_length: int = 10,
                 max_length: Optional[int] = None,
                 remove_duplicates: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Initialize dataset loader.

        Args:
            preprocessing: Apply automatic text cleaning
            min_length: Minimum character length for samples (shorter ones filtered)
            max_length: Maximum character length for samples (longer ones truncated)
            remove_duplicates: Remove exact duplicate samples
        """
        self.preprocessing = preprocessing
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.cache_dir = cache_dir

    def load_huggingface(self,
                        dataset_name: str,
                        config_name: Optional[str] = None,
                        split: Optional[str] = 'train',
                        streaming: bool = False,
                        trust_remote_code: bool = False) -> Union[Dataset, DatasetDict]:
        """
        Load dataset from HuggingFace Hub.

        Args:
            dataset_name: Dataset identifier (e.g., 'wikitext', 'openwebtext')
            config_name: Dataset configuration (e.g., 'wikitext-2-raw-v1')
            split: Dataset split ('train', 'validation', 'test', or None for all)
            streaming: Use streaming mode for large datasets
            trust_remote_code: Allow datasets with custom code

        Returns:
            Dataset or DatasetDict

        Example:
            >>> loader = DatasetLoader()
            >>> dataset = loader.load_huggingface('wikitext', 'wikitext-2-raw-v1')
            📊 Loading HuggingFace dataset: wikitext (wikitext-2-raw-v1)
            ✓ Loaded 36,718 samples
        """
        print(f"📊 Loading HuggingFace dataset: {dataset_name}", end="")
        if config_name:
            print(f" ({config_name})", end="")
        print()

        # Retry with exponential backoff for transient network errors
        last_exc = None
        for attempt in range(3):
            try:
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=trust_remote_code,
                    cache_dir=self.cache_dir
                )
                break
            except Exception as e:
                last_exc = e
                if attempt == 2:
                    print(f"❌ Failed to load dataset after retries: {e}")
                    raise
                wait = 2 ** attempt
                print(f"⚠️  Network error loading dataset, retrying in {wait}s... ({attempt + 1}/3)")
                time.sleep(wait)

            if not streaming:
                if isinstance(dataset, Dataset):
                    print(f"✓ Loaded {len(dataset):,} samples")
                else:
                    print(f"✓ Loaded dataset with splits: {list(dataset.keys())}")

        return dataset


class TinyVisionDataset(TorchDataset):
    """
    Lightweight vision dataset for tiny image classification tasks.

    Expects a directory with a ``labels.json`` file mapping image filenames
    to integer class labels. If image files or torchvision/PIL are not
    available, it falls back to randomly generated tensors with the
    configured image size, so that vision workflows remain usable in
    minimal environments.
    """

    def __init__(
        self,
        data_dir: Union[Path, str],
        image_size: Tuple[int, int, int] = (3, 64, 64),
        transforms: Optional[Callable[[Any], torch.Tensor]] = None,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing images and an optional labels.json.
            image_size: (C, H, W) target size for images.
            transforms: Optional custom torchvision-style transform pipeline.
            normalize: Whether to apply normalization when building default transforms.
            mean: Per-channel mean for normalization (default: [0.5, 0.5, 0.5]).
            std: Per-channel std for normalization (default: [0.5, 0.5, 0.5]).
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        labels_path = self.data_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels_map: Dict[str, int] = json.load(f)
            self.image_files: List[str] = sorted(self.labels_map.keys())
        else:
            # Synthetic fallback: small balanced label set with no on-disk images.
            num_samples = 16
            num_classes = 4
            self.image_files = [f"sample_{i:03d}.png" for i in range(num_samples)]
            self.labels_map = {
                fname: i % num_classes for i, fname in enumerate(self.image_files)
            }

        self.mean = mean or [0.5, 0.5, 0.5]
        self.std = std or [0.5, 0.5, 0.5]
        self._transforms = transforms
        self._has_torchvision = False

        if self._transforms is None:
            try:
                from torchvision import transforms as T  # type: ignore[import]

                _, h, w = self.image_size
                transform_list: List[Any] = [
                    T.Resize((h, w)),
                    T.ToTensor(),
                ]
                if normalize:
                    transform_list.append(T.Normalize(mean=self.mean, std=self.std))
                self._transforms = T.Compose(transform_list)
                self._has_torchvision = True
            except Exception:
                # torchvision is optional; fall back to random tensors in __getitem__.
                self._transforms = None

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                - 'pixel_values': Tensor[C, H, W]
                - 'labels': int
        """
        img_file = self.image_files[idx]
        img_path = self.data_dir / img_file
        c, h, w = self.image_size

        pixel_values: torch.Tensor

        if self._transforms is not None and img_path.exists():
            try:
                from PIL import Image  # type: ignore[import]

                image = Image.open(img_path).convert("RGB")
                pixel_values = self._transforms(image)
            except Exception:
                pixel_values = torch.rand(c, h, w)
        else:
            pixel_values = torch.rand(c, h, w)

        label = int(self.labels_map[img_file])

        return {"pixel_values": pixel_values, "labels": label}

    def load_local_file(self,
                       file_path: Union[str, Path],
                       file_format: Optional[Literal['txt', 'json', 'csv']] = None,
                       text_column: str = 'text',
                       encoding: str = 'utf-8') -> Dataset:
        """
        Load dataset from local file.

        Supports:
        - TXT: One sample per line or paragraph
        - JSON: List of dicts or JSONL format
        - CSV: Pandas-compatible CSV with text column

        Args:
            file_path: Path to local file
            file_format: File format (auto-detected if None)
            text_column: Column name containing text
            encoding: Text encoding (default: utf-8)

        Returns:
            Dataset with text samples

        Example:
            >>> dataset = loader.load_local_file('data.txt')
            📂 Loading local file: data.txt
            ✓ Loaded 1,000 samples
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format
        if file_format is None:
            file_format = file_path.suffix.lstrip('.').lower()
            if file_format not in ['txt', 'json', 'csv', 'jsonl']:
                raise ValueError(f"Unsupported file format: {file_format}")

        print(f"📂 Loading local file: {file_path.name}")

        # Load based on format
        if file_format == 'txt':
            samples = self._load_txt(file_path, encoding)
        elif file_format in ['json', 'jsonl']:
            samples = self._load_json(file_path, text_column, encoding)
        elif file_format == 'csv':
            samples = self._load_csv(file_path, text_column, encoding)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Create dataset
        dataset = Dataset.from_dict({text_column: samples})

        # Apply preprocessing
        if self.preprocessing:
            dataset = self._preprocess_dataset(dataset, text_column)

        print(f"✓ Loaded {len(dataset):,} samples")

        return dataset

    def load_from_drive(self,
                       drive_path: Union[str, Path],
                       text_column: str = 'text',
                       mount_point: str = '/content/drive') -> Dataset:
        """
        Load dataset from Google Drive (Colab environment).

        Automatically mounts drive if not already mounted.

        Args:
            drive_path: Path relative to drive root or absolute path
            text_column: Column name for text data
            mount_point: Drive mount point (default: /content/drive)

        Returns:
            Dataset with text samples

        Example:
            >>> # In Colab
            >>> dataset = loader.load_from_drive('/content/drive/MyDrive/data.txt')
            🔗 Mounting Google Drive...
            ✓ Drive mounted
            📂 Loading from Drive: data.txt
            ✓ Loaded 5,000 samples
        """
        drive_path = Path(drive_path)

        # Check if we're in Colab
        try:
            from google.colab import drive as colab_drive

            # Mount if not already mounted
            if not Path(mount_point).exists():
                print("🔗 Mounting Google Drive...")
                colab_drive.mount(mount_point)
                print("✓ Drive mounted")
            else:
                print("✓ Drive already mounted")

        except ImportError:
            print("⚠️  Not in Colab environment - treating as local path")

        # If path is not absolute, assume it's relative to MyDrive
        if not drive_path.is_absolute():
            drive_path = Path(mount_point) / 'MyDrive' / drive_path

        # Load as local file
        print(f"📂 Loading from Drive: {drive_path.name}")
        return self.load_local_file(drive_path, text_column=text_column)

    def _load_txt(self, file_path: Path, encoding: str) -> List[str]:
        """Load text file (one sample per line or paragraph)."""
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()

        # Try splitting by double newline (paragraphs)
        samples = content.split('\n\n')

        # If very few samples, split by single newline instead
        if len(samples) < 10:
            samples = content.split('\n')

        # Filter empty lines
        samples = [s.strip() for s in samples if s.strip()]

        return samples

    def _load_json(self, file_path: Path, text_column: str, encoding: str) -> List[str]:
        """Load JSON or JSONL file."""
        with open(file_path, 'r', encoding=encoding) as f:
            # Try loading as single JSON object
            try:
                data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    # List of dicts
                    if isinstance(data[0], dict):
                        samples = [item.get(text_column, str(item)) for item in data]
                    else:
                        # List of strings
                        samples = [str(item) for item in data]
                elif isinstance(data, dict):
                    # Single dict - extract text column
                    samples = data.get(text_column, [])
                    if not isinstance(samples, list):
                        samples = [str(samples)]
                else:
                    samples = [str(data)]

            except json.JSONDecodeError:
                # Try JSONL format (one JSON per line)
                f.seek(0)
                samples = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict):
                                samples.append(item.get(text_column, str(item)))
                            else:
                                samples.append(str(item))
                        except json.JSONDecodeError:
                            continue

        return samples

    def _load_csv(self, file_path: Path, text_column: str, encoding: str) -> List[str]:
        """Load CSV file using pandas."""
        df = pd.read_csv(file_path, encoding=encoding)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

        samples = df[text_column].astype(str).tolist()

        return samples

    def _preprocess_dataset(self, dataset: Dataset, text_column: str) -> Dataset:
        """
        Apply preprocessing to dataset.

        Steps:
        1. Clean text (remove excess whitespace, control characters)
        2. Filter by length
        3. Remove duplicates (optional)
        """
        initial_size = len(dataset)

        print("🔧 Preprocessing dataset...")

        # Clean text
        def clean_text(example):
            text = example[text_column]

            # Remove control characters (except newlines and tabs)
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Truncate if max_length specified
            if self.max_length and len(text) > self.max_length:
                text = text[:self.max_length]

            example[text_column] = text
            return example

        dataset = dataset.map(clean_text, desc="Cleaning text")

        # Filter by minimum length
        dataset = dataset.filter(
            lambda x: len(x[text_column]) >= self.min_length,
            desc="Filtering by length"
        )

        # Remove duplicates
        if self.remove_duplicates:
            seen = set()

            def is_unique(example):
                text = example[text_column]
                if text in seen:
                    return False
                seen.add(text)
                return True

            dataset = dataset.filter(is_unique, desc="Removing duplicates")

        filtered = initial_size - len(dataset)
        if filtered > 0:
            print(f"  Filtered {filtered:,} samples ({filtered/initial_size*100:.1f}%)")

        return dataset

    def get_statistics(self, dataset: Union[Dataset, DatasetDict],
                      text_column: str = 'text') -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Args:
            dataset: Dataset or DatasetDict
            text_column: Column containing text

        Returns:
            Dictionary with statistics:
            - num_samples: Total samples
            - total_chars: Total characters
            - total_words: Total words (approximate)
            - avg_chars: Average characters per sample
            - avg_words: Average words per sample
            - min_chars: Minimum sample length
            - max_chars: Maximum sample length

        Example:
            >>> stats = loader.get_statistics(dataset)
            >>> print(f"Samples: {stats['num_samples']:,}")
            >>> print(f"Avg length: {stats['avg_chars']:.0f} chars")
        """
        if isinstance(dataset, DatasetDict):
            # Combine all splits for statistics
            all_samples = []
            for split_name, split_data in dataset.items():
                all_samples.extend(split_data[text_column])
            samples = all_samples
        else:
            samples = dataset[text_column]

        # Compute statistics
        lengths = [len(s) for s in samples]
        word_counts = [len(s.split()) for s in samples]

        stats = {
            'num_samples': len(samples),
            'total_chars': sum(lengths),
            'total_words': sum(word_counts),
            'avg_chars': sum(lengths) / len(lengths) if lengths else 0,
            'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_chars': min(lengths) if lengths else 0,
            'max_chars': max(lengths) if lengths else 0,
        }

        return stats

    def print_statistics(self, dataset: Union[Dataset, DatasetDict],
                        text_column: str = 'text'):
        """
        Print formatted dataset statistics.

        Args:
            dataset: Dataset or DatasetDict
            text_column: Column containing text
        """
        stats = self.get_statistics(dataset, text_column)

        print("\n📊 Dataset Statistics:")
        print(f"  Samples: {stats['num_samples']:,}")
        print(f"  Total characters: {stats['total_chars']:,}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Average length: {stats['avg_chars']:.0f} chars ({stats['avg_words']:.0f} words)")
        print(f"  Length range: {stats['min_chars']:,} - {stats['max_chars']:,} chars")

    def preview_samples(self, dataset: Dataset,
                       num_samples: int = 3,
                       text_column: str = 'text'):
        """
        Print preview of dataset samples.

        Args:
            dataset: Dataset to preview
            num_samples: Number of samples to show
            text_column: Column containing text
        """
        print(f"\n👀 Sample Preview ({num_samples} samples):")
        print("─" * 80)

        for i in range(min(num_samples, len(dataset))):
            text = dataset[i][text_column]
            # Truncate long samples
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"\nSample {i+1}:")
            print(text)
            print("─" * 80)


class DatasetUploader:
    """
    User-friendly dataset upload for Colab environment.

    Provides:
    - File upload widget
    - Drag-and-drop support (via widget)
    - Format validation
    - Preview before processing
    - Size limits and warnings

    Example:
        >>> # In Colab
        >>> uploader = DatasetUploader()
        >>> dataset = uploader.upload_and_load()
        ┌──────────────────────────┐
        │  Upload Dataset File     │
        │  (TXT, JSON, CSV)        │
        │  Max size: 500 MB        │
        └──────────────────────────┘
        [Upload widget appears]
        ✓ Uploaded: my_data.txt (1.2 MB)
        ✓ Loaded 10,000 samples
    """

    def __init__(self,
                 max_size_mb: int = 500,
                 text_column: str = 'text'):
        """
        Initialize dataset uploader.

        Args:
            max_size_mb: Maximum file size in MB
            text_column: Column name for text data
        """
        self.max_size_mb = max_size_mb
        self.text_column = text_column

    def upload_and_load(self,
                       preprocessing: bool = True,
                       preview: bool = True) -> Optional[Dataset]:
        """
        Upload file and load as dataset.

        Args:
            preprocessing: Apply automatic preprocessing
            preview: Show sample preview before loading

        Returns:
            Dataset or None if upload cancelled
        """
        try:
            from google.colab import files
        except ImportError:
            print("❌ This feature requires Google Colab environment")
            return None

        print("┌" + "─" * 40 + "┐")
        print("│  📤 Upload Dataset File" + " " * 16 + "│")
        print("│  Supported: TXT, JSON, CSV" + " " * 12 + "│")
        print(f"│  Max size: {self.max_size_mb} MB" + " " * (28 - len(str(self.max_size_mb))) + "│")
        print("└" + "─" * 40 + "┘")
        print()

        # Upload file
        uploaded = files.upload()

        if not uploaded:
            print("⚠️  No file uploaded")
            return None

        # Get uploaded file
        filename = list(uploaded.keys())[0]
        file_size_mb = len(uploaded[filename]) / (1024 * 1024)

        print(f"✓ Uploaded: {filename} ({file_size_mb:.1f} MB)")

        # Check size
        if file_size_mb > self.max_size_mb:
            print(f"❌ File too large ({file_size_mb:.1f} MB > {self.max_size_mb} MB)")
            return None


# -----------------------------------------------------------------------------
# Task-aware dataloader builder (Workstream C placeholder)
# -----------------------------------------------------------------------------

class _IntSeqDataset(TorchDataset):
    def __init__(self, samples: List[List[int]], labels: Optional[List[int]] = None):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.long)
        if self.labels is None:
            return {"input_ids": x, "labels": x.clone()}
        else:
            return {"input_ids": x, "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)}


def _char_to_ids(s: str, vocab_size: int, max_len: int) -> List[int]:
    ids = [(ord(ch) % max(vocab_size, 2)) for ch in s]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))


def _load_lm_tiny(path: Union[str, Path], vocab_size: int, max_len: int, limit: Optional[int]) -> _IntSeqDataset:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Map to ids deterministically
            ids = _char_to_ids(line, vocab_size, max_len)
            lines.append(ids)
            if limit and len(lines) >= limit:
                break
    return _IntSeqDataset(lines, None)


def _load_cls_tiny(path: Union[str, Path], vocab_size: int, max_len: int, limit: Optional[int]) -> _IntSeqDataset:
    import csv
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '')
            label = int(row.get('label', 0))
            ids = _char_to_ids(text, vocab_size, max_len)
            texts.append(ids)
            labels.append(label)
            if limit and len(texts) >= limit:
                break
    return _IntSeqDataset(texts, labels)


def _load_seq2seq_tiny(path: Union[str, Path], vocab_size: int, max_len: int, limit: Optional[int]) -> TorchDataset:
    # Returns dict with input_ids, decoder_input_ids, labels
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = _char_to_ids(str(obj.get('input', '')), vocab_size, max_len)
            tgt = _char_to_ids(str(obj.get('target', '')), vocab_size, max_len)
            items.append({
                'input_ids': torch.tensor(src, dtype=torch.long),
                'decoder_input_ids': torch.tensor(tgt[:-1] + [0], dtype=torch.long),
                'labels': torch.tensor(tgt, dtype=torch.long),
            })
            if limit and len(items) >= limit:
                break

    class _Seq2SeqDataset(TorchDataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, idx):
            return items[idx]

    return _Seq2SeqDataset()


def _collate_lm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = (input_ids != 0).long()
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def _collate_cls(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = (input_ids != 0).long()
    labels = torch.stack([b['labels'] for b in batch]).long()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def _collate_seq2seq(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = (input_ids != 0).long()
    decoder_input_ids = torch.stack([b['decoder_input_ids'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }


def build_dataloader(task_spec, eval_config, training_config):
    """
    Build a task-aware DataLoader for eval.

    Loads tiny example datasets when dataset_id matches known presets.
    Falls back to synthetic mapping for unsupported cases.
    """
    base_dir = Path('examples/datasets')
    vocab_size = getattr(training_config, 'vocab_size', 256)
    max_len = getattr(eval_config, 'max_seq_length', getattr(training_config, 'max_seq_len', 128))
    limit = getattr(eval_config, 'max_eval_examples', None)
    batch_size = getattr(eval_config, 'batch_size', 4)

    # Vision classification branch (multimodal extension)
    modality = getattr(task_spec, "modality", "text")
    if modality == "vision" and getattr(task_spec, "task_type", None) == "vision_classification":
        image_size_value = task_spec.input_schema.get("image_size", [3, 64, 64])
        if not isinstance(image_size_value, (list, tuple)) or len(image_size_value) != 3:
            raise ValueError(f"Expected input_schema['image_size'] to be [C, H, W], got {image_size_value!r}")
        c, h, w = (int(image_size_value[0]), int(image_size_value[1]), int(image_size_value[2]))

        preprocessing = getattr(task_spec, "preprocessing_config", None) or {}
        data_dir = base_dir / "vision" / getattr(training_config, "task_name", task_spec.name)

        dataset = TinyVisionDataset(
            data_dir=data_dir,
            image_size=(c, h, w),
            normalize=bool(preprocessing.get("normalize", True)),
            mean=preprocessing.get("mean", [0.5, 0.5, 0.5]),
            std=preprocessing.get("std", [0.5, 0.5, 0.5]),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if task_spec.task_type == 'lm':
        path = base_dir / 'lm_tiny.txt'
        if path.exists():
            dataset = _load_lm_tiny(path, vocab_size, max_len, limit)
        else:
            # Synthetic fallback
            samples = [[(i + j) % vocab_size for j in range(max_len)] for i in range(limit or 16)]
            dataset = _IntSeqDataset(samples, None)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_lm)

    if task_spec.task_type == 'classification':
        path = base_dir / 'cls_tiny.csv'
        if path.exists():
            dataset = _load_cls_tiny(path, vocab_size, max_len, limit)
        else:
            texts = [[(i * 7 + j) % vocab_size for j in range(max_len)] for i in range(limit or 16)]
            labels = [i % int(task_spec.additional_config.get('num_classes', 2)) for i in range(len(texts))]
            dataset = _IntSeqDataset(texts, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_cls)

    if task_spec.task_type == 'seq2seq':
        path = base_dir / 'seq2seq_tiny.jsonl'
        if path.exists():
            dataset = _load_seq2seq_tiny(path, vocab_size, max_len, limit)
        else:
            # Minimal synthetic
            class _Tmp(TorchDataset):
                def __len__(self):
                    return limit or 8
                def __getitem__(self, idx):
                    src = torch.tensor([(idx + j) % vocab_size for j in range(max_len)], dtype=torch.long)
                    tgt = torch.tensor([(idx * 3 + j) % vocab_size for j in range(max_len)], dtype=torch.long)
                    return {
                        'input_ids': src,
                        'decoder_input_ids': torch.cat([tgt[:-1], torch.tensor([0])]),
                        'labels': tgt,
                    }
            dataset = _Tmp()
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_seq2seq)

    raise ValueError(f"Unsupported task type: {task_spec.task_type}")
