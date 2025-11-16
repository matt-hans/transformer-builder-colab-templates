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
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Literal
from datasets import Dataset, load_dataset, DatasetDict
from tqdm.auto import tqdm
import re


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

        try:
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                cache_dir=self.cache_dir
            )

            if not streaming:
                if isinstance(dataset, Dataset):
                    print(f"✓ Loaded {len(dataset):,} samples")
                else:
                    print(f"✓ Loaded dataset with splits: {list(dataset.keys())}")

            return dataset

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise

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

        # Validate format
        file_ext = Path(filename).suffix.lstrip('.').lower()
        if file_ext not in ['txt', 'json', 'csv', 'jsonl']:
            print(f"❌ Unsupported format: {file_ext}")
            print("   Supported: TXT, JSON, CSV, JSONL")
            return None

        # Load dataset
        loader = DatasetLoader(preprocessing=preprocessing)
        try:
            dataset = loader.load_local_file(
                filename,
                text_column=self.text_column
            )

            # Show preview
            if preview and len(dataset) > 0:
                loader.preview_samples(dataset, num_samples=2, text_column=self.text_column)
                loader.print_statistics(dataset, text_column=self.text_column)

            return dataset

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return None
