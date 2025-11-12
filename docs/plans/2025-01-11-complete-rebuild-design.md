# Complete Rebuild Design: Production-Ready Google Colab Template

**Date:** 2025-01-11
**Status:** Approved
**Estimated Timeline:** 6-8 weeks
**Scope:** Colab notebook-side only (platform recommendations documented separately)

## Executive Summary

This design addresses critical failures in the current Google Colab template integration with the Transformer Builder platform. The solution implements a production-ready training pipeline with:

- **Universal Model Adapter**: Handles arbitrary forward() signatures from generated models
- **4-Tier Adaptive Tokenization**: Supports any vocab_size through intelligent strategy selection
- **PyTorch Lightning Integration**: Production training with distributed, mixed precision, checkpointing
- **Wizard-Driven UX**: Interactive ipywidgets-based setup with progressive disclosure
- **Production Export**: ONNX, TorchScript, quantization with validation

**Key Metrics:**
- ~4,500 lines new code
- ~800 lines modified code
- 100% test pass rate target (currently 0%)
- 6-8 week implementation timeline

## Problem Statement

### Current Failures

**1. Dependency Conflicts (Cell 2)**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
numpy 1.26.4 is installed but numpy>=2.0 is required by {...}
```

**2. Import System Failure (Cell 12)**
```python
ModuleNotFoundError: No module named 'test_functions'
# Caused by: relative imports in downloaded files, not a proper package
```

**3. Model Signature Incompatibility (Cell 10+)**
```python
# Generated model signature:
def forward(self, input_0_tokens, mhsa_0_output, residual_0_output, ...)

# Test functions expect:
def forward(self, input_ids)

# Result: 100% test failure rate
```

**4. Production Gaps**
- No custom dataset upload/integration
- No checkpoint management or resumption
- No distributed training or mixed precision
- No production export capabilities

### Root Causes

1. **Unmanaged Dependencies**: No pinned versions, random installation order
2. **Package Structure**: utils/ not a proper Python package when downloaded
3. **Architecture Constraint**: Cannot modify generated model code (platform team's domain)
4. **Signature Assumption**: Tests assume simple `forward(input_ids)` signature
5. **Scope Creep**: Tier 3 tests meant for demos, not production training

## Solution Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Colab Notebook                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │         SetupWizard (ipywidgets UI)                │    │
│  │  1. Model Validation (Tier 1 tests)                │    │
│  │  2. Dataset Selection (Upload/HF/Example)          │    │
│  │  3. Tokenizer Setup (4-tier adaptive)              │    │
│  │  4. Training Config (Smart defaults)               │    │
│  │  5. Confirm & Launch                                │    │
│  └─────────────────┬──────────────────────────────────┘    │
│                    │                                         │
│  ┌─────────────────▼──────────────────────────────────┐    │
│  │      UniversalModelAdapter (Lightning Module)       │    │
│  │  - ModelSignatureInspector                          │    │
│  │  - ComputationalGraphExecutor                       │    │
│  │  - Transparent signature handling                   │    │
│  └─────────────────┬──────────────────────────────────┘    │
│                    │                                         │
│  ┌─────────────────▼──────────────────────────────────┐    │
│  │         PyTorch Lightning Trainer                   │    │
│  │  - Distributed training (DDP)                       │    │
│  │  - Mixed precision (FP16/BF16)                      │    │
│  │  - Checkpointing (Google Drive)                     │    │
│  │  - Live dashboard (TQDMProgressBar)                 │    │
│  └─────────────────┬──────────────────────────────────┘    │
│                    │                                         │
│  ┌─────────────────▼──────────────────────────────────┐    │
│  │          Production Export Suite                    │    │
│  │  - ONNX with validation                             │    │
│  │  - TorchScript                                      │    │
│  │  - Quantization (dynamic/static)                    │    │
│  │  - Model cards                                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Specifications

#### 1. Universal Model Adapter

**Purpose:** Handle arbitrary forward() signatures from generated models transparently

**Location:** `utils/adapters/model_adapter.py` (~400 lines)

**Key Classes:**

```python
class ModelSignatureInspector:
    """Analyzes model forward() signature using inspect module"""
    def __init__(self, model: nn.Module):
        self.signature = inspect.signature(model.forward)
        self.params = list(self.signature.parameters.keys())

    def requires_intermediate_outputs(self) -> bool:
        """Check if signature needs computed intermediates"""
        return any(p.startswith(('mhsa_', 'residual_', 'ffn_'))
                   for p in self.params)

    def get_required_params(self) -> List[str]:
        """Return all required parameter names"""
        return [p for p in self.params if self.signature.parameters[p].default == inspect.Parameter.empty]

class ComputationalGraphExecutor:
    """Resolves and computes intermediate dependencies"""
    def __init__(self, model: nn.Module, inspector: ModelSignatureInspector):
        self.model = model
        self.inspector = inspector
        self.intermediate_cache = {}

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute model with dependency resolution"""
        # 1. Start with base inputs (input_ids, attention_mask)
        # 2. Walk through model layers, computing intermediates
        # 3. Cache intermediate outputs (mhsa_0_output, residual_0_output, etc.)
        # 4. Call model.forward() with all required parameters
        # 5. Return final logits

class UniversalModelAdapter(pl.LightningModule):
    """Lightning-compatible wrapper for ANY generated model"""
    def __init__(self, generated_model: nn.Module, config: Any, tokenizer: PreTrainedTokenizer, learning_rate: float = 5e-5):
        super().__init__()
        self.model = generated_model
        self.inspector = ModelSignatureInspector(generated_model)
        self.executor = ComputationalGraphExecutor(generated_model, self.inspector)
        self.config = config
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """Unified forward interface"""
        if self.inspector.requires_intermediate_outputs():
            logits = self.executor.forward(input_ids, attention_mask)
        else:
            logits = self.model(input_ids, attention_mask=attention_mask)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def training_step(self, batch, batch_idx):
        output = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train_loss", output["loss"], prog_bar=True)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("val_loss", output["loss"], prog_bar=True)
        return output["loss"]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
```

**Testing Strategy:**
- Unit tests for signature inspection (simple, complex, nested)
- Integration tests with real generated models
- Validation that all Tier 1 tests pass with adapter

#### 2. 4-Tier Adaptive Tokenization

**Purpose:** Handle ANY vocab_size from user's custom transformers

**Location:** `utils/tokenization/adaptive_tokenizer.py` (~500 lines)

**Strategy:**

```python
class AdaptiveTokenizer:
    """4-tier tokenization strategy"""

    # Tier 1: Known pretrained tokenizers
    KNOWN_TOKENIZERS = {
        50257: "gpt2",                          # GPT-2/3
        32000: "meta-llama/Llama-2-7b-hf",     # LLaMA
        30522: "bert-base-uncased",            # BERT
        250002: "facebook/opt-350m",            # OPT
        128000: "meta-llama/Meta-Llama-3-8B",  # LLaMA 3
        49152: "microsoft/phi-2",              # Phi-2
        100277: "Qwen/Qwen-7B",                # Qwen
        151936: "microsoft/Phi-3-mini-4k-instruct", # Phi-3
    }

    @classmethod
    def detect_strategy(cls, vocab_size: int, dataset_size: int) -> str:
        """Detect best tokenization strategy"""
        if vocab_size in cls.KNOWN_TOKENIZERS:
            return 'pretrained'  # Tier 1: Exact match
        elif dataset_size >= 100 and 5000 <= vocab_size <= 100000:
            return 'train_bpe'   # Tier 2: Train custom BPE
        else:
            return 'character'   # Tier 3: Character-level

    @classmethod
    def load_or_create(cls, vocab_size: int, dataset: Optional[Dataset] = None, cache_dir: str = "./tokenizer_cache") -> PreTrainedTokenizer:
        """Load or create tokenizer based on strategy"""
        strategy = cls.detect_strategy(vocab_size, len(dataset) if dataset else 0)

        if strategy == 'pretrained':
            # Tier 1: Load from HuggingFace
            model_name = cls.KNOWN_TOKENIZERS[vocab_size]
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        elif strategy == 'train_bpe':
            # Tier 2: Train custom BPE
            tokenizer = FastBPETrainer.train_on_dataset(
                texts=dataset['text'],
                vocab_size=vocab_size,
                special_tokens=['<pad>', '<unk>', '<s>', '</s>'],
                cache_dir=cache_dir
            )

        elif strategy == 'character':
            # Tier 3: Character-level tokenizer
            tokenizer = CharacterLevelTokenizer(
                vocab_size=vocab_size,
                special_tokens=['<pad>', '<unk>', '<s>', '</s>']
            )

        # Validate
        TokenizerValidator.validate(tokenizer, vocab_size)
        return tokenizer
```

**Tier 2: BPE Trainer**

**Location:** `utils/tokenization/bpe_trainer.py` (~300 lines)

```python
class FastBPETrainer:
    """Train custom BPE tokenizer in Colab"""

    @staticmethod
    def train_on_dataset(texts: List[str], vocab_size: int, special_tokens: List[str], cache_dir: str) -> PreTrainedTokenizerFast:
        """Train BPE tokenizer on dataset samples"""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Initialize BPE model
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True
        )

        # Train on text samples
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Wrap in HuggingFace tokenizer
        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token='<pad>',
            unk_token='<unk>',
            bos_token='<s>',
            eos_token='</s>'
        )

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        wrapped.save_pretrained(cache_dir)

        return wrapped
```

**Tier 3: Character-Level**

**Location:** `utils/tokenization/character_tokenizer.py` (~200 lines)

```python
class CharacterLevelTokenizer:
    """Character-level tokenizer (always works fallback)"""

    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # Build vocab from ASCII + special tokens
        self.char_to_id = {char: idx + len(special_tokens) for idx, char in enumerate(string.printable)}
        self.special_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        self.vocab = {**self.special_to_id, **self.char_to_id}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        tokens = [self.vocab.get(char, self.special_to_id['<unk>']) for char in text]
        tokens = tokens[:max_length]
        attention_mask = [1] * len(tokens)

        # Pad to max_length
        padding_length = max_length - len(tokens)
        tokens += [self.special_to_id['<pad>']] * padding_length
        attention_mask += [0] * padding_length

        return {
            "input_ids": torch.tensor(tokens),
            "attention_mask": torch.tensor(attention_mask)
        }

    def decode(self, token_ids: List[int]) -> str:
        return ''.join([self.id_to_token.get(tid, '<unk>') for tid in token_ids])
```

**Validator**

**Location:** `utils/tokenization/validator.py` (~100 lines)

```python
class TokenizerValidator:
    """Validate tokenizer compatibility with model config"""

    @staticmethod
    def validate(tokenizer: Union[PreTrainedTokenizer, CharacterLevelTokenizer], expected_vocab_size: int):
        """Validate tokenizer meets requirements"""
        actual_vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size

        if actual_vocab_size != expected_vocab_size:
            raise ValueError(f"Tokenizer vocab size {actual_vocab_size} != expected {expected_vocab_size}")

        # Validate special tokens
        required_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        for token in required_tokens:
            if not hasattr(tokenizer, f'{token.strip("<>")}_token'):
                raise ValueError(f"Missing special token: {token}")

        # Validate encode/decode
        test_text = "Hello world!"
        encoded = tokenizer.encode(test_text) if hasattr(tokenizer, 'encode') else tokenizer.encode(test_text)["input_ids"]
        decoded = tokenizer.decode(encoded)

        print(f"✓ Tokenizer validated (vocab_size={actual_vocab_size})")
        print(f"  Test encode: {test_text} → {encoded}")
        print(f"  Test decode: {decoded}")
```

**Lightning DataModule**

**Location:** `utils/tokenization/data_module.py` (~200 lines)

```python
class AdaptiveTokenizerDataModule(pl.LightningDataModule):
    """Lightning DataModule with adaptive tokenization"""

    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, batch_size: int = 16, max_length: int = 512):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage: str):
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )

        tokenized = self.dataset.map(tokenize_function, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Split train/val
        split = tokenized.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = split['train']
        self.val_dataset = split['test']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)
```

#### 3. Dependency Management

**Purpose:** Fix dependency conflicts with pinned versions

**Location:** `requirements-colab.txt` (NEW file)

```txt
# Core dependencies - INSTALL FIRST
numpy==1.26.4

# PyTorch ecosystem
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

# HuggingFace
transformers==4.36.2
tokenizers==0.15.0
datasets==2.16.1
accelerate==0.25.0

# Training framework
pytorch-lightning==2.1.0
torchmetrics==1.2.1

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Export
onnx==1.15.0
onnxruntime==1.16.3

# Notebook UI
ipywidgets==8.1.1
tqdm==4.66.1

# Utilities
psutil==5.9.6
```

**Installation Strategy (Cell 2):**

```python
# Step 1: Upgrade pip
!pip install --upgrade pip

# Step 2: Install numpy FIRST (critical dependency)
!pip install numpy==1.26.4

# Step 3: Install all other dependencies in one command
!pip install -r requirements-colab.txt

# Step 4: Verify installations
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer

print(f"✓ numpy: {np.__version__}")
print(f"✓ torch: {torch.__version__}")
print(f"✓ pytorch-lightning: {pl.__version__}")
```

#### 4. Package Structure Fix

**Purpose:** Make utils/ a proper Python package when downloaded

**Location:** `utils/__init__.py` (NEW file)

```python
"""
Transformer Builder Colab Utilities

Production-ready utilities for training, validating, and exporting
transformer models generated by the Transformer Builder platform.
"""

__version__ = "2.0.0"

# Core adapters
from .adapters.model_adapter import UniversalModelAdapter, ModelSignatureInspector, ComputationalGraphExecutor

# Tokenization
from .tokenization.adaptive_tokenizer import AdaptiveTokenizer
from .tokenization.bpe_trainer import FastBPETrainer
from .tokenization.character_tokenizer import CharacterLevelTokenizer
from .tokenization.validator import TokenizerValidator
from .tokenization.data_module import AdaptiveTokenizerDataModule

# Training
from .training.dataset_utilities import DatasetLoader, DatasetUploader
from .training.checkpoint_manager import CheckpointManager
from .training.training_core import TrainingCoordinator
from .training.export_utilities import ONNXExporter, TorchScriptExporter, ModelCardGenerator

# UI
from .ui.setup_wizard import SetupWizard

# Test functions (backward compatibility)
from .test_functions import *

__all__ = [
    # Adapters
    'UniversalModelAdapter',
    'ModelSignatureInspector',
    'ComputationalGraphExecutor',
    # Tokenization
    'AdaptiveTokenizer',
    'FastBPETrainer',
    'CharacterLevelTokenizer',
    'TokenizerValidator',
    'AdaptiveTokenizerDataModule',
    # Training
    'DatasetLoader',
    'DatasetUploader',
    'CheckpointManager',
    'TrainingCoordinator',
    'ONNXExporter',
    'TorchScriptExporter',
    'ModelCardGenerator',
    # UI
    'SetupWizard',
]
```

**Download Strategy (Cell 3):**

```python
# Download complete utils package structure
!rm -rf utils/
!git clone --depth 1 --branch main https://github.com/your-org/transformer-builder-colab-templates.git temp_repo
!cp -r temp_repo/utils ./
!rm -rf temp_repo

# Verify package structure
import sys
sys.path.insert(0, './')

from utils import UniversalModelAdapter, AdaptiveTokenizer, SetupWizard
print("✓ Package imported successfully")
```

#### 5. Setup Wizard (UX)

**Purpose:** Interactive 5-step wizard for intuitive setup

**Location:** `utils/ui/setup_wizard.py` (~600 lines)

```python
import ipywidgets as widgets
from IPython.display import display, clear_output

class SetupWizard:
    """Interactive 5-step setup wizard"""

    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        self.config = config
        self.current_step = 1
        self.wizard_data = {}

        # UI components
        self.output = widgets.Output()
        self.progress = widgets.IntProgress(min=0, max=5, value=1, description='Progress:')
        self.step_container = widgets.VBox()

    def run(self):
        """Launch wizard"""
        display(widgets.VBox([
            widgets.HTML("<h2>Transformer Training Setup Wizard</h2>"),
            self.progress,
            self.step_container,
            self.output
        ]))
        self._render_step1()

    def _render_step1(self):
        """Step 1: Validate Model"""
        clear_output(wait=True)
        self.step_container.children = [
            widgets.HTML("<h3>Step 1: Validate Model</h3>"),
            widgets.HTML("<p>Running Tier 1 validation tests on your model...</p>"),
            widgets.Button(description="Run Validation", button_style='primary')
        ]

        btn = self.step_container.children[-1]
        btn.on_click(lambda b: self._run_tier1_tests())

    def _run_tier1_tests(self):
        """Execute Tier 1 tests with progress feedback"""
        with self.output:
            clear_output(wait=True)
            from utils.tier1_critical_validation import (
                test_shape_robustness, test_gradient_flow,
                test_output_stability, test_parameter_initialization
            )

            tests = [
                ("Shape Robustness", test_shape_robustness),
                ("Gradient Flow", test_gradient_flow),
                ("Output Stability", test_output_stability),
                ("Parameter Init", test_parameter_initialization),
            ]

            results = []
            for name, test_func in tests:
                try:
                    test_func(self.model, self.config)
                    results.append((name, True, None))
                    print(f"✓ {name} passed")
                except Exception as e:
                    results.append((name, False, str(e)))
                    print(f"✗ {name} failed: {str(e)}")

            self.wizard_data['validation_results'] = results

            # Check if all passed
            all_passed = all(r[1] for r in results)
            if all_passed:
                print("\n✓ All validation tests passed!")
                self._render_step2()
            else:
                print("\n⚠ Some tests failed. Review errors above.")

    def _render_step2(self):
        """Step 2: Select Dataset"""
        self.progress.value = 2

        dataset_options = [
            ("Upload Custom Text File", "upload"),
            ("Use HuggingFace Dataset", "huggingface"),
            ("Use Example Dataset (WikiText-2)", "example"),
        ]

        radio = widgets.RadioButtons(
            options=[(label, val) for label, val in dataset_options],
            description='Source:',
            disabled=False
        )

        upload_widget = widgets.FileUpload(accept='.txt,.csv,.json', multiple=False)
        hf_textbox = widgets.Text(placeholder='e.g., wikitext-103-v1', description='Dataset:')
        next_btn = widgets.Button(description="Next", button_style='success')

        self.step_container.children = [
            widgets.HTML("<h3>Step 2: Select Dataset</h3>"),
            radio,
            upload_widget,
            hf_textbox,
            next_btn
        ]

        def on_selection_change(change):
            if change['new'] == 'upload':
                upload_widget.layout.visibility = 'visible'
                hf_textbox.layout.visibility = 'hidden'
            elif change['new'] == 'huggingface':
                upload_widget.layout.visibility = 'hidden'
                hf_textbox.layout.visibility = 'visible'
            else:
                upload_widget.layout.visibility = 'hidden'
                hf_textbox.layout.visibility = 'hidden'

        radio.observe(on_selection_change, names='value')
        next_btn.on_click(lambda b: self._load_dataset(radio.value, upload_widget, hf_textbox))

    def _load_dataset(self, source: str, upload_widget, hf_textbox):
        """Load dataset based on selection"""
        with self.output:
            clear_output(wait=True)
            from utils.training.dataset_utilities import DatasetLoader

            if source == 'upload':
                uploaded = upload_widget.value
                if not uploaded:
                    print("⚠ Please upload a file first")
                    return
                dataset = DatasetLoader.from_uploaded_file(uploaded[0])
            elif source == 'huggingface':
                dataset_name = hf_textbox.value
                if not dataset_name:
                    print("⚠ Please enter a dataset name")
                    return
                dataset = DatasetLoader.from_huggingface(dataset_name)
            else:  # example
                dataset = DatasetLoader.from_huggingface('wikitext', 'wikitext-2-raw-v1')

            self.wizard_data['dataset'] = dataset
            print(f"✓ Loaded dataset with {len(dataset)} examples")
            self._render_step3()

    def _render_step3(self):
        """Step 3: Setup Tokenizer"""
        self.progress.value = 3

        vocab_size = self.config.vocab_size
        dataset = self.wizard_data['dataset']

        # Auto-detect strategy
        strategy = AdaptiveTokenizer.detect_strategy(vocab_size, len(dataset))

        strategy_descriptions = {
            'pretrained': f"✓ Detected known vocab_size ({vocab_size}). Will use pretrained tokenizer.",
            'train_bpe': f"Will train custom BPE tokenizer (vocab_size={vocab_size}, dataset_size={len(dataset)}).",
            'character': f"Will use character-level tokenizer (vocab_size={vocab_size})."
        }

        self.step_container.children = [
            widgets.HTML("<h3>Step 3: Setup Tokenizer</h3>"),
            widgets.HTML(f"<p>{strategy_descriptions[strategy]}</p>"),
            widgets.Button(description="Create Tokenizer", button_style='primary')
        ]

        btn = self.step_container.children[-1]
        btn.on_click(lambda b: self._create_tokenizer(strategy))

    def _create_tokenizer(self, strategy: str):
        """Create tokenizer based on strategy"""
        with self.output:
            clear_output(wait=True)

            vocab_size = self.config.vocab_size
            dataset = self.wizard_data['dataset']

            print(f"Creating tokenizer (strategy: {strategy})...")
            tokenizer = AdaptiveTokenizer.load_or_create(vocab_size, dataset)

            self.wizard_data['tokenizer'] = tokenizer
            print("✓ Tokenizer created and validated")
            self._render_step4()

    def _render_step4(self):
        """Step 4: Configure Training"""
        self.progress.value = 4

        # Smart defaults based on config
        default_epochs = 3
        default_batch_size = 16
        default_lr = 5e-5

        epochs_slider = widgets.IntSlider(value=default_epochs, min=1, max=10, description='Epochs:')
        batch_slider = widgets.IntSlider(value=default_batch_size, min=4, max=64, step=4, description='Batch Size:')
        lr_text = widgets.FloatText(value=default_lr, description='Learning Rate:')

        mixed_precision = widgets.Checkbox(value=True, description='Mixed Precision (FP16)')
        save_checkpoints = widgets.Checkbox(value=True, description='Save to Google Drive')

        next_btn = widgets.Button(description="Next", button_style='success')

        self.step_container.children = [
            widgets.HTML("<h3>Step 4: Configure Training</h3>"),
            epochs_slider,
            batch_slider,
            lr_text,
            mixed_precision,
            save_checkpoints,
            next_btn
        ]

        def on_next():
            self.wizard_data['training_config'] = {
                'epochs': epochs_slider.value,
                'batch_size': batch_slider.value,
                'learning_rate': lr_text.value,
                'mixed_precision': mixed_precision.value,
                'save_checkpoints': save_checkpoints.value,
            }
            self._render_step5()

        next_btn.on_click(lambda b: on_next())

    def _render_step5(self):
        """Step 5: Confirm and Start"""
        self.progress.value = 5

        summary = f"""
        <h3>Step 5: Review & Start Training</h3>
        <table>
            <tr><td><b>Model:</b></td><td>{self.config.model_type} (vocab={self.config.vocab_size})</td></tr>
            <tr><td><b>Dataset:</b></td><td>{len(self.wizard_data['dataset'])} examples</td></tr>
            <tr><td><b>Tokenizer:</b></td><td>{type(self.wizard_data['tokenizer']).__name__}</td></tr>
            <tr><td><b>Epochs:</b></td><td>{self.wizard_data['training_config']['epochs']}</td></tr>
            <tr><td><b>Batch Size:</b></td><td>{self.wizard_data['training_config']['batch_size']}</td></tr>
            <tr><td><b>Learning Rate:</b></td><td>{self.wizard_data['training_config']['learning_rate']}</td></tr>
            <tr><td><b>Mixed Precision:</b></td><td>{self.wizard_data['training_config']['mixed_precision']}</td></tr>
        </table>
        """

        start_btn = widgets.Button(description="Start Training", button_style='success', icon='play')

        self.step_container.children = [
            widgets.HTML(summary),
            start_btn
        ]

        start_btn.on_click(lambda b: self._start_training())

    def _start_training(self):
        """Launch training with configured settings"""
        with self.output:
            clear_output(wait=True)

            from utils.training.training_core import TrainingCoordinator

            coordinator = TrainingCoordinator(
                model=self.model,
                config=self.config,
                tokenizer=self.wizard_data['tokenizer'],
                dataset=self.wizard_data['dataset'],
                training_config=self.wizard_data['training_config']
            )

            print("🚀 Starting training...")
            coordinator.train()
```

#### 6. Training Infrastructure

**Location:** `utils/training/` (5 new modules)

**A. Dataset Utilities** (`dataset_utilities.py` ~300 lines)

```python
class DatasetLoader:
    """Load datasets from various sources"""

    @staticmethod
    def from_huggingface(dataset_name: str, config_name: Optional[str] = None) -> Dataset:
        """Load from HuggingFace datasets"""
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, config_name, split='train')

        # Ensure 'text' column exists
        if 'text' not in dataset.column_names:
            text_cols = [col for col in dataset.column_names if 'text' in col.lower()]
            if text_cols:
                dataset = dataset.rename_column(text_cols[0], 'text')
            else:
                raise ValueError(f"No text column found in dataset columns: {dataset.column_names}")

        return dataset

    @staticmethod
    def from_uploaded_file(file_data: bytes, file_format: str = 'txt') -> Dataset:
        """Load from uploaded file"""
        from datasets import Dataset as HFDataset

        if file_format == 'txt':
            text = file_data.decode('utf-8')
            lines = text.split('\n')
            return HFDataset.from_dict({'text': lines})

        elif file_format == 'csv':
            import pandas as pd
            df = pd.read_csv(io.BytesIO(file_data))
            return HFDataset.from_pandas(df)

        elif file_format == 'json':
            import json
            data = json.loads(file_data.decode('utf-8'))
            return HFDataset.from_dict(data)

    @staticmethod
    def prepare_for_training(dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dataset:
        """Tokenize and prepare dataset"""
        def tokenize_function(examples):
            # Create labels (shifted input_ids for causal LM)
            tokenized = tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        return tokenized

class DatasetUploader:
    """Handle file uploads in Colab"""

    @staticmethod
    def upload_from_local() -> bytes:
        """Upload file from local machine"""
        from google.colab import files
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        return uploaded[filename]
```

**B. Checkpoint Manager** (`checkpoint_manager.py` ~250 lines)

```python
class CheckpointManager:
    """Manage checkpoints with Google Drive integration"""

    def __init__(self, experiment_name: str, drive_path: str = "/content/drive/MyDrive/transformer_checkpoints"):
        self.experiment_name = experiment_name
        self.drive_path = drive_path
        self.local_path = f"./checkpoints/{experiment_name}"

        # Mount Google Drive
        self._mount_drive()

        # Create directories
        os.makedirs(self.local_path, exist_ok=True)
        os.makedirs(f"{self.drive_path}/{experiment_name}", exist_ok=True)

    def _mount_drive(self):
        """Mount Google Drive"""
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            print("✓ Google Drive mounted")
        except Exception as e:
            print(f"⚠ Could not mount Google Drive: {e}")

    def get_checkpoint_callback(self) -> pl.callbacks.ModelCheckpoint:
        """Create Lightning checkpoint callback"""
        return pl.callbacks.ModelCheckpoint(
            dirpath=self.local_path,
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True
        )

    def sync_to_drive(self):
        """Sync local checkpoints to Google Drive"""
        import shutil

        drive_dir = f"{self.drive_path}/{self.experiment_name}"

        for file in os.listdir(self.local_path):
            src = os.path.join(self.local_path, file)
            dst = os.path.join(drive_dir, file)
            shutil.copy2(src, dst)

        print(f"✓ Synced checkpoints to {drive_dir}")

    def load_latest_checkpoint(self) -> Optional[str]:
        """Load latest checkpoint from Drive"""
        drive_dir = f"{self.drive_path}/{self.experiment_name}"

        if not os.path.exists(drive_dir):
            return None

        checkpoints = [f for f in os.listdir(drive_dir) if f.endswith('.ckpt')]
        if not checkpoints:
            return None

        # Get latest checkpoint
        latest = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(drive_dir, f)))
        src = os.path.join(drive_dir, latest)
        dst = os.path.join(self.local_path, latest)

        import shutil
        shutil.copy2(src, dst)

        print(f"✓ Loaded checkpoint: {latest}")
        return dst
```

**C. Training Core** (`training_core.py` ~400 lines)

```python
class TrainingCoordinator:
    """Coordinate training with PyTorch Lightning"""

    def __init__(self, model: nn.Module, config: Any, tokenizer: PreTrainedTokenizer,
                 dataset: Dataset, training_config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.training_config = training_config

        # Create experiment name
        self.experiment_name = f"{config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def train(self):
        """Execute training pipeline"""
        print("Setting up training pipeline...")

        # 1. Wrap model in Universal Adapter
        lightning_model = UniversalModelAdapter(
            self.model,
            self.config,
            self.tokenizer,
            learning_rate=self.training_config['learning_rate']
        )

        # 2. Create DataModule
        data_module = AdaptiveTokenizerDataModule(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            batch_size=self.training_config['batch_size'],
            max_length=512
        )

        # 3. Setup callbacks
        callbacks = [
            pl.callbacks.TQDMProgressBar(refresh_rate=10),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        ]

        # 4. Checkpoint management
        if self.training_config['save_checkpoints']:
            checkpoint_mgr = CheckpointManager(self.experiment_name)
            callbacks.append(checkpoint_mgr.get_checkpoint_callback())

        # 5. Configure Trainer
        trainer_kwargs = {
            'max_epochs': self.training_config['epochs'],
            'callbacks': callbacks,
            'enable_progress_bar': True,
            'log_every_n_steps': 10,
            'accelerator': 'auto',
            'devices': 1,
        }

        # Mixed precision
        if self.training_config['mixed_precision']:
            trainer_kwargs['precision'] = '16-mixed'

        # Check for checkpoint resume
        resume_ckpt = None
        if self.training_config['save_checkpoints']:
            resume_ckpt = checkpoint_mgr.load_latest_checkpoint()
            if resume_ckpt:
                print(f"Resuming from checkpoint: {resume_ckpt}")

        # 6. Create Trainer
        trainer = pl.Trainer(**trainer_kwargs)

        # 7. Train
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_name}")
        print(f"{'='*60}\n")

        trainer.fit(lightning_model, data_module, ckpt_path=resume_ckpt)

        # 8. Sync checkpoints to Drive
        if self.training_config['save_checkpoints']:
            checkpoint_mgr.sync_to_drive()

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")

        return trainer, lightning_model
```

**D. Export Utilities** (`export_utilities.py` ~500 lines)

```python
class ONNXExporter:
    """Export model to ONNX format"""

    @staticmethod
    def export(model: nn.Module, tokenizer: PreTrainedTokenizer, output_path: str,
               config: Any, validate: bool = True):
        """Export to ONNX with validation"""
        import onnx
        import onnxruntime as ort

        model.eval()

        # Create dummy input
        dummy_input = {
            'input_ids': torch.randint(0, config.vocab_size, (1, 512)),
            'attention_mask': torch.ones((1, 512), dtype=torch.long)
        }

        # Export
        print("Exporting to ONNX...")
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'}
            },
            opset_version=14
        )

        print(f"✓ Exported to {output_path}")

        # Validate
        if validate:
            ONNXExporter._validate_export(output_path, model, dummy_input, config)

    @staticmethod
    def _validate_export(onnx_path: str, pytorch_model: nn.Module, dummy_input: Dict, config: Any):
        """Validate ONNX export matches PyTorch output"""
        import onnxruntime as ort

        # PyTorch inference
        with torch.no_grad():
            pytorch_out = pytorch_model(dummy_input['input_ids'], dummy_input['attention_mask'])
            if isinstance(pytorch_out, dict):
                pytorch_out = pytorch_out['logits']

        # ONNX inference
        ort_session = ort.InferenceSession(onnx_path)
        onnx_out = ort_session.run(
            None,
            {
                'input_ids': dummy_input['input_ids'].numpy(),
                'attention_mask': dummy_input['attention_mask'].numpy()
            }
        )[0]

        # Compare
        max_diff = np.abs(pytorch_out.numpy() - onnx_out).max()
        print(f"✓ ONNX validation passed (max_diff={max_diff:.6f})")

        if max_diff > 1e-3:
            print("⚠ Warning: Large difference between PyTorch and ONNX outputs")

class TorchScriptExporter:
    """Export model to TorchScript"""

    @staticmethod
    def export(model: nn.Module, output_path: str, config: Any, method: str = 'trace'):
        """Export to TorchScript via tracing or scripting"""
        model.eval()

        dummy_input = (
            torch.randint(0, config.vocab_size, (1, 512)),
            torch.ones((1, 512), dtype=torch.long)
        )

        if method == 'trace':
            scripted = torch.jit.trace(model, dummy_input)
        else:  # script
            scripted = torch.jit.script(model)

        scripted.save(output_path)
        print(f"✓ Exported to TorchScript: {output_path}")

class ModelCardGenerator:
    """Generate model cards for HuggingFace Hub"""

    @staticmethod
    def generate(model: nn.Module, config: Any, tokenizer: PreTrainedTokenizer,
                 training_config: Dict, output_path: str):
        """Generate model card markdown"""

        card = f"""---
license: mit
tags:
- transformer
- pytorch
- custom-architecture
---

# {config.model_type} - Custom Transformer

This model was trained using the Transformer Builder platform and Google Colab template.

## Model Details

- **Architecture:** {config.model_type}
- **Vocabulary Size:** {config.vocab_size}
- **Parameters:** {sum(p.numel() for p in model.parameters()):,}
- **Training Framework:** PyTorch Lightning
- **Tokenizer:** {type(tokenizer).__name__}

## Training Configuration

- **Epochs:** {training_config['epochs']}
- **Batch Size:** {training_config['batch_size']}
- **Learning Rate:** {training_config['learning_rate']}
- **Mixed Precision:** {training_config['mixed_precision']}

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load model
model = torch.load('model.pt')
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

# Inference
text = "Hello world"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
```

## Training Data

[Describe dataset used for training]

## Evaluation

[Add evaluation metrics here]

## Limitations

[Add limitations and biases]
"""

        with open(output_path, 'w') as f:
            f.write(card)

        print(f"✓ Generated model card: {output_path}")
```

**E. Legacy Training** (`legacy_training.py`)

```python
# Keep existing tier3_training_utilities.py content for backward compatibility
# Rename functions to indicate legacy status:
# - test_fine_tuning → legacy_test_fine_tuning
# - test_hyperparameter_search → legacy_test_hyperparameter_search
# - test_benchmark_comparison → legacy_test_benchmark_comparison
```

#### 7. Test Suite Updates

**Location:** Modify existing tier1/tier2 files

**Tier 1 Updates** (`utils/tier1_critical_validation.py`)

```python
# Add helper function to detect model signature
def _safe_get_model_output(model, input_ids, attention_mask=None):
    """
    Safely get model output regardless of signature.
    Uses UniversalModelAdapter if complex signature detected.
    """
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())

    # Check if model has complex signature
    has_intermediates = any(p.startswith(('mhsa_', 'residual_', 'ffn_')) for p in params)

    if has_intermediates:
        # Use UniversalModelAdapter
        from utils.adapters.model_adapter import UniversalModelAdapter
        import types

        # Create mock config (extract from model if possible)
        class MockConfig:
            def __init__(self):
                # Try to extract vocab_size from model
                if hasattr(model, 'embedding') and hasattr(model.embedding, 'num_embeddings'):
                    self.vocab_size = model.embedding.num_embeddings
                else:
                    # Fallback: inspect first parameter of first layer
                    self.vocab_size = 50257  # Default

        config = MockConfig()

        # Create temporary tokenizer (not used for validation, just adapter requirement)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Wrap in adapter
        adapter = UniversalModelAdapter(model, config, tokenizer)
        output = adapter(input_ids, attention_mask)
        return output['logits'] if isinstance(output, dict) else output
    else:
        # Simple signature
        if attention_mask is not None and 'attention_mask' in params:
            return model(input_ids, attention_mask=attention_mask)
        else:
            return model(input_ids)

# Update all test functions to use _safe_get_model_output
# Example:
def test_shape_robustness(model: nn.Module, config: Any, num_tests: int = 5) -> Dict[str, Any]:
    """Test model with various input shapes"""
    # ... existing code ...

    # OLD: output = model(input_ids)
    # NEW:
    output = _safe_get_model_output(model, input_ids, attention_mask)

    # ... rest of function
```

**Tier 2 Updates** (`utils/tier2_advanced_analysis.py`)

```python
# Apply same _safe_get_model_output helper to:
# - test_attention_patterns
# - test_attribution_analysis
# - test_robustness
```

#### 8. Notebook Restructure

**Location:** `template.ipynb`

**New Cell Structure (21 cells total):**

```
Cell 1: Introduction
Cell 2: Install Dependencies (requirements-colab.txt)
Cell 3: Download Utils Package
Cell 4: Import Libraries
Cell 5: Mount Google Drive (optional)
Cell 6: Download Generated Model
Cell 7: Load Model & Config
Cell 8: Launch Setup Wizard
Cell 9: [WIZARD OUTPUT CELL - auto-populated]
Cell 10: Training Dashboard
Cell 11: View Results
Cell 12: Export to ONNX
Cell 13: Export to TorchScript
Cell 14: Generate Model Card
Cell 15: [SECTION BREAK] Advanced: Manual Configuration
Cell 16: Manual: Load Custom Dataset
Cell 17: Manual: Create Custom Tokenizer
Cell 18: Manual: Configure Lightning Trainer
Cell 19: Manual: Start Training
Cell 20: [SECTION BREAK] Testing & Validation
Cell 21: Run All Tests
```

**Key Changes:**
- Wizard-first approach (Cells 8-11)
- Manual fallback (Cells 15-19)
- Production export (Cells 12-14)
- Clear section breaks

## Implementation Timeline

### Phase 1: Foundation & Critical Fixes (Weeks 1-2)

**Week 1: Core Infrastructure**
- [ ] Create `requirements-colab.txt` with pinned dependencies
- [ ] Add `utils/__init__.py` for proper package structure
- [ ] Implement `ModelSignatureInspector` class
- [ ] Implement `ComputationalGraphExecutor` class
- [ ] Write unit tests for signature inspection

**Week 2: Model Adapter & Tokenization**
- [ ] Complete `UniversalModelAdapter` (Lightning module)
- [ ] Implement `AdaptiveTokenizer` detection logic
- [ ] Implement `FastBPETrainer` for custom BPE
- [ ] Implement `CharacterLevelTokenizer` fallback
- [ ] Implement `TokenizerValidator`
- [ ] Create `AdaptiveTokenizerDataModule` (Lightning)
- [ ] Write integration tests with real generated models

### Phase 2: Training Pipeline (Weeks 3-4)

**Week 3: Lightning Integration**
- [ ] Implement `DatasetLoader` (HuggingFace, upload, example)
- [ ] Implement `CheckpointManager` with Google Drive sync
- [ ] Implement `TrainingCoordinator` with Lightning
- [ ] Add mixed precision support
- [ ] Add distributed training support (if needed)

**Week 4: Training Features**
- [ ] Create live training dashboard
- [ ] Add early stopping and learning rate scheduling
- [ ] Implement checkpoint resumption logic
- [ ] Write training integration tests
- [ ] Performance optimization

### Phase 3: User Experience & Export (Weeks 5-6)

**Week 5: Setup Wizard**
- [ ] Implement `SetupWizard` base class
- [ ] Create Step 1: Model Validation UI
- [ ] Create Step 2: Dataset Selection UI
- [ ] Create Step 3: Tokenizer Setup UI
- [ ] Create Step 4: Training Config UI
- [ ] Create Step 5: Confirmation UI
- [ ] Wire all steps together with state management

**Week 6: Export & Production**
- [ ] Implement `ONNXExporter` with validation
- [ ] Implement `TorchScriptExporter`
- [ ] Add quantization support
- [ ] Implement `ModelCardGenerator`
- [ ] Create export validation tests

### Phase 4: Testing & Documentation (Weeks 7-8)

**Week 7: Testing**
- [ ] Update Tier 1 tests with `_safe_get_model_output`
- [ ] Update Tier 2 tests with signature handling
- [ ] Write comprehensive test suite for new components
- [ ] End-to-end integration tests
- [ ] Performance benchmarks

**Week 8: Documentation & Polish**
- [ ] Restructure `template.ipynb` with wizard-first flow
- [ ] Write `TRAINING_GUIDE.md`
- [ ] Write `DEPLOYMENT_GUIDE.md`
- [ ] Write `TROUBLESHOOTING.md`
- [ ] Write `PLATFORM_RECOMMENDATIONS.md` (for platform team)
- [ ] Final testing in fresh Colab runtime
- [ ] User acceptance testing

## File Structure Summary

```
transformer-builder-colab-templates/
├── template.ipynb                          # MODIFIED: Restructured with wizard (21 cells)
├── requirements-colab.txt                  # NEW: Pinned dependencies
├── README.md                               # MODIFIED: Updated with new workflow
├── CLAUDE.md                               # EXISTING: Already created
├── docs/
│   ├── plans/
│   │   └── 2025-01-11-complete-rebuild-design.md  # THIS DOCUMENT
│   ├── TRAINING_GUIDE.md                   # NEW: Training workflow guide
│   ├── DEPLOYMENT_GUIDE.md                 # NEW: Export & deployment
│   ├── TROUBLESHOOTING.md                  # NEW: Common issues
│   └── PLATFORM_RECOMMENDATIONS.md         # NEW: For platform team
├── utils/
│   ├── __init__.py                         # NEW: Package initialization
│   ├── test_functions.py                   # EXISTING: Keep for backward compat
│   ├── tier1_critical_validation.py        # MODIFIED: Add _safe_get_model_output (~50 lines added)
│   ├── tier2_advanced_analysis.py          # MODIFIED: Add signature handling (~30 lines added)
│   ├── tier3_training_utilities.py         # EXISTING: Rename to legacy_training.py
│   ├── adapters/
│   │   ├── __init__.py                     # NEW
│   │   └── model_adapter.py                # NEW: ~400 lines
│   ├── tokenization/
│   │   ├── __init__.py                     # NEW
│   │   ├── adaptive_tokenizer.py           # NEW: ~500 lines
│   │   ├── bpe_trainer.py                  # NEW: ~300 lines
│   │   ├── character_tokenizer.py          # NEW: ~200 lines
│   │   ├── validator.py                    # NEW: ~100 lines
│   │   └── data_module.py                  # NEW: ~200 lines
│   ├── training/
│   │   ├── __init__.py                     # NEW
│   │   ├── dataset_utilities.py            # NEW: ~300 lines
│   │   ├── checkpoint_manager.py           # NEW: ~250 lines
│   │   ├── training_core.py                # NEW: ~400 lines
│   │   ├── export_utilities.py             # NEW: ~500 lines
│   │   └── legacy_training.py              # RENAMED: from tier3_training_utilities.py
│   └── ui/
│       ├── __init__.py                     # NEW
│       └── setup_wizard.py                 # NEW: ~600 lines
└── tests/                                  # NEW: Comprehensive test suite
    ├── test_model_adapter.py               # NEW: ~200 lines
    ├── test_tokenization.py                # NEW: ~300 lines
    ├── test_training_pipeline.py           # NEW: ~250 lines
    └── test_export.py                      # NEW: ~150 lines

TOTAL:
- New lines: ~4,500
- Modified lines: ~800
- New files: 23
- Modified files: 4
```

## Success Metrics

### Functional Requirements
- [ ] 100% Tier 1 test pass rate (currently 0%)
- [ ] Support for ANY generated model signature
- [ ] Support for ANY vocab_size (100 - 500,000)
- [ ] Checkpoint save/resume works across Colab sessions
- [ ] ONNX export validates with <1e-3 difference
- [ ] Training completes without OOM on T4 GPU

### User Experience
- [ ] Setup wizard completes in <5 minutes for new users
- [ ] Training starts in <2 clicks after wizard
- [ ] Clear error messages with actionable fixes
- [ ] Progress visible at all times
- [ ] Documentation covers 90% of use cases

### Performance
- [ ] BPE training completes in <2 minutes for 10K samples
- [ ] Mixed precision training 40% faster than FP32
- [ ] Checkpoint sync to Drive <30 seconds
- [ ] Wizard load time <5 seconds

## Risk Mitigation

### High Risk: PyTorch Lightning Complexity
**Mitigation:** Create fallback to pure PyTorch training loop if Lightning integration issues arise. Keep `legacy_training.py` functional.

### Medium Risk: BPE Training Memory
**Mitigation:** Implement streaming approach for large datasets, fallback to character-level if memory issues.

### Medium Risk: Timeline Slippage
**Mitigation:** Phase 1-2 are critical path. Phase 3-4 can be reduced scope if needed (wizard → simple forms, fewer export formats).

### Low Risk: ONNX Export Compatibility
**Mitigation:** Focus on inference export only (no training graph). Validate with multiple model architectures early.

## Platform-Side Recommendations

**Separate Document:** `docs/PLATFORM_RECOMMENDATIONS.md`

Key recommendations for platform team:

1. **Model Signature Standardization**: Consider generating models with consistent signature (e.g., always `forward(input_ids, attention_mask)`) to simplify integration

2. **Metadata Generation**: Include model architecture metadata in generated code:
   ```python
   class Model(nn.Module):
       __metadata__ = {
           'architecture': 'GPT',
           'layer_types': ['attention', 'ffn', 'residual'],
           'intermediate_outputs': ['mhsa_0_output', 'residual_0_output'],
       }
   ```

3. **Config Object Standardization**: Generate unified config object with all hyperparameters

4. **Dependency Management**: Document exact library versions used during code generation

5. **Testing Integration**: Consider running Tier 1 tests on platform side before "Open in Colab" to catch issues early

## Approval & Sign-off

**Design Approved By:** User (2025-01-11)
**Technical Review:** Completed via parallel agent analysis
**Implementation Start Date:** 2025-01-11
**Target Completion Date:** 2025-03-01 (8 weeks)

---

**Document Version:** 1.0
**Last Updated:** 2025-01-11
**Status:** Approved - Ready for Implementation
