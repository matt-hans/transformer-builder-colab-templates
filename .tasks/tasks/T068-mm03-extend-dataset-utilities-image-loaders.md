---
id: T068
enhancement_id: MM-03
title: Extend Dataset Utilities with Image Loaders for Vision Tasks
status: pending
priority: 1
agent: backend
dependencies: [T066, T067]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [multimodal, datasets, vision, enhancement1.0, critical-path]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/DEVELOPER_GUIDE_TASKS_EVAL.md

est_tokens: 14000
actual_tokens: null
---

## Description

Extend `dataset_utilities.py` to support image datasets for vision classification tasks, enabling the training pipeline to load and preprocess images from local directories or HuggingFace datasets. This task creates a `TinyVisionDataset` class and integrates it into the `build_dataloader` factory function.

The implementation adds a modality-aware branch to the existing text-centric dataset loader, using `task_spec.modality` to route to appropriate dataset classes. For vision tasks, images are loaded from disk (PNG/JPEG), transformed via torchvision (resize, normalize, tensor conversion), and batched with labels.

**Technical Approach**: Create a lightweight `TinyVisionDataset` PyTorch Dataset class, add sample images to `examples/datasets/vision/`, and extend `build_dataloader` with vision-specific logic. Lazy-import torchvision to avoid extra dependencies in text-only workflows.

**Integration Points**:
- `utils/training/dataset_utilities.py` (primary changes)
- `examples/datasets/vision/` (sample data)
- `utils/training/task_spec.py` (consumes input_schema for transform config)
- `utils/training/training_core.py` (calls build_dataloader for vision tasks)

## Business Context

**User Story**: As a vision model developer, I want to load CIFAR-10-style image datasets using the same `build_dataloader` interface as text datasets, so that I can train models without writing custom data loading code.

**Why This Matters**:
- **Enables vision training**: Unlocks end-to-end training pipeline for vision models
- **Consistent API**: Same function call for text and vision datasets reduces learning curve
- **Colab-friendly**: Tiny sample dataset (<1MB) for fast experimentation without downloads

**What It Unblocks**:
- MM-04: Vision evaluation (needs DataLoader to run eval loop)
- MM-05: CLI integration (needs working dataset for run_training.py)
- Tier 3 vision training (fine-tuning on vision_tiny dataset)

**Priority Justification**: Priority 1 - Required for any vision model training; blocks MM-04, MM-05, and vision fine-tuning workflows.

## Acceptance Criteria

- [ ] `TinyVisionDataset` class created in `utils/training/dataset_utilities.py` with `__getitem__` returning `{"pixel_values": Tensor[C, H, W], "labels": int}`
- [ ] Sample dataset created in `examples/datasets/vision/vision_tiny/` with 16-32 images (3-4 classes, 4-8 images per class)
- [ ] `labels.json` file maps image filenames to class labels (e.g., `{"cat_001.jpg": 0, "dog_001.jpg": 1, ...}`)
- [ ] `build_dataloader` extended with vision branch: `if task_spec.modality == "vision"`
- [ ] Transforms configured from `task_spec.input_schema["image_size"]` dynamically
- [ ] Optional torchvision dependency (lazy import with graceful fallback if not installed)
- [ ] DataLoader returns batches with keys `{"pixel_values": [B, C, H, W], "labels": [B]}`
- [ ] Works with `TrainingConfig(task_name="vision_tiny")` + `build_task_spec` + `build_dataloader`
- [ ] Type hints added to TinyVisionDataset.__init__ and __getitem__
- [ ] Unit test validates dataset length == number of images in labels.json
- [ ] Unit test validates single item shape matches input_schema
- [ ] Example notebook cell demonstrates vision DataLoader usage in training.ipynb

## Test Scenarios

**Test Case 1: TinyVisionDataset Creation**
- Given: `examples/datasets/vision/vision_tiny/` with 16 images, labels.json with 16 entries
- When: `dataset = TinyVisionDataset(data_dir="examples/datasets/vision/vision_tiny", image_size=(3, 64, 64))`
- Then: `len(dataset) == 16`, `dataset[0]` returns dict with pixel_values shape [3, 64, 64]

**Test Case 2: DataLoader Batching**
- Given: TinyVisionDataset with batch_size=4
- When: `loader = DataLoader(dataset, batch_size=4, shuffle=True)`
- Then: First batch has pixel_values shape [4, 3, 64, 64], labels shape [4]

**Test Case 3: Build DataLoader for Vision Task**
- Given: `task_spec` with `modality="vision"`, `input_schema={"image_size": [3, 32, 32]}`
- When: `loader = build_dataloader(task_spec, task_name="vision_tiny", batch_size=4)`
- Then: Returns DataLoader yielding vision batches with correct shapes

**Test Case 4: Dynamic Image Resize from Input Schema**
- Given: `input_schema={"image_size": [3, 128, 128]}` (larger than stored images)
- When: Dataset loads image and applies transforms
- Then: Returned pixel_values are resized to [3, 128, 128] via torchvision.Resize

**Test Case 5: Normalization from Preprocessing Config**
- Given: `preprocessing_config={"normalize": True, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}`
- When: Dataset applies transforms
- Then: Pixel values normalized to [-1, 1] range (verified by checking mean ≈ 0)

**Test Case 6: Graceful Fallback Without Torchvision**
- Given: Torchvision not installed in environment
- When: `build_dataloader` called for vision task
- Then: Raises `ImportError` with message: "torchvision required for vision tasks. Install with: pip install torchvision"

**Test Case 7: Training Loop Integration**
- Given: Vision model + VisionClassificationAdapter + vision DataLoader
- When: Training loop iterates over DataLoader and calls `adapter.forward(model, batch, task_spec)`
- Then: Training runs for 1 epoch without errors, loss decreases

**Test Case 8: Labels.json Validation**
- Given: labels.json missing a filename that exists in directory
- When: Dataset initialization called
- Then: Warning logged: "Image cat_005.jpg found in directory but not in labels.json, skipping"

## Technical Implementation

**Required Components:**

1. **`examples/datasets/vision/vision_tiny/`** (sample data)
   - Create 16-32 small images (32x32 or 64x64 PNG/JPEG)
   - 3-4 classes (e.g., "cat", "dog", "bird", "fish")
   - `labels.json`:
     ```json
     {
       "cat_001.png": 0,
       "cat_002.png": 0,
       "dog_001.png": 1,
       "dog_002.png": 1,
       "bird_001.png": 2,
       ...
     }
     ```
   - Total size <500KB for fast git clone

2. **`utils/training/dataset_utilities.py`** (extend with vision support)
   ```python
   class TinyVisionDataset(Dataset):
       """Lightweight vision dataset for tiny image classification tasks."""

       def __init__(
           self,
           data_dir: Path | str,
           image_size: tuple[int, int, int] = (3, 64, 64),
           transforms: Callable | None = None,
           normalize: bool = True,
           mean: list[float] = [0.5, 0.5, 0.5],
           std: list[float] = [0.5, 0.5, 0.5],
       ):
           """
           Args:
               data_dir: Directory containing images and labels.json
               image_size: (C, H, W) target size for images
               transforms: Optional custom transforms (if None, uses default resize+normalize)
               normalize: Whether to apply normalization
               mean/std: Normalization parameters (default: [-1, 1] range)
           """
           self.data_dir = Path(data_dir)
           self.image_size = image_size

           # Load labels
           with open(self.data_dir / "labels.json") as f:
               self.labels_map = json.load(f)

           self.image_files = list(self.labels_map.keys())

           # Build transforms
           if transforms is None:
               try:
                   from torchvision import transforms as T
                   transform_list = [
                       T.Resize((image_size[1], image_size[2])),
                       T.ToTensor(),
                   ]
                   if normalize:
                       transform_list.append(T.Normalize(mean=mean, std=std))
                   self.transforms = T.Compose(transform_list)
               except ImportError:
                   raise ImportError(
                       "torchvision required for vision tasks. "
                       "Install with: pip install torchvision"
                   )
           else:
               self.transforms = transforms

       def __len__(self) -> int:
           return len(self.image_files)

       def __getitem__(self, idx: int) -> dict[str, Any]:
           """
           Returns:
               {"pixel_values": Tensor[C, H, W], "labels": int}
           """
           from PIL import Image

           img_file = self.image_files[idx]
           img_path = self.data_dir / img_file
           image = Image.open(img_path).convert("RGB")

           pixel_values = self.transforms(image)
           label = self.labels_map[img_file]

           return {"pixel_values": pixel_values, "labels": label}
   ```

3. **Extend `build_dataloader`**
   ```python
   def build_dataloader(
       task_spec: TaskSpec,
       task_name: str | None = None,
       batch_size: int = 4,
       shuffle: bool = True,
       num_workers: int = 0,
   ) -> DataLoader:
       """Build dataloader for text or vision tasks."""

       if task_spec.modality == "vision" and task_spec.task_type == "vision_classification":
           # Extract config
           image_size = task_spec.input_schema.get("image_size", [3, 64, 64])
           preprocess = task_spec.preprocessing_config or {}

           # Determine dataset path
           data_dir = Path(f"examples/datasets/vision/{task_name}")
           if not data_dir.exists():
               raise FileNotFoundError(f"Vision dataset not found: {data_dir}")

           dataset = TinyVisionDataset(
               data_dir=data_dir,
               image_size=tuple(image_size),
               normalize=preprocess.get("normalize", True),
               mean=preprocess.get("mean", [0.5, 0.5, 0.5]),
               std=preprocess.get("std", [0.5, 0.5, 0.5]),
           )

           return DataLoader(
               dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               num_workers=num_workers,
           )

       elif task_spec.modality == "text":
           # Existing text logic
           ...
   ```

**Validation Commands:**

```bash
# Manual unit test
python -c "
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.training.dataset_utilities import TinyVisionDataset, build_dataloader
from utils.training.task_spec import TaskSpec

# Test TinyVisionDataset
dataset = TinyVisionDataset(
    data_dir='examples/datasets/vision/vision_tiny',
    image_size=(3, 64, 64)
)
print(f'Dataset length: {len(dataset)}')
assert len(dataset) > 0, 'Dataset should not be empty'

# Test single item
item = dataset[0]
assert 'pixel_values' in item, 'Item must have pixel_values key'
assert 'labels' in item, 'Item must have labels key'
assert item['pixel_values'].shape == (3, 64, 64), f'Wrong shape: {item[\"pixel_values\"].shape}'
print(f'✓ Single item shape: {item[\"pixel_values\"].shape}')

# Test DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
assert batch['pixel_values'].shape[0] == min(4, len(dataset)), 'Wrong batch size'
assert batch['pixel_values'].shape[1:] == (3, 64, 64), 'Wrong image shape in batch'
print(f'✓ Batch shape: {batch[\"pixel_values\"].shape}')

# Test build_dataloader integration
task_spec = TaskSpec(
    task_name='vision_tiny',
    modality='vision',
    task_type='vision_classification',
    input_schema={'image_size': [3, 64, 64]}
)
loader = build_dataloader(task_spec, task_name='vision_tiny', batch_size=2)
batch = next(iter(loader))
assert batch['pixel_values'].shape == (2, 3, 64, 64), 'Build_dataloader failed'
print('✓ build_dataloader integration successful')
"
```

**Code Patterns:**

- Use `Path` for file path handling (cross-platform)
- Lazy import torchvision (avoid forcing installation for text-only users)
- Return dicts from `__getitem__` (not tuples) for clarity
- Validate data_dir exists before dataset creation
- Log warnings for missing labels, don't crash

## Dependencies

**Hard Dependencies** (must be complete first):
- [T066] Extend TaskSpec to Support Modalities - Provides input_schema, modality fields
- [T067] Add VisionClassificationAdapter - Will consume batches produced by this dataset

**Soft Dependencies** (nice to have):
- None

**External Dependencies:**
- torchvision >= 0.15.0 (for transforms, optional dependency)
- Pillow (PIL) for image loading (usually bundled with PyTorch)

## Design Decisions

**Decision 1: Store images as files, not in-memory arrays**
- **Rationale**: Git-friendly (<500KB total), easy to inspect/modify, standard format
- **Alternatives**: Store as numpy arrays in .npy file (faster but opaque)
- **Trade-offs**: Slight I/O overhead, but negligible for tiny dataset

**Decision 2: Use labels.json instead of directory-based labels**
- **Rationale**: Explicit mapping, easier to reorder classes, fewer directories
- **Alternatives**: ImageFolder-style (images/cat/*.jpg, images/dog/*.jpg) - more standard but rigid
- **Trade-offs**: Manual JSON creation, but more control over splits and subsets

**Decision 3: Lazy import torchvision**
- **Rationale**: Don't force installation for text-only workflows; reduces dependency bloat
- **Alternatives**: Add torchvision to requirements.txt (simpler but increases install size by ~100MB)
- **Trade-offs**: Potential import errors if user forgets to install, but clear error message mitigates

**Decision 4: Default normalization to [-1, 1] range**
- **Rationale**: Common for GANs and some vision models; mean=0.5, std=0.5 achieves this
- **Alternatives**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) - standard for pretrained models
- **Trade-offs**: Not optimal for pretrained models, but user can override via preprocessing_config

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Torchvision not installed in Colab | H - Vision training blocked | L | Document installation in training.ipynb; add lazy import with clear error message |
| Sample images too large (>1MB total) | M - Slow git clone | M | Compress to PNG, resize to 32x32 or 64x64; target <500KB total |
| Labels.json out of sync with image files | M - Dataset fails to load | M | Add validation on init: warn if files missing from JSON, skip gracefully |
| Image file corruption | M - Training crashes mid-epoch | L | Add try/except in `__getitem__` to catch corrupt images, log warning and skip |
| Transforms cause memory issues on GPU | M - OOM errors during training | L | Document num_workers=0 for Colab (avoid multiprocessing overhead); use small batch sizes |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Third task in multimodal foundation (MM-03 from enhancement1.0.md)
**Dependencies:** T066 (TaskSpec), T067 (VisionAdapter)
**Estimated Complexity:** Standard (dataset class + sample data creation + integration)

## Completion Checklist

**Code Implementation:**
- [ ] `TinyVisionDataset` class created in dataset_utilities.py
- [ ] `__init__` configures transforms from image_size and preprocessing_config
- [ ] `__getitem__` loads image, applies transforms, returns dict with pixel_values + labels
- [ ] `build_dataloader` extended with vision modality branch
- [ ] Lazy torchvision import with ImportError message

**Sample Data:**
- [ ] `examples/datasets/vision/vision_tiny/` directory created
- [ ] 16-32 images added (PNG or JPEG, 32x32 or 64x64)
- [ ] 3-4 classes represented (balanced distribution)
- [ ] `labels.json` created with filename → class_id mapping
- [ ] Total directory size <500KB (verified with `du -sh`)

**Testing:**
- [ ] Unit test: `len(dataset)` matches number of entries in labels.json
- [ ] Unit test: `dataset[0]` returns correct shape from input_schema
- [ ] Unit test: DataLoader batching works (batch shape [B, C, H, W])
- [ ] Integration test: `build_dataloader` creates working DataLoader for vision_tiny
- [ ] Manual test: Training loop runs 1 epoch with vision model + adapter + dataset

**Documentation:**
- [ ] Docstrings added to TinyVisionDataset with Args/Returns
- [ ] training.ipynb updated with vision dataset usage example cell
- [ ] `docs/DEVELOPER_GUIDE_TASKS_EVAL.md` updated with dataset creation guide
- [ ] labels.json format documented

**Integration:**
- [ ] Works with TrainingConfig(task_name="vision_tiny")
- [ ] Compatible with VisionClassificationAdapter from T067
- [ ] No regressions in text dataset loading

**Quality Gates:**
- [ ] All 12 acceptance criteria checked
- [ ] All 8 test scenarios validated
- [ ] 4 design decisions documented
- [ ] 5 risks with mitigations
- [ ] Token estimate (14,000) appropriate

**Definition of Done:**
Task is complete when TinyVisionDataset loads sample images correctly, build_dataloader creates vision DataLoaders, sample dataset is committed to examples/datasets/, and training loop can iterate over vision batches without errors.
