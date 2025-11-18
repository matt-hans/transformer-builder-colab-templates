# Performance Optimization in Google Colab

Best practices for maximizing GPU utilization, memory efficiency, and runtime performance.

## GPU Optimization

### 1. Select the Right Runtime

```python
# Check current GPU allocation
!nvidia-smi

# Recommended runtimes by use case:
# - T4 (15GB): Fine-tuning, inference, medium models
# - V100 (16GB): Training large models
# - A100 (40GB): Very large models, high-throughput training
# - L4 (24GB): Latest generation, good balance
```

**Pro Tip:** Use T4 for most tasks - it's the default free tier GPU and sufficient for 90% of workflows.

### 2. Mixed Precision Training (FP16/BF16)

```python
import torch

# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass with autocasting
        with autocast():
            output = model(batch)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Benefits:**
- ~2x faster training
- ~50% less memory usage
- Minimal accuracy loss (<1%)

**When to use:** Large models, limited VRAM, batch processing

### 3. Gradient Accumulation

```python
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)

    # Scale loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()

    # Update weights only every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefits:**
- Larger effective batch sizes without OOM errors
- Improved gradient estimates
- Better convergence

**When to use:** Limited VRAM, need large batch sizes

### 4. Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class Model(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return self.output(x)
```

**Benefits:**
- ~30-50% memory reduction
- Trade compute for memory

**When to use:** Very deep models, OOM errors

### 5. Efficient Attention Mechanisms

```python
# Flash Attention 2 (fastest)
from torch.nn.functional import scaled_dot_product_attention

def forward(self, q, k, v):
    return scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True
    )

# xFormers (memory efficient)
from xformers.ops import memory_efficient_attention

def forward(self, q, k, v):
    return memory_efficient_attention(q, k, v)
```

**Benefits:**
- 2-4x faster than vanilla attention
- Lower memory usage
- Longer sequence lengths

## Memory Management

### 1. Monitor Memory Usage

```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")

# Call after major operations
print_memory_stats()
```

### 2. Clear Memory Cache

```python
# Free unused memory
torch.cuda.empty_cache()

# More aggressive cleanup
import gc
gc.collect()
torch.cuda.empty_cache()

# Nuclear option (delete model and rebuild)
del model
gc.collect()
torch.cuda.empty_cache()
model = create_model()
```

### 3. Optimize Batch Sizes

```python
def find_optimal_batch_size(model, input_shape, min_bs=1, max_bs=128):
    """Binary search for maximum batch size that fits in memory."""
    device = next(model.parameters()).device

    def test_batch_size(bs):
        try:
            dummy_input = torch.randn(bs, *input_shape).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            del dummy_input
            torch.cuda.empty_cache()
            return True
        except RuntimeError:  # OOM
            torch.cuda.empty_cache()
            return False

    low, high = min_bs, max_bs
    optimal = min_bs

    while low <= high:
        mid = (low + high) // 2
        if test_batch_size(mid):
            optimal = mid
            low = mid + 1
        else:
            high = mid - 1

    return optimal

# Usage
batch_size = find_optimal_batch_size(model, (3, 224, 224))
print(f"Optimal batch size: {batch_size}")
```

### 4. Model Quantization

```python
# Dynamic Quantization (CPU inference)
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

# Static Quantization (requires calibration)
model_fp32 = model.to('cpu')
model_fp32.eval()

# Fuse modules
model_fp32_fused = torch.quantization.fuse_modules(
    model_fp32,
    [['conv', 'bn', 'relu']]
)

# Prepare for quantization
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# Calibrate with representative data
for data in calibration_data:
    model_fp32_prepared(data)

# Convert to quantized model
model_int8 = torch.quantization.convert(model_fp32_prepared)
```

**Benefits:**
- 4x smaller model size (fp32 → int8)
- 2-4x faster inference
- Lower memory usage

### 5. Offload to CPU

```python
# Offload optimizer states to CPU
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

# Manual offloading
model = model.to('cuda')
optimizer_state = optimizer.state_dict()

# Move optimizer state to CPU
optimizer.state.clear()
torch.cuda.empty_cache()

# Load back when needed
optimizer.load_state_dict(optimizer_state)
```

## Data Loading Optimization

### 1. Efficient DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,  # Use 2-4 workers in Colab
    pin_memory=True,  # Faster CPU→GPU transfer
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

**Settings for Colab:**
- `num_workers=2`: Colab has 2 CPU cores
- `pin_memory=True`: Always enable for GPU training
- `persistent_workers=True`: Reduces worker startup overhead

### 2. Data Preprocessing

```python
import torch
from torchvision import transforms

# Precompute transforms on CPU
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Apply on CPU before moving to GPU
for batch in dataloader:
    # batch is already transformed and on CPU
    batch = batch.to('cuda', non_blocking=True)  # Async transfer
    output = model(batch)
```

### 3. Caching Preprocessed Data

```python
import pickle
from pathlib import Path

CACHE_DIR = Path("/content/cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_with_cache(dataset_fn, cache_key):
    """Load dataset with caching."""
    cache_path = CACHE_DIR / f"{cache_key}.pkl"

    if cache_path.exists():
        print(f"Loading from cache: {cache_key}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Processing dataset: {cache_key}")
    dataset = dataset_fn()

    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset

# Usage
train_data = load_with_cache(
    lambda: prepare_dataset("train"),
    cache_key="train_processed"
)
```

## Inference Optimization

### 1. Disable Gradient Computation

```python
# For inference only
model.eval()

with torch.no_grad():
    output = model(input)
```

**Savings:** ~50% memory, ~20% faster

### 2. Use torch.jit (TorchScript)

```python
# Trace model
example_input = torch.randn(1, 3, 224, 224).cuda()
traced_model = torch.jit.trace(model, example_input)

# Use traced model for inference
with torch.no_grad():
    output = traced_model(input)

# Save for later use
traced_model.save("model_traced.pt")
```

**Benefits:**
- ~10-30% faster inference
- Can run without Python interpreter
- Production deployment ready

### 3. Batch Inference

```python
# Instead of processing one at a time
for item in items:
    output = model(item)  # Slow

# Batch multiple items
batch_size = 32
for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    outputs = model(batch)  # Much faster
```

**Benefits:**
- 5-10x faster throughput
- Better GPU utilization

### 4. Model Compilation (PyTorch 2.0+)

```python
import torch

# Compile model for faster inference
compiled_model = torch.compile(model, mode='max-autotune')

# Use compiled model
with torch.no_grad():
    output = compiled_model(input)
```

**Benefits:**
- 30-50% faster inference
- Zero code changes
- Automatic kernel fusion

## Runtime Management

### 1. Prevent Disconnections

```python
# Install keepalive extension
!pip install jupyter_http_over_ws

# Or use JavaScript to prevent idle timeout
from IPython.display import Javascript

Javascript('''
    function ClickConnect(){
        console.log("Clicking 'Connect' button...");
        document.querySelector("colab-connect-button").shadowRoot.querySelector("#connect").click()
    }
    setInterval(ClickConnect, 60000)
''')
```

### 2. Monitor Runtime Limits

```python
import psutil
import time

def print_runtime_stats():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    # Memory usage
    mem = psutil.virtual_memory()
    mem_used = mem.used / 1024**3
    mem_total = mem.total / 1024**3

    # Disk usage
    disk = psutil.disk_usage('/content')
    disk_used = disk.used / 1024**3
    disk_total = disk.total / 1024**3

    print(f"CPU: {cpu_percent}%")
    print(f"RAM: {mem_used:.1f} / {mem_total:.1f} GB ({mem.percent}%)")
    print(f"Disk: {disk_used:.1f} / {disk_total:.1f} GB ({disk.percent}%)")

    # GPU (if available)
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU: {gpu_used:.1f} / {gpu_mem:.1f} GB")

print_runtime_stats()
```

### 3. Checkpointing Strategy

```python
from pathlib import Path
import torch

CHECKPOINT_DIR = Path("/content/drive/MyDrive/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pt"):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    path = CHECKPOINT_DIR / filename
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pt"):
    """Load training checkpoint."""
    path = CHECKPOINT_DIR / filename

    if not path.exists():
        print(f"No checkpoint found: {filename}")
        return 0

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded: {filename} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

# Save every N epochs
for epoch in range(start_epoch, num_epochs):
    train_one_epoch()

    if (epoch + 1) % 5 == 0:  # Save every 5 epochs
        save_checkpoint(model, optimizer, epoch, loss,
                       filename=f"checkpoint_epoch_{epoch}.pt")
```

## Profiling and Debugging

### 1. PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

### 2. Measure Latency

```python
import time

def measure_latency(fn, num_iterations=100, warmup=10):
    """Measure function latency."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Measure
    torch.cuda.synchronize()  # Ensure GPU finished
    start = time.time()

    for _ in range(num_iterations):
        fn()

    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / num_iterations * 1000  # ms
    return avg_latency

# Usage
latency = measure_latency(lambda: model(input))
print(f"Average latency: {latency:.2f} ms")
```

## Best Practices Summary

1. **Use mixed precision training** (FP16/BF16) for 2x speedup
2. **Enable gradient accumulation** for larger effective batch sizes
3. **Monitor memory usage** and clear cache regularly
4. **Optimize DataLoader** with num_workers=2, pin_memory=True
5. **Use torch.compile()** (PyTorch 2.0+) for 30-50% faster inference
6. **Save checkpoints to Google Drive** to survive disconnections
7. **Profile code** to identify bottlenecks
8. **Batch inference** for 5-10x throughput improvement
9. **Use Flash Attention** for transformer models
10. **Quantize models** for 4x smaller size and faster inference
