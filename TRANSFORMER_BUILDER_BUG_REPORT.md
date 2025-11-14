# CRITICAL BUG: Transformer Builder Code Generation

**Date:** 2025-01-13
**Severity:** CRITICAL - Breaks all generated models
**Gist ID:** 8c78c86843e7253f6d66f4339ae15275
**Status:** Blocks all Transformer Builder exports

---

## Executive Summary

The Transformer Builder code generator produces **syntactically valid but semantically broken** PyTorch models. The forward method signature is incorrect, intermediate outputs are treated as inputs, and critical components (residual connections, output projection) are missing or malformed.

**Impact:** 100% of exported models fail to run in Colab or any other environment.

---

## The Bug

### Generated Code (BROKEN)

```python
def forward(self, input_0_tokens: torch.Tensor, mhsa_0_output: torch.Tensor,
            residual_0_output: torch.Tensor, residual_1_output: torch.Tensor) -> torch.Tensor:
    # Embedding: embedding_0
    B, T = input_0_tokens.shape
    positions = torch.arange(0, T, device=input_0_tokens.device)
    tok_emb = self.embedding_0_token(input_0_tokens)
    pos_emb = self.embedding_0_pos(positions)
    embedding_0_x = self.embedding_0_dropout(tok_emb + pos_emb)

    mhsa_0_output, _ = self.mhsa_0(embedding_0_x, embedding_0_x, embedding_0_x)  # ← Overwrites argument!
    layernorm_0_output = self.layernorm_0(residual_0_output)  # ← Uses undefined argument
    ffn_0_output = self.ffn_0(layernorm_0_output)
    layernorm_1_output = self.layernorm_1(residual_1_output)  # ← Uses undefined argument

    return output_0_logits  # ← Variable not defined!
```

### Error When Running

```
TypeError: CustomTransformer.forward() missing 3 required positional arguments:
'mhsa_0_output', 'residual_0_output', and 'residual_1_output'
```

---

## Root Cause Analysis

### Issue 1: Forward Signature Treats Intermediate Outputs as Inputs

**Problem:** The code generator adds intermediate node outputs to the forward signature:
```python
def forward(self, input_0_tokens, mhsa_0_output, residual_0_output, residual_1_output):
```

**Why this is wrong:** In PyTorch, the forward method should only accept:
- Input tensors (e.g., `input_ids`, `attention_mask`)
- Optional configuration flags

Intermediate outputs like `mhsa_0_output` are **computed during the forward pass**, not passed in as arguments.

**Correct signature:**
```python
def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
```

---

### Issue 2: Residual Connections Not Computed

**Problem:** The code references `residual_0_output` and `residual_1_output` as arguments but never computes them.

**What residual connections should do:**
```python
# After attention
mhsa_output, _ = self.mhsa_0(x, x, x)
residual_0_output = x + mhsa_output  # ← Add input to output (residual connection)

# After FFN
ffn_output = self.ffn_0(normalized)
residual_1_output = residual_0_output + ffn_output  # ← Add previous layer
```

**Current behavior:** Treats them as magical inputs that appear from nowhere.

---

### Issue 3: Undefined Output Variable

**Problem:** The forward method returns `output_0_logits`, which is never defined.

**Missing code:**
```python
# Need a final linear projection to vocabulary size
self.output_projection = nn.Linear(768, 50257)  # In __init__

# In forward:
logits = self.output_projection(final_layer_output)
return logits
```

---

### Issue 4: Logic Overwrites Argument

**Problem:** The code computes `mhsa_0_output` inside the forward method:
```python
mhsa_0_output, _ = self.mhsa_0(embedding_0_x, embedding_0_x, embedding_0_x)
```

But `mhsa_0_output` is also a required function argument! This suggests the code generator is confused about whether `mhsa_0_output` is:
- An input to the function (wrong)
- An intermediate computation result (correct)

---

## Correct Implementation

Here's what the code generator **should** produce:

```python
class CustomTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Embedding layers
        self.embedding_0_token = nn.Embedding(50257, 768)
        self.embedding_0_pos = nn.Embedding(512, 768)
        self.embedding_0_dropout = nn.Dropout(0.1)

        # Multi-head self-attention
        self.mhsa_0 = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization
        self.layernorm_0 = nn.LayerNorm(768, eps=1e-05)

        # Feed-forward network
        self.ffn_0 = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(3072, 768),
            nn.Dropout(0.1)
        )

        # Layer normalization
        self.layernorm_1 = nn.LayerNorm(768, eps=1e-05)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(768, 50257)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Optional attention mask, shape (batch_size, seq_len)

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device)
        tok_emb = self.embedding_0_token(input_ids)
        pos_emb = self.embedding_0_pos(positions)
        x = self.embedding_0_dropout(tok_emb + pos_emb)

        # Multi-head self-attention
        mhsa_output, _ = self.mhsa_0(x, x, x, attn_mask=attention_mask)

        # Residual connection + LayerNorm
        residual_0 = x + mhsa_output  # ← Residual connection 1
        x = self.layernorm_0(residual_0)

        # Feed-forward network
        ffn_output = self.ffn_0(x)

        # Residual connection + LayerNorm
        residual_1 = residual_0 + ffn_output  # ← Residual connection 2
        x = self.layernorm_1(residual_1)

        # Output projection
        logits = self.output_projection(x)

        return logits
```

---

## Code Generator Fix Requirements

### 1. Forward Signature Generation

**Current (WRONG):**
```python
def forward(self, input_0_tokens, mhsa_0_output, residual_0_output, residual_1_output):
```

**Required:**
```python
def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
```

**Rule:** Only input nodes should become function parameters. Intermediate nodes (MHSA, residual, FFN, etc.) should be computed inside the forward method.

---

### 2. Node Processing Order

The code generator needs to:

1. **Topologically sort nodes** based on dependencies
2. **Generate sequential computations** in correct order
3. **Store intermediate results in local variables** (not function arguments)

**Example execution order:**
```
input → embedding → mhsa → residual_add → layernorm → ffn → residual_add → layernorm → output
```

**Generated code structure:**
```python
def forward(self, input_ids):
    # Step 1: Process input node
    x = self.embedding(input_ids)

    # Step 2: Process mhsa node (depends on embedding output)
    mhsa_output = self.mhsa(x, x, x)

    # Step 3: Process residual node (depends on embedding + mhsa)
    residual_0 = x + mhsa_output

    # Step 4: Process layernorm node (depends on residual_0)
    x = self.layernorm_0(residual_0)

    # ... continue for remaining nodes

    # Final: Return output node result
    return final_output
```

---

### 3. Residual Node Implementation

**Current behavior:** Generates as function argument

**Required behavior:** Generate as addition operation

**Code template:**
```python
# For a residual node connecting node_A and node_B:
residual_output = node_A_output + node_B_output
```

---

### 4. Output Node Implementation

**Current behavior:** Returns undefined variable `output_0_logits`

**Required behavior:**
1. Add output projection layer in `__init__`
2. Apply projection in forward
3. Return the result

**Code template:**
```python
# In __init__:
self.output_projection = nn.Linear(d_model, vocab_size)

# In forward (at the end):
logits = self.output_projection(final_layer_output)
return logits
```

---

## Testing the Fix

### Validation Steps

1. **Export a simple transformer** (1 layer, no residuals)
2. **Check forward signature:**
   ```python
   import inspect
   sig = inspect.signature(model.forward)
   params = list(sig.parameters.keys())
   assert params == ['input_ids'] or params == ['input_ids', 'attention_mask']
   ```

3. **Run a forward pass:**
   ```python
   import torch
   model = CustomTransformer()
   input_ids = torch.randint(0, 50257, (2, 32))
   output = model(input_ids)
   assert output.shape == (2, 32, 50257)  # (batch, seq, vocab)
   ```

4. **Test with residual connections:**
   - Export model with residual nodes
   - Verify residuals are computed as `x + layer_output`
   - Verify no residual variables appear in forward signature

---

## Recommended Code Generator Architecture

```
Canvas Nodes → Dependency Graph → Topological Sort → Code Generation
                                                      ↓
                                                 __init__ generation:
                                                 - Input nodes → skip
                                                 - Layer nodes → nn.Module components
                                                 - Output nodes → projection layers
                                                      ↓
                                                 forward() generation:
                                                 - Input nodes → function parameters
                                                 - Layer nodes → sequential computations
                                                 - Residual nodes → addition operations
                                                 - Output nodes → return statement
```

---

## Priority

**CRITICAL - P0**

This bug blocks **100% of Transformer Builder exports**. No generated model can run successfully. All development, testing, and user workflows are blocked until this is fixed.

---

## Reproduction

1. Go to Transformer Builder
2. Create any transformer architecture (even minimal: input → embedding → output)
3. Export to Colab
4. Paste Gist ID in Cell 3
5. Run notebook
6. **Observe:** TypeError about missing positional arguments

**Every single export will fail.**

---

## Contact

- **Colab Template Repository:** https://github.com/matt-hans/transformer-builder-colab-templates
- **Bug Report File:** `TRANSFORMER_BUILDER_BUG_REPORT.md`
- **Test Gist ID:** 8c78c86843e7253f6d66f4339ae15275
- **Date Reported:** 2025-01-13

---

## Appendix: Full Generated Code (Broken)

```python
"""
Generated model: CustomTransformer
Auto-generated by Transformer Builder.
DO NOT EDIT - regenerate from canvas.
"""

import torch
import torch.nn as nn

class CustomTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # input: input_0
        # embedding: embedding_0
        self.embedding_0_token = nn.Embedding(50257, 768)
        self.embedding_0_pos = nn.Embedding(512, 768)
        self.embedding_0_dropout = nn.Dropout(0.1)
        # mhsa: mhsa_0
        self.mhsa_0 = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        # residual: residual_0
        # layernorm: layernorm_0
        self.layernorm_0 = nn.LayerNorm(768, eps=1e-05)
        # ffn: ffn_0
        self.ffn_0 = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(3072, 768),
            nn.Dropout(0.1)
        )
        # residual: residual_1
        # layernorm: layernorm_1
        self.layernorm_1 = nn.LayerNorm(768, eps=1e-05)
        # output: output_0

    def forward(self, input_0_tokens: torch.Tensor, mhsa_0_output: torch.Tensor, residual_0_output: torch.Tensor, residual_1_output: torch.Tensor) -> torch.Tensor:
        # Embedding: embedding_0
        B, T = input_0_tokens.shape
        positions = torch.arange(0, T, device=input_0_tokens.device)
        tok_emb = self.embedding_0_token(input_0_tokens)
        pos_emb = self.embedding_0_pos(positions)
        embedding_0_x = self.embedding_0_dropout(tok_emb + pos_emb)
        mhsa_0_output, _ = self.mhsa_0(embedding_0_x, embedding_0_x, embedding_0_x)
        layernorm_0_output = self.layernorm_0(residual_0_output)
        ffn_0_output = self.ffn_0(layernorm_0_output)
        layernorm_1_output = self.layernorm_1(residual_1_output)

        return output_0_logits
```

**Issues:**
1. ❌ Forward signature includes intermediate outputs as arguments
2. ❌ `mhsa_0_output` computed but also required as argument
3. ❌ `residual_0_output` and `residual_1_output` used but never computed
4. ❌ `output_0_logits` returned but never defined
5. ❌ No output projection layer in `__init__`
6. ❌ Residual connections not implemented
7. ❌ Cannot be called with just `input_ids`

---

**End of Bug Report**
