# Transformer Builder → Colab Integration Guide

**Version:** 3.4.0 (Simple Modal Approach)
**Date:** 2025-01-13
**Status:** Ready for Implementation

---

## Overview

This document describes the simple, clean integration between Transformer Builder and Google Colab for exporting custom transformer models.

**User Experience:**
1. User clicks "Export to Colab" in Transformer Builder
2. Modal appears with Gist ID and one-click copy button
3. User clicks Copy → OK
4. Colab opens in new tab
5. User pastes Gist ID in prominent Cell 3 input form
6. Run all cells → Custom model loads and tests automatically

**Total user effort:** One copy/paste (5 seconds)

---

## Why This Approach?

We evaluated complex auto-injection solutions but chose this simple modal approach because:

- ✅ **10 minutes implementation** (vs 9 hours for auto-injection)
- ✅ **Zero maintenance** (no template syncing, no injection bugs)
- ✅ **Crystal clear UX** (user sees exactly what's happening)
- ✅ **No edge cases** (no sharing issues, no expiry bugs)
- ✅ **One copy/paste is trivial** (not "confusing" - it's transparent)

---

## Technical Implementation

### 1. Gist Creation

When the user clicks "Export to Colab", create a GitHub Gist with **exactly 2 files:**

```javascript
const gist = await createGist({
    files: {
        'model.py': {
            content: generateModelCode(model)  // Your generated Python code
        },
        'config.json': {
            content: JSON.stringify({
                vocab_size: model.vocab_size,
                d_model: model.d_model,
                nhead: model.nhead,
                num_layers: model.num_layers,
                // ... all model configuration parameters
            })
        }
    },
    description: `${model.name} - Transformer Builder Export`,
    public: true  // Must be public for Colab to access
});

const gistId = gist.id;  // e.g., "abc123def456"
```

**Requirements:**
- Gist must be **public** (Colab API requires public Gists)
- Must contain **exactly** `model.py` and `config.json`
- File names are case-sensitive

---

### 2. Modal UI Implementation

Show a modal with the Gist ID and copy functionality:

```javascript
async function exportToColab(model, config) {
    // Create Gist
    const gist = await createGist({
        files: {
            'model.py': { content: generateModelCode(model) },
            'config.json': { content: JSON.stringify(config) }
        },
        description: `${model.name} - Transformer Builder Export`,
        public: true
    });

    const gistId = gist.id;

    // Show modal with copy button
    showModal({
        title: '📋 Ready to Test in Colab',
        html: `
            <div class="export-modal">
                <p class="success-message">
                    ✅ Your model has been exported successfully!
                </p>

                <div class="gist-id-section">
                    <label>Your Gist ID:</label>
                    <div class="gist-id-box">
                        <code id="gist-id-value">${gistId}</code>
                        <button
                            class="copy-button"
                            onclick="copyGistId('${gistId}')"
                        >
                            📋 Copy
                        </button>
                    </div>
                </div>

                <div class="instructions">
                    <p><strong>Next steps:</strong></p>
                    <ol>
                        <li>Click the <strong>Copy</strong> button above</li>
                        <li>Click <strong>Open in Colab</strong> below</li>
                        <li>Paste the Gist ID in <strong>Cell 3</strong></li>
                        <li>Click <strong>Runtime → Run all</strong></li>
                    </ol>
                </div>
            </div>
        `,
        buttons: [
            {
                text: 'Cancel',
                variant: 'secondary',
                onClick: () => closeModal()
            },
            {
                text: '🚀 Open in Colab',
                variant: 'primary',
                onClick: () => {
                    window.open(
                        'https://colab.research.google.com/github/matt-hans/transformer-builder-colab-templates/blob/main/template.ipynb',
                        '_blank'
                    );
                    closeModal();
                }
            }
        ]
    });
}

function copyGistId(gistId) {
    navigator.clipboard.writeText(gistId).then(() => {
        // Show success feedback
        showToast({
            message: '✅ Gist ID copied!',
            type: 'success',
            duration: 2000
        });

        // Optional: Change button text temporarily
        const button = document.querySelector('.copy-button');
        const originalText = button.innerHTML;
        button.innerHTML = '✅ Copied!';
        button.disabled = true;

        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 2000);
    }).catch(err => {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = gistId;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);

        showToast({
            message: '✅ Gist ID copied!',
            type: 'success',
            duration: 2000
        });
    });
}
```

---

### 3. Modal Styling (CSS)

```css
.export-modal {
    max-width: 500px;
    padding: 20px;
}

.success-message {
    font-size: 16px;
    margin-bottom: 20px;
    color: #2e7d32;
}

.gist-id-section {
    margin: 20px 0;
}

.gist-id-section label {
    display: block;
    font-weight: 600;
    margin-bottom: 8px;
    color: #333;
}

.gist-id-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 6px;
}

#gist-id-value {
    flex: 1;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 14px;
    color: #1976d2;
    background: white;
    padding: 8px 12px;
    border-radius: 4px;
    border: 1px solid #ccc;
    user-select: all;  /* Makes text easy to select */
}

.copy-button {
    padding: 8px 16px;
    background: #1976d2;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
    white-space: nowrap;
    transition: background 0.2s;
}

.copy-button:hover {
    background: #1565c0;
}

.copy-button:disabled {
    background: #4caf50;
    cursor: not-allowed;
}

.instructions {
    margin-top: 20px;
    padding: 15px;
    background: #e3f2fd;
    border-left: 4px solid #1976d2;
    border-radius: 4px;
}

.instructions strong {
    color: #1976d2;
}

.instructions ol {
    margin: 10px 0 0 0;
    padding-left: 20px;
}

.instructions li {
    margin: 6px 0;
}
```

---

## Colab Template Integration

The Colab template (v3.4.0) now has:

### **Cell 0:** Introduction
Explains 3-step quick start with emphasis on pasting Gist ID in Cell 3

### **Cell 2:** Markdown Instructions
Clear heading: "STEP 1: Paste Your Gist ID"

### **Cell 3:** 📥 Gist ID Input Form (NEW)
```python
#@title 📥 **Paste Your Gist ID Here**
GIST_ID = ""  #@param {type:"string"}
```

- Prominent form with validation
- Clear error messages if empty or invalid format
- Success message with next steps
- Stores GIST_ID variable for Cell 7

### **Cell 7:** Model Loading
- Simplified to just use `GIST_ID` variable
- Clear error if Cell 3 wasn't run first
- Fetches model.py and config.json from Gist
- Comprehensive error messages for troubleshooting

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No Gist ID provided" | User didn't run Cell 3 | Clear message: "Go back to Cell 3" |
| "Invalid Gist ID format" | Malformed ID | Show expected format (alphanumeric) |
| "HTTP 404" | Gist not found | Double-check Gist ID, verify Gist is public |
| "HTTP 403 - Rate limit" | >60 requests/hour | Wait 1 hour or authenticate with GitHub |
| "Gist missing model.py" | Export incomplete | Re-export from Transformer Builder |

All errors include:
- Clear description of what went wrong
- Troubleshooting steps
- Link to Gist URL for manual verification

---

## Testing Checklist

### Before Releasing:

- [ ] **Gist Creation Works**
  - [ ] Creates public Gist
  - [ ] Contains model.py with valid Python code
  - [ ] Contains config.json with valid JSON
  - [ ] Gist ID is captured correctly

- [ ] **Modal UI Works**
  - [ ] Modal appears after Gist creation
  - [ ] Gist ID is displayed correctly
  - [ ] Copy button works (test in Chrome, Firefox, Safari)
  - [ ] "Open in Colab" button opens correct URL
  - [ ] Modal can be closed/cancelled

- [ ] **End-to-End Workflow**
  - [ ] Click "Export to Colab"
  - [ ] Copy Gist ID from modal
  - [ ] Click "Open in Colab"
  - [ ] Colab opens in new tab
  - [ ] Paste Gist ID in Cell 3
  - [ ] Run Cell 3 → Success message appears
  - [ ] Run all cells → Model loads successfully
  - [ ] Tests execute without errors

- [ ] **Error Cases**
  - [ ] Test with invalid Gist ID (shows error)
  - [ ] Test without running Cell 3 first (shows error)
  - [ ] Test with Gist missing files (shows error)
  - [ ] Test with private Gist (shows 404 error)

---

## Example Gist Structure

After export, the Gist should look like this:

**URL:** `https://gist.github.com/username/abc123def456`

**Files:**

**`model.py`:**
```python
"""
Generated model: CustomTransformer
Auto-generated by Transformer Builder.
"""

import torch
import torch.nn as nn

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # ... rest of model architecture

    def forward(self, input_ids):
        # ... forward pass
        return logits
```

**`config.json`:**
```json
{
  "vocab_size": 50257,
  "d_model": 512,
  "nhead": 8,
  "num_layers": 6,
  "model_name": "CustomTransformer"
}
```

---

## Implementation Timeline

**Estimated Time:** 2-3 hours

1. **Hour 1:** Implement Gist creation logic
   - Add GitHub Gist API integration
   - Generate model.py from canvas
   - Generate config.json from model parameters

2. **Hour 2:** Implement modal UI
   - Create modal component
   - Add copy functionality
   - Add "Open in Colab" button
   - Style modal

3. **Hour 3:** Testing
   - Test Gist creation
   - Test modal UI across browsers
   - Test end-to-end workflow
   - Fix any bugs

---

## API Reference

### GitHub Gist API

**Create Gist:**
```http
POST https://api.github.com/gists
Content-Type: application/json
Authorization: Bearer YOUR_GITHUB_TOKEN

{
  "description": "Model Name - Transformer Builder Export",
  "public": true,
  "files": {
    "model.py": {
      "content": "... Python code ..."
    },
    "config.json": {
      "content": "... JSON config ..."
    }
  }
}
```

**Response:**
```json
{
  "id": "abc123def456",
  "html_url": "https://gist.github.com/username/abc123def456",
  "files": { ... }
}
```

**Rate Limits:**
- Authenticated: 5,000 requests/hour
- Unauthenticated: 60 requests/hour

**Recommendation:** Use GitHub token authentication to avoid rate limits

---

## Support

If you encounter issues during implementation:

1. **Test Gist manually:** Visit the Gist URL and verify files exist
2. **Check Gist visibility:** Ensure Gist is public (not secret)
3. **Validate JSON:** Ensure config.json is valid JSON
4. **Test in Colab:** Manually paste Gist ID in Cell 3 to isolate issues

**Contact:** Reference this document and provide:
- Gist ID that's failing
- Error message from Colab
- Screenshots of modal UI

---

## Appendix: Alternative Approaches Considered

We evaluated several approaches before choosing the simple modal:

| Approach | Time | Pros | Cons | Decision |
|----------|------|------|------|----------|
| **Simple Modal** | 2-3 hrs | Simple, maintainable | One copy/paste | ✅ **CHOSEN** |
| Auto-injection | 9 hrs | Zero copy/paste | Complex, brittle | ❌ Rejected |
| URL parameters | 4 hrs | No modal needed | Colab strips params | ❌ Rejected |
| Metadata injection | 6 hrs | No URL tricks | Can't read metadata | ❌ Rejected |

The simple modal approach was chosen because:
- **10x faster implementation** (2-3 hours vs 9+ hours)
- **Zero maintenance burden** (no template syncing)
- **Crystal clear UX** (user knows exactly what they're doing)
- **One copy/paste is trivial** (5 seconds of user time)

---

**Ready to implement? Start with Section 1 (Gist Creation) and work through sequentially.**

**Questions?** Review the Testing Checklist and API Reference sections.
