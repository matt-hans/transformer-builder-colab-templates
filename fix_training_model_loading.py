#!/usr/bin/env python3
"""
Fix training.ipynb model loading to use GitHub API (like template.ipynb).

PROBLEM: training.ipynb uses raw URLs which fail
SOLUTION: Use GitHub API like template.ipynb
"""

import json
import shutil
from datetime import datetime

# Backup
backup = f'training.ipynb.backup-model-loading-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy('training.ipynb', backup)
print(f"✅ Backup: {backup}")

with open('training.ipynb', 'r') as f:
    nb = json.load(f)

print(f"\n📖 Current training.ipynb: {len(nb['cells'])} cells")

# Find the model loading cells (should be around index 10-11)
model_source_idx = None
load_model_idx = None
init_model_idx = None

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        if 'Model Source Configuration' in source_text:
            model_source_idx = idx
        elif 'Load Model from Gist' in source_text:
            load_model_idx = idx
        elif 'Initialize Model' in source_text and 'Detect device' in source_text:
            init_model_idx = idx

print(f"\n🔍 Found model loading cells:")
print(f"  Cell {model_source_idx}: Model Source Configuration")
print(f"  Cell {load_model_idx}: Load Model from Gist (BROKEN - needs fixing)")
print(f"  Cell {init_model_idx}: Initialize Model")

# ==============================================================================
# WORKING CODE FROM TEMPLATE.IPYNB (GitHub API approach)
# ==============================================================================

working_load_model_cell = """# @title 📦 Load Model from Gist { display-mode: "form" }

import urllib.request
import json
import sys
import tempfile
import shutil

print("=" * 70)
print("MODEL LOADING")
print("=" * 70)
print()

# ==============================================================================
# VERIFY GIST ID WAS PROVIDED
# ==============================================================================

if 'gist_id' not in globals() or not gist_id:
    print("❌ ERROR: No Gist ID found!")
    print()
    print("==" * 35)
    print("🔙 GO BACK TO PREVIOUS CELL")
    print("==" * 35)
    print()
    print("You must run the Model Source Configuration cell first.")
    print()
    raise ValueError("Gist ID required - run previous cell first")

print(f"📥 Loading model from GitHub Gist: {gist_id}")
print()

# ==============================================================================
# FETCH GIST AND LOAD MODEL FILES - GitHub API Approach
# ==============================================================================

def _fetch_gist(gid: str) -> dict:
    \"\"\"Fetch Gist data from GitHub API.\"\"\"
    url = f"https://api.github.com/gists/{gid}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "transformer-builder-colab"
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = f"HTTP {e.code}"
        try:
            body = e.read().decode("utf-8")
            if "rate limit" in body.lower():
                detail += " - GitHub API rate limit (try again in an hour)"
            elif e.code == 404:
                detail += " - Gist not found (check your Gist ID)"
        except:
            pass
        raise RuntimeError(f"GitHub API error: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"Network error: {e}") from e

def _write(path: str, text: str):
    \"\"\"Write text to file.\"\"\"
    with open(path, "w") as f:
        f.write(text)

# Fetch Gist
try:
    gist_data = _fetch_gist(gist_id)
    files = gist_data.get("files") or {}

    # Check for required files
    if "model.py" not in files:
        raise RuntimeError("Gist is missing 'model.py' - please re-export from Transformer Builder")
    if "config.json" not in files:
        raise RuntimeError("Gist is missing 'config.json' - please re-export from Transformer Builder")

    model_code = files["model.py"].get("content", "")
    config_json = files["config.json"].get("content", "")

    if not model_code or not config_json:
        raise RuntimeError("Empty content in model.py or config.json")

    # Write to files
    _write("model.py", model_code)
    _write("config.json", config_json)

    print(f"✅ Model loaded successfully!")
    print(f"✅ Gist URL: {gist_data.get('html_url', 'N/A')}")
    print(f"✅ Model code: {len(model_code):,} bytes")
    print(f"✅ Config: {len(config_json):,} bytes")
    print()

    # Parse model name from config if available
    try:
        model_config = json.loads(config_json)
        if 'model_name' in model_config:
            model_name = model_config['model_name']
            print(f"✅ Model name: {model_name}")
        else:
            model_name = 'CustomTransformer'
            print(f"ℹ️  Using default name: {model_name}")
        print()
    except:
        model_name = 'CustomTransformer'
        print(f"⚠️  Could not parse config, using default name: {model_name}")

    # Store for next cell
    gist_loaded = True

except Exception as e:
    print(f"❌ Failed to load model from Gist!")
    print()
    print(f"Error: {e}")
    print()
    print("=" * 70)
    print("TROUBLESHOOTING")
    print("=" * 70)
    print()
    print("Common issues:")
    print("  1. Check your Gist ID is correct (go back to previous cell)")
    print("  2. Ensure you exported from Transformer Builder successfully")
    print("  3. Check you're not hitting GitHub rate limit (60 requests/hour)")
    print("  4. Try re-exporting from Transformer Builder")
    print()
    print("If the problem persists:")
    print(f"  • Gist URL: https://gist.github.com/{gist_id}")
    print("  • Verify the Gist contains model.py and config.json")
    print()

    # Fallback to example model
    print("⚠️  Falling back to example model for demonstration...")
    gist_loaded = False
    model_name = 'ExampleTransformer'

print("=" * 70)
print("✅ MODEL LOADING COMPLETE")
print("=" * 70)
print()
print("Model will be instantiated in the next cell.")
print()

# Display downloaded model code preview
if gist_loaded:
    print("\\n📄 Model Code Preview:")
    print("=" * 60)
    with open('model.py', 'r') as f:
        model_lines = f.read().split('\\n')
        # Show first 20 lines
        for i, line in enumerate(model_lines[:20], 1):
            print(f"{i:3d} | {line}")
        if len(model_lines) > 20:
            print(f"... ({len(model_lines) - 20} more lines)")
    print("=" * 60)

print(f"\\n📊 Model: {model_name}")
if gist_loaded:
    print(f"   Config: {json.dumps(model_config, indent=2)}")
"""

# Convert to list with newlines
working_cell_lines = [line + '\n' for line in working_load_model_cell.split('\n')]
working_cell_lines[-1] = working_cell_lines[-1].rstrip('\n')  # Last line no newline

# Replace the broken cell
if load_model_idx is not None:
    nb['cells'][load_model_idx]['source'] = working_cell_lines
    print(f"\n✅ Replaced cell {load_model_idx} with working GitHub API code")
else:
    print(f"\n❌ Could not find 'Load Model from Gist' cell!")
    exit(1)

# Save
with open('training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ training.ipynb updated successfully!")
print(f"\n📊 Changes:")
print(f"  - Cell {load_model_idx}: Now uses GitHub API (like template.ipynb)")
print(f"  - Fetches from: https://api.github.com/gists/{{gist_id}}")
print(f"  - Extracts files from JSON response")
print(f"  - More reliable than raw URL approach")

