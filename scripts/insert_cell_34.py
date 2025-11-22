#!/usr/bin/env python3
"""Insert new Cell 34 (Load Model Weights) into training.ipynb."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

# Read notebook
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

# Create new Cell 34 structure (placeholder content)
new_cell_34 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "cellView": "form"
    },
    "outputs": [],
    "source": [
        "# @title 🔧 Optional: Load Model Weights from Checkpoint { display-mode: \"form\" }\n",
        "\n",
        "# PLACEHOLDER - Will be populated in next task\n",
        "print(\"Cell 34 placeholder\")\n"
    ]
}

# Insert at index 34 (between Cell 33 and current Cell 34)
nb['cells'].insert(34, new_cell_34)

# Save notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✅ Inserted new Cell 34 at index 34")
print(f"   Total cells: {len(nb['cells'])}")
print(f"   Cell 33: {nb['cells'][33]['source'][0][:60]}")
print(f"   Cell 34 (NEW): {nb['cells'][34]['source'][0][:60]}")
print(f"   Cell 35 (was 34): {nb['cells'][35]['source'][0][:60]}")
