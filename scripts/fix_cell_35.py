#!/usr/bin/env python3
"""Fix Cell 35 to always show output."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')
CELL_INDEX = 35

# Fixed Cell 35 source - always prints status
FIXED_SOURCE = [
    "# @title 🔧 Extract Session Variables { display-mode: \"form\" }\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(\"EXTRACT SESSION VARIABLES\")\n",
    "print(\"=\" * 70)\n",
    "print()\n",
    "\n",
    "# Extract workspace_root, run_name, metrics_df from results dict\n",
    "# Works for both training (Cell 32) and recovery (Cell 33) workflows\n",
    "\n",
    "if 'results' in globals():\n",
    "    # Extract workspace_root (for file paths)\n",
    "    workspace_root = results.get('workspace_root', './workspace')\n",
    "    print(f\"✅ workspace_root: {workspace_root}\")\n",
    "    \n",
    "    # Extract run_name (for file naming)\n",
    "    run_name = results.get('run_name', 'training_run')\n",
    "    print(f\"✅ run_name: {run_name}\")\n",
    "    \n",
    "    # Extract metrics_df (for analysis)\n",
    "    metrics_df = results.get('metrics_summary')\n",
    "    if metrics_df is not None:\n",
    "        print(f\"✅ metrics_df: {len(metrics_df)} epochs\")\n",
    "    else:\n",
    "        print(\"⚠️  metrics_df: Not available\")\n",
    "    \n",
    "    print()\n",
    "    print(\"💡 Variables extracted and ready to use\")\n",
    "    \n",
    "else:\n",
    "    print(\"⚠️  'results' not found\")\n",
    "    print()\n",
    "    print(\"💡 To fix:\")\n",
    "    print(\"   1. Run Cell 32 (Training) OR Cell 33 (Recovery) first\")\n",
    "    print(\"   2. Re-run this cell\")\n",
    "\n",
    "print()\n",
    "print(\"=\" * 70)\n",
]

# Update Cell 35
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

nb['cells'][CELL_INDEX]['source'] = FIXED_SOURCE

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✅ Fixed Cell {CELL_INDEX}")
print(f"   Now always prints extraction status")
