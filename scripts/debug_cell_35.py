#!/usr/bin/env python3
"""Debug why Cell 35 doesn't produce output."""

import json

nb = json.load(open('training.ipynb'))
cell_35 = nb['cells'][35]

print("Cell 35 logic flow:")
print("1. if 'results' in globals() → checks for results dict")
print("2.   if 'workspace_root' not in globals() → only prints if missing")
print("3.   if 'run_name' not in globals() → only prints if missing")
print("4.   if 'metrics_df' not in globals() → only prints if missing")
print()
print("Issue: If all variables already exist, nothing prints!")
print("Solution: Always print variable values, not just when creating them.")
