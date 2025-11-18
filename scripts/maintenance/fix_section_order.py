#!/usr/bin/env python3
"""Fix section order: Move Model Loading (Section 2) before Data Loading (Section 3)."""

import json

with open('training.ipynb', 'r') as f:
    nb = json.load(f)

print("🔄 Fixing section order...")

# Find section boundaries
section_2_start = None  # Model Loading
section_2_end = None
section_3_start = None  # Data Loading
section_3_end = None

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        if 'Section 2: Model Loading' in source_text:
            section_2_start = idx
        elif 'Section 3: Data Loading' in source_text:
            section_3_start = idx
            if section_2_start is not None:
                section_2_end = idx  # Section 2 ends where Section 3 starts
        elif 'Section 4: Training Configuration' in source_text:
            section_3_end = idx  # Section 3 ends where Section 4 starts
            break

print(f"  Section 2 (Model Loading): cells {section_2_start} to {section_2_end}")
print(f"  Section 3 (Data Loading): cells {section_3_start} to {section_3_end}")

if section_2_start > section_3_start:
    print(f"\n  ❌ Wrong order detected - swapping sections...")

    # Extract Section 3 cells (Data Loading)
    section_3_cells = nb['cells'][section_3_start:section_3_end]

    # Extract Section 2 cells (Model Loading)
    section_2_cells = nb['cells'][section_2_start:section_2_end]

    # Remove both sections
    # Remove Section 2 first (it's later)
    del nb['cells'][section_2_start:section_2_end]

    # Now remove Section 3 (indices shifted after removing Section 2)
    del nb['cells'][section_3_start:section_3_start + len(section_3_cells)]

    # Insert Section 2 first (at position where Section 3 was)
    for i, cell in enumerate(section_2_cells):
        nb['cells'].insert(section_3_start + i, cell)

    # Then insert Section 3 after Section 2
    insert_pos = section_3_start + len(section_2_cells)
    for i, cell in enumerate(section_3_cells):
        nb['cells'].insert(insert_pos + i, cell)

    print(f"  ✅ Sections swapped!")
    print(f"     Section 2 (Model Loading) now at: cells {section_3_start} onwards")
    print(f"     Section 3 (Data Loading) now at: cells {insert_pos} onwards")
else:
    print(f"  ✅ Sections already in correct order!")

# Save
with open('training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Notebook saved with correct section order")
