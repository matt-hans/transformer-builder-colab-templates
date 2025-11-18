#!/usr/bin/env python3
"""
Fix training.ipynb formatting and reorganize sections.

Issues fixed:
1. Add missing newlines to cells 20-22 (Model loading cells)
2. Move model loading cells to Section 2 (after Drive setup)
3. Update Table of Contents
4. Ensure proper section ordering
"""

import json
import shutil
from datetime import datetime

def add_newlines_to_cell(cell):
    """Add newlines to cell source if missing."""
    if cell['cell_type'] != 'code':
        return cell

    source = cell.get('source', [])
    if not isinstance(source, list):
        return cell

    # Add newlines to all lines except the last one
    fixed_source = []
    for i, line in enumerate(source):
        if i < len(source) - 1:  # Not the last line
            if not line.endswith('\n'):
                fixed_source.append(line + '\n')
            else:
                fixed_source.append(line)
        else:  # Last line
            fixed_source.append(line)

    cell['source'] = fixed_source
    return cell

def create_section_header(section_num, title, anchor_id):
    """Create a markdown cell for section header."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            f'<a id="{anchor_id}"></a>\n',
            f'# {title}\n'
        ]
    }

def main():
    # Backup original file
    backup_path = f'training.ipynb.backup-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy('training.ipynb', backup_path)
    print(f"✅ Backup created: {backup_path}")

    # Load notebook
    with open('training.ipynb', 'r') as f:
        nb = json.load(f)

    print(f"📖 Loaded notebook with {len(nb['cells'])} cells")

    # Fix cells 20-22 (model loading cells)
    print("\n🔧 Fixing cell formatting...")
    cells_to_fix = [20, 21, 22]
    for idx in cells_to_fix:
        if idx < len(nb['cells']):
            before_lines = len(nb['cells'][idx]['source'])
            before_newlines = sum(1 for line in nb['cells'][idx]['source'] if line.endswith('\n'))

            nb['cells'][idx] = add_newlines_to_cell(nb['cells'][idx])

            after_newlines = sum(1 for line in nb['cells'][idx]['source'] if line.endswith('\n'))
            print(f"  Cell {idx}: {before_lines} lines, {before_newlines} → {after_newlines} newlines")

    # Reorganize: Move model loading cells to Section 2
    print("\n📦 Reorganizing sections...")
    print("  Moving model loading cells (20-22) to Section 2...")

    # Extract model loading cells
    model_loading_cells = [nb['cells'][i] for i in range(20, 23)]

    # Remove them from current position
    for _ in range(3):
        nb['cells'].pop(20)

    # Find insertion point (after Section 1 - Drive setup)
    # Section 1 ends after the experiment DB cell (around index 7)
    insert_idx = 8  # After cell id "37c65122" (ExperimentDB initialization)

    # Insert new section header
    section_2_header = create_section_header(
        2,
        "📦 Section 2: Model Loading",
        "section-2"
    )

    section_2_description = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            'Load your transformer model from Transformer Builder or use the example model.\n',
            '\n',
            '**Options:**\n',
            '- **Custom Model**: Provide Gist ID from Transformer Builder (auto-detected from URL)\n',
            '- **Example Model**: GPT-2 style architecture for testing\n',
            '\n',
            '**You will see:**\n',
            '1. Model code preview\n',
            '2. Architecture summary (layers, parameters, size)\n',
            '3. GPU compatibility check\n'
        ]
    }

    # Insert section header, description, and model cells
    nb['cells'].insert(insert_idx, section_2_header)
    nb['cells'].insert(insert_idx + 1, section_2_description)
    for i, cell in enumerate(model_loading_cells):
        nb['cells'].insert(insert_idx + 2 + i, cell)

    print(f"  ✅ Inserted model loading section at index {insert_idx}")

    # Update section markers in subsequent cells
    print("\n🔄 Updating section markers...")

    # Find and update data loading section (was section-2, now section-3)
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell['source'])

            if '<a id="section-2"></a>' in source_text and '📊 Section 2: Data Loading' in source_text:
                cell['source'] = [
                    '<a id="section-3"></a>\n',
                    '# 📊 Section 3: Data Loading\n',
                    '\n',
                    'Choose your data source (run ONE of the following cells):\n',
                    '- **Option 1**: HuggingFace Datasets (recommended)\n',
                    '- **Option 2**: Google Drive Upload\n',
                    '- **Option 3**: File Upload (small datasets)\n',
                    '- **Option 4**: Local Files (from previous sessions)\n',
                    '- **Option 5**: Synthetic Data (testing only)\n'
                ]
                print(f"  Updated data loading section at index {idx}")

            # Update section-3 to section-4 (Training Configuration)
            elif '<a id="section-3"></a>' in source_text:
                source_text = source_text.replace('section-3', 'section-4')
                source_text = source_text.replace('Section 3:', 'Section 4:')
                cell['source'] = source_text.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                print(f"  Updated section-3 → section-4 at index {idx}")

            # Update section-4 to section-5 (W&B)
            elif '<a id="section-4"></a>' in source_text:
                source_text = source_text.replace('section-4', 'section-5')
                source_text = source_text.replace('Section 4:', 'Section 5:')
                cell['source'] = source_text.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                print(f"  Updated section-4 → section-5 at index {idx}")

            # Update section-5 to section-6 (Training Loop)
            elif '<a id="section-5"></a>' in source_text:
                source_text = source_text.replace('section-5', 'section-6')
                source_text = source_text.replace('Section 5:', 'Section 6:')
                cell['source'] = source_text.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                print(f"  Updated section-5 → section-6 at index {idx}")

            # Update remaining sections (6→7, 7→8, 8→9)
            elif '<a id="section-6"></a>' in source_text:
                source_text = source_text.replace('section-6', 'section-7')
                source_text = source_text.replace('Section 6:', 'Section 7:')
                cell['source'] = source_text.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                print(f"  Updated section-6 → section-7 at index {idx}")
            elif '<a id="section-7"></a>' in source_text:
                source_text = source_text.replace('section-7', 'section-8')
                source_text = source_text.replace('Section 7:', 'Section 8:')
                cell['source'] = source_text.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                print(f"  Updated section-7 → section-8 at index {idx}")
            elif '<a id="section-8"></a>' in source_text:
                source_text = source_text.replace('section-8', 'section-9')
                source_text = source_text.replace('Section 8:', 'Section 9:')
                cell['source'] = source_text.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                print(f"  Updated section-8 → section-9 at index {idx}")

    # Update Table of Contents
    print("\n📋 Updating Table of Contents...")
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell['source'])
            if '## 📋 Table of Contents' in source_text:
                cell['source'] = [
                    '## 📋 Table of Contents\n',
                    '\n',
                    '1. [Section 0: Quick Start](#section-0) ← You are here\n',
                    '2. [Section 1: Setup & Drive Workspace](#section-1) (2 min)\n',
                    '3. [Section 2: Model Loading](#section-2) (Load custom or example model)\n',
                    '4. [Section 3: Data Loading](#section-3) (5 sources)\n',
                    '5. [Section 4: Training Configuration](#section-4) (Hyperparameters)\n',
                    '6. [Section 5: W&B Tracking Setup](#section-5) (Optional)\n',
                    '7. [Section 6: Training Loop](#section-6) (Main training)\n',
                    '8. [Section 7: Analysis & Visualization](#section-7) (Dashboards)\n',
                    '9. [Section 8: Export & Results](#section-8) (Download checkpoints)\n',
                    '10. [Section 9: Advanced Features](#section-9) (Hyperparameter search)\n',
                    '\n',
                    '⏱️ **Total Time**: ~20-60 minutes depending on mode\n'
                ]
                print(f"  Updated TOC at index {idx}")
                break

    # Save fixed notebook
    with open('training.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\n✅ Fixed notebook saved with {len(nb['cells'])} cells")
    print(f"\n📊 New structure:")
    print(f"  Section 0: Quick Start")
    print(f"  Section 1: Setup & Drive Workspace (cells 1-7)")
    print(f"  Section 2: Model Loading (cells 8-12) ← MOVED HERE")
    print(f"  Section 3: Data Loading")
    print(f"  Section 4: Training Configuration")
    print(f"  Section 5: W&B Tracking")
    print(f"  Section 6: Training Loop (actual training code)")
    print(f"  Section 7: Analysis & Visualization")
    print(f"  Section 8: Export & Results")
    print(f"  Section 9: Advanced Features")

if __name__ == '__main__':
    main()
