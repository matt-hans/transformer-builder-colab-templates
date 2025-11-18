#!/usr/bin/env python3
"""
Surgical fix for training.ipynb cells 21 and 22 formatting issue.

Root Cause: Cells have source stored improperly without line breaks.
Fix: Convert single-string source to proper Jupyter list-of-lines format.

SOLID Principles:
- Single Responsibility: Only fixes cell formatting
- Open/Closed: Can extend to fix other cells without modifying core logic
- Minimal Change: Only modifies cells 21 and 22, preserves all other data
"""

import json

def fix_cell_formatting(notebook_path):
    """Fix cells 21 and 22 by properly splitting source into lines."""

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells")
    fixed_count = 0

    for i, cell in enumerate(nb['cells']):
        cell_id = cell.get('metadata', {}).get('id')

        if cell_id not in ['cell-21', 'cell-22']:
            continue

        source = cell.get('source', [])

        if not isinstance(source, list) or len(source) == 0:
            print(f"⚠️  {cell_id}: invalid source, skipping")
            continue

        # Concatenate all source elements
        code_blob = ''.join(source)

        # Check if already properly formatted (has many \n-terminated lines)
        if len(source) > 20 and source[0].endswith('\n'):
            print(f"✅ {cell_id}: already formatted ({len(source)} lines)")
            continue

        print(f"\n🔧 Fixing {cell_id}")
        print(f"   Before: {len(source)} elements, {len(code_blob)} chars")

        # Split into lines (Jupyter format: each line ends with \n except last)
        if '\n' not in code_blob:
            # Single line, no fixing needed
            formatted_source = [code_blob]
        else:
            lines = code_blob.split('\n')
            formatted_source = []
            for j, line in enumerate(lines):
                if j < len(lines) - 1:
                    # All lines except last get \n appended
                    formatted_source.append(line + '\n')
                else:
                    # Last line: only add if non-empty
                    if line.strip():
                        formatted_source.append(line)

        # Update cell
        cell['source'] = formatted_source
        nb['cells'][i] = cell
        fixed_count += 1

        print(f"   After: {len(formatted_source)} lines")
        print(f"   First line: {formatted_source[0][:60]}...")
        if len(formatted_source) > 1:
            print(f"   Last line: {formatted_source[-1][:60]}...")

    # Save if we fixed anything
    if fixed_count > 0:
        # Backup original (load fresh from disk before modifications)
        backup_path = notebook_path + '.backup-cells-21-22'
        with open(notebook_path, 'r', encoding='utf-8') as f:
            original_nb = json.load(f)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(original_nb, f, indent=1)
        print(f"\n💾 Backup saved: {backup_path}")

        # Save fixed version
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"✅ SUCCESS: Fixed {fixed_count} cells in {notebook_path}")
        print("\n📋 Next Steps:")
        print("1. Open training.ipynb in Google Colab")
        print("2. Navigate to Section 5 (cells 21 and 22)")
        print("3. Verify cells display as properly formatted multi-line code")
        return True
    else:
        print("\n⚠️  No cells needed fixing")
        return False

if __name__ == '__main__':
    import sys
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'training.ipynb'
    fix_cell_formatting(notebook_path)
