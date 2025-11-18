#!/usr/bin/env python3
"""
EMERGENCY FIX: training.ipynb has catastrophic character-level corruption.

Root Cause: Cells 20, 21, 22 have source stored as individual characters
instead of lines. Example: ['#', ' ', '@', 't', 'i', ...] instead of
['# @title\\n', 'import urllib\\n', ...]

This fix:
1. Joins characters back into a string
2. Splits on actual newlines
3. Reformats as proper Jupyter notebook lines
"""

import json

def fix_character_corruption(notebook_path):
    """Fix cells with character-level source corruption."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells\n")
    fixed_count = 0

    for i, cell in enumerate(nb['cells']):
        source = cell.get('source', [])

        # Detect corruption: source is a string instead of list, OR list with single chars
        is_string_corruption = isinstance(source, str) and len(source) > 100
        is_char_corrupted = (isinstance(source, list) and len(source) > 100 and
                            all(len(elem) == 1 for elem in source[:10]))

        if is_string_corruption or is_char_corrupted:
            if is_string_corruption:
                print(f"🔧 Fixing cell {i} (string corruption - source is str not list)")
                print(f"   Before: {len(source)} character string")
                code_blob = source  # Already a string
                print(f"   Source type: string ({len(code_blob)} chars)")
            else:  # is_char_corrupted
                print(f"🔧 Fixing cell {i} (character-level list corruption)")
                print(f"   Before: {len(source)} individual character elements")
                code_blob = ''.join(source)  # Join characters
                print(f"   Reconstructed: {len(code_blob)} chars total")

            # Split into proper lines (runs for BOTH corruption types)
            lines = code_blob.split('\n')
            formatted_source = []
            for j, line in enumerate(lines):
                if j < len(lines) - 1:
                    formatted_source.append(line + '\n')
                else:
                    if line.strip():  # Only add non-empty last line
                        formatted_source.append(line)

            # Update cell
            cell['source'] = formatted_source
            nb['cells'][i] = cell
            fixed_count += 1

            print(f"   After: {len(formatted_source)} proper lines")
            print(f"   First line: {formatted_source[0][:70]}...")
            print()

    if fixed_count > 0:
        # Backup
        backup_path = notebook_path + '.backup-char-corruption'
        with open(notebook_path, 'r', encoding='utf-8') as f:
            original_nb = json.load(f)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(original_nb, f, indent=1)
        print(f"💾 Backup saved: {backup_path}\n")

        # Save fixed version
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"✅ SUCCESS: Fixed {fixed_count} cells with character corruption")
        print("\n📋 Next Steps:")
        print("1. Reload training.ipynb in Colab")
        print("2. Cells should now display as properly formatted code")
        print("3. Commit to GitHub to make fix permanent")
        return True
    else:
        print("⚠️  No character-level corruption detected")
        return False

if __name__ == '__main__':
    import sys
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'training.ipynb'
    fix_character_corruption(notebook_path)
