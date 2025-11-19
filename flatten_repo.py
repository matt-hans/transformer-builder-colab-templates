#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
from datetime import datetime
import fnmatch

# Exact directory names to ignore
DEFAULT_IGNORES_EXACT = {
    # Version control
    ".git", ".hg", ".svn", ".bzr",
    # Python cache and build artifacts
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".coverage", "htmlcov", ".tox", ".eggs", "dist", "build",
    "site-packages", ".cache", ".local",
    # Archive directories
    "archive", "archives",
    # Jupyter
    ".ipynb_checkpoints", ".jupyter",
    # Node.js
    "node_modules", "bower_components",
    # IDE
    ".idea", ".vscode", ".vs", ".sublime-project", ".sublime-workspace",
    # OS files
    ".DS_Store", "Thumbs.db", ".directory",
    # Project-specific
    ".claude", ".playwright-mcp", ".tasks",
}

# Pattern-based ignores (supports wildcards)
DEFAULT_IGNORES_PATTERNS = [
    # Virtual environments (any variation - catch all venv patterns)
    "*venv*", "venv*", ".venv*", "env*", ".env*", "ENV*", ".ENV*",
    "virtualenv*", ".virtualenv*",
    # Also catch directories ending in env (but be careful not to match too broadly)
    "*_env", "*_ENV",
    # Python package artifacts
    "*.egg-info", "*.egg", "*.pyc", "*.pyo", "*.pyd", ".Python",
    # IDE patterns
    ".sublime-*", "*.swp", "*.swo", "*~",
    # Build artifacts
    "*.so", "*.dylib", "*.dll",
    # Coverage and testing
    ".coverage.*", ".pytest_cache", ".hypothesis",
    # Temporary files
    "*.tmp", "*.temp", "*.log",
    # Archive files
    "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2", "*.tbz2", "*.tar.xz", "*.txz",
    "*.gz", "*.bz2", "*.xz", "*.rar", "*.7z", "*.cab", "*.deb", "*.rpm",
    "*.dmg", "*.iso", "*.whl",
]

def should_ignore(name: str, exact_ignores: set = None, pattern_ignores: list = None) -> bool:
    """
    Check if a file/directory name should be ignored.
    Supports both exact matches and wildcard patterns.
    """
    if exact_ignores is None:
        exact_ignores = DEFAULT_IGNORES_EXACT
    if pattern_ignores is None:
        pattern_ignores = DEFAULT_IGNORES_PATTERNS
    
    # Check exact matches first (faster)
    if name in exact_ignores:
        return True
    
    # Check pattern matches
    for pattern in pattern_ignores:
        if fnmatch.fnmatch(name, pattern):
            return True
    
    return False

def is_binary_file(path: Path, blocksize: int = 1024) -> bool:
    """
    Rough heuristic: if there is a null byte in the first block, treat as binary.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(blocksize)
        if b"\0" in chunk:
            return True
        return False
    except Exception:
        # If we can't read it, just treat as binary and skip
        return True

def collect_files(root: Path, exact_ignores: set = None, pattern_ignores: list = None):
    """
    Walk the repo and yield (full_path, rel_path) for files we care about.
    """
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # Remove ignored directories in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if not should_ignore(d, exact_ignores, pattern_ignores)]

        for fname in filenames:
            if should_ignore(fname, exact_ignores, pattern_ignores):
                continue
            full_path = Path(dirpath) / fname
            rel_path = full_path.relative_to(root)
            yield full_path, rel_path

def generate_ascii_tree(root: Path, exact_ignores: set = None, pattern_ignores: list = None) -> str:
    """
    Generate an ASCII tree of the directory structure, similar to the `tree` command,
    skipping ignored directories.
    """
    root = root.resolve()
    lines = [root.name + "/"]

    def inner(dir_path: Path, prefix: str = ""):
        # List entries and sort: directories first, then files (both alphabetically)
        entries = sorted(
            dir_path.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )

        # Skip ignored directories and files
        entries = [
            e for e in entries
            if not should_ignore(e.name, exact_ignores, pattern_ignores)
        ]

        total = len(entries)
        for idx, entry in enumerate(entries):
            connector = "└── " if idx == total - 1 else "├── "
            line = prefix + connector + entry.name
            if entry.is_dir():
                line += "/"
            lines.append(line)

            if entry.is_dir():
                new_prefix = prefix + ("    " if idx == total - 1 else "│   ")
                inner(entry, new_prefix)

    inner(root)
    return "\n".join(lines)

def write_flat_file(repo_root: Path, output_path: Path, max_size_mb: float | None = None,
                    exact_ignores: set = None, pattern_ignores: list = None):
    if exact_ignores is None:
        exact_ignores = DEFAULT_IGNORES_EXACT
    if pattern_ignores is None:
        pattern_ignores = DEFAULT_IGNORES_PATTERNS
    
    files = list(collect_files(repo_root, exact_ignores, pattern_ignores))
    files.sort(key=lambda t: str(t[1]))  # sort by relative path

    max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb is not None else None

    with output_path.open("w", encoding="utf-8", errors="replace") as out:
        out.write(f"# Repository snapshot\n")
        out.write(f"# Root: {repo_root.resolve()}\n")
        out.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        out.write(f"# Total files considered: {len(files)}\n\n")

        # 1) ASCII directory tree
        out.write("## Directory Tree\n\n")
        out.write("```text\n")
        out.write(generate_ascii_tree(repo_root, exact_ignores, pattern_ignores))
        out.write("\n```\n\n")

        # 2) File-by-file contents
        out.write("## File Contents\n")

        for full_path, rel_path in files:
            # Size check
            if max_size_bytes is not None and full_path.stat().st_size > max_size_bytes:
                out.write("\n\n===== FILE SKIPPED (too large) =====\n")
                out.write(f"PATH: {rel_path}\n")
                out.write(f"SIZE: {full_path.stat().st_size} bytes\n")
                out.write(f"REASON: exceeds {max_size_mb} MB limit\n")
                continue

            # Binary check
            if is_binary_file(full_path):
                out.write("\n\n===== BINARY FILE SKIPPED =====\n")
                out.write(f"PATH: {rel_path}\n")
                continue

            out.write("\n\n")
            out.write("============================================================\n")
            out.write(f"FILE: {rel_path}\n")
            out.write("============================================================\n\n")

            try:
                with full_path.open("r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"<< Error reading file: {e} >>\n")

def main():
    parser = argparse.ArgumentParser(
        description="Flatten a git repo into a single text file (tree + contents)."
    )
    parser.add_argument("repo_root", help="Path to the root of the repo")
    parser.add_argument(
        "-o", "--output",
        help="Output text file (default: repo_snapshot.txt in repo root)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=5.0,
        help="Maximum file size in MB to include (default: 5 MB per file)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser()
    if not repo_root.is_dir():
        raise SystemExit(f"Repo root does not exist or is not a directory: {repo_root}")

    output_path = Path(args.output) if args.output else (repo_root / "repo_snapshot.txt")
    write_flat_file(repo_root, output_path, max_size_mb=args.max_size_mb)

    print(f"Done. Output written to: {output_path}")

if __name__ == "__main__":
    main()
