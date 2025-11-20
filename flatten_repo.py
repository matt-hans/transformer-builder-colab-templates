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

def collect_files(root: Path, exact_ignores: set = None, pattern_ignores: list = None,
                  include_paths: list = None, exclude_paths: list = None):
    """
    Walk the repo and yield (full_path, rel_path) for files we care about.
    
    Args:
        root: Root directory to walk
        exact_ignores: Set of exact directory/file names to ignore
        pattern_ignores: List of wildcard patterns to ignore
        include_paths: List of relative paths (from root) to include. If None, include all.
        exclude_paths: List of relative paths (from root) to exclude. Applied after include_paths.
    """
    root = root.resolve()
    
    # Convert include/exclude paths to Path objects relative to root
    include_paths_set = None
    if include_paths:
        include_paths_set = {root / Path(p).as_posix() for p in include_paths}
    
    exclude_paths_set = None
    if exclude_paths:
        exclude_paths_set = {root / Path(p).as_posix() for p in exclude_paths}
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_path = Path(dirpath)
        
        # Check if this directory should be included/excluded
        if include_paths_set:
            # Check if any include path is a parent of this directory or this directory itself
            if not any(inc_path == dirpath_path or inc_path in dirpath_path.parents for inc_path in include_paths_set):
                # Also check if this directory is a parent of any include path
                if not any(dirpath_path in inc_path.parents or dirpath_path == inc_path for inc_path in include_paths_set):
                    continue
        
        if exclude_paths_set:
            # Skip if this directory or any parent is in exclude_paths
            if any(exc_path == dirpath_path or exc_path in dirpath_path.parents for exc_path in exclude_paths_set):
                continue
        
        # Remove ignored directories in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if not should_ignore(d, exact_ignores, pattern_ignores)]

        for fname in filenames:
            if should_ignore(fname, exact_ignores, pattern_ignores):
                continue
            
            full_path = Path(dirpath) / fname
            
            # Check if this file should be included/excluded
            if include_paths_set:
                if not any(inc_path == full_path or inc_path in full_path.parents for inc_path in include_paths_set):
                    continue
            
            if exclude_paths_set:
                if any(exc_path == full_path or exc_path in full_path.parents for exc_path in exclude_paths_set):
                    continue
            
            rel_path = full_path.relative_to(root)
            yield full_path, rel_path

def generate_tree_from_files(root: Path, files: list) -> str:
    """
    Generate an ASCII tree from a list of collected files.
    This ensures the tree matches exactly what's included in the snapshot.
    """
    root = root.resolve()
    lines = [root.name + "/"]
    
    # Build a set of all directories and files that exist in our file list
    dirs = set()
    file_set = set()
    for full_path, rel_path in files:
        file_set.add(rel_path)
        # Add all parent directories
        parts = rel_path.parts
        for i in range(len(parts)):
            dirs.add(Path(*parts[:i]))
    
    def inner(current_rel: Path, prefix: str = ""):
        # Find all children (files and dirs) at this level
        children = {}
        for rel_path in file_set | dirs:
            if rel_path == current_rel:
                continue
            if rel_path.parent == current_rel:
                children[rel_path.name] = (rel_path, rel_path in file_set)
        
        if not children:
            return
        
        # Sort: directories first, then files (both alphabetically)
        sorted_children = sorted(
            children.items(),
            key=lambda x: (not x[1][1], x[0].lower())  # dirs first (not is_file), then alphabetically
        )
        
        total = len(sorted_children)
        for idx, (name, (rel_path, is_file)) in enumerate(sorted_children):
            connector = "└── " if idx == total - 1 else "├── "
            line = prefix + connector + name
            if not is_file:
                line += "/"
            lines.append(line)
            
            if not is_file:
                new_prefix = prefix + ("    " if idx == total - 1 else "│   ")
                inner(rel_path, new_prefix)
    
    inner(Path("."))
    return "\n".join(lines)

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
                    exact_ignores: set = None, pattern_ignores: list = None,
                    include_paths: list = None, exclude_paths: list = None):
    if exact_ignores is None:
        exact_ignores = DEFAULT_IGNORES_EXACT
    if pattern_ignores is None:
        pattern_ignores = DEFAULT_IGNORES_PATTERNS
    
    files = list(collect_files(repo_root, exact_ignores, pattern_ignores, include_paths, exclude_paths))
    files.sort(key=lambda t: str(t[1]))  # sort by relative path

    max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb is not None else None

    with output_path.open("w", encoding="utf-8", errors="replace") as out:
        out.write(f"# Repository snapshot\n")
        out.write(f"# Root: {repo_root.resolve()}\n")
        out.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        if include_paths:
            out.write(f"# Included paths: {', '.join(include_paths)}\n")
        if exclude_paths:
            out.write(f"# Excluded paths: {', '.join(exclude_paths)}\n")
        out.write(f"# Total files considered: {len(files)}\n\n")

        # 1) ASCII directory tree (generate from collected files for accuracy)
        out.write("## Directory Tree\n\n")
        out.write("```text\n")
        # Generate tree from the actual files we collected
        tree_lines = generate_tree_from_files(repo_root, files)
        out.write(tree_lines)
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
    parser.add_argument(
        "--include",
        nargs="+",
        help="Specific paths/directories to include (relative to repo_root). Can specify multiple.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Specific paths/directories to exclude (relative to repo_root). Can specify multiple.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser()
    if not repo_root.is_dir():
        raise SystemExit(f"Repo root does not exist or is not a directory: {repo_root}")

    output_path = Path(args.output) if args.output else (repo_root / "repo_snapshot.txt")
    write_flat_file(
        repo_root, 
        output_path, 
        max_size_mb=args.max_size_mb,
        include_paths=args.include,
        exclude_paths=args.exclude
    )

    print(f"Done. Output written to: {output_path}")

if __name__ == "__main__":
    main()
