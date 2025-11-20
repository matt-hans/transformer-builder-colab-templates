#!/usr/bin/env python3
"""
Requirements Sync Validation Script

Validates that the three requirements files remain in sync:
1. requirements.txt - Local development with exact pins
2. requirements-training.txt - Training notebook exact pins
3. requirements-colab-v3.4.0.txt - Colab with range pins

Checks:
- training.txt ⊆ colab.txt (training section only)
- All imports in utils/ are declared in requirements
- Version compatibility (exact pins vs range pins)
- No missing dependencies

Exit codes:
- 0: All checks pass
- 1: Validation failures detected
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from packaging import version as pkg_version
from packaging.specifiers import SpecifierSet


# Python standard library modules (exempt from requirements check)
STDLIB_MODULES = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
    'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'functools',
    'gc', 'glob', 'hashlib', 'heapq', 'importlib', 'inspect', 'io', 'itertools',
    'json', 'logging', 'math', 'multiprocessing', 'operator', 'os', 'pathlib',
    'pickle', 'platform', 'pprint', 'queue', 'random', 're', 'shutil', 'socket',
    'sqlite3', 'statistics', 'string', 'struct', 'subprocess', 'sys', 'tempfile',
    'threading', 'time', 'traceback', 'types', 'typing', 'unittest', 'urllib',
    'uuid', 'warnings', 'weakref', 'xml', '__future__',
}

# Internal project modules (exempt from requirements check)
INTERNAL_MODULES = {
    'utils', 'training', 'tokenization', 'ui', 'adapters', 'cli',
    'test_functions', 'tier1_critical_validation', 'tier2_advanced_analysis',
    'tier3_training_utilities', 'model_helpers', 'wandb_helpers',
    'amp_utils', 'benchmark_utils', 'checkpoint_manager', 'early_stopping',
    'environment_snapshot', 'eval_config', 'eval_runner', 'experiment_db',
    'export_utilities', 'metrics_tracker', 'metrics_utils', 'seed_manager',
    'task_spec', 'training_config', 'training_core', 'drift_metrics',
    'hf_hub', 'regression_testing', 'bpe_trainer', 'character_tokenizer',
    'validator', 'adaptive_tokenizer', 'data_collator', 'data_module',
    'dataset_utilities', 'model_adapter', 'gist_loader', 'presets',
    'setup_wizard', 'resume_utils', 'amp_benchmark', 'live_plotting',
    'sweep_runner', 'tier4_export_validation', 'tier5_monitoring',
    'dashboard',
}

# Intentionally omitted packages (documented in CLAUDE.md)
INTENTIONALLY_OMITTED = {
    'datasets',  # Tests use synthetic data generation
    'huggingface-hub',  # Models loaded from Gist, not Hub
    'huggingface_hub',  # Alternative import name
}

# Optional dependencies (may not be installed in all environments)
OPTIONAL_PACKAGES = {
    'google',  # Google Colab specific, only available in Colab
    'ipython',  # Jupyter/IPython specific (included in jupyter package)
    'pynvml',  # Optional GPU monitoring
    'onnx',  # Optional model export
    'onnxruntime',  # Optional model export
    'psutil',  # Optional system monitoring
    'captum',  # Optional feature attribution (Tier 2 tests)
    'requests',  # Optional for Gist loader (has fallback)
    'pytorch_lightning',  # Training-specific (in requirements-training.txt)
}

# Package aliases (import name -> package name)
PACKAGE_ALIASES = {
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'IPython': 'ipython',  # Provided by jupyter/ipykernel packages
}


@dataclass
class PackageSpec:
    """Represents a package with version specification."""
    name: str
    version: Optional[str] = None  # e.g., "2.0.0"
    operator: Optional[str] = None  # e.g., "==", ">=", "<"

    @property
    def specifier(self) -> Optional[str]:
        """Returns full specifier string (e.g., '>=2.0.0')."""
        if self.operator and self.version:
            return f"{self.operator}{self.version}"
        return None

    def __str__(self) -> str:
        if self.specifier:
            return f"{self.name}{self.specifier}"
        return self.name


class RequirementsParser:
    """Parses requirements.txt files and extracts package specifications."""

    # Regex for parsing requirement lines: package==1.2.3 or package>=1.0.0
    REQUIREMENT_PATTERN = re.compile(
        r'^([a-zA-Z0-9_-]+)\s*([><=!~]+)?\s*([0-9.]+)?'
    )

    @classmethod
    def parse_file(cls, filepath: Path) -> Dict[str, PackageSpec]:
        """
        Parse a requirements file and return dict of package specs.

        Args:
            filepath: Path to requirements file

        Returns:
            Dict mapping package name to PackageSpec
        """
        packages = {}

        if not filepath.exists():
            return packages

        for line in filepath.read_text().splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse package specification
            match = cls.REQUIREMENT_PATTERN.match(line)
            if match:
                name, operator, ver = match.groups()
                name = name.lower()  # Normalize package names
                packages[name] = PackageSpec(
                    name=name,
                    version=ver,
                    operator=operator
                )

        return packages

    @classmethod
    def parse_colab_training_section(cls, filepath: Path) -> Dict[str, PackageSpec]:
        """
        Parse only the training section of requirements-colab file.

        This extracts packages between:
        # TRAINING.IPYNB - AUTOMATIC INSTALLATION
        and the next section marker.

        Args:
            filepath: Path to requirements-colab file

        Returns:
            Dict mapping package name to PackageSpec
        """
        packages = {}
        in_training_section = False

        if not filepath.exists():
            return packages

        for line in filepath.read_text().splitlines():
            line = line.strip()

            # Start of training section
            if 'TRAINING.IPYNB - AUTOMATIC INSTALLATION' in line:
                in_training_section = True
                continue

            # End of training section (next section marker)
            if in_training_section and line.startswith('# ===='):
                # Check if this is a new section (not continuation)
                if 'DEVELOPMENT' in line or 'VERSION' in line:
                    break

            # Parse packages in training section
            if in_training_section and line and not line.startswith('#'):
                match = cls.REQUIREMENT_PATTERN.match(line)
                if match:
                    name, operator, ver = match.groups()
                    name = name.lower()
                    packages[name] = PackageSpec(
                        name=name,
                        version=ver,
                        operator=operator
                    )

        return packages


class ImportScanner:
    """Scans Python files for import statements."""

    @classmethod
    def extract_imports(cls, filepath: Path) -> Set[str]:
        """
        Extract all top-level imports from a Python file.

        Args:
            filepath: Path to Python file

        Returns:
            Set of imported module names (top-level only)
        """
        imports = set()

        try:
            content = filepath.read_text()
            tree = ast.parse(content, filename=str(filepath))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Extract top-level module (e.g., 'torch' from 'torch.nn')
                        top_level = alias.name.split('.')[0]
                        imports.add(top_level)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Extract top-level module
                        top_level = node.module.split('.')[0]
                        imports.add(top_level)

        except SyntaxError as e:
            print(f"⚠️  Syntax error in {filepath}: {e}")

        return imports

    @classmethod
    def scan_directory(cls, dirpath: Path) -> Set[str]:
        """
        Recursively scan directory for all imports.

        Args:
            dirpath: Directory to scan

        Returns:
            Set of all unique imported module names
        """
        all_imports = set()

        for pyfile in dirpath.rglob('*.py'):
            imports = cls.extract_imports(pyfile)
            all_imports.update(imports)

        return all_imports


class VersionValidator:
    """Validates version compatibility between requirements files."""

    @staticmethod
    def is_compatible(exact_spec: PackageSpec, range_spec: PackageSpec) -> bool:
        """
        Check if exact version pin satisfies range specifier.

        Args:
            exact_spec: Package with exact pin (e.g., torch==2.0.0)
            range_spec: Package with range pin (e.g., torch>=1.9.0)

        Returns:
            True if exact version satisfies range, False otherwise
        """
        if not exact_spec.version or not range_spec.specifier:
            return True  # Cannot validate without versions

        try:
            exact_ver = pkg_version.parse(exact_spec.version)
            specifier_set = SpecifierSet(range_spec.specifier)
            return exact_ver in specifier_set
        except Exception as e:
            print(f"⚠️  Version parsing error for {exact_spec.name}: {e}")
            return True  # Assume compatible on error

    @staticmethod
    def format_version_fix(package_name: str, exact_ver: str, range_spec: str) -> str:
        """
        Generate fix command for version mismatch.

        Args:
            package_name: Name of package
            exact_ver: Current exact version
            range_spec: Required range specifier

        Returns:
            Shell command to fix version mismatch
        """
        return (
            f"sed -i 's/{package_name}=={exact_ver}/{package_name}{range_spec}/' "
            f"requirements.txt"
        )


class RequirementsSyncValidator:
    """Main validation orchestrator."""

    def __init__(self, repo_root: Path):
        """
        Initialize validator.

        Args:
            repo_root: Root directory of repository
        """
        self.repo_root = repo_root
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all checks pass, False otherwise
        """
        print("🔍 Starting requirements sync validation...\n")

        # Parse requirements files
        req_file = self.repo_root / 'requirements.txt'
        train_file = self.repo_root / 'requirements-training.txt'
        colab_file = self.repo_root / 'requirements-colab-v3.4.0.txt'

        requirements = RequirementsParser.parse_file(req_file)
        training = RequirementsParser.parse_file(train_file)
        colab_training = RequirementsParser.parse_colab_training_section(colab_file)

        print(f"📦 Parsed {len(requirements)} packages from requirements.txt")
        print(f"📦 Parsed {len(training)} packages from requirements-training.txt")
        print(f"📦 Parsed {len(colab_training)} packages from requirements-colab (training section)")
        print()

        # Check 1: training.txt ⊆ colab.txt (training section)
        self._check_training_subset(training, colab_training)

        # Check 2: Version compatibility
        self._check_version_compatibility(training, colab_training)

        # Check 3: All imports declared
        self._check_imports_declared(requirements, training)

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _check_training_subset(
        self,
        training: Dict[str, PackageSpec],
        colab_training: Dict[str, PackageSpec]
    ) -> None:
        """Check that all training packages are in colab training section."""
        print("✓ Checking training.txt ⊆ colab.txt (training section)...")

        missing = []
        for pkg_name, pkg_spec in training.items():
            if pkg_name not in colab_training:
                missing.append(pkg_name)

        if missing:
            self.errors.append(
                f"❌ Missing packages in requirements-colab training section:\n"
                f"   {', '.join(missing)}\n"
                f"   \n"
                f"   Fix: Add to TRAINING.IPYNB section in requirements-colab-v3.4.0.txt"
            )
        else:
            print("   ✅ All training packages present in colab training section")
        print()

    def _check_version_compatibility(
        self,
        training: Dict[str, PackageSpec],
        colab_training: Dict[str, PackageSpec]
    ) -> None:
        """Check version compatibility between training and colab."""
        print("✓ Checking version compatibility...")

        conflicts = []
        for pkg_name in training:
            if pkg_name in colab_training:
                train_spec = training[pkg_name]
                colab_spec = colab_training[pkg_name]

                if not VersionValidator.is_compatible(train_spec, colab_spec):
                    conflicts.append((pkg_name, train_spec, colab_spec))

        if conflicts:
            error_msg = "❌ Version conflicts detected:\n"
            for pkg_name, train_spec, colab_spec in conflicts:
                error_msg += f"\n   Package: {pkg_name}\n"
                error_msg += f"   requirements-training.txt: {train_spec}\n"
                error_msg += f"   requirements-colab.txt: {colab_spec}\n"
                error_msg += f"   \n"
                error_msg += f"   Fix: Update requirements-training.txt:\n"
                error_msg += f"   sed -i 's/{train_spec}/{pkg_name}{colab_spec.specifier}/' requirements-training.txt\n"

            self.errors.append(error_msg)
        else:
            print("   ✅ All versions compatible")
        print()

    def _check_imports_declared(
        self,
        requirements: Dict[str, PackageSpec],
        training: Dict[str, PackageSpec]
    ) -> None:
        """Check that all imports in utils/ are declared in requirements."""
        print("✓ Checking imports are declared in requirements...")

        utils_dir = self.repo_root / 'utils'
        if not utils_dir.exists():
            self.warnings.append("⚠️  utils/ directory not found, skipping import check")
            return

        # Scan for imports
        imports = ImportScanner.scan_directory(utils_dir)

        # Apply package aliases first (before filtering)
        resolved_imports = set()
        for imp in imports:
            pkg_name = PACKAGE_ALIASES.get(imp, imp)
            resolved_imports.add(pkg_name.lower())

        # Filter out stdlib, internal modules, intentionally omitted, and optional packages
        third_party = (
            resolved_imports
            - {m.lower() for m in STDLIB_MODULES}
            - {m.lower() for m in INTERNAL_MODULES}
            - {m.lower() for m in INTENTIONALLY_OMITTED}
            - {m.lower() for m in OPTIONAL_PACKAGES}
        )

        # Normalize for comparison
        resolved = third_party

        # Check against requirements
        all_packages = set(requirements.keys()) | set(training.keys())
        missing = resolved - all_packages

        if missing:
            self.errors.append(
                f"❌ Undeclared imports found in utils/:\n"
                f"   {', '.join(sorted(missing))}\n"
                f"   \n"
                f"   These packages are imported but not in requirements.txt or requirements-training.txt\n"
                f"   \n"
                f"   Fix: Add missing packages:\n"
                f"   pip install {' '.join(sorted(missing))}\n"
                f"   pip freeze | grep -E '({'|'.join(sorted(missing))})' >> requirements.txt"
            )
        else:
            print(f"   ✅ All {len(resolved)} third-party imports declared")
        print()

    def _print_results(self) -> None:
        """Print validation results summary."""
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80 + "\n")

        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"\n{warning}")
            print()

        if self.errors:
            print("❌ FAILURES:")
            for error in self.errors:
                print(f"\n{error}")
            print()
            print(f"Total errors: {len(self.errors)}")
            print("\nRequirements files are OUT OF SYNC. Please fix the issues above.")
        else:
            print("✅ SUCCESS: All requirements files are in sync!")
            print("\n✓ training.txt ⊆ colab.txt")
            print("✓ Version compatibility verified")
            print("✓ All imports declared")

        print("\n" + "="*80 + "\n")


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    # Determine repository root (2 levels up from scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    validator = RequirementsSyncValidator(repo_root)
    success = validator.validate()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
