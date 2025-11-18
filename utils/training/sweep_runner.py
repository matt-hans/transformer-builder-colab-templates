"""
Simple hyperparameter sweep runner (grid).

Generates combinations of parameters and invokes a provided function per config.
Integrates lightly with ExperimentDB by returning run_ids from the provided
run function.
"""

from __future__ import annotations

from typing import List, Dict, Callable, Any
import itertools
import copy


def _set_nested_attr(obj: Any, key_path: str, value: Any) -> None:
    """Set attribute on dataclass/namespace/dict via dotted path."""
    parts = key_path.split('.')
    target = obj
    for k in parts[:-1]:
        target = getattr(target, k) if hasattr(target, k) else target[k]
    last = parts[-1]
    if hasattr(target, last):
        setattr(target, last, value)
    else:
        target[last] = value


def run_grid_sweep(
    base_config: Any,
    param_grid: Dict[str, List[Any]],
    run_fn: Callable[[Any], str],
) -> List[str]:
    """
    Generate cartesian product of params and call run_fn(config) for each.

    Args:
        base_config: TrainingConfig-like object (dataclass or SimpleNamespace)
        param_grid: Mapping of dotted key path to list of values
        run_fn: Callable that takes a config and returns a run_id (str)

    Returns:
        List of run_ids from run_fn calls.
    """
    keys = list(param_grid.keys())
    values_product = list(itertools.product(*[param_grid[k] for k in keys]))

    run_ids: List[str] = []
    for combo in values_product:
        cfg = copy.deepcopy(base_config)
        for k, v in zip(keys, combo):
            _set_nested_attr(cfg, k, v)
        run_id = run_fn(cfg)
        run_ids.append(str(run_id))
    return run_ids

