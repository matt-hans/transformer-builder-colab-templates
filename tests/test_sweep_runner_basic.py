from types import SimpleNamespace
from utils.training.sweep_runner import run_grid_sweep


def test_sweep_runner_returns_ids():
    base = SimpleNamespace(learning_rate=1e-4, num_layers=2)
    grid = {"learning_rate": [1e-4, 5e-4], "num_layers": [2, 3]}
    def run_fn(cfg):
        return f"{cfg.learning_rate}-{cfg.num_layers}"
    run_ids = run_grid_sweep(base, grid, run_fn)
    assert len(run_ids) == 4
