import pytest

from utils.training.training_core import TrainingCoordinator


def test_training_coordinator_accepts_strategy_and_devices():
    # Basic smoke test: ensure we can construct with new parameters
    coord = TrainingCoordinator(
        output_dir="./tmp_training_output",
        use_gpu=False,
        precision="32",
        gradient_clip_val=0.5,
        strategy="auto",
        devices=None,
        num_nodes=1,
    )
    assert coord.strategy == "auto"
    assert coord.devices is None
    assert coord.num_nodes == 1

