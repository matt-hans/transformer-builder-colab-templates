from types import SimpleNamespace
from utils.training.export_utilities import create_repro_bundle
from utils.training.task_spec import get_default_task_specs
from utils.training.eval_config import EvalConfig


def test_create_repro_bundle(tmp_path):
    training_cfg = SimpleNamespace(to_dict=lambda: {"epochs": 1, "batch_size": 2})
    task = get_default_task_specs()["lm_tiny"]
    eval_cfg = EvalConfig(
        dataset_id="lm_tiny_v1",
        split="validation",
        max_eval_examples=4,
        batch_size=2,
        num_workers=0,
        max_seq_length=16,
        eval_interval_steps=0,
        eval_on_start=True,
    )
    archive = create_repro_bundle(
        run_id='test123',
        training_config=training_cfg,
        task_spec=task,
        eval_config=eval_cfg,
        environment_snapshot={},
        experiment_db=None,
        dashboard_paths=None,
        output_path=str(tmp_path),
    )
    assert archive.endswith('.zip')
