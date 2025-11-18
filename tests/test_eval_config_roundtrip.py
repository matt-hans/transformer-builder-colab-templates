from utils.training.eval_config import EvalConfig


def test_eval_config_roundtrip_defaults():
    cfg = EvalConfig(
        dataset_id="lm_tiny_v1",
        split="validation",
        max_eval_examples=256,
        batch_size=4,
        num_workers=0,
        max_seq_length=128,
        eval_interval_steps=50,
        eval_on_start=True,
    )

    d = cfg.to_dict()
    loaded = EvalConfig.from_dict(d)

    assert loaded.dataset_id == cfg.dataset_id
    assert loaded.split == cfg.split
    assert loaded.max_eval_examples == cfg.max_eval_examples
    assert loaded.batch_size == cfg.batch_size
    assert loaded.num_workers == cfg.num_workers
    assert loaded.max_seq_length == cfg.max_seq_length
    assert loaded.eval_interval_steps == cfg.eval_interval_steps
    assert loaded.eval_on_start is True

