from cli.run_training import run_from_config


def test_cli_run_training_stub_executes():
    cfg = {
        "task_name": "lm_tiny",
        "epochs": 1,
        "batch_size": 2,
        "vocab_size": 77,
        "max_seq_len": 16,
        "learning_rate": 0.0005,
    }
    out = run_from_config(cfg)
    assert isinstance(out, dict)
    assert 'eval_summary' in out or 'metrics_summary' in out
