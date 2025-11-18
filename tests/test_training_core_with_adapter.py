from types import SimpleNamespace
import torch
import matplotlib
matplotlib.use("Agg")
import torch.nn as nn

from utils.training import build_task_spec, build_eval_config, TrainingConfig
from utils.training.training_core import run_training
from utils.adapters import DecoderOnlyLMAdapter


class LMStub(nn.Module):
    def __init__(self, vocab_size=77, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.head(x)


def test_run_training_with_adapter_succeeds():
    model = LMStub(vocab_size=77)
    cfg = TrainingConfig(epochs=1, batch_size=2, vocab_size=77, max_seq_len=16)
    task = build_task_spec(cfg)  # defaults to lm_tiny
    eval_cfg = build_eval_config(cfg)
    adapter = DecoderOnlyLMAdapter()

    results = run_training(
        model=model,
        adapter=adapter,
        training_config=cfg,
        task_spec=task,
        eval_config=eval_cfg,
        experiment_db=None,
        metrics_tracker=None,
    )

    assert isinstance(results, dict)
    assert 'metrics_summary' in results
    # Check summary has expected columns
    summary = results['metrics_summary']
    if hasattr(summary, 'columns'):
        assert 'val/loss' in summary.columns
