import torch
import torch.nn as nn
from types import SimpleNamespace

from utils.training.task_spec import get_default_task_specs
from utils.training.eval_config import EvalConfig
from utils.training.dataset_utilities import build_dataloader
from utils.training.eval_runner import run_evaluation
from utils.adapters import EncoderOnlyClassificationAdapter


class CLSStub(nn.Module):
    def __init__(self, vocab_size=101, d_model=32, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.head(x)  # (B, T, C); adapter pools


def test_eval_runner_cls_tiny():
    device = torch.device('cpu')
    vocab_size = 101
    model = CLSStub(vocab_size=vocab_size, num_classes=2).to(device)

    task = get_default_task_specs()["cls_tiny"]
    adapter = EncoderOnlyClassificationAdapter()
    train_cfg = SimpleNamespace(vocab_size=vocab_size, max_seq_len=16)
    eval_cfg = EvalConfig(
        dataset_id="cls_tiny_v1",
        split="validation",
        max_eval_examples=8,
        batch_size=2,
        num_workers=0,
        max_seq_length=16,
        eval_interval_steps=0,
        eval_on_start=True,
    )

    dl = build_dataloader(task, eval_cfg, train_cfg)
    summary = run_evaluation(model, adapter, task, eval_cfg, train_cfg, dl, metrics_tracker=None)

    assert "loss" in summary
    assert "accuracy" in summary
    assert isinstance(summary["accuracy"], float)

