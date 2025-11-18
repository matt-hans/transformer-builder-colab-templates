from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.training.task_spec import TaskSpec
from utils.training.eval_config import EvalConfig
from utils.training.tier5_monitoring import run_tier5_monitoring
from utils.training.experiment_db import ExperimentDB
from utils.training.drift_metrics import compute_dataset_profile, log_profile_to_db
from utils.training.drift_metrics import compare_profiles
from utils.training.dataset_utilities import TinyVisionDataset
from utils.training.drift_metrics import _js_distance
from utils.adapters import DecoderOnlyLMAdapter


class TinyLMDataset:
    def __init__(self, length: int = 16, size: int = 32) -> None:
        self.length = length
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        seq = list(range(self.length))
        return {"input_ids": seq, "labels": seq}


class TinyLMStub(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.head(x)


def test_run_tier5_monitoring_eval_only(tmp_path, monkeypatch):
    # Simple text task spec
    task = TaskSpec(
        name="lm_tiny_tier5",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "perplexity"],
        modality="text",
        input_schema={"max_seq_len": 16, "vocab_size": 32},
        output_schema={"vocab_size": 32},
    )

    eval_cfg = EvalConfig(
        dataset_id="lm_tiny_tier5",
        split="validation",
        max_eval_examples=16,
        batch_size=4,
        num_workers=0,
        max_seq_length=16,
        eval_interval_steps=0,
        eval_on_start=True,
    )
    train_cfg = SimpleNamespace(vocab_size=32, max_seq_len=16, task_name="lm_tiny_tier5")
    setattr(eval_cfg, "training_config", train_cfg)

    # Patch build_dataloader to use in-memory dataset for this test
    from utils.training import dataset_utilities

    def _fake_build_dataloader(task_spec, eval_config, training_config):
        ds = TinyLMDataset(length=16, size=16)
        return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    monkeypatch.setattr(dataset_utilities, "build_dataloader", _fake_build_dataloader)

    model = TinyLMStub(vocab_size=32)
    adapter = DecoderOnlyLMAdapter()

    db_path = tmp_path / "experiments.db"
    db = ExperimentDB(db_path)

    result = run_tier5_monitoring(
        model=model,
        adapter=adapter,
        task_spec=task,
        eval_cfg=eval_cfg,
        db=db,
        baseline_run_id=None,
        reference_profile_id=None,
    )

    assert "eval_metrics" in result
    assert result["comparison"] is None
    assert result["drift"] is None
    assert result["status"] in {"ok", "warn", "fail"}

