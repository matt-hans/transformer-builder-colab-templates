import argparse
from types import SimpleNamespace

from utils.test_functions import test_shape_robustness, test_gradient_flow
from utils.training import build_task_spec, build_eval_config, TrainingConfig
from utils.adapters import DecoderOnlyLMAdapter

import torch
import torch.nn as nn


class LMStub(nn.Module):
    def __init__(self, vocab_size=101, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.head(x)


def run_from_config(cfg: dict) -> dict:
    # Minimal, stub-based tiers runner
    task_name = cfg.get('task_name', 'lm_tiny')
    mode = cfg.get('mode', 'FAST_DEV')
    vocab_size = int(cfg.get('vocab_size', 101))
    max_seq_len = int(cfg.get('max_seq_len', 16))

    training_cfg = TrainingConfig(vocab_size=vocab_size, max_seq_len=max_seq_len)
    training_cfg.task_name = task_name
    task = build_task_spec(training_cfg)
    adapter = DecoderOnlyLMAdapter()

    model = LMStub(vocab_size=vocab_size)
    config_ns = SimpleNamespace(vocab_size=vocab_size, max_seq_len=max_seq_len, max_batch_size=4)

    tier1 = {
        'shape': test_shape_robustness(model, config_ns, adapter=adapter, task_spec=task),
        'gradients': test_gradient_flow(model, config_ns, adapter=adapter, task_spec=task),
    }
    return {'tier1': 'ok', 'details': tier1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, help='Path to config JSON (optional)')
    args = ap.parse_args()

    cfg = {}
    if args.config:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
    out = run_from_config(cfg)
    print(out)


if __name__ == '__main__':
    main()

