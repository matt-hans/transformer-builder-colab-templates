import argparse
import json
from utils.training import TrainingConfig, build_task_spec, build_eval_config
from utils.training.training_core import run_training
from utils.adapters import DecoderOnlyLMAdapter
from utils.adapters.gist_loader import load_gist_model
from utils.training.experiment_db import ExperimentDB
from pathlib import Path
import importlib.util
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


def _load_model_from_cfg(cfg: dict) -> nn.Module:
    # Local model path specified
    model_file = cfg.get('model_file') or cfg.get('model_path')
    if model_file:
        p = Path(model_file)
        if p.is_dir():
            p = p / 'model.py'
        if p.exists():
            spec = importlib.util.spec_from_file_location('user_model', str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'build_model'):
                return mod.build_model()
            if hasattr(mod, 'Model'):
                return mod.Model()
    # Gist specified
    if cfg.get('gist_id'):
        md = load_gist_model(cfg['gist_id'], cfg.get('gist_revision'))
        root = Path('./external/gists') / md.gist_id / (md.revision or 'latest')
        mf = root / 'model.py'
        if mf.exists():
            spec = importlib.util.spec_from_file_location('gist_model', str(mf))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'build_model'):
                return mod.build_model()
            if hasattr(mod, 'Model'):
                return mod.Model()
    # Fallback stub
    return LMStub(vocab_size=int(cfg.get('vocab_size', 101)))


def run_from_config(cfg: dict) -> dict:
    cfg_obj = TrainingConfig(
        epochs=int(cfg.get('epochs', 1)),
        batch_size=int(cfg.get('batch_size', 2)),
        vocab_size=int(cfg.get('vocab_size', 101)),
        max_seq_len=int(cfg.get('max_seq_len', 16)),
        learning_rate=float(cfg.get('learning_rate', 5e-4)),
    )
    if 'task_name' in cfg:
        cfg_obj.task_name = cfg['task_name']

    task = build_task_spec(cfg_obj)
    # Allow overrides for eval config
    eval_cfg = build_eval_config(cfg_obj)
    if 'eval' in cfg:
        ev = cfg['eval']
        from utils.training.eval_config import EvalConfig
        eval_cfg = EvalConfig.from_dict({
            'dataset_id': ev.get('dataset_id', eval_cfg.dataset_id),
            'split': ev.get('split', eval_cfg.split),
            'max_eval_examples': int(ev.get('max_eval_examples', eval_cfg.max_eval_examples)),
            'batch_size': int(ev.get('batch_size', eval_cfg.batch_size)),
            'num_workers': int(ev.get('num_workers', eval_cfg.num_workers)),
            'max_seq_length': int(ev.get('max_seq_length', eval_cfg.max_seq_length)),
            'eval_interval_steps': int(ev.get('eval_interval_steps', eval_cfg.eval_interval_steps)),
            'eval_on_start': bool(ev.get('eval_on_start', eval_cfg.eval_on_start)),
        })
    adapter = DecoderOnlyLMAdapter()
    model = _load_model_from_cfg(cfg)
    out = run_training(model, adapter, cfg_obj, task, eval_cfg)
    # Optional DB logging if requested
    if cfg.get('log_to_db'):
        db = ExperimentDB(cfg.get('db_path', 'experiments.db'))
        run_id = db.log_run(
            run_name=cfg.get('run_name', 'cli-run'),
            config=cfg_obj.to_dict(),
            notes=cfg.get('notes', ''),
            sweep_id=cfg.get('sweep_id'),
            sweep_params=cfg.get('sweep_params'),
            gist_id=cfg.get('gist_id'),
            gist_revision=cfg.get('gist_revision'),
            gist_sha256=None,
        )
        db.update_run_status(run_id, 'completed')
        out['run_id'] = run_id
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, help='Path to config JSON (optional)')
    args = ap.parse_args()
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    out = run_from_config(cfg)
    print(json.dumps({k: ('...' if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == '__main__':
    main()
