import torch
import torch.nn as nn
from types import SimpleNamespace

from utils.training.task_spec import get_default_task_specs
from utils.adapters import DecoderOnlyLMAdapter
import utils.test_functions as _tf


class DecoderOnlyLMStub(nn.Module):
    def __init__(self, vocab_size=101, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        return self.linear(x)


def test_decoder_only_lm_adapter_forward_and_tier1():
    device = torch.device('cpu')
    vocab_size = 101
    model = DecoderOnlyLMStub(vocab_size=vocab_size).to(device)

    task = get_default_task_specs()["lm_tiny"]
    adapter = DecoderOnlyLMAdapter()

    batch = {
        'input_ids': torch.randint(0, vocab_size, (2, 8), device=device),
        'attention_mask': torch.ones(2, 8, device=device),
        'labels': torch.randint(0, vocab_size, (2, 8), device=device),
    }

    loss, outputs = adapter.forward_for_loss(model, adapter.prepare_inputs(batch, task), task)
    assert loss is not None
    logits = outputs['logits']
    assert logits.shape == (2, 8, vocab_size)

    # Run Tier 1 shape test with adapter
    config = SimpleNamespace(vocab_size=vocab_size, max_seq_len=16, max_batch_size=4)
    df_or_list = _tf.test_shape_robustness(model, config, adapter=adapter, task_spec=task)
    # Expect at least one PASS result
    if isinstance(df_or_list, list):
        assert any('PASS' in r.get('status', '') for r in df_or_list)
    else:
        assert (df_or_list['status'].astype(str).str.contains('PASS')).any()
