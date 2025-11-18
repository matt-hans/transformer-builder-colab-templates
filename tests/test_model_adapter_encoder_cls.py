import torch
import torch.nn as nn

from utils.training.task_spec import get_default_task_specs
from utils.adapters import EncoderOnlyClassificationAdapter


class EncoderOnlyCLSStub(nn.Module):
    def __init__(self, vocab_size=101, d_model=32, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        # Return token-level logits (B, T, C) so adapter pools
        return self.head(x)


def test_encoder_only_cls_adapter_forward_loss():
    device = torch.device('cpu')
    vocab_size = 101
    num_classes = 3
    model = EncoderOnlyCLSStub(vocab_size=vocab_size, num_classes=num_classes).to(device)

    task = get_default_task_specs()["cls_tiny"]
    adapter = EncoderOnlyClassificationAdapter()

    batch = {
        'input_ids': torch.randint(0, vocab_size, (4, 7), device=device),
        'attention_mask': torch.ones(4, 7, device=device),
        'labels': torch.randint(0, num_classes, (4,), device=device),
    }

    loss, outputs = adapter.forward_for_loss(model, adapter.prepare_inputs(batch, task), task)
    assert loss is not None
    logits = outputs['logits']
    assert logits.shape == (4, num_classes)

