import torch
import torch.nn as nn

from utils.training.task_spec import get_default_task_specs
from utils.adapters import EncoderDecoderSeq2SeqAdapter


class Seq2SeqStub(nn.Module):
    def __init__(self, vocab_size=101, d_model=32):
        super().__init__()
        self.encoder_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids=None, decoder_input_ids=None, attention_mask=None):
        # Ignore encoder outputs for simplicity; just map decoder embeddings
        d = self.decoder_embed(decoder_input_ids)
        return self.proj(d)


def test_seq2seq_adapter_forward_loss():
    device = torch.device('cpu')
    vocab_size = 101
    model = Seq2SeqStub(vocab_size=vocab_size).to(device)

    task = get_default_task_specs()["seq2seq_tiny"]
    adapter = EncoderDecoderSeq2SeqAdapter()

    batch = {
        'input_ids': torch.randint(0, vocab_size, (2, 5), device=device),
        'decoder_input_ids': torch.randint(0, vocab_size, (2, 6), device=device),
        'labels': torch.randint(0, vocab_size, (2, 6), device=device),
    }

    loss, outputs = adapter.forward_for_loss(model, adapter.prepare_inputs(batch, task), task)
    assert loss is not None
    logits = outputs['logits']
    assert logits.shape == (2, 6, vocab_size)

