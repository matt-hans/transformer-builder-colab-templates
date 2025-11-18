import math
import torch

from utils.tier3_training_utilities import get_cosine_schedule_with_warmup


def test_lr_warmup_phase():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < 1e-5
    assert abs(lrs[-1] - 1e-4) < 1e-6
    assert lrs[50] < lrs[-1]


def test_lr_cosine_decay():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    # Finish warmup
    for _ in range(100):
        optimizer.step()
        scheduler.step()

    lrs = []
    for _ in range(900):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    assert lrs[0] > lrs[-1]
    assert lrs[-1] < 1e-5


def test_schedule_determinism():
    def lr_seq(seed):
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 50)
        seq = []
        for _ in range(50):
            seq.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        return seq

    a = lr_seq(42)
    b = lr_seq(42)
    assert a == b

