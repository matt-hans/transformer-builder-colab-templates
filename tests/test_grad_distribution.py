import torch
import torch.nn as nn


class _Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2, bias=True)

    def forward(self, x):
        return self.lin(x)


class _DummyTracker:
    def __init__(self):
        self.logged = []
        self.use_wandb = False

    def log_scalar(self, name, value, step=None):
        self.logged.append((name, float(value), step))


def test_log_gradient_distribution_logs_per_param_norm():
    import utils.tier3_training_utilities as t3

    model = _Mini()
    x = torch.randn(3, 4)
    y = model(x).sum()
    y.backward()

    tracker = _DummyTracker()
    t3._log_gradient_distribution(model, tracker, step=0, log_histogram=False)

    keys = [k for (k, _, _) in tracker.logged]
    # Expect norms for 'lin.weight' and 'lin.bias'
    assert any('gradients/lin.weight/norm' == k for k in keys)
    assert any('gradients/lin.bias/norm' == k for k in keys)

