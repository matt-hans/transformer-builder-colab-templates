import torch

from utils.training.tier4_export_validation import _max_abs_and_rel_error


def test_max_abs_and_rel_error_simple():
    ref = torch.tensor([1.0, 2.0, 4.0])
    cand = torch.tensor([1.001, 1.999, 3.996])

    max_abs, max_rel = _max_abs_and_rel_error(ref, cand)

    assert max_abs > 0.0
    assert max_abs < 0.01
    assert max_rel < 0.01

