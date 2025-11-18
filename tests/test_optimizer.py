import torch.nn as nn

from utils.tier3_training_utilities import _get_optimizer_grouped_parameters


def test_parameter_grouping_bias_and_layernorm_excluded():
    model = nn.Sequential(
        nn.Linear(10, 10, bias=True),
        nn.LayerNorm(10),
        nn.Linear(10, 5, bias=True),
    )

    groups = _get_optimizer_grouped_parameters(model, weight_decay=0.01)

    assert isinstance(groups, list) and len(groups) == 2
    decay_group, no_decay_group = groups
    assert decay_group['weight_decay'] == 0.01
    assert no_decay_group['weight_decay'] == 0.0

    # Count parameters by name to assert correct grouping
    names = dict(model.named_parameters())
    # Linear weights should be in decay group
    decay_params = set(decay_group['params'])
    no_decay_params = set(no_decay_group['params'])

    assert names['0.weight'] in decay_params
    assert names['2.weight'] in decay_params

    # Biases and LayerNorm should be in no-decay group
    assert names['0.bias'] in no_decay_params
    assert names['1.weight'] in no_decay_params  # LayerNorm weight
    assert names['1.bias'] in no_decay_params    # LayerNorm bias
    assert names['2.bias'] in no_decay_params


def test_all_parameters_accounted_for():
    model = nn.Linear(8, 4, bias=True)
    groups = _get_optimizer_grouped_parameters(model, 0.01)
    total_grouped = sum(len(g['params']) for g in groups)
    total_params = sum(1 for _ in model.parameters())
    assert total_grouped == total_params

