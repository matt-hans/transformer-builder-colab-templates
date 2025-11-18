from utils.training.task_spec import TaskSpec, get_default_task_specs, load_task_spec_from_dict


def test_task_spec_roundtrip_default_presets():
    presets = get_default_task_specs()
    assert "lm_tiny" in presets
    spec = presets["lm_tiny"]

    d = spec.to_dict()
    spec2 = TaskSpec.from_dict(d)

    assert spec2.name == spec.name
    assert spec2.task_type == spec.task_type
    assert spec2.model_family == spec.model_family
    assert spec2.input_fields == spec.input_fields
    assert spec2.target_field == spec.target_field
    assert spec2.loss_type == spec.loss_type
    assert spec2.metrics == spec.metrics
    assert spec2.special_tokens == spec.special_tokens
    assert spec2.additional_config == spec.additional_config


def test_load_task_spec_from_dict_alias():
    src = {
        "name": "cls_custom",
        "task_type": "classification",
        "model_family": "encoder_only",
        "input_fields": ["input_ids", "attention_mask"],
        "target_field": "labels",
        "loss_type": "cross_entropy",
        "metrics": ["loss", "accuracy"],
        "special_tokens": {"pad_token_id": 0},
        "additional_config": {"num_classes": 3},
    }
    spec = load_task_spec_from_dict(src)
    assert spec.name == "cls_custom"
    assert spec.additional_config.get("num_classes") == 3

