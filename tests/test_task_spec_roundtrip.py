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


def test_task_spec_vision_modality_fields():
    vision_spec = TaskSpec(
        name="vision_tiny",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"],
        modality="vision",
        input_schema={"image_size": [3, 32, 32], "channels_first": True},
        output_schema={"num_classes": 10},
        preprocessing_config={"normalize": True},
    )

    assert vision_spec.is_vision()
    assert not vision_spec.is_text()
    assert vision_spec.modality == "vision"
    assert vision_spec.input_schema["image_size"] == [3, 32, 32]
    assert vision_spec.output_schema["num_classes"] == 10
    # get_input_shape should surface the image_size
    assert vision_spec.get_input_shape() == [3, 32, 32]


def test_task_spec_text_defaults_backward_compatible():
    # Existing text tasks should default to modality="text" and expose a basic input schema
    presets = get_default_task_specs()
    lm_spec = presets["lm_tiny"]

    assert lm_spec.is_text()
    assert lm_spec.modality == "text"
    # Ensure input_schema is present and contains max_seq_len
    assert "max_seq_len" in lm_spec.input_schema
    assert isinstance(lm_spec.input_schema["max_seq_len"], int)
