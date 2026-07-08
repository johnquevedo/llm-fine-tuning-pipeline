"""Offline tests for prompt formatting and config overrides."""
from finetune import config as C
from finetune import prompts


def test_prompt_formatting_wraps_text():
    out = prompts.format_prompt("i feel great", use_prompt_formatting=True)
    assert "i feel great" in out
    assert "Emotion:" in out
    assert out.strip().endswith("Emotion:")


def test_prompt_formatting_can_be_disabled():
    assert prompts.format_prompt("  hi  ", use_prompt_formatting=False) == "hi"


def test_label_helpers():
    assert prompts.num_labels() == 6
    assert prompts.id_to_name(1) == "joy"
    assert prompts.LABEL_NAMES[0] == "sadness"


def test_config_override_coercion():
    cfg = {"train": {"epochs": 3}, "wandb": {"mode": "disabled"}}
    out = C.apply_overrides(cfg, ["train.epochs=1", "wandb.mode=offline",
                                  "data.max_train_samples=null"])
    assert out["train"]["epochs"] == 1          # int, not "1"
    assert out["wandb"]["mode"] == "offline"
    assert out["data"]["max_train_samples"] is None
    # original untouched
    assert cfg["train"]["epochs"] == 3


def test_config_get_default():
    cfg = {"a": {"b": 2}}
    assert C.get(cfg, "a.b") == 2
    assert C.get(cfg, "a.missing", 7) == 7
    assert C.get(cfg, "nope.here", "d") == "d"
