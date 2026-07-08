"""Model/tokenizer loading and checkpoint save/load helpers."""
from __future__ import annotations

import os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from . import config as C
from .prompts import LABEL_NAMES, num_labels


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(C.get(cfg, "model.name"))


def load_model(cfg: dict, from_checkpoint: str | None = None):
    """Load a sequence-classification model.

    - from_checkpoint=None: fresh pretrained backbone + randomly initialized head
      (this is exactly the "untrained head" baseline before fine-tuning).
    - from_checkpoint=path: load a fine-tuned checkpoint.
    """
    source = from_checkpoint or C.get(cfg, "model.name")
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=num_labels(),
        id2label=id2label,
        label2id=label2id,
    )
    return model


def best_checkpoint_dir(cfg: dict) -> str:
    return os.path.join(C.get(cfg, "checkpoint_dir", "checkpoints"), "best")


def save_best(model, tokenizer, cfg: dict) -> str:
    path = best_checkpoint_dir(cfg)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def load_best(cfg: dict):
    path = best_checkpoint_dir(cfg)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No fine-tuned checkpoint at {path}. Run `python -m finetune.cli train` first."
        )
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
