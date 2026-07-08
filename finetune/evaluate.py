"""Baseline (pre-fine-tuning) and fine-tuned evaluation, incl. calibration."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from . import config as C
from . import metrics as M
from .data import build_tokenized
from .model import load_best, load_model, load_tokenizer
from .prompts import LABEL_NAMES, id_to_name

_MODEL_COLS = {"input_ids", "attention_mask", "labels"}


def _predictor(model, tokenizer, cfg):
    args = TrainingArguments(
        output_dir=os.path.join(C.get(cfg, "checkpoint_dir", "checkpoints"), "_pred_tmp"),
        per_device_eval_batch_size=C.get(cfg, "train.eval_batch_size", 32),
        report_to=[],
        dataloader_drop_last=False,
    )
    return Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer))


def _logits_on(split, model, tokenizer, cfg) -> np.ndarray:
    model_ds = split.remove_columns([c for c in split.column_names if c not in _MODEL_COLS])
    out = _predictor(model, tokenizer, cfg).predict(model_ds)
    logits = out.predictions
    if isinstance(logits, tuple):  # some models return tuples
        logits = logits[0]
    return np.asarray(logits)


# --------------------------------------------------------------------------- #
# Baselines (before fine-tuning)
# --------------------------------------------------------------------------- #
def run_baseline(cfg: dict) -> dict:
    out_dir = C.get(cfg, "output_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = load_tokenizer(cfg)
    ds = build_tokenized(cfg, tokenizer)

    train_labels = np.array(ds["train"]["labels"])
    test_labels = np.array(ds["test"]["labels"])

    # 1) Majority-class baseline
    majority = int(np.bincount(train_labels, minlength=len(LABEL_NAMES)).argmax())
    maj_preds = np.full_like(test_labels, majority)
    from sklearn.metrics import accuracy_score, f1_score
    majority_metrics = {
        "strategy": f"predict-majority ({id_to_name(majority)})",
        "accuracy": float(accuracy_score(test_labels, maj_preds)),
        "macro_f1": float(f1_score(test_labels, maj_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_labels, maj_preds, average="weighted", zero_division=0)),
        "n": int(len(test_labels)),
    }

    # 2) Untrained-head baseline (pretrained backbone, random classifier head)
    model = load_model(cfg)  # no checkpoint -> random head
    logits = _logits_on(ds["test"], model, tokenizer, cfg)
    n_bins = C.get(cfg, "eval.calibration_bins", 15)
    untrained_metrics = M.full_report(logits, test_labels, n_bins)
    untrained_metrics["strategy"] = "pretrained-backbone + untrained-head"

    result = {"majority": majority_metrics, "untrained_head": untrained_metrics}
    with open(os.path.join(out_dir, "baseline_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[baseline] -> {os.path.join(out_dir, 'baseline_metrics.json')}")
    print(f"[baseline] majority acc={majority_metrics['accuracy']:.4f} | "
          f"untrained-head acc={untrained_metrics['accuracy']:.4f}")
    return result


# --------------------------------------------------------------------------- #
# Fine-tuned evaluation (+ temperature scaling)
# --------------------------------------------------------------------------- #
def run_eval(cfg: dict) -> dict:
    out_dir = C.get(cfg, "output_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    n_bins = C.get(cfg, "eval.calibration_bins", 15)

    model, tokenizer = load_best(cfg)
    ds = build_tokenized(cfg, tokenizer)

    val_labels = np.array(ds["validation"]["labels"])
    test_labels = np.array(ds["test"]["labels"])

    # Fit temperature on validation, apply to test (proper protocol).
    val_logits = _logits_on(ds["validation"], model, tokenizer, cfg)
    temperature = M.fit_temperature(val_logits, val_labels)

    test_logits = _logits_on(ds["test"], model, tokenizer, cfg)
    pre = M.full_report(test_logits, test_labels, n_bins)
    scaled_logits = M.apply_temperature(test_logits, temperature)
    post = M.full_report(scaled_logits, test_labels, n_bins)

    metrics = {
        "temperature": float(temperature),
        "test": pre,
        "test_temperature_scaled": post,
        "ece_pre": pre["ece"],
        "ece_post": post["ece"],
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Reliability bins (pre-scaling) for the diagram / report.
    pd.DataFrame(M.reliability_bins(test_logits, test_labels, n_bins)).to_csv(
        os.path.join(out_dir, "reliability.csv"), index=False)

    # Per-example predictions for error analysis.
    probs = M.softmax(test_logits)
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    text_col = C.get(cfg, "data.text_column", "text")
    texts = ds["test"][text_col] if text_col in ds["test"].column_names else ds["test"]["prompt"]
    pred_df = pd.DataFrame({
        "text": texts,
        "true_label_id": test_labels,
        "true_label": [id_to_name(i) for i in test_labels],
        "pred_label_id": preds,
        "pred_label": [id_to_name(i) for i in preds],
        "confidence": conf,
        "correct": (preds == test_labels),
    })
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print(f"[eval] -> {os.path.join(out_dir, 'eval_metrics.json')}")
    print(f"[eval] test acc={pre['accuracy']:.4f} macro_f1={pre['macro_f1']:.4f} "
          f"loss={pre['loss_cross_entropy']:.4f}")
    print(f"[eval] ECE {pre['ece']:.4f} -> {post['ece']:.4f} (T={temperature:.3f})")
    return metrics
