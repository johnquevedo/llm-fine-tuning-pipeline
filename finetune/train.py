"""Supervised fine-tuning via the Hugging Face Trainer.

Handles: checkpointing (save best), W&B tracking (online/offline/disabled),
and exporting the per-step train/eval loss log used for overfitting checks.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import config as C
from . import metrics as M
from .data import build_tokenized
from .model import load_model, load_tokenizer, save_best
from .wandb_utils import configure_wandb


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    out = M.classification_metrics(logits, labels)
    out["ece"] = M.expected_calibration_error(logits, labels)
    return out


def export_training_log(trainer, out_path: str) -> pd.DataFrame:
    """Flatten Trainer.state.log_history into a tidy CSV for overfitting analysis."""
    rows = trainer.state.log_history
    df = pd.DataFrame(rows)
    # Columns of interest if present
    keep = [c for c in ["epoch", "step", "loss", "eval_loss", "eval_accuracy",
                        "eval_macro_f1", "eval_ece", "learning_rate"] if c in df.columns]
    df = df[keep] if keep else df
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def train(cfg: dict) -> dict:
    set_seed(C.get(cfg, "seed", 42))
    report_to = configure_wandb(cfg)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)  # fresh backbone + new head
    ds = build_tokenized(cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    ckpt_dir = C.get(cfg, "checkpoint_dir", "checkpoints")
    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=C.get(cfg, "train.epochs", 3),
        per_device_train_batch_size=C.get(cfg, "train.batch_size", 16),
        per_device_eval_batch_size=C.get(cfg, "train.eval_batch_size", 32),
        learning_rate=float(C.get(cfg, "train.learning_rate", 2e-5)),
        weight_decay=C.get(cfg, "train.weight_decay", 0.01),
        warmup_ratio=C.get(cfg, "train.warmup_ratio", 0.06),
        logging_steps=C.get(cfg, "train.logging_steps", 20),
        eval_strategy=C.get(cfg, "train.eval_strategy", "epoch"),
        save_strategy=C.get(cfg, "train.save_strategy", "epoch"),
        load_best_model_at_end=C.get(cfg, "train.load_best_model_at_end", True),
        metric_for_best_model=C.get(cfg, "train.metric_for_best_model", "eval_loss"),
        greater_is_better=C.get(cfg, "train.greater_is_better", False),
        save_total_limit=2,
        report_to=report_to,
        run_name=C.get(cfg, "wandb.run_name") or None,
        seed=C.get(cfg, "seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"].remove_columns(
            [c for c in ds["train"].column_names
             if c not in {"input_ids", "attention_mask", "labels"}]
        ),
        eval_dataset=ds["validation"].remove_columns(
            [c for c in ds["validation"].column_names
             if c not in {"input_ids", "attention_mask", "labels"}]
        ),
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # Persist best model + tokenizer where evaluate.py expects it.
    save_path = save_best(trainer.model, tokenizer, cfg)

    out_dir = C.get(cfg, "output_dir", "reports")
    log_df = export_training_log(trainer, os.path.join(out_dir, "training_log.csv"))
    final_eval = trainer.evaluate()

    summary = {
        "checkpoint": save_path,
        "epochs": C.get(cfg, "train.epochs", 3),
        "train_samples": len(ds["train"]),
        "val_samples": len(ds["validation"]),
        "final_eval": {k: float(v) for k, v in final_eval.items()
                       if isinstance(v, (int, float))},
        "log_rows": int(len(log_df)),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    print(f"[train] best checkpoint -> {save_path}")
    print(f"[train] training log -> {os.path.join(out_dir, 'training_log.csv')}")
    return summary
