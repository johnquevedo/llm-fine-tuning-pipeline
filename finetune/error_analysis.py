"""Group model errors by category and check for overfitting.

Reads reports/predictions.csv and reports/training_log.csv (produced earlier)
and writes:
- reports/confusion_matrix.csv
- reports/per_class_metrics.csv
- reports/error_analysis.md   (human-readable grouped errors + overfitting note)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C
from .prompts import LABEL_NAMES


def _confusion(df: pd.DataFrame) -> pd.DataFrame:
    mat = pd.crosstab(df["true_label"], df["pred_label"],
                      rownames=["true"], colnames=["pred"])
    return mat.reindex(index=LABEL_NAMES, columns=LABEL_NAMES, fill_value=0)


def _per_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in LABEL_NAMES:
        sub = df[df["true_label"] == name]
        support = len(sub)
        correct = int(sub["correct"].sum()) if support else 0
        # precision needs predicted==name across all rows
        pred_as = df[df["pred_label"] == name]
        tp = int((pred_as["true_label"] == name).sum())
        precision = tp / len(pred_as) if len(pred_as) else 0.0
        recall = correct / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({
            "label": name,
            "support": support,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "n_errors": support - correct,
        })
    return pd.DataFrame(rows)


def _top_confusions(conf: pd.DataFrame, k: int = 5) -> list:
    pairs = []
    for t in LABEL_NAMES:
        for p in LABEL_NAMES:
            if t != p and conf.loc[t, p] > 0:
                pairs.append((t, p, int(conf.loc[t, p])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def _confidence_buckets(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0.0, 0.5, 0.7, 0.9, 1.01]
    labels = ["0.0-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
    df = df.copy()
    df["bucket"] = pd.cut(df["confidence"], bins=edges, labels=labels, right=False)
    out = df.groupby("bucket", observed=False).agg(
        count=("correct", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()
    # "confidently wrong" = high confidence but incorrect
    conf_wrong = df[(df["confidence"] >= 0.9) & (~df["correct"])]
    return out, conf_wrong


def overfitting_check(train_log_path: str) -> dict:
    """Compare train vs eval loss trajectory across epochs."""
    if not os.path.exists(train_log_path):
        return {"available": False, "note": "No training_log.csv found."}
    df = pd.read_csv(train_log_path)

    train_loss = df.dropna(subset=["loss"])[["epoch", "loss"]] if "loss" in df else pd.DataFrame()
    eval_loss = df.dropna(subset=["eval_loss"])[["epoch", "eval_loss"]] if "eval_loss" in df else pd.DataFrame()
    if eval_loss.empty:
        return {"available": False, "note": "No eval_loss logged."}

    eval_loss = eval_loss.sort_values("epoch")
    best_epoch = float(eval_loss.loc[eval_loss["eval_loss"].idxmin(), "epoch"])
    last_epoch = float(eval_loss["epoch"].max())
    min_eval = float(eval_loss["eval_loss"].min())
    last_eval = float(eval_loss["eval_loss"].iloc[-1])

    # Overfitting signal: eval loss rose after its minimum while training continued.
    rose_after_min = last_eval > min_eval + 1e-4 and last_epoch > best_epoch
    final_train = float(train_loss["loss"].iloc[-1]) if not train_loss.empty else None

    return {
        "available": True,
        "best_eval_epoch": best_epoch,
        "min_eval_loss": round(min_eval, 4),
        "final_eval_loss": round(last_eval, 4),
        "final_train_loss": round(final_train, 4) if final_train is not None else None,
        "overfitting_detected": bool(rose_after_min),
        "note": (
            f"Eval loss bottomed at epoch {best_epoch:g} ({min_eval:.4f}) and "
            + ("rose afterward -> overfitting after that point."
               if rose_after_min else
               "did not rise afterward -> no clear overfitting in this run.")
        ),
    }


def run_analysis(cfg: dict) -> dict:
    out_dir = C.get(cfg, "output_dir", "reports")
    pred_path = os.path.join(out_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"{pred_path} not found. Run `python -m finetune.cli eval` first.")

    df = pd.read_csv(pred_path)
    df["correct"] = df["correct"].astype(bool)

    conf = _confusion(df)
    conf.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    per_class = _per_class(df)
    per_class.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    top_conf = _top_confusions(conf)
    buckets, conf_wrong = _confidence_buckets(df)
    over = overfitting_check(os.path.join(out_dir, "training_log.csv"))

    # ---- write markdown ----
    n = len(df)
    acc = float(df["correct"].mean())
    lines = []
    lines.append("# Error Analysis\n")
    lines.append(f"- Test examples: **{n}**")
    lines.append(f"- Overall accuracy: **{acc:.4f}**")
    lines.append(f"- Total errors: **{int((~df['correct']).sum())}**\n")

    lines.append("## Errors by true class (task/category breakdown)\n")
    lines.append(per_class.to_markdown(index=False))
    lines.append("")

    lines.append("## Top confusion pairs (reasoning-failure patterns)\n")
    lines.append("These are the most frequent `true -> predicted` mistakes:\n")
    for t, p, c in top_conf:
        lines.append(f"- `{t}` misread as `{p}`: **{c}** times")
    if not top_conf:
        lines.append("- No confusions (perfect on this subset).")
    lines.append("")

    lines.append("## Confidence buckets (calibration of errors)\n")
    lines.append(buckets.to_markdown(index=False))
    lines.append(f"\n**Confidently wrong** (confidence ≥ 0.9 but incorrect): "
                 f"**{len(conf_wrong)}** examples.\n")
    if len(conf_wrong):
        sample = conf_wrong.sort_values("confidence", ascending=False).head(5)
        lines.append("Worst offenders:\n")
        for _, r in sample.iterrows():
            txt = str(r["text"])[:100].replace("\n", " ")
            lines.append(f"- ({r['confidence']:.2f}) true=`{r['true_label']}` "
                         f"pred=`{r['pred_label']}` — \"{txt}\"")
        lines.append("")

    lines.append("## Overfitting check (train vs. validation loss)\n")
    lines.append(f"- {over['note']}")
    if over.get("available"):
        lines.append(f"- Min eval loss: {over['min_eval_loss']} at epoch {over['best_eval_epoch']:g}")
        lines.append(f"- Final eval loss: {over['final_eval_loss']} | "
                     f"final train loss: {over['final_train_loss']}")
    lines.append("")

    md_path = os.path.join(out_dir, "error_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[analyze] -> {md_path}")
    print(f"[analyze] confusion_matrix.csv, per_class_metrics.csv written")
    return {
        "accuracy": acc,
        "top_confusions": top_conf,
        "confidently_wrong": int(len(conf_wrong)),
        "overfitting": over,
    }
