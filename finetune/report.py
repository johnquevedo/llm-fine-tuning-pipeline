"""Generate the final markdown + CSV report from measured artifacts only.

Nothing here fabricates numbers: it reads whatever the earlier stages wrote and
leaves fields blank/None if a stage has not been run yet.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import config as C


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_report(cfg: dict) -> dict:
    out_dir = C.get(cfg, "output_dir", "reports")
    baseline = _load_json(os.path.join(out_dir, "baseline_metrics.json"))
    evalm = _load_json(os.path.join(out_dir, "eval_metrics.json"))
    trainm = _load_json(os.path.join(out_dir, "train_summary.json"))
    claudem = _load_json(os.path.join(out_dir, "claude_baseline.json"))

    model_name = C.get(cfg, "model.name")
    train_n = trainm["train_samples"] if trainm else C.get(cfg, "data.max_train_samples")
    epochs = trainm["epochs"] if trainm else C.get(cfg, "train.epochs")

    # ---- summary.csv (one row per config we know about) ----
    rows = []
    if baseline:
        maj = baseline.get("majority", {})
        rows.append({
            "run": "baseline-majority", "model": "-", "train_n": 0, "epochs": 0,
            "test_acc": maj.get("accuracy"), "macro_f1": maj.get("macro_f1"),
            "test_loss": None, "ece_pre": None, "ece_post": None, "temperature": None,
        })
        unt = baseline.get("untrained_head", {})
        rows.append({
            "run": "baseline-untrained-head", "model": model_name, "train_n": 0, "epochs": 0,
            "test_acc": unt.get("accuracy"), "macro_f1": unt.get("macro_f1"),
            "test_loss": unt.get("loss_cross_entropy"), "ece_pre": unt.get("ece"),
            "ece_post": None, "temperature": None,
        })
    if claudem:
        rows.append({
            "run": "zero-shot-claude", "model": claudem.get("model", "claude"),
            "train_n": 0, "epochs": 0,
            "test_acc": claudem.get("accuracy"), "macro_f1": claudem.get("macro_f1"),
            "test_loss": None, "ece_pre": None, "ece_post": None, "temperature": None,
        })
    if evalm:
        test = evalm.get("test", {})
        rows.append({
            "run": "fine-tuned", "model": model_name, "train_n": train_n, "epochs": epochs,
            "test_acc": test.get("accuracy"), "macro_f1": test.get("macro_f1"),
            "test_loss": test.get("loss_cross_entropy"),
            "ece_pre": evalm.get("ece_pre"), "ece_post": evalm.get("ece_post"),
            "temperature": evalm.get("temperature"),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # ---- report.md ----
    lines = ["# LLM Fine-Tuning Pipeline — Report\n"]
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Task: emotion classification (6 classes), instruction-formatted")
    lines.append(f"- Train examples: {train_n} | Epochs: {epochs}\n")

    lines.append("## Results summary\n")
    if not summary_df.empty:
        show = summary_df.copy()
        for col in ["test_acc", "macro_f1", "test_loss", "ece_pre", "ece_post", "temperature"]:
            show[col] = show[col].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "")
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("_No stages have been run yet._")
    lines.append("")

    if baseline and evalm:
        b = baseline["untrained_head"]["accuracy"]
        f = evalm["test"]["accuracy"]
        lines.append(f"**Fine-tuning lift over untrained head:** "
                     f"{b:.4f} → {f:.4f} (+{f - b:.4f} absolute accuracy).\n")

    if evalm:
        lines.append("## Calibration\n")
        lines.append(f"- ECE before temperature scaling: **{evalm['ece_pre']:.4f}**")
        lines.append(f"- ECE after temperature scaling (T={evalm['temperature']:.3f}): "
                     f"**{evalm['ece_post']:.4f}**")
        lines.append(f"- Brier score: **{evalm['test'].get('brier_score'):.4f}**")
        lines.append("- Reliability bins: `reports/reliability.csv`\n")

    # Fold in the error-analysis markdown if present.
    ea_path = os.path.join(out_dir, "error_analysis.md")
    if os.path.exists(ea_path):
        lines.append("## Error analysis\n")
        lines.append(f"Full detail in `reports/error_analysis.md`. Summary:\n")
        with open(ea_path) as f:
            ea = f.read()
        # pull the overfitting section note
        for line in ea.splitlines():
            if line.strip().startswith("- Eval loss") or "overfitting" in line.lower():
                lines.append(line)
        lines.append("")

    lines.append("## Artifacts\n")
    for name in ["baseline_metrics.json", "claude_baseline.json", "train_summary.json",
                 "training_log.csv", "eval_metrics.json", "predictions.csv",
                 "reliability.csv", "confusion_matrix.csv", "per_class_metrics.csv",
                 "error_analysis.md", "summary.csv"]:
        p = os.path.join(out_dir, name)
        mark = "x" if os.path.exists(p) else " "
        lines.append(f"- [{mark}] `reports/{name}`")
    lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[report] -> {report_path}")
    print(f"[report] -> {summary_path}")
    return {"report": report_path, "summary": summary_path, "rows": len(summary_df)}
