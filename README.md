# LLM Fine-Tuning Pipeline

A small, end-to-end supervised fine-tuning (SFT) pipeline for a **small language model**
on an **instruction-formatted text-classification task**. It is designed to run on a normal
laptop (CPU works; a free Colab/T4 GPU is faster) and to make model behavior *measurable*
across runs: baseline vs. fine-tuned, accuracy/F1/loss, calibration (ECE + temperature
scaling), grouped error analysis, and train-vs-validation overfitting checks.

- **Task:** 6-way emotion classification, formatted as an instruction-following prompt.
- **Model:** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (~66M params).
- **Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (public, small).
- **Tracking:** Weights & Biases (works in `online`, `offline`, or `disabled` mode).

Everything writes real files under `reports/`. The report generator only fills in numbers
that were actually measured on your machine — **no results are hard-coded**.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

(Optional) Weights & Biases:

```bash
wandb login            # online tracking; skip this to run offline/disabled
```

The pipeline runs fine with **no** W&B account. Control it in `config.yaml`:
`wandb.mode: online | offline | disabled`. If `wandb` is not installed or login fails,
it automatically falls back to `disabled`.

---

## 2. Commands

All commands share `config.yaml`. Override any field with `--set key=value`.

```bash
# 0) Sanity check on a tiny subset (fast, downloads model + data once)
python -m finetune.cli smoke

# 1) Baseline evaluation BEFORE fine-tuning (majority-class + untrained head)
python -m finetune.cli baseline

# 2) Supervised fine-tuning (checkpoints + W&B tracking + train/val loss log)
python -m finetune.cli train

# 3) Evaluate the fine-tuned checkpoint (accuracy/F1/loss + calibration)
python -m finetune.cli eval

# 4) Error analysis grouped by category + overfitting check
python -m finetune.cli analyze

# 5) Generate the final markdown + CSV report
python -m finetune.cli report

# Run 1-5 end to end
python -m finetune.cli all

# (Optional) Zero-shot Claude baseline to compare the fine-tuned model against a
# strong LLM. Requires an Anthropic API key (or `ant auth login`); skipped
# cleanly if unavailable. Uses the Batch API by default; --sync for a quick check.
python -m finetune.cli claude-baseline
```

Fast example override (smaller, quicker):

```bash
python -m finetune.cli all --set data.max_train_samples=1000 --set train.epochs=1
```

---

## 3. Outputs (point to these in interviews)

| File | What it contains |
|------|------------------|
| `reports/baseline_metrics.json` | Accuracy/F1 of majority-class and untrained-head baselines |
| `checkpoints/best/` | Best fine-tuned model + tokenizer (loadable) |
| `reports/training_log.csv` | Per-step/epoch train loss & eval loss (overfitting evidence) |
| `reports/eval_metrics.json` | Fine-tuned accuracy, macro/weighted F1, loss, ECE, Brier, temperature |
| `reports/predictions.csv` | Per-example text, true label, prediction, confidence, correctness |
| `reports/reliability.csv` | Calibration reliability-diagram bins (confidence vs. accuracy) |
| `reports/error_analysis.md` | Errors grouped by true class, confusion pairs, confidence buckets |
| `reports/confusion_matrix.csv` | Full confusion matrix |
| `reports/report.md` | Final human-readable report tying it all together |
| `reports/summary.csv` | One-row-per-run summary for comparing runs |
| `reports/claude_baseline.json` | (Optional) zero-shot Claude accuracy/F1 on the test set |

---

## 4. Results

Measured locally with the default config (`2,000` training examples, `3` epochs).

| Run | Model | Train N | Epochs | Test Acc | Macro F1 | Test Loss | ECE (pre-T) | ECE (post-T) | Temp T |
|-----|-------|---------|--------|----------|----------|-----------|-------------|--------------|--------|
| baseline (majority) | — | 0 | 0 | 0.359 | 0.0881 | — | — | — | — |
| baseline (untrained head) | distilbert-base-uncased | 0 | 0 | 0.121 | 0.0654 | 1.7971 | 0.0737 | — | — |
| fine-tuned | distilbert-base-uncased | 2000 | 3 | 0.783 | 0.5344 | 0.6742 | 0.0968 | 0.0373 | 0.7894 |

**Fine-tuning lift over untrained head:** 0.1210 → 0.7830 (+0.6620 absolute accuracy).

**Overfitting note (from `training_log.csv`):** Eval loss bottomed at epoch 3
(0.7677) and did not rise afterward, so there was no clear overfitting signal in
this run.

---

## 5. Project layout

```
llm-fine-tuning-pipeline/
├── config.yaml              # all knobs
├── requirements.txt
├── finetune/
│   ├── config.py            # load/override config
│   ├── prompts.py           # instruction prompt formatting
│   ├── data.py              # load, preprocess, split, tokenize
│   ├── model.py             # model/tokenizer load + checkpoint save/load
│   ├── metrics.py           # accuracy, F1, ECE, Brier, temperature scaling
│   ├── train.py             # SFT via HF Trainer + W&B + checkpoints
│   ├── evaluate.py          # baseline + fine-tuned evaluation
│   ├── error_analysis.py    # grouped errors + overfitting check
│   ├── report.py            # markdown/CSV report generation
│   └── cli.py               # subcommands: smoke/baseline/train/eval/analyze/report/all
├── tests/                   # offline smoke tests (no network needed)
└── reports/                 # all generated artifacts
```

## 6. Notes / honesty

- CPU-only works; expect a few minutes per epoch on ~2k examples. GPU is auto-detected.
- Calibration is meaningful here because the task is classification with class probabilities.
- The "untrained head" baseline is near-chance by construction — that is expected and is the
  point of showing fine-tuning gains.
