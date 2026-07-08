# LLM Fine-Tuning Pipeline — Report

- Model: `distilbert-base-uncased`
- Task: emotion classification (6 classes), instruction-formatted
- Train examples: 2000 | Epochs: 3

## Results summary

| run                     | model                   |   train_n |   epochs |   test_acc |   macro_f1 |   test_loss |   ece_pre |   ece_post |   temperature |
|:------------------------|:------------------------|----------:|---------:|-----------:|-----------:|------------:|----------:|-----------:|--------------:|
| baseline-majority       | -                       |         0 |        0 |      0.359 |     0.0881 |    nan      |  nan      |   nan      |      nan      |
| baseline-untrained-head | distilbert-base-uncased |         0 |        0 |      0.121 |     0.0654 |      1.7971 |    0.0737 |   nan      |      nan      |
| fine-tuned              | distilbert-base-uncased |      2000 |        3 |      0.783 |     0.5344 |      0.6742 |    0.0968 |     0.0373 |        0.7894 |

**Fine-tuning lift over untrained head:** 0.1210 → 0.7830 (+0.6620 absolute accuracy).

## Calibration

- ECE before temperature scaling: **0.0968**
- ECE after temperature scaling (T=0.789): **0.0373**
- Brier score: **0.3192**
- Reliability bins: `reports/reliability.csv`

## Error analysis

Full detail in `reports/error_analysis.md`. Summary:

## Overfitting check (train vs. validation loss)
- Eval loss bottomed at epoch 3 (0.7677) and did not rise afterward -> no clear overfitting in this run.

## Artifacts

- [x] `reports/baseline_metrics.json`
- [ ] `reports/claude_baseline.json`
- [x] `reports/train_summary.json`
- [x] `reports/training_log.csv`
- [x] `reports/eval_metrics.json`
- [x] `reports/predictions.csv`
- [x] `reports/reliability.csv`
- [x] `reports/confusion_matrix.csv`
- [x] `reports/per_class_metrics.csv`
- [x] `reports/error_analysis.md`
- [x] `reports/summary.csv`
