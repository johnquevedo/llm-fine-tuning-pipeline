# Error Analysis

- Test examples: **1000**
- Overall accuracy: **0.7830**
- Total errors: **217**

## Errors by true class (task/category breakdown)

| label    |   support |   recall |   precision |     f1 |   n_errors |
|:---------|----------:|---------:|------------:|-------:|-----------:|
| sadness  |       287 |   0.9199 |      0.8024 | 0.8571 |         23 |
| joy      |       359 |   0.9387 |      0.7874 | 0.8564 |         22 |
| love     |        73 |   0.0137 |      0.5    | 0.0267 |         72 |
| anger    |       137 |   0.6058 |      0.8384 | 0.7034 |         54 |
| fear     |       115 |   0.8522 |      0.6901 | 0.7626 |         17 |
| surprise |        29 |   0      |      0      | 0      |         29 |

## Top confusion pairs (reasoning-failure patterns)

These are the most frequent `true -> predicted` mistakes:

- `love` misread as `joy`: **65** times
- `anger` misread as `sadness`: **35** times
- `surprise` misread as `fear`: **16** times
- `anger` misread as `fear`: **15** times
- `joy` misread as `sadness`: **12** times

## Confidence buckets (calibration of errors)

| bucket   |   count |   accuracy |
|:---------|--------:|-----------:|
| 0.0-0.5  |     145 |   0.413793 |
| 0.5-0.7  |     270 |   0.707407 |
| 0.7-0.9  |     585 |   0.909402 |
| 0.9-1.0  |       0 | nan        |

**Confidently wrong** (confidence ≥ 0.9 but incorrect): **0** examples.

## Overfitting check (train vs. validation loss)

- Eval loss bottomed at epoch 3 (0.7677) and did not rise afterward -> no clear overfitting in this run.
- Min eval loss: 0.7677 at epoch 3
- Final eval loss: 0.7677 | final train loss: 0.6845
