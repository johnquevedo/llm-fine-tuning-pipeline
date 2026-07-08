"""Offline tests for error-analysis + overfitting logic on synthetic data."""
import pandas as pd

from finetune import error_analysis as EA


def _synthetic_predictions():
    # 4 examples: 3 correct, 1 confident-wrong (joy predicted as sadness)
    return pd.DataFrame({
        "text": ["a", "b", "c", "d"],
        "true_label_id": [1, 1, 3, 0],
        "true_label": ["joy", "joy", "anger", "sadness"],
        "pred_label_id": [1, 0, 3, 0],
        "pred_label": ["joy", "sadness", "anger", "sadness"],
        "confidence": [0.95, 0.92, 0.80, 0.60],
        "correct": [True, False, True, True],
    })


def test_confusion_and_per_class(tmp_path):
    df = _synthetic_predictions()
    conf = EA._confusion(df)
    assert conf.loc["joy", "sadness"] == 1
    per = EA._per_class(df)
    joy = per[per["label"] == "joy"].iloc[0]
    assert joy["support"] == 2
    assert joy["n_errors"] == 1


def test_confidence_buckets_flag_confident_wrong():
    df = _synthetic_predictions()
    buckets, conf_wrong = EA._confidence_buckets(df)
    # one example (0.92) is >=0.9 and wrong
    assert len(conf_wrong) == 1
    assert buckets["count"].sum() == 4


def test_overfitting_detected_when_eval_loss_rises(tmp_path):
    log = pd.DataFrame([
        {"epoch": 1, "loss": 1.0, "eval_loss": 0.9},
        {"epoch": 2, "loss": 0.5, "eval_loss": 0.6},   # min here
        {"epoch": 3, "loss": 0.2, "eval_loss": 0.8},   # rose -> overfit
    ])
    p = tmp_path / "training_log.csv"
    log.to_csv(p, index=False)
    res = EA.overfitting_check(str(p))
    assert res["available"] is True
    assert res["overfitting_detected"] is True
    assert res["best_eval_epoch"] == 2


def test_no_overfitting_when_eval_loss_monotonic_down(tmp_path):
    log = pd.DataFrame([
        {"epoch": 1, "loss": 1.0, "eval_loss": 0.9},
        {"epoch": 2, "loss": 0.5, "eval_loss": 0.6},
        {"epoch": 3, "loss": 0.2, "eval_loss": 0.5},
    ])
    p = tmp_path / "training_log.csv"
    log.to_csv(p, index=False)
    res = EA.overfitting_check(str(p))
    assert res["overfitting_detected"] is False


def test_overfitting_check_handles_missing_file(tmp_path):
    res = EA.overfitting_check(str(tmp_path / "nope.csv"))
    assert res["available"] is False
