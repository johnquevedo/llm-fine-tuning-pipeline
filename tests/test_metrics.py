"""Offline unit tests for metrics/calibration (no model download needed)."""
import numpy as np

from finetune import metrics as M


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = M.softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], 1 / 3)  # uniform when logits equal


def test_accuracy_and_f1_perfect():
    logits = np.eye(3) * 10  # confident correct predictions on diagonal
    labels = np.array([0, 1, 2])
    m = M.classification_metrics(logits, labels)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_cross_entropy_zero_when_perfect_and_confident():
    logits = np.array([[100.0, 0.0], [0.0, 100.0]])
    labels = np.array([0, 1])
    assert M.cross_entropy(logits, labels) < 1e-6


def test_ece_zero_for_perfectly_calibrated_confident_model():
    # fully confident and fully correct -> confidence == accuracy == 1 -> ECE 0
    logits = np.array([[100.0, 0.0], [0.0, 100.0], [100.0, 0.0]])
    labels = np.array([0, 1, 0])
    assert M.expected_calibration_error(logits, labels, n_bins=10) < 1e-6


def test_ece_positive_for_overconfident_wrong_model():
    # very confident but wrong -> large calibration error
    logits = np.array([[100.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
    labels = np.array([1, 1, 1])  # all wrong
    assert M.expected_calibration_error(logits, labels, n_bins=10) > 0.9


def test_temperature_scaling_reduces_nll_for_overconfident_model():
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 3, size=n)
    # build overconfident logits: correct class favored but exaggerated
    logits = rng.normal(0, 1, size=(n, 3))
    logits[np.arange(n), labels] += 2.0
    logits *= 5.0  # exaggerate -> overconfident
    T = M.fit_temperature(logits, labels)
    nll_before = M.cross_entropy(logits, labels)
    nll_after = M.cross_entropy(M.apply_temperature(logits, T), labels)
    assert T > 1.0            # overconfident -> temperature should soften (>1)
    assert nll_after <= nll_before + 1e-6


def test_brier_bounds():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1])
    b = M.brier_score(logits, labels)
    assert 0.0 <= b < 0.01


def test_reliability_bins_cover_all_examples():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 4))
    labels = rng.integers(0, 4, size=50)
    bins = M.reliability_bins(logits, labels, n_bins=10)
    assert sum(b["count"] for b in bins) == 50
