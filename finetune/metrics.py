"""Classification + calibration metrics.

All functions take plain numpy arrays so they are easy to unit-test offline:
- logits: (N, C) raw model outputs
- labels: (N,) integer gold labels
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict:
    preds = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = softmax(logits)
    n = probs.shape[0]
    p_true = probs[np.arange(n), np.asarray(labels)]
    p_true = np.clip(p_true, 1e-12, 1.0)
    return float(-np.log(p_true).mean())


def brier_score(logits: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs. one-hot targets."""
    probs = softmax(logits)
    n, c = probs.shape
    onehot = np.zeros((n, c))
    onehot[np.arange(n), np.asarray(labels)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_bins(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> List[Dict]:
    """Per-bin confidence vs. accuracy for a reliability diagram."""
    probs = softmax(logits)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == np.asarray(labels)).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # include right edge in the last bin
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            out.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0,
                        "avg_confidence": None, "accuracy": None})
            continue
        out.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": count,
            "avg_confidence": float(conf[mask].mean()),
            "accuracy": float(correct[mask].mean()),
        })
    return out


def expected_calibration_error(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: sum over bins of (bin_size/N) * |accuracy - confidence|."""
    bins = reliability_bins(logits, labels, n_bins)
    n = len(labels)
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        ece += (b["count"] / n) * abs(b["accuracy"] - b["avg_confidence"])
    return float(ece)


def max_calibration_error(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(logits, labels, n_bins)
    gaps = [abs(b["accuracy"] - b["avg_confidence"]) for b in bins if b["count"] > 0]
    return float(max(gaps)) if gaps else 0.0


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lr: float = 0.05, max_iter: int = 200) -> float:
    """Fit a single temperature T by minimizing NLL of softmax(logits / T).

    Simple, dependency-free gradient descent on log(T) (keeps T > 0). Returns T.
    Fit this on the validation set, then apply to test logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    n = logits.shape[0]
    log_t = 0.0  # T = 1.0 initially

    for _ in range(max_iter):
        t = np.exp(log_t)
        scaled = logits / t
        probs = softmax(scaled)
        # NLL as a function of u = log(T), with scaled = logits / T and dscaled/du = -scaled.
        # d(NLL)/du = mean_i ( z_true_i - sum_k p_k z_k ).
        z_true = scaled[np.arange(n), labels]
        expected_z = (probs * scaled).sum(axis=1)
        grad = (z_true - expected_z).mean()  # d(NLL)/d(log_t)
        log_t -= lr * grad                   # gradient descent on NLL
        # clamp to a sane range
        log_t = float(np.clip(log_t, np.log(0.05), np.log(20.0)))
    return float(np.exp(log_t))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)


def full_report(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict:
    """Bundle the metrics most useful for the report."""
    m = classification_metrics(logits, labels)
    m.update({
        "loss_cross_entropy": cross_entropy(logits, labels),
        "brier_score": brier_score(logits, labels),
        "ece": expected_calibration_error(logits, labels, n_bins),
        "mce": max_calibration_error(logits, labels, n_bins),
        "n": int(len(labels)),
    })
    return m
