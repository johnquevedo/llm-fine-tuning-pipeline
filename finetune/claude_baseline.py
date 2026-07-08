"""Zero-shot Claude baseline for the emotion-classification task.

This is a *reference* baseline: it classifies the same test set with Claude
(no fine-tuning) so you can compare your fine-tuned DistilBERT against a strong
LLM. It writes reports/claude_baseline.json with real, measured accuracy/F1 —
nothing is hard-coded.

Design (per the Anthropic API guidance):
- Official `anthropic` SDK, model from config (default claude-opus-4-8).
- Structured outputs via `output_config.format` with a 6-label enum, so every
  prediction is one of the valid emotion labels (no free-text parsing).
- Batch API by default (50% cheaper) for the bulk test set; a `--sync` path is
  available for quick checks on a few examples.

Runs only when you have Anthropic credentials. If the `anthropic` package isn't
installed or no credentials are configured, it prints a clear message and exits
without touching the rest of the (offline) pipeline.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from . import config as C
from .data import load_splits
from .prompts import INSTRUCTION, LABEL_NAMES

# JSON schema constraining Claude's output to exactly one emotion label.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"emotion": {"type": "string", "enum": LABEL_NAMES}},
    "required": ["emotion"],
    "additionalProperties": False,
}
_OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}}


def _prompt(text: str) -> str:
    return f"{INSTRUCTION}\n\nMessage: {(text or '').strip()}"


def _extract_label(message) -> Optional[str]:
    """Pull the emotion string out of a structured-output Message."""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            try:
                emotion = json.loads(block.text).get("emotion")
            except (json.JSONDecodeError, AttributeError):
                return None
            return emotion if emotion in LABEL_NAMES else None
    return None


def _client_or_none():
    """Construct an Anthropic client, or return None with a helpful message."""
    try:
        import anthropic
    except ImportError:
        print("[claude-baseline] `anthropic` not installed. "
              "Run `pip install anthropic` to enable this optional baseline.")
        return None
    try:
        return anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile
    except Exception as e:  # noqa: BLE001 - surface any auth/config problem cleanly
        print(f"[claude-baseline] could not create Anthropic client: {e}")
        print("[claude-baseline] set ANTHROPIC_API_KEY (or run `ant auth login`) "
              "and retry, or skip this optional baseline.")
        return None


def _load_test(cfg: dict):
    """Return (texts, gold_label_ids) for the test split."""
    splits = load_splits(cfg)
    ds = splits["test"]
    text_col = C.get(cfg, "data.text_column", "text")
    label_col = C.get(cfg, "data.label_column", "label")
    texts = list(ds[text_col])
    labels = list(ds[label_col])

    max_n = C.get(cfg, "claude.max_samples")
    if max_n is not None and max_n < len(texts):
        texts, labels = texts[:max_n], labels[:max_n]
    return texts, labels


# --------------------------------------------------------------------------- #
# Prediction backends
# --------------------------------------------------------------------------- #
def _predict_sync(client, cfg, texts: List[str]) -> List[Optional[str]]:
    model = C.get(cfg, "claude.model", "claude-opus-4-8")
    max_tokens = C.get(cfg, "claude.max_tokens", 64)
    preds = []
    for i, text in enumerate(texts):
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            output_config=_OUTPUT_CONFIG,
            messages=[{"role": "user", "content": _prompt(text)}],
        )
        if msg.stop_reason == "refusal":
            preds.append(None)
        else:
            preds.append(_extract_label(msg))
        if (i + 1) % 20 == 0:
            print(f"[claude-baseline] sync {i + 1}/{len(texts)}")
    return preds


def _predict_batch(client, cfg, texts: List[str]) -> List[Optional[str]]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    model = C.get(cfg, "claude.model", "claude-opus-4-8")
    max_tokens = C.get(cfg, "claude.max_tokens", 64)

    requests = [
        Request(
            custom_id=f"test-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                output_config=_OUTPUT_CONFIG,
                messages=[{"role": "user", "content": _prompt(text)}],
            ),
        )
        for i, text in enumerate(texts)
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"[claude-baseline] batch {batch.id} submitted "
          f"({len(requests)} requests); polling...")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(15)

    # Results arrive in any order — key by custom_id, never by position.
    preds: List[Optional[str]] = [None] * len(texts)
    for result in client.messages.batches.results(batch.id):
        idx = int(result.custom_id.split("-")[1])
        if result.result.type == "succeeded":
            msg = result.result.message
            preds[idx] = None if msg.stop_reason == "refusal" else _extract_label(msg)
    return preds


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_claude_baseline(cfg: dict, sync: bool = False) -> Optional[dict]:
    client = _client_or_none()
    if client is None:
        return None

    from sklearn.metrics import accuracy_score, f1_score

    texts, gold = _load_test(cfg)
    model = C.get(cfg, "claude.model", "claude-opus-4-8")
    use_batch = C.get(cfg, "claude.use_batch", True) and not sync
    print(f"[claude-baseline] model={model} n={len(texts)} "
          f"mode={'batch' if use_batch else 'sync'}")

    if use_batch:
        pred_names = _predict_batch(client, cfg, texts)
    else:
        pred_names = _predict_sync(client, cfg, texts)

    # Map to ids; unresolved/invalid predictions -> -1 (always counts as wrong).
    pred_ids = [LABEL_NAMES.index(p) if p in LABEL_NAMES else -1 for p in pred_names]
    covered = sum(1 for p in pred_ids if p != -1)

    accuracy = float(accuracy_score(gold, pred_ids))  # -1 preds count as wrong
    # Average F1 only over the real classes so unresolved preds don't add a phantom class.
    macro_f1 = float(f1_score(gold, pred_ids, labels=list(range(len(LABEL_NAMES))),
                              average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gold, pred_ids, labels=list(range(len(LABEL_NAMES))),
                                 average="weighted", zero_division=0))

    result = {
        "strategy": f"zero-shot Claude ({model})",
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n": len(texts),
        "coverage": covered / len(texts) if texts else 0.0,
        "used_batch": use_batch,
    }

    out_dir = C.get(cfg, "output_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "claude_baseline.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[claude-baseline] -> {os.path.join(out_dir, 'claude_baseline.json')}")
    print(f"[claude-baseline] acc={accuracy:.4f} macro_f1={macro_f1:.4f} "
          f"coverage={result['coverage']:.3f}")
    return result
