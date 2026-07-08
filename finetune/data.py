"""Dataset loading, prompt formatting, splitting, and tokenization."""
from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from . import config as C
from .prompts import format_prompt, num_labels


def _subsample(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_splits(cfg: dict) -> DatasetDict:
    """Load train/validation/test and subsample per config.

    dair-ai/emotion ships with train/validation/test. If a source ever lacks a
    validation split, we carve one out of train so the pipeline still runs.
    """
    seed = C.get(cfg, "seed", 42)
    name = C.get(cfg, "data.dataset_name")
    subcfg = C.get(cfg, "data.dataset_config")

    raw = load_dataset(name, subcfg) if subcfg else load_dataset(name)

    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.1, seed=seed)
        raw = DatasetDict(
            train=split["train"],
            validation=split["test"],
            test=raw.get("test", split["test"]),
        )

    train = _subsample(raw["train"], C.get(cfg, "data.max_train_samples"), seed)
    val = _subsample(raw["validation"], C.get(cfg, "data.max_eval_samples"), seed)
    test = _subsample(raw["test"], C.get(cfg, "data.max_test_samples"), seed)
    return DatasetDict(train=train, validation=val, test=test)


def add_prompt_column(ds: Dataset, cfg: dict) -> Dataset:
    text_col = C.get(cfg, "data.text_column", "text")
    use_pf = C.get(cfg, "data.use_prompt_formatting", True)

    def _fmt(batch):
        return {"prompt": [format_prompt(t, use_pf) for t in batch[text_col]]}

    return ds.map(_fmt, batched=True)


def build_tokenized(cfg: dict, tokenizer):
    """Return a DatasetDict with 'input_ids','attention_mask','labels','prompt'.

    'prompt' and the original text are kept for later error analysis.
    """
    splits = load_splits(cfg)
    label_col = C.get(cfg, "data.label_column", "label")
    max_len = C.get(cfg, "data.max_length", 128)

    tokenized = {}
    for name, ds in splits.items():
        ds = add_prompt_column(ds, cfg)

        def _tok(batch):
            enc = tokenizer(
                batch["prompt"],
                truncation=True,
                max_length=max_len,
                padding=False,  # dynamic padding via collator at train time
            )
            enc["labels"] = batch[label_col]
            return enc

        keep = ds.column_names  # drop nothing yet; we prune columns below
        ds_tok = ds.map(_tok, batched=True)
        # Keep model inputs + text/prompt/label for downstream analysis.
        text_col = C.get(cfg, "data.text_column", "text")
        drop = [c for c in ds_tok.column_names
                if c not in {"input_ids", "attention_mask", "labels", "prompt", text_col}]
        tokenized[name] = ds_tok.remove_columns(drop)
    return DatasetDict(tokenized)


def sanity_summary(cfg: dict) -> dict:
    """Cheap summary used by the smoke test / logging."""
    splits = load_splits(cfg)
    return {
        "num_labels": num_labels(),
        "sizes": {k: len(v) for k, v in splits.items()},
    }
