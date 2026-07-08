"""Config loading with dotted-key CLI overrides.

Config is a plain nested dict loaded from YAML. Access via helpers or the small
attribute wrapper. Overrides look like ``train.epochs=1`` and are type-coerced.
"""
from __future__ import annotations

import ast
import copy
import os
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG_PATH = os.environ.get("FINETUNE_CONFIG", "config.yaml")


def load_config(path: str | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return cfg


def _coerce(value: str) -> Any:
    """Turn a CLI string into an int/float/bool/None/str as appropriate.

    Accepts YAML-style literals too (null / true / false) since config is YAML.
    """
    low = value.strip().lower()
    if low in ("null", "none", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` style overrides, returning a new dict."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad --set override (need key=value): {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override into non-mapping at {p!r}")
        node[parts[-1]] = _coerce(raw)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
