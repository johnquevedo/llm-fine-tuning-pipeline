"""Weights & Biases setup that degrades gracefully.

Modes (from config `wandb.mode`):
- online:   normal logging (requires `wandb login` or WANDB_API_KEY)
- offline:  logs to local ./wandb dir, no network needed (`wandb sync` later)
- disabled: no wandb at all

If wandb is not installed, or online mode has no credentials, we fall back to
disabled so the pipeline always runs.
"""
from __future__ import annotations

import os

from . import config as C


def configure_wandb(cfg: dict) -> list:
    """Set env vars and return the Trainer `report_to` list."""
    mode = str(C.get(cfg, "wandb.mode", "disabled")).lower()

    try:
        import wandb  # noqa: F401
    except Exception:
        if mode != "disabled":
            print("[wandb] not installed -> falling back to disabled mode.")
        mode = "disabled"

    if mode == "online" and not (os.environ.get("WANDB_API_KEY") or _netrc_has_wandb()):
        print("[wandb] no credentials found -> falling back to offline mode.")
        mode = "offline"

    if mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        return []

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = mode  # "online" or "offline"
    os.environ["WANDB_PROJECT"] = C.get(cfg, "wandb.project", "llm-finetuning-pipeline")
    run_name = C.get(cfg, "wandb.run_name")
    if run_name:
        os.environ["WANDB_NAME"] = str(run_name)
    return ["wandb"]


def _netrc_has_wandb() -> bool:
    try:
        import netrc

        auth = netrc.netrc()
        return any("wandb" in host for host in auth.hosts)
    except Exception:
        return False
