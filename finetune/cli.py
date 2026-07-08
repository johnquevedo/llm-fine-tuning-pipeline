"""Command-line entry point.

Usage:
    python -m finetune.cli <command> [--config config.yaml] [--set k=v ...]

Commands: smoke | baseline | train | eval | analyze | report | all
"""
from __future__ import annotations

import argparse
import sys

from . import config as C


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="finetune", description=__doc__)
    p.add_argument("command",
                   choices=["smoke", "baseline", "train", "eval", "analyze",
                            "report", "all", "claude-baseline"])
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Override a config field, e.g. --set train.epochs=1")
    p.add_argument("--sync", action="store_true",
                   help="claude-baseline only: use synchronous calls instead of the Batch API")
    return p


def _load(args) -> dict:
    cfg = C.load_config(args.config)
    return C.apply_overrides(cfg, args.overrides)


def _run_pipeline(cfg: dict):
    from .evaluate import run_baseline, run_eval
    from .train import train
    from .error_analysis import run_analysis
    from .report import run_report

    run_baseline(cfg)
    train(cfg)
    run_eval(cfg)
    run_analysis(cfg)
    run_report(cfg)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = _load(args)
    cmd = args.command

    if cmd == "smoke":
        # Tiny, fast end-to-end sanity check (downloads model+data once).
        smoke_overrides = [
            "data.max_train_samples=200",
            "data.max_eval_samples=100",
            "data.max_test_samples=100",
            "train.epochs=1",
            "train.batch_size=16",
        ]
        cfg = C.apply_overrides(cfg, smoke_overrides)
        from .data import sanity_summary
        print("[smoke] dataset summary:", sanity_summary(cfg))
        _run_pipeline(cfg)
        print("[smoke] OK — end-to-end pipeline ran and wrote reports/.")
        return

    if cmd == "baseline":
        from .evaluate import run_baseline
        run_baseline(cfg)
    elif cmd == "train":
        from .train import train
        train(cfg)
    elif cmd == "eval":
        from .evaluate import run_eval
        run_eval(cfg)
    elif cmd == "analyze":
        from .error_analysis import run_analysis
        run_analysis(cfg)
    elif cmd == "report":
        from .report import run_report
        run_report(cfg)
    elif cmd == "claude-baseline":
        from .claude_baseline import run_claude_baseline
        run_claude_baseline(cfg, sync=args.sync)
    elif cmd == "all":
        _run_pipeline(cfg)


if __name__ == "__main__":
    sys.exit(main())
