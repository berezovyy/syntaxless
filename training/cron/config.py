"""Seeded run configuration and CLI parsing for the cron training harness.

Every knob the harness needs is one dataclass field so the report can serialize
the exact run that produced a checkpoint. The device is resolved once, here, so
the resolved value — not the request — is what the report records.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class Config:
    """One reproducible training run."""

    run: str = "tiny"
    seed: int = 20260913
    epochs: int = 2
    samples: int = 2000
    eval_samples: int = 500
    batch: int = 1024
    learning_rate: float = 3.0e-3
    warmup_steps: int = 50
    qat_start: int = 1
    quantization_bits: int = 6
    hidden: int = 32
    storage: str = "training/cron/runs"
    device: str = "cpu"
    device_requested: str = "auto"
    data_prefix: str | None = None
    balance_frames: bool = False
    balance_kinds: bool = False
    balance_schedule_kinds: bool = False
    frame_balance_power: float = 0.5
    early_stopping_patience: int = 0
    min_epochs: int = 1
    fresh_each_epoch: bool = False
    fresh_epoch_size: int = 300_000
    val_max_sequences: int = 20_000
    init: str | None = None
    h_per_epoch: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize for the report without aliasing the dataclass."""
        return asdict(self)


def resolve_device(requested: str) -> str:
    """Honor an explicit device; `auto` prefers a GPU when one is visible."""
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train the tiny quantization-aware cron role tagger.",
    )
    parser.add_argument("--run", default="tiny", help="run name under --storage")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--qat-start", type=int, default=1)
    parser.add_argument("--quantization-bits", type=int, default=6)
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="shared model width",
    )
    parser.add_argument("--storage", default="training/cron/runs")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or any torch device string",
    )
    parser.add_argument(
        "--balance-frames",
        action="store_true",
        help="temper raw sequence counts toward equal wording-frame mass",
    )
    parser.add_argument(
        "--balance-kinds",
        action="store_true",
        help="give each sentence kind equal total mass before sampling frames",
    )
    parser.add_argument(
        "--balance-schedule-kinds",
        action="store_true",
        help="give each global schedule class equal total sampling mass",
    )
    parser.add_argument(
        "--frame-balance-power",
        type=float,
        default=0.5,
        help="0 preserves raw frame mass, 1 samples frames uniformly (default: 0.5)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop after this many non-improving epochs (0 disables)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="minimum epochs before early stopping may end the run",
    )
    parser.add_argument(
        "--data-prefix",
        default=None,
        help=(
            "pre-featurized split base; loads <prefix>-train, "
            "<prefix>-validation, and <prefix>-heldout instead of generate.ts"
        ),
    )
    parser.add_argument(
        "--fresh-each-epoch",
        action="store_true",
        help="resample --fresh-epoch-size rows from the train pool with seed+epoch",
    )
    parser.add_argument(
        "--fresh-epoch-size",
        type=int,
        default=300_000,
        help="draw size for --fresh-each-epoch (default 300000)",
    )
    parser.add_argument(
        "--val-max-sequences",
        type=int,
        default=20_000,
        help="per-epoch validation cap (default 20000); full val is scored at run end",
    )
    parser.add_argument(
        "--init",
        default=None,
        help="checkpoint to initialize from (A2 warm start)",
    )
    parser.add_argument(
        "--h-per-epoch",
        action="store_true",
        help="score human-regression H after each epoch and log it on the history row",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.eval_samples < 1:
        raise ValueError("--eval-samples must be at least 1")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")
    if not 2 <= args.quantization_bits <= 8:
        raise ValueError("--quantization-bits must be in [2, 8]")
    if args.hidden < 1:
        raise ValueError("--hidden must be at least 1")
    if args.fresh_epoch_size < 1:
        raise ValueError("--fresh-epoch-size must be at least 1")
    if args.val_max_sequences < 1:
        raise ValueError("--val-max-sequences must be at least 1")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must be in [1, --epochs]")
    if not 0 <= args.frame_balance_power <= 1:
        raise ValueError("--frame-balance-power must be in [0, 1]")
    if args.balance_kinds and args.balance_schedule_kinds:
        raise ValueError("choose either --balance-kinds or --balance-schedule-kinds")
    return Config(
        run=args.run,
        seed=args.seed,
        epochs=args.epochs,
        samples=args.samples,
        eval_samples=args.eval_samples,
        batch=args.batch,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        qat_start=args.qat_start,
        quantization_bits=args.quantization_bits,
        hidden=args.hidden,
        storage=args.storage,
        device=resolve_device(args.device),
        device_requested=args.device,
        data_prefix=args.data_prefix,
        balance_frames=args.balance_frames,
        balance_kinds=args.balance_kinds,
        balance_schedule_kinds=args.balance_schedule_kinds,
        frame_balance_power=args.frame_balance_power,
        early_stopping_patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
        fresh_each_epoch=args.fresh_each_epoch,
        fresh_epoch_size=args.fresh_epoch_size,
        val_max_sequences=args.val_max_sequences,
        init=args.init,
        h_per_epoch=args.h_per_epoch,
    )
