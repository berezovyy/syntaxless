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
    batch: int = 64
    learning_rate: float = 3.0e-3
    warmup_steps: int = 50
    qat_start: int = 1
    quantization_bits: int = 6
    storage: str = "training/cron/runs"
    device: str = "cpu"
    device_requested: str = "auto"

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
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--qat-start", type=int, default=1)
    parser.add_argument("--quantization-bits", type=int, default=6)
    parser.add_argument("--storage", default="training/cron/runs")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or any torch device string",
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
        storage=args.storage,
        device=resolve_device(args.device),
        device_requested=args.device,
    )
