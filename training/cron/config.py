"""The current compact training recipe, serialized with every checkpoint.

The TypeScript coordinator supplies verified paths. Training parameters live
here; there is no training-mode or hyperparameter command-line interface.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    run: str
    storage: str
    data_prefix: str
    device: str
    seed: int = 20260913
    epochs: int = 2
    batch: int = 512
    hidden: int = 64
    learning_rate: float = 3.0e-3
    warmup_steps: int = 50
    qat_start: int = 1
    quantization_bits: int = 6
    amp: bool = False
    prefetch: bool = False
    compile_model: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def production_config() -> Config:
    """Resolve paths and hardware for the coordinator's fixed recipe."""
    import torch

    prefix_value = os.environ.get("SYNTAXLESS_TRAIN_DATA_PREFIX")
    run_value = os.environ.get("SYNTAXLESS_TRAIN_RUN_DIRECTORY")
    if not prefix_value or not run_value:
        raise ValueError("Start training from the repository root with: bun run train")
    prefix = Path(prefix_value)
    run = Path(run_value)
    if not prefix.is_absolute() or not run.is_absolute():
        raise ValueError("The training coordinator must supply absolute artifact paths")
    cuda = torch.cuda.is_available()
    return Config(
        run=run.name,
        storage=str(run.parent),
        data_prefix=str(prefix),
        device="cuda" if cuda else "cpu",
        seed=20261012,
        epochs=24,
        batch=128,
        hidden=64,
        warmup_steps=10,
        amp=cuda,
        prefetch=cuda,
        compile_model=cuda,
    )
