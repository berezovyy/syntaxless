"""Training harness: regenerate splits, train the tagger, select on validation.

One epoch draws a fresh training split (`seed + epoch * STRIDE`), keeps
validation and heldout frozen, and evaluates every epoch with `qat=True` so the
checkpoint that wins is the quantized model that ships. The run report records
hashes, digests, seeds, and the label histogram as evidence.
"""
from __future__ import annotations

import hashlib
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import config as config_module
from config import Config
from dataset import Dataset, fingerprints_path, prepare, repo_root
from model import Tagger, parameter_counts
from report import write_report

#: Weights the boundary term against the role term.
BOUNDARY_LOSS_WEIGHT = 1.0
#: Seed stride between regenerated training splits; large enough to never alias.
EPOCH_SEED_STRIDE = 1_000_003
#: Validation and heldout derive from the run seed with fixed offsets.
VALIDATION_SEED_OFFSET = 1
HELDOUT_SEED_OFFSET = 2
#: AdamW weight decay; not exposed as a CLI knob in this phase.
WEIGHT_DECAY = 0.01
#: Gradient clipping norm.
CLIP_NORM = 1.0

HARNESS_SOURCES = (
    "config.py",
    "dataset.py",
    "model.py",
    "report.py",
    "train.py",
    "test_model.py",
    "pyproject.toml",
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    hashes: dict[str, str] = {}
    for name in HARNESS_SOURCES:
        path = base / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def assert_disjoint(train: Dataset, heldout: Dataset) -> None:
    overlap = set(train.fingerprints) & set(heldout.fingerprints)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(
            f"held-out leak: {len(overlap)} fingerprint(s) shared by train and "
            f"heldout (e.g. {sample})"
        )


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: Tagger,
    optimizer,
    scheduler,
    dataset: Dataset,
    batch_size: int,
    device: str,
    qat: bool,
) -> dict[str, float]:
    model.train()
    model.qat = qat
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for indices in dataset.batches(batch_size):
        batch = dataset.batch(indices, device)
        labels = batch["labels"]
        mask = labels != -100

        optimizer.zero_grad(set_to_none=True)
        roles, boundary = model(batch["rows"], batch["valid"], batch["neighbors"])
        role_loss = F.cross_entropy(
            roles.reshape(-1, roles.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        boundary_loss = F.binary_cross_entropy_with_logits(
            boundary, batch["boundaries"], reduction="none"
        )
        boundary_loss = (boundary_loss * mask).sum() / mask.sum().clamp(min=1)
        loss = role_loss + BOUNDARY_LOSS_WEIGHT * boundary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        scheduler.step()

        token_count = int(mask.sum())
        total_loss += float(loss.detach()) * max(token_count, 1)
        total_tokens += token_count
        batches += 1

    return {
        "trainLoss": total_loss / max(total_tokens, 1),
        "trainTokens": total_tokens,
        "learningRate": float(scheduler.get_last_lr()[0]),
        "batches": batches,
    }


@torch.no_grad()
def evaluate(
    model: Tagger, dataset: Dataset, batch_size: int, device: str
) -> dict[str, float | int]:
    model.eval()
    model.qat = True

    correct = 0
    total_tokens = 0
    exact_sequences = 0
    sequences = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for indices in dataset.batches(batch_size):
        batch = dataset.batch(indices, device)
        labels = batch["labels"]
        mask = labels != -100
        roles, boundary = model(batch["rows"], batch["valid"], batch["neighbors"])

        predictions = roles.argmax(dim=-1)
        correct += int(((predictions == labels) & mask).sum())
        total_tokens += int(mask.sum())

        probability = torch.sigmoid(boundary)
        predicted = (probability >= 0.5) & mask
        target = (batch["boundaries"] >= 0.5) & mask
        true_positive += int((predicted & target).sum())
        false_positive += int((predicted & ~target).sum())
        false_negative += int((~predicted & target).sum())

        label_ok = (predictions == labels) | ~mask
        boundary_ok = ((probability >= 0.5) == (batch["boundaries"] >= 0.5)) | ~mask
        exact_sequences += int((label_ok & boundary_ok).all(dim=1).sum())
        sequences += len(indices)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tokenAccuracy": correct / max(total_tokens, 1),
        "exactLabelAndBoundarySequence": exact_sequences / max(sequences, 1),
        "boundaryPrecision": precision,
        "boundaryRecall": recall,
        "boundaryF1": f1,
        "tokens": total_tokens,
        "sequences": sequences,
        "boundaryTruePositive": true_positive,
        "boundaryFalsePositive": false_positive,
        "boundaryFalseNegative": false_negative,
    }


def save_checkpoint(
    path: Path,
    model: Tagger,
    config: Config,
    epoch: int,
    criterion: float,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "criterion": criterion,
            "metrics": metrics,
            "config": config.to_dict(),
            "quantizationBits": config.quantization_bits,
            "roleClasses": model.role_classes,
            "rowsPerToken": model.rows_per_token,
            "featureRows": model.feature_rows,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    config = config_module.parse_args(argv)
    set_seeds(config.seed)

    repo = repo_root()
    storage = Path(config.storage)
    if not storage.is_absolute():
        storage = repo / storage
    run_dir = storage / config.run
    data_dir = run_dir / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    heldout_prefix = prepare(
        "heldout", config.eval_samples, config.seed + HELDOUT_SEED_OFFSET, data_dir
    )
    heldout = Dataset(heldout_prefix)
    print(
        f"heldout: {len(heldout)} sequences, {heldout.token_count} tokens "
        f"({len(heldout.fingerprints)} fingerprints)"
    )

    validation_prefix = prepare(
        "validation",
        config.eval_samples,
        config.seed + VALIDATION_SEED_OFFSET,
        data_dir,
        exclude=fingerprints_path(heldout_prefix),
    )
    validation = Dataset(validation_prefix)
    print(
        f"validation: {len(validation)} sequences, {validation.token_count} tokens "
        f"({len(validation.fingerprints)} fingerprints)"
    )

    model = Tagger(
        feature_rows=validation.feature_rows,
        rows_per_token=validation.rows_per_token,
        role_classes=validation.role_classes,
        hidden=32,
        quantization_bits=config.quantization_bits,
    ).to(config.device)
    counts = parameter_counts(model)
    print(f"parameters: {counts['total']} total, {counts['trainable']} trainable")
    print(f"device: {config.device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = max(1, math.ceil(config.samples / config.batch))
    scheduler = make_scheduler(
        optimizer, total_steps=config.epochs * steps_per_epoch, warmup_steps=config.warmup_steps
    )

    history: list[dict] = []
    tokens_seen = 0
    best_criterion: float | None = None
    best_epoch: int | None = None
    latest_train = None
    start = time.time()

    for epoch in range(config.epochs):
        train_prefix = prepare(
            "train",
            config.samples,
            config.seed + epoch * EPOCH_SEED_STRIDE,
            data_dir,
            exclude=fingerprints_path(heldout_prefix),
        )
        train = Dataset(train_prefix)
        assert_disjoint(train, heldout)
        latest_train = train

        qat = epoch >= config.qat_start
        train_metrics = train_epoch(
            model, optimizer, scheduler, train, config.batch, config.device, qat
        )
        tokens_seen += int(train_metrics["trainTokens"])

        metrics = evaluate(model, validation, config.batch, config.device)
        criterion = float(metrics["tokenAccuracy"]) + float(
            metrics["exactLabelAndBoundarySequence"]
        )

        save_checkpoint(run_dir / "last.pt", model, config, epoch, criterion, metrics)
        if best_criterion is None or criterion > best_criterion:
            best_criterion = criterion
            best_epoch = epoch
            save_checkpoint(run_dir / "best.pt", model, config, epoch, criterion, metrics)

        entry = {
            "epoch": epoch,
            "qat": qat,
            "trainShard": train.digests(),
            "selectionCriterion": criterion,
            "validation": metrics,
            "train": train_metrics,
        }
        history.append(entry)
        print(
            f"epoch {epoch}: qat={qat} loss={train_metrics['trainLoss']:.4f} "
            f"tokenAcc={metrics['tokenAccuracy']:.4f} "
            f"exactSeq={metrics['exactLabelAndBoundarySequence']:.4f} "
            f"criterion={criterion:.4f}"
        )

    label_histogram = {
        "train": latest_train.label_counts if latest_train is not None else {},
        "validation": validation.label_counts,
        "heldout": heldout.label_counts,
    }
    totals: dict[str, int] = {}
    for counts_by_label in label_histogram.values():
        for label, value in counts_by_label.items():
            totals[label] = totals.get(label, 0) + int(value)
    label_histogram["total"] = totals

    report = {
        "selectionCriterion": (
            "validation.tokenAccuracy + validation.exactLabelAndBoundarySequence"
        ),
        "config": config.to_dict(),
        "sourceHashes": source_hashes(),
        "gitRev": git_revision(),
        "datasetDigests": {
            "train": latest_train.digests() if latest_train is not None else {},
            "validation": validation.digests(),
            "heldout": heldout.digests(),
        },
        "labelHistogram": label_histogram,
        "history": history,
        "tokensSeen": tokens_seen,
        "elapsedSeconds": time.time() - start,
        "parameters": counts,
        "bestCriterion": best_criterion,
        "bestEpoch": best_epoch,
    }
    write_report(run_dir / "report.json", report)
    print(f"best epoch {best_epoch} criterion {best_criterion}")
    print(f"report -> {run_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
