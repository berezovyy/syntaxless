"""The fixed trainer that `bun run train:cron` runs after corpus preparation.

Train each verified row once per epoch, select with actual calibrated TypeScript
intent parsing on complete validation and export the selected checkpoint. The
TypeScript coordinator owns final human regression and workflow completion.
"""
from __future__ import annotations

import hashlib
import math
import random
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import Config, production_config
from dataset import Dataset, POPULATION_NAMES, ROLE_LABELS, repo_root
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger, parameter_counts
from report import write_report
from selection import (
    REGRESSION_Z, decoded_preserves, decoded_rank, score_checkpoint, score_module,
    selects_checkpoint, significant_count_increase, significant_rate_drop, verify_preparation,
)

BOUNDARY_LOSS_WEIGHT = 1.0
SCHEDULE_KIND_LOSS_WEIGHT = 0.35
SPEECH_ACT_LOSS_WEIGHT = 0.35
ACTIONABLE_LOSS_WEIGHT = 1.0
EPOCH_SEED_STRIDE = 1_000_003
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0

HARNESS_SOURCES = (
    "config.py", "dataset.py", "model.py", "report.py", "train.py",
    "export.py", "selection.py", "abstain.py", "calibrate.py", "pyproject.toml", "uv.lock",
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


def assert_disjoint(
    left: Dataset, right: Dataset, left_name: str = "train", right_name: str = "heldout"
) -> None:
    """Reject fingerprints shared by two splits when the manifests list them.

    Shards may write an empty list to keep manifests small; their split
    disjointness is then checked during corpus preparation.
    """
    overlap = set(left.fingerprints) & set(right.fingerprints)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(
            f"split leak: {len(overlap)} fingerprint(s) shared by {left_name} and "
            f"{right_name} (e.g. {sample})"
        )


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _prepared_batches(
    dataset: Dataset,
    indices: Iterable[np.ndarray],
    device: str,
    *,
    prefetch: bool,
) -> Iterator[tuple[np.ndarray, dict[str, torch.Tensor]]]:
    """Overlap vectorized memmap gathering/pinning with the previous GPU step."""
    if not prefetch or not device.startswith("cuda"):
        for batch_indices in indices:
            yield batch_indices, dataset.batch(batch_indices, device)
        return
    iterator = iter(indices)
    first = next(iterator, None)
    if first is None:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-prefetch") as pool:
        future = pool.submit(dataset.batch, first, None, pin_memory=True)
        current = first
        while True:
            cpu_batch = future.result()
            following = next(iterator, None)
            if following is not None:
                future = pool.submit(dataset.batch, following, None, pin_memory=True)
            gpu_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
            }
            yield current, gpu_batch
            if following is None:
                break
            current = following


def train_epoch(
    model: Tagger,
    optimizer,
    scheduler,
    dataset: Dataset,
    batch_size: int,
    device: str,
    qat: bool,
    shuffle_seed: int,
    amp: bool,
    scaler: torch.amp.GradScaler,
    prefetch: bool,
    expected_sequences: int,
) -> dict[str, object]:
    model.train()
    base_model = getattr(model, "_orig_mod", model)
    base_model.qat = qat
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.int64)
    total_input_tokens = 0
    total_padded_tokens = 0
    head_loss_sums = torch.zeros(5, device=device, dtype=torch.float64)
    head_loss_rows = 0
    batches = 0
    sequences = 0
    started = time.perf_counter()
    population_counts = np.zeros(len(POPULATION_NAMES), dtype=np.int64)
    index_batches = dataset.batches(batch_size, shuffle_seed=shuffle_seed)
    for indices, batch in _prepared_batches(
        dataset, index_batches, device, prefetch=prefetch
    ):
        population_counts += np.bincount(
            dataset.populations[indices], minlength=len(POPULATION_NAMES),
        )
        labels = batch["labels"]
        mask = labels != -100

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            roles, boundary, schedule_kind, speech_act, actionable = model(
                batch["rows"], batch["valid"], batch["neighbors"]
            )
            role_loss = F.cross_entropy(
                roles.reshape(-1, roles.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary, batch["boundaries"], reduction="none"
            )
            boundary_loss = (boundary_loss * mask).sum() / mask.sum().clamp(min=1)
            schedule_kind_loss = F.cross_entropy(
                schedule_kind, batch["schedule_kinds"]
            )
            speech_act_loss = F.cross_entropy(speech_act, batch["speech_acts"])
            actionable_loss = F.binary_cross_entropy_with_logits(
                actionable, batch["actionable"], reduction="none"
            )
            actionable_loss = (
                actionable_loss * mask
            ).sum() / mask.sum().clamp(min=1)
            loss = (
                role_loss
                + BOUNDARY_LOSS_WEIGHT * boundary_loss
                + SCHEDULE_KIND_LOSS_WEIGHT * schedule_kind_loss
                + SPEECH_ACT_LOSS_WEIGHT * speech_act_loss
                + ACTIONABLE_LOSS_WEIGHT * actionable_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        token_count = mask.sum()
        total_loss += loss.detach().double() * token_count
        total_tokens += token_count
        total_input_tokens += int(dataset.lengths[indices].sum())
        total_padded_tokens += batch["valid"].numel()
        head_loss_sums += torch.stack((role_loss, boundary_loss, schedule_kind_loss,
                                      speech_act_loss, actionable_loss)).detach().double() * len(indices)
        head_loss_rows += len(indices)
        batches += 1
        sequences += len(indices)
        if batches % 100 == 0 or sequences >= expected_sequences:
            elapsed = time.perf_counter() - started
            rate = sequences / max(elapsed, 1e-9)
            remaining = max(expected_sequences - sequences, 0)
            peak = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.startswith("cuda")
                else 0.0
            )
            print(
                f"train {sequences:,}/{expected_sequences:,} sequences "
                f"({rate:,.0f}/s, ETA {remaining / max(rate, 1e-9) / 60:.1f}m, "
                f"CUDA peak {peak:.2f} GiB)",
                flush=True,
            )

    return {
        "trainLoss": float(total_loss / total_tokens.clamp(min=1)),
        "trainTokens": int(total_tokens),
        "inputTokens": total_input_tokens,
        "paddedTokens": total_padded_tokens,
        "sequencesPresented": sequences,
        "elapsedSeconds": time.perf_counter() - started,
        "headLosses": dict(zip(("role", "boundary", "scheduleKind", "speechAct", "actionable"),
                               (head_loss_sums / max(head_loss_rows, 1)).tolist(), strict=True)),
        "headLossAggregation": "sequence-weighted batch means; not per-example dynamics",
        "learningRate": float(scheduler.get_last_lr()[0]),
        "batches": batches,
        "trainPopulations": {
            name: int(population_counts[index])
            for index, name in enumerate(POPULATION_NAMES)
        },
    }


@torch.no_grad()
def evaluate(
    model: Tagger,
    dataset: Dataset,
    batch_size: int,
    device: str,
    *,
    amp: bool = False,
    prefetch: bool = False,
) -> dict[str, object]:
    model.eval()
    base_model = getattr(model, "_orig_mod", model)
    base_model.qat = True

    correct = 0
    total_tokens = 0
    exact_sequences = 0
    sequences = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    schedule_kind_correct = 0
    schedule_kind_totals = np.zeros(len(SCHEDULE_KIND_NAMES), dtype=np.int64)
    schedule_kind_correct_by_class = np.zeros(
        len(SCHEDULE_KIND_NAMES), dtype=np.int64
    )
    speech_act_correct = 0
    actionable_true_positive = 0
    actionable_false_positive = 0
    actionable_false_negative = 0
    evaluation_batches = 0
    evaluation_started = time.perf_counter()
    expected_sequences = len(dataset)
    frame_kind_names = [name.split("/", 1)[0] for name in dataset.manifest.frame_names]
    kind_names = list(dict.fromkeys(frame_kind_names))
    kind_index = {name: index for index, name in enumerate(kind_names)}
    frame_kind_ids = np.asarray(
        [kind_index[name] for name in frame_kind_names], dtype=np.int64
    )
    kind_totals = np.zeros(len(kind_names), dtype=np.int64)
    kind_correct = np.zeros(len(kind_names), dtype=np.int64)
    kind_joint_correct = np.zeros(len(kind_names), dtype=np.int64)
    schedule_joint_correct = np.zeros(len(SCHEDULE_KIND_NAMES), dtype=np.int64)
    speech_totals = np.zeros(len(SPEECH_ACT_NAMES), dtype=np.int64)
    speech_joint_correct = np.zeros(len(SPEECH_ACT_NAMES), dtype=np.int64)
    joint_sequences = 0
    positive_total = positive_joint_correct = negative_total = request_false_accepts = 0

    index_batches = dataset.batches(batch_size, shuffle_seed=0)
    for indices, batch in _prepared_batches(
        dataset, index_batches, device, prefetch=prefetch
    ):
        labels = batch["labels"]
        mask = labels != -100
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            roles, boundary, schedule_kind, speech_act, actionable = model(
                batch["rows"], batch["valid"], batch["neighbors"]
            )
        schedule_ok = schedule_kind.argmax(dim=-1) == batch["schedule_kinds"]
        speech_predictions = speech_act.argmax(dim=-1)
        speech_ok = speech_predictions == batch["speech_acts"]
        actionable_ok = (((actionable >= 0) == (batch["actionable"] >= 0.5)) | ~mask).all(dim=1)
        schedule_kind_correct += int(schedule_ok.sum())
        target_numpy = batch["schedule_kinds"].detach().cpu().numpy()
        correct_numpy = schedule_ok.detach().cpu().numpy()
        schedule_kind_totals += np.bincount(
            target_numpy, minlength=len(SCHEDULE_KIND_NAMES)
        )
        schedule_kind_correct_by_class += np.bincount(
            target_numpy,
            weights=correct_numpy.astype(np.int64),
            minlength=len(SCHEDULE_KIND_NAMES),
        ).astype(np.int64)
        speech_act_correct += int(speech_ok.sum())
        predicted_actionable = (actionable >= 0) & mask
        target_actionable = (batch["actionable"] >= 0.5) & mask
        actionable_true_positive += int(
            (predicted_actionable & target_actionable).sum()
        )
        actionable_false_positive += int(
            (predicted_actionable & ~target_actionable).sum()
        )
        actionable_false_negative += int(
            (~predicted_actionable & target_actionable).sum()
        )

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
        exact = (label_ok & boundary_ok).all(dim=1)
        exact_sequences += int(exact.sum())
        joint = exact & schedule_ok & speech_ok & actionable_ok
        joint_sequences += int(joint.sum())
        joint_numpy = joint.detach().cpu().numpy().astype(np.int64)
        schedule_joint_correct += np.bincount(
            batch["schedule_kinds"].detach().cpu().numpy(),
            weights=joint_numpy, minlength=len(SCHEDULE_KIND_NAMES),
        ).astype(np.int64)
        speech_numpy = batch["speech_acts"].detach().cpu().numpy()
        speech_totals += np.bincount(speech_numpy, minlength=len(SPEECH_ACT_NAMES))
        speech_joint_correct += np.bincount(
            speech_numpy, weights=joint_numpy, minlength=len(SPEECH_ACT_NAMES),
        ).astype(np.int64)
        positive = (batch["speech_acts"] == 1) | (batch["speech_acts"] == 5)
        requested = (speech_predictions == 1) | (speech_predictions == 5)
        positive_total += int(positive.sum())
        positive_joint_correct += int((positive & joint).sum())
        negative_total += int((~positive).sum())
        request_false_accepts += int((~positive & requested).sum())
        sequences += len(indices)
        evaluation_batches += 1
        if len(kind_names) > 0:
            batch_kind_ids = frame_kind_ids[np.asarray(dataset.frames[indices])]
            kind_totals += np.bincount(batch_kind_ids, minlength=len(kind_names))
            kind_correct += np.bincount(
                batch_kind_ids,
                weights=exact.detach().cpu().numpy().astype(np.int64),
                minlength=len(kind_names),
            ).astype(np.int64)
            kind_joint_correct += np.bincount(
                batch_kind_ids, weights=joint_numpy, minlength=len(kind_names),
            ).astype(np.int64)
        if evaluation_batches % 100 == 0 or sequences >= expected_sequences:
            elapsed = time.perf_counter() - evaluation_started
            rate = sequences / max(elapsed, 1e-9)
            print(
                f"evaluate {sequences:,}/{expected_sequences:,} sequences "
                f"({rate:,.0f}/s)",
                flush=True,
            )

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    by_kind = {
        name: {
            "sequences": int(kind_totals[index]),
            "exact": int(kind_correct[index]),
            "exactRate": float(kind_correct[index] / max(kind_totals[index], 1)),
            "jointExactRate": float(kind_joint_correct[index] / max(kind_totals[index], 1)),
        }
        for index, name in enumerate(kind_names)
    }
    macro_kind_exact = (
        float(np.mean([entry["exactRate"] for entry in by_kind.values() if entry["sequences"] > 0]))
        if by_kind
        else exact_sequences / max(sequences, 1)
    )
    by_schedule_kind = {
        name: {
            "sequences": int(schedule_kind_totals[index]),
            "correct": int(schedule_kind_correct_by_class[index]),
            "jointExactRate": float(schedule_joint_correct[index] / max(schedule_kind_totals[index], 1)),
            "accuracy": float(
                schedule_kind_correct_by_class[index]
                / max(schedule_kind_totals[index], 1)
            ),
        }
        for index, name in enumerate(SCHEDULE_KIND_NAMES)
    }
    schedule_kind_rates = [
        entry["accuracy"]
        for entry in by_schedule_kind.values()
        if entry["sequences"] > 0
    ]
    macro_schedule_kind_accuracy = (
        float(np.mean(schedule_kind_rates)) if schedule_kind_rates else 0.0
    )
    actionable_precision = actionable_true_positive / max(
        actionable_true_positive + actionable_false_positive, 1
    )
    actionable_recall = actionable_true_positive / max(
        actionable_true_positive + actionable_false_negative, 1
    )
    by_speech_act = {
        name: {
            "sequences": int(speech_totals[index]),
            "jointExactRate": float(speech_joint_correct[index] / max(speech_totals[index], 1)),
        }
        for index, name in enumerate(SPEECH_ACT_NAMES)
    }
    capability_rates = [
        entry["jointExactRate"]
        for group in (by_schedule_kind, by_speech_act)
        for entry in group.values() if entry["sequences"] > 0
    ]
    return {
        "jointHeadExactSequence": joint_sequences / max(sequences, 1),
        "macroJointHeadExactSequence": float(np.mean(capability_rates)) if capability_rates else 0.0,
        "positiveJointHeadExactSequence": positive_joint_correct / max(positive_total, 1),
        "requestDecisionFalseAccepts": request_false_accepts,
        "requestDecisionFalseAcceptRate": request_false_accepts / max(negative_total, 1),
        "positiveSequences": positive_total,
        "negativeSequences": negative_total,
        "bySpeechAct": by_speech_act,
        "tokenAccuracy": correct / max(total_tokens, 1),
        "exactLabelAndBoundarySequence": exact_sequences / max(sequences, 1),
        "macroKindExactSequence": macro_kind_exact,
        "speechActAccuracy": speech_act_correct / max(sequences, 1),
        "actionableF1": (
            2
            * actionable_precision
            * actionable_recall
            / max(actionable_precision + actionable_recall, 1e-12)
        ),
        "scheduleKindAccuracy": schedule_kind_correct / max(sequences, 1),
        "macroScheduleKindAccuracy": macro_schedule_kind_accuracy,
        "byScheduleKind": by_schedule_kind,
        "byKind": by_kind,
        "boundaryPrecision": precision,
        "boundaryRecall": recall,
        "boundaryF1": f1,
        "tokens": total_tokens,
        "sequences": sequences,
        "boundaryTruePositive": true_positive,
        "boundaryFalsePositive": false_positive,
        "boundaryFalseNegative": false_negative,
    }


def checkpoint_rank(metrics: dict) -> tuple[float, ...]:
    """One selection rule; all-head capability exactness, then request safety."""
    values = (
        metrics["macroJointHeadExactSequence"],
        -metrics["requestDecisionFalseAcceptRate"],
        metrics["positiveJointHeadExactSequence"],
        metrics["tokenAccuracy"],
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite validation metrics cannot select a checkpoint")
    return values


def checkpoint_preserves_capabilities(candidate: dict, current: dict) -> bool:
    """A higher average cannot pay for a measured validation regression."""
    if significant_count_increase(
        current["requestDecisionFalseAccepts"], candidate["requestDecisionFalseAccepts"]
    ):
        return False
    if candidate["positiveSequences"] != current["positiveSequences"]:
        raise ValueError("Checkpoint comparison requires identical validation supports")
    if significant_rate_drop(
        current["positiveJointHeadExactSequence"],
        candidate["positiveJointHeadExactSequence"],
        current["positiveSequences"],
    ):
        return False
    for category in ("byScheduleKind", "bySpeechAct", "byKind"):
        if candidate[category].keys() != current[category].keys():
            raise ValueError("Checkpoint comparison requires identical validation coverage")
        for name, previous in current[category].items():
            following = candidate[category][name]
            if following["sequences"] != previous["sequences"]:
                raise ValueError("Checkpoint comparison requires identical validation supports")
            if significant_rate_drop(
                previous["jointExactRate"], following["jointExactRate"], previous["sequences"]
            ):
                return False
    return True


def save_checkpoint(
    path: Path, model: Tagger, config: Config, epoch: int,
    rank: tuple[float, ...], metrics: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "criterion": rank[0],
            "rank": list(rank),
            "metrics": metrics,
            "config": config.to_dict(),
            "quantizationBits": config.quantization_bits,
            "roleClasses": model.role_classes,
            "rowsPerToken": model.rows_per_token,
            "featureRows": model.feature_rows,
            "scheduleKindClasses": model.schedule_kind_classes,
            "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
            "speechActClasses": model.speech_act_classes,
            "speechActLabels": list(SPEECH_ACT_NAMES),
            "actionableHead": model.actionable_head_enabled,
        },
        path,
    )


def validate_training_splits(datasets: tuple[Dataset, ...]) -> None:
    reference = datasets[0]
    for dataset in datasets:
        if len(dataset) == 0:
            raise ValueError(f"{dataset.prefix}: training requires non-empty splits")
        if (
            dataset.role_classes != len(ROLE_LABELS)
            or dataset.role_classes != reference.role_classes
            or dataset.feature_rows != reference.feature_rows
            or dataset.rows_per_token != reference.rows_per_token
            or dataset.manifest.speech_act_names != list(SPEECH_ACT_NAMES)
            or dataset.manifest.speech_act_classes != len(SPEECH_ACT_NAMES)
        ):
            raise ValueError(f"{dataset.prefix}: incompatible production supervision")


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("Training takes no flags. Use: bun run train:cron")
    started = time.perf_counter()
    config = production_config()
    if config.device == "cpu":
        torch.set_num_threads(2)
    set_seeds(config.seed)
    run_dir = Path(config.storage) / config.run
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / name).exists() for name in ("best.pt", "last.pt", "report.json", "weights.gen.ts")):
        raise ValueError(f"Training output already exists: {run_dir}")

    prefix = Path(config.data_prefix)
    corpus = verify_preparation(prefix)
    train = Dataset(Path(f"{prefix}-train"))
    validation = Dataset(Path(f"{prefix}-validation"))
    heldout = Dataset(Path(f"{prefix}-heldout"))
    validate_training_splits((train, validation, heldout))
    assert_disjoint(train, validation, "train", "validation")
    assert_disjoint(train, heldout)
    assert_disjoint(validation, heldout, "validation", "heldout")
    print(f"corpus: {prefix}", flush=True)
    print(f"train: {len(train):,}; validation: {len(validation):,}; heldout: {len(heldout):,}", flush=True)

    model = Tagger(
        feature_rows=train.feature_rows,
        rows_per_token=train.rows_per_token,
        role_classes=train.role_classes,
        schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
        speech_act_classes=len(SPEECH_ACT_NAMES),
        actionable_head=True,
        hidden=config.hidden,
        quantization_bits=config.quantization_bits,
    ).to(config.device)
    counts = parameter_counts(model)
    runtime_model = (
        torch.compile(model, mode="reduce-overhead", dynamic=False)
        if config.compile_model else model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = train.batch_count(config.batch)
    scheduler = make_scheduler(
        optimizer, total_steps=config.epochs * steps_per_epoch,
        warmup_steps=config.warmup_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    print(f"device: {config.device}; parameters: {counts['total']:,}; batch: {config.batch}", flush=True)
    print(f"epoch: one pass over {len(train):,} rows, {steps_per_epoch:,} optimizer steps", flush=True)

    history = []
    best_rank = None
    best_epoch = None
    for epoch in range(config.epochs):
        train_metrics = train_epoch(
            runtime_model, optimizer, scheduler, train, config.batch,
            config.device, epoch >= config.qat_start,
            config.seed + epoch * EPOCH_SEED_STRIDE, config.amp, scaler,
            config.prefetch, len(train),
        )
        if train_metrics["sequencesPresented"] != len(train) or train_metrics["batches"] != steps_per_epoch:
            raise RuntimeError("Training did not consume exactly one full corpus pass")
        if not math.isfinite(train_metrics["trainLoss"]):
            raise RuntimeError("Training produced a non-finite loss")
        metrics = evaluate(
            runtime_model, validation, config.batch, config.device,
            amp=config.amp, prefetch=config.prefetch,
        )
        save_checkpoint(run_dir / "last.pt", model, config, epoch, checkpoint_rank(metrics), metrics)
        decoded = score_checkpoint(
            run_dir / "last.pt", validation, corpus,
            device=config.device, batch_size=config.batch,
        )
        rank = decoded_rank(decoded)
        scored_metrics = {**metrics, "decodedValidation": decoded}
        save_checkpoint(run_dir / "last.pt", model, config, epoch, rank, scored_metrics)
        reference_epoch = best_epoch
        preserves = best_epoch is None or decoded_preserves(
            decoded, history[best_epoch]["decodedValidation"],
        )
        improved = selects_checkpoint(
            decoded, history[best_epoch]["decodedValidation"] if best_epoch is not None else None,
        )
        if improved:
            best_rank, best_epoch = rank, epoch
            save_checkpoint(run_dir / "best.pt", model, config, epoch, rank, scored_metrics)
        history.append({
            "epoch": epoch, "qat": epoch >= config.qat_start,
            "validation": metrics, "rank": list(rank), "train": train_metrics,
            "decodedValidation": decoded,
            "checkpointSelection": {
                "referenceEpoch": reference_epoch,
                "preservesValidationCapabilities": preserves,
                "referencePassedSyntheticGate": reference_epoch is not None and history[reference_epoch]["decodedValidation"]["syntheticGatePassed"],
                "predictionHeadPreservationDiagnostic": reference_epoch is None or checkpoint_preserves_capabilities(metrics, history[reference_epoch]["validation"]),
                "selected": improved,
            },
        })
        print(
            f"epoch {epoch + 1}/{config.epochs}: loss={train_metrics['trainLoss']:.4f} "
            f"jointHeadExact={metrics['jointHeadExactSequence']:.4f} "
            f"decodedIntentExact={decoded['positiveExactRate']:.4f} "
            f"macroDecodedIntentExact={rank[0]:.4f} "
            f"requestDecisionFalseAcceptRate={metrics['requestDecisionFalseAcceptRate']:.4f}",
            flush=True,
        )

    if best_epoch is None:
        raise RuntimeError("No valid checkpoint was produced")
    selected = torch.load(run_dir / "best.pt", map_location=config.device, weights_only=False)
    model.load_state_dict(selected["model"])
    full_heldout = evaluate(
        runtime_model, heldout, config.batch, config.device,
        amp=config.amp, prefetch=config.prefetch,
    )
    report = {
        "status": "trained",
        "config": config.to_dict(),
        "sourceHashes": source_hashes(),
        "gitRev": git_revision(),
        "parameters": counts,
        "datasetDigests": {
            "train": train.digests(), "validation": validation.digests(),
            "heldout": heldout.digests(),
        },
        "labelHistogram": {
            "train": train.label_counts, "validation": validation.label_counts,
            "heldout": heldout.label_counts,
        },
        "exposurePolicy": "each unique training row once per epoch, without replacement",
        "selectionCriterion": (
            "actual calibrated TypeScript parsing of the complete validation split; "
            "passing synthetic checkpoints take priority over development candidates; "
            "macro exact positive intent across semantic families, then overall exact "
            "positive intent, lower wrong accepted meanings, lower negative false "
            "accepts and higher raw positive intent accuracy; replacement of a passing checkpoint requires "
            "no safety increase or positive intent loss overall or within a semantic "
            "family, generator kind or speech act larger than "
            f"{REGRESSION_Z:g} standard errors (and one sequence)"
        ),
        "selectionPassed": history[best_epoch]["decodedValidation"]["syntheticGatePassed"],
        "bestEpoch": best_epoch,
        "bestCriterion": best_rank[0],
        "epochsCompleted": len(history),
        "history": history,
        "tokensSeen": sum(row["train"]["trainTokens"] for row in history),
        "inputTokensSeen": sum(row["train"]["inputTokens"] for row in history),
        "paddedTokensSeen": sum(row["train"]["paddedTokens"] for row in history),
        "trainingSeconds": sum(row["train"]["elapsedSeconds"] for row in history),
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "fullValidation": history[best_epoch]["validation"],
        "decodedValidation": history[best_epoch]["decodedValidation"],
        "fullHeldout": full_heldout,
    }
    write_report(run_dir / "report.json", report)

    from export import write_candidate_module

    candidate = write_candidate_module(
        run_dir / "best.pt", run_dir / "weights.gen.ts", validation,
        device=config.device, batch_size=config.batch,
        expected_artifact_sha256=history[best_epoch]["decodedValidation"]["artifactSha256"],
    )
    decoded_heldout = score_module(corpus, candidate, "heldout", run_dir / "heldout-runtime.json")
    selected_decoded = history[best_epoch]["decodedValidation"]
    if decoded_heldout["total"] != len(heldout) or any(
        decoded_heldout[key] != selected_decoded[key]
        for key in ("artifactSha256", "corpusSha256", "sourceHashes")
    ):
        raise ValueError("Heldout scoring differs from the frozen selected checkpoint")
    report.update({
        "status": "exported",
        "candidateModule": str(candidate),
        "decodedHeldout": decoded_heldout,
        "totalRunSeconds": time.perf_counter() - started,
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    write_report(run_dir / "report.json", report)
    print(f"checkpoint -> {run_dir / 'best.pt'}", flush=True)
    print(f"weights -> {candidate}", flush=True)
    print(f"report -> {run_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
