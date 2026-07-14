"""Training harness: load or regenerate splits, train, select on validation.

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
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import config as config_module
from config import Config
from dataset import (
    Dataset,
    FRESH_EPOCH_SIZE,
    POPULATION_NAMES,
    VAL_MAX_SEQUENCES,
    fingerprints_path,
    merge_fingerprint_excludes,
    prepare,
    repo_root,
)
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger, parameter_counts
from report import write_report

#: Weights the boundary term against the role term.
BOUNDARY_LOSS_WEIGHT = 1.0
SCHEDULE_KIND_LOSS_WEIGHT = 0.35
#: Utterance CE; same order as schedule-kind. Do not mix with role accuracy.
SPEECH_ACT_LOSS_WEIGHT = 0.35
#: Per-token BCE-with-logits, masked like boundary.
ACTIONABLE_LOSS_WEIGHT = 1.0
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


def assert_disjoint(
    left: Dataset, right: Dataset, left_name: str = "train", right_name: str = "heldout"
) -> None:
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
    balance_frames: bool,
    balance_kinds: bool,
    balance_schedule_kinds: bool,
    frame_balance_power: float,
    population_mix: tuple[float, float, float] | None,
    population_cover_all: bool,
    amp: bool,
    scaler: torch.amp.GradScaler,
    prefetch: bool,
    expected_sequences: int,
    max_sequences: int | None = None,
) -> dict[str, object]:
    model.train()
    base_model = getattr(model, "_orig_mod", model)
    base_model.qat = qat
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.int64)
    batches = 0
    sequences = 0
    started = time.perf_counter()
    population_counts = np.zeros(len(POPULATION_NAMES), dtype=np.int64)
    index_batches = dataset.batches(
        batch_size,
        shuffle_seed=shuffle_seed,
        balance_frames=balance_frames,
        balance_kinds=balance_kinds,
        balance_schedule_kinds=balance_schedule_kinds,
        frame_balance_power=frame_balance_power,
        population_mix=population_mix,
        population_cover_all=population_cover_all,
        max_sequences=max_sequences,
    )
    for indices, batch in _prepared_batches(
        dataset, index_batches, device, prefetch=prefetch
    ):
        if dataset.populations is not None:
            population_counts += np.bincount(
                np.asarray(dataset.populations)[indices],
                minlength=len(POPULATION_NAMES),
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
            schedule_kind_loss = (
                F.cross_entropy(schedule_kind, batch["schedule_kinds"])
                if schedule_kind.shape[-1] > 0
                else roles.new_zeros(())
            )
            speech_act_loss = (
                F.cross_entropy(speech_act, batch["speech_acts"])
                if speech_act.shape[-1] > 0
                else roles.new_zeros(())
            )
            if actionable.shape[-1] > 0:
                actionable_loss = F.binary_cross_entropy_with_logits(
                    actionable, batch["actionable"], reduction="none"
                )
                actionable_loss = (
                    actionable_loss * mask
                ).sum() / mask.sum().clamp(min=1)
            else:
                actionable_loss = roles.new_zeros(())
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
    max_sequences: int | None = None,
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
    expected_sequences = (
        len(dataset) if max_sequences is None else min(len(dataset), max_sequences)
    )
    frame_kind_names = [name.split("/", 1)[0] for name in dataset.manifest.frame_names]
    kind_names = list(dict.fromkeys(frame_kind_names))
    kind_index = {name: index for index, name in enumerate(kind_names)}
    frame_kind_ids = np.asarray(
        [kind_index[name] for name in frame_kind_names], dtype=np.int64
    )
    kind_totals = np.zeros(len(kind_names), dtype=np.int64)
    kind_correct = np.zeros(len(kind_names), dtype=np.int64)

    index_batches = dataset.batches(
        batch_size, shuffle_seed=0, max_sequences=max_sequences
    )
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
        if schedule_kind.shape[-1] > 0:
            schedule_kind_correct += int(
                (schedule_kind.argmax(dim=-1) == batch["schedule_kinds"]).sum()
            )
            schedule_predictions = schedule_kind.argmax(dim=-1)
            schedule_targets = batch["schedule_kinds"]
            target_numpy = schedule_targets.detach().cpu().numpy()
            correct_numpy = (
                schedule_predictions == schedule_targets
            ).detach().cpu().numpy()
            schedule_kind_totals += np.bincount(
                target_numpy, minlength=len(SCHEDULE_KIND_NAMES)
            )
            schedule_kind_correct_by_class += np.bincount(
                target_numpy,
                weights=correct_numpy.astype(np.int64),
                minlength=len(SCHEDULE_KIND_NAMES),
            ).astype(np.int64)
        if speech_act.shape[-1] > 0:
            speech_act_correct += int(
                (speech_act.argmax(dim=-1) == batch["speech_acts"]).sum()
            )
        if actionable.shape[-1] > 0:
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
        sequences += len(indices)
        evaluation_batches += 1
        if dataset.frames is not None and len(kind_names) > 0:
            batch_kind_ids = frame_kind_ids[np.asarray(dataset.frames[indices])]
            kind_totals += np.bincount(batch_kind_ids, minlength=len(kind_names))
            kind_correct += np.bincount(
                batch_kind_ids,
                weights=exact.detach().cpu().numpy().astype(np.int64),
                minlength=len(kind_names),
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
        }
        for index, name in enumerate(kind_names)
    }
    macro_kind_exact = (
        float(np.mean([entry["exactRate"] for entry in by_kind.values()]))
        if by_kind
        else exact_sequences / max(sequences, 1)
    )
    by_schedule_kind = {
        name: {
            "sequences": int(schedule_kind_totals[index]),
            "correct": int(schedule_kind_correct_by_class[index]),
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
    return {
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


def h_selection_score(h_metrics: dict[str, object] | None) -> float | None:
    """H-pos rate for checkpoint/early-stop; None when H was not scored.

    Never the composite H: it rewards a reject-all model (723/1646 = 44 %), so
    selecting on it stopped a1-full at epoch 3 while H-pos was still rising.
    """
    if not h_metrics or not h_metrics.get("ok"):
        return None
    total = h_metrics.get("positiveTotal")
    correct = h_metrics.get("positiveCorrect")
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    if not isinstance(correct, (int, float)):
        return None
    return float(correct) / float(total)


def score_h_if_requested(
    config: Config, run_dir: Path, model: Tagger, epoch: int
) -> dict[str, object] | None:
    """Log H after an epoch when `--h-per-epoch` is set.

    Export + human-evaluate is the real scorer. A missing corpus or a failed
    subprocess is recorded, not raised — training must still finish a ledger.
    """
    if not config.h_per_epoch:
        return None
    checkpoint = run_dir / f"epoch-{epoch}.pt"
    save_checkpoint(checkpoint, model, config, epoch, 0.0, {})
    repo = repo_root()
    candidate = run_dir / f"epoch-{epoch}.weights.gen.ts"
    export_cmd = [
        "uv",
        "run",
        "python",
        "export.py",
        "--checkpoint",
        str(checkpoint),
        "--candidate",
        str(candidate),
        "--candidate-only",
    ]
    exported = subprocess.run(
        export_cmd,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if exported.returncode != 0:
        return {
            "ok": False,
            "error": exported.stderr.strip() or exported.stdout.strip() or "export failed",
        }
    command = [
        "bun",
        "run",
        "packages/cron/training/src/human-evaluate.ts",
        "--partition",
        "regression",
        "--weights",
        str(candidate),
        "--out",
        str(run_dir / f"h-epoch-{epoch}.json"),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return {"ok": False, "error": str(error)}
    payload_path = run_dir / f"h-epoch-{epoch}.json"
    if payload_path.exists():
        import json

        try:
            parsed = json.loads(payload_path.read_text())
            off = parsed.get("overlayOff") or parsed
            return {
                "ok": True,
                # Composite (correct + correctly rejected): diagnostic only.
                "correct": parsed.get("correct"),
                "total": parsed.get("total"),
                # H-pos / H-negOK: the decision and safety metrics.
                "positiveCorrect": off.get("positiveCorrect"),
                "positiveTotal": off.get("positiveTotal"),
                "negativeCorrect": off.get("negativeCorrect"),
                "negativeTotal": off.get("negativeTotal"),
                "negatives": off.get("negativeFalseAccepts"),
                "silent": parsed.get("positiveSilentMeaningChanges"),
            }
        except (OSError, ValueError) as error:
            return {"ok": False, "error": str(error)}
    return {
        "ok": False,
        "error": result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}",
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
            "scheduleKindClasses": model.schedule_kind_classes,
            "scheduleKindLabels": list(SCHEDULE_KIND_NAMES),
            "speechActClasses": model.speech_act_classes,
            "speechActLabels": list(SPEECH_ACT_NAMES) if model.speech_act_classes > 0 else [],
            "actionableHead": model.actionable_head_enabled,
        },
        path,
    )


def autotune_batch_size(model: Tagger, dataset: Dataset, starting_batch: int) -> int:
    """Choose the fastest safe batch on representative long production rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("automatic batch tuning requires CUDA")
    # Width 64 covers the expensive tail while still containing enough rows to
    # test every candidate. The 3,501 width-96 rows are separately checked at
    # their real maximum batch below.
    long_indices = np.flatnonzero((dataset.lengths > 48) & (dataset.lengths <= 64))
    tail_indices = np.flatnonzero((dataset.lengths > 64) & (dataset.lengths <= 96))
    if long_indices.size == 0:
        long_indices = np.arange(min(len(dataset), 8192), dtype=np.int64)
    candidates: list[int] = []
    value = starting_batch
    while value <= 4096:
        candidates.append(value)
        value *= 2
    model.train()
    model.qat = True
    best = starting_batch
    best_rate = 0.0
    print(f"auto-batch GPU-only candidates: {candidates}", flush=True)
    for candidate in candidates:
        indices = np.asarray(long_indices[:candidate])
        if indices.size < candidate:
            break
        try:
            batch = dataset.batch(indices, "cuda")
            # Warm up allocator/kernels, then time two complete forward/backward passes.
            elapsed = 0.0
            for iteration in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(batch["rows"], batch["valid"], batch["neighbors"])
                    loss = sum(output.float().square().mean() for output in outputs)
                loss.backward()
                torch.cuda.synchronize()
                if iteration:
                    elapsed += time.perf_counter() - started
            rate = candidate * 2 / max(elapsed, 1e-9)
            del batch, outputs, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # Validate the actual width-96 tail too; it can contain at most 3,501 rows.
            if tail_indices.size:
                tail = dataset.batch(
                    np.asarray(tail_indices[: min(candidate, tail_indices.size)]), "cuda"
                )
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(tail["rows"], tail["valid"], tail["neighbors"])
                    tail_loss = sum(
                        output.float().square().mean() for output in outputs
                    )
                tail_loss.backward()
                torch.cuda.synchronize()
                del tail, outputs, tail_loss
            peak = torch.cuda.max_memory_allocated() / 1024**3
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            headroom = free / 1024**3 - peak
            print(
                f"auto-batch GPU-only {candidate}: {rate:,.0f} sequences/s, "
                f"{peak:.2f} GiB peak, {headroom:.2f} GiB headroom",
                flush=True,
            )
            if headroom < 2.0:
                print(
                    "auto-batch stopped: less than 2 GiB CUDA headroom", flush=True
                )
                break
            if rate > best_rate * 1.02:
                best = candidate
                best_rate = rate
        except torch.OutOfMemoryError:
            print(f"auto-batch GPU-only {candidate}: out of memory", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
    print(f"auto-batch selected: {best} (GPU compute benchmark)", flush=True)
    return best


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

    external_train: Dataset | None = None
    if config.data_prefix is not None:
        external_base = Path(config.data_prefix)
        if not external_base.is_absolute():
            external_base = repo / external_base
        heldout_prefix = Path(f"{external_base}-heldout")
        validation_prefix = Path(f"{external_base}-validation")
        train_prefix = Path(f"{external_base}-train")
        external_train = Dataset(train_prefix)
        heldout = Dataset(heldout_prefix)
        validation = Dataset(validation_prefix)
        assert_disjoint(external_train, heldout)
        assert_disjoint(external_train, validation, "train", "validation")
        assert_disjoint(validation, heldout, "validation", "heldout")
        print(f"external corpus: {external_base}-{{train,validation,heldout}}")
    else:
        heldout_prefix = prepare(
            "heldout", config.eval_samples, config.seed + HELDOUT_SEED_OFFSET, data_dir
        )
        heldout = Dataset(heldout_prefix)
        validation_prefix = prepare(
            "validation",
            config.eval_samples,
            config.seed + VALIDATION_SEED_OFFSET,
            data_dir,
            exclude=fingerprints_path(heldout_prefix),
        )
        validation = Dataset(validation_prefix)
    print(
        f"heldout: {len(heldout)} sequences, {heldout.token_count} tokens "
        f"({len(heldout.fingerprints)} fingerprints)"
    )

    print(
        f"validation: {len(validation)} sequences, {validation.token_count} tokens "
        f"({len(validation.fingerprints)} fingerprints)"
    )

    model = Tagger(
        feature_rows=validation.feature_rows,
        rows_per_token=validation.rows_per_token,
        role_classes=validation.role_classes,
        schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
        speech_act_classes=validation.manifest.speech_act_classes,
        actionable_head="actionable" in validation.manifest.format,
        hidden=config.hidden,
        quantization_bits=config.quantization_bits,
    ).to(config.device)
    if config.init is not None:
        init_path = Path(config.init)
        if not init_path.is_absolute():
            init_path = repo / init_path
        loaded = torch.load(init_path, map_location=config.device)
        model.load_state_dict(loaded["model"])
        print(f"init from {init_path}")
    counts = parameter_counts(model)
    print(f"parameters: {counts['total']} total, {counts['trainable']} trainable")
    print(f"device: {config.device}")

    if config.auto_batch:
        if external_train is None:
            raise ValueError("--auto-batch requires --data-prefix")
        config.batch = autotune_batch_size(model, external_train, config.batch)
    runtime_model = (
        torch.compile(model, mode="reduce-overhead", dynamic=False)
        if config.compile_model
        else model
    )
    if config.compile_model:
        print("model compilation enabled (reduce-overhead)", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    if external_train is not None and config.fresh_each_epoch:
        training_sequences = min(
            len(external_train), config.fresh_epoch_size, FRESH_EPOCH_SIZE
        )
    elif (
        external_train is not None
        and config.population_mix is not None
        and config.population_cover_all
    ):
        training_sequences = external_train.population_epoch_size(
            config.population_mix, cover_all=True
        )
    elif external_train is not None:
        training_sequences = min(len(external_train), config.samples)
    else:
        training_sequences = config.samples
    steps_per_epoch = max(1, math.ceil(training_sequences / config.batch))
    scheduler = make_scheduler(
        optimizer, total_steps=config.epochs * steps_per_epoch, warmup_steps=config.warmup_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    print(
        f"epoch plan: {training_sequences:,} sequences, batch {config.batch}, "
        f"~{steps_per_epoch:,} optimizer steps",
        flush=True,
    )

    history: list[dict] = []
    tokens_seen = 0
    best_criterion: float | None = None
    best_epoch: int | None = None
    non_improving_epochs = 0
    stopped_early = False
    latest_train = None
    start = time.time()

    for epoch in range(config.epochs):
        if external_train is not None:
            train = external_train
        else:
            train_exclude = merge_fingerprint_excludes(
                [
                    fingerprints_path(heldout_prefix),
                    fingerprints_path(validation_prefix),
                ],
                data_dir / "train-exclude.fingerprints.json",
            )
            train_prefix = prepare(
                "train",
                config.samples,
                config.seed + epoch * EPOCH_SEED_STRIDE,
                data_dir,
                exclude=train_exclude,
            )
            train = Dataset(train_prefix)
        assert_disjoint(train, heldout)
        assert_disjoint(train, validation, "train", "validation")
        latest_train = train

        qat = epoch >= config.qat_start
        epoch_seed = config.seed + epoch * EPOCH_SEED_STRIDE
        if config.fresh_each_epoch:
            train_cap = min(config.fresh_epoch_size, len(train), FRESH_EPOCH_SIZE)
        elif config.population_cover_all:
            train_cap = None
        elif external_train is not None:
            train_cap = training_sequences
        else:
            train_cap = None
        train_metrics = train_epoch(
            runtime_model,
            optimizer,
            scheduler,
            train,
            config.batch,
            config.device,
            qat,
            epoch_seed,
            config.balance_frames,
            config.balance_kinds,
            config.balance_schedule_kinds,
            config.frame_balance_power,
            config.population_mix,
            config.population_cover_all,
            config.amp,
            scaler,
            config.prefetch,
            training_sequences,
            train_cap,
        )
        tokens_seen += int(train_metrics["trainTokens"])

        val_cap = min(len(validation), config.val_max_sequences, VAL_MAX_SEQUENCES)
        metrics = evaluate(
            runtime_model,
            validation,
            config.batch,
            config.device,
            val_cap,
            amp=config.amp,
            prefetch=config.prefetch,
        )
        # Equalize checkpoint selection across sentence kinds. The global
        # sequence metric is dominated by large Cartesian schedule lattices;
        # macro-kind exactness makes terse and negative language count too.
        criterion = (
            float(metrics["tokenAccuracy"])
            + float(metrics["macroKindExactSequence"])
            + 0.25 * float(metrics["macroScheduleKindAccuracy"])
        )

        h_metrics = score_h_if_requested(config, run_dir, model, epoch)
        h_score = h_selection_score(h_metrics)
        if h_score is not None:
            criterion = h_score

        save_checkpoint(run_dir / "last.pt", model, config, epoch, criterion, metrics)
        improved = best_criterion is None or criterion > best_criterion
        if improved:
            best_criterion = criterion
            best_epoch = epoch
            non_improving_epochs = 0
            save_checkpoint(run_dir / "best.pt", model, config, epoch, criterion, metrics)
        else:
            non_improving_epochs += 1
        entry = {
            "epoch": epoch,
            "qat": qat,
            "trainShard": train.digests(),
            "selectionCriterion": criterion,
            "validation": metrics,
            "train": train_metrics,
            "h": h_metrics,
        }
        history.append(entry)
        h_note = ""
        if h_metrics is not None:
            h_note = (
                f" H={h_metrics.get('correct', '?')}/{h_metrics.get('total', '?')}"
            )
        print(
            f"epoch {epoch}: qat={qat} loss={train_metrics['trainLoss']:.4f} "
            f"tokenAcc={metrics['tokenAccuracy']:.4f} "
            f"exactSeq={metrics['exactLabelAndBoundarySequence']:.4f} "
            f"macroKind={metrics['macroKindExactSequence']:.4f} "
            f"criterion={criterion:.4f}{h_note}"
        )
        if (
            config.early_stopping_patience > 0
            and epoch + 1 >= config.min_epochs
            and non_improving_epochs >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early stop after {epoch + 1} epochs: "
                f"{non_improving_epochs} epochs without validation improvement"
            )
            break

    if best_epoch is not None:
        selected = torch.load(
            run_dir / "best.pt", map_location=config.device, weights_only=False
        )
        model.load_state_dict(selected["model"])
        print(f"selected best checkpoint from epoch {best_epoch}", flush=True)
    final_cap = None if config.full_final_eval else config.val_max_sequences
    full_validation = evaluate(
        runtime_model,
        validation,
        config.batch,
        config.device,
        final_cap,
        amp=config.amp,
        prefetch=config.prefetch,
    )
    full_heldout = (
        evaluate(
            runtime_model,
            heldout,
            config.batch,
            config.device,
            None,
            amp=config.amp,
            prefetch=config.prefetch,
        )
        if config.full_final_eval
        else None
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
            "validation.tokenAccuracy + validation.macroKindExactSequence + "
            "0.25 * validation.scheduleKindAccuracy"
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
        "epochsCompleted": len(history),
        "stoppedEarly": stopped_early,
        "fullValidation": full_validation,
        "fullHeldout": full_heldout,
    }
    write_report(run_dir / "report.json", report)
    print(f"best epoch {best_epoch} criterion {best_criterion}")
    print(f"report -> {run_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
