"""Export the selected checkpoint as hash-verified int6 weights.

The exporter is the only producer of `packages/cron/src/model/weights.gen.ts`.
It symmetrically quantizes every parameter, packs the integers into a compact
alphabet string, then *decodes that exact string back into a fresh PyTorch
module* and measures that module. An in-memory checkpoint is not the artifact
that ships, so calibration, evaluation, and the frozen parity fixture are all
computed from the decoded wire artifact.

The int6 range is `[-31, 31]`, so a symbol is `value + 31` in `[0, 62]`. One
segment per parameter records `name`, `offset`, `length`, `shape`, and `scale`.
`artifactSha256` hashes the packed `q` payload. The report is written last and
is the commit marker: an export without a report is not valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from calibrate import Calibration, calibrate
from dataset import Dataset, fingerprints_path, prepare, repo_root
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger, parameter_counts

#: The shipped format is int6 by contract.
QUANTIZATION_BITS = 6
QUANTIZATION_CEILING = (1 << (QUANTIZATION_BITS - 1)) - 1  # 31
#: `value + SYMBOL_BIAS` indexes the alphabet; 0..62 covers -31..31.
SYMBOL_BIAS = 31
#: Compact 64-symbol alphabet. One character per quantized parameter element.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(ALPHABET)}
FORMAT_VERSION = 1
STORAGE = "alphabet"
#: Fixtures freeze at most this many held-out sequences for Phase 16 parity.
MAX_PARITY_SEQUENCES = 512
#: Default output locations, relative to the repository root. The exporter never
#: writes the active module directly: it emits a candidate, runs the Phase 19
#: gate, and promotes only on acceptance.
DEFAULT_CANDIDATE = "packages/cron/training/candidate/weights.gen.ts"
DEFAULT_ACTIVE_MODULE = "packages/cron/src/model/weights.gen.ts"
DEFAULT_CANDIDATE_REPORT = "training/cron/candidate/export-report.json"
DEFAULT_REPORT = "training/cron/active/export-report.json"
DEFAULT_FIXTURES = "packages/cron/tests/fixtures"
DEFAULT_DIST = "packages/cron/dist"
DEFAULT_CORPUS = "packages/cron/training/data/gold/reserved-carriers.jsonl"
DEFAULT_TRAIN_SPLIT = (
    "packages/cron/training/data/synth/kinds.jsonl.fingerprints.json"
)
#: Files hashed into the report provenance block.
EXPORT_SOURCES = (
    "calibrate.py",
    "config.py",
    "dataset.py",
    "export.py",
    "model.py",
    "report.py",
    "test_export.py",
    "train.py",
    "pyproject.toml",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Repo-relative where possible so a committed report stays portable."""
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    return {
        name: sha256_file(base / name)
        for name in EXPORT_SOURCES
        if (base / name).exists()
    }


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


def load_labels() -> list[str]:
    """Read the one label vocabulary from the Phase 07 TypeScript source."""
    source = repo_root() / "packages" / "cron" / "src" / "model" / "labels.ts"
    text = source.read_text()
    start = text.index("export const LABELS")
    open_bracket = text.index("[", start)
    close = text.index("] as const", open_bracket)
    labels = re.findall(r'"([^"]+)"', text[open_bracket + 1 : close])
    if not labels:
        raise ValueError(f"could not parse LABELS from {source}")
    return labels


def quantize_parameter(
    parameter: torch.Tensor, ceiling: int = QUANTIZATION_CEILING
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Symmetric per-tensor int-`bits` quantization of one parameter.

    Returns the integer codes, their dequantized float tensor, and the scale.
    """
    values = parameter.detach().to(torch.float32)
    max_abs = float(values.abs().max().item()) if values.numel() else 0.0
    scale = max(max_abs, 1e-8) / ceiling
    codes = torch.round(values / scale).clamp(-ceiling, ceiling).to(torch.int64)
    dequantized = codes.to(torch.float32) * scale
    return codes.reshape(-1), dequantized.reshape(parameter.shape), scale


def pack_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[list[dict], str]:
    """Quantize every parameter and pack its codes into the alphabet string."""
    segments: list[dict] = []
    symbols: list[str] = []
    offset = 0
    for name, parameter in state.items():
        codes, _, scale = quantize_parameter(parameter)
        length = int(codes.numel())
        symbols.append(
            "".join(ALPHABET[int(code) + SYMBOL_BIAS] for code in codes.tolist())
        )
        segments.append(
            {
                "name": name,
                "offset": offset,
                "length": length,
                "shape": list(parameter.shape),
                "scale": scale,
            }
        )
        offset += length
    return segments, "".join(symbols)


def decode_payload(
    q: str, offset: int, length: int, scale: float, shape: list[int]
) -> torch.Tensor:
    """Decode one segment of the packed payload into a torch tensor."""
    if offset < 0 or length < 0 or offset + length > len(q):
        raise ValueError("segment range exceeds the packed payload")
    product = 1
    for dimension in shape:
        product *= int(dimension)
    if product != length:
        raise ValueError(f"shape {shape} does not match length {length}")
    values = np.empty(length, dtype=np.float32)
    scale32 = np.float32(scale)
    for index, symbol in enumerate(q[offset : offset + length]):
        position = SYMBOL_INDEX.get(symbol)
        if position is None:
            raise ValueError(f"unknown symbol {symbol!r} in packed payload")
        values[index] = np.float32(position - SYMBOL_BIAS) * scale32
    return torch.from_numpy(values).reshape(shape)


def decode_artifact(artifact: dict) -> dict[str, torch.Tensor]:
    """Decode every segment of a wire artifact into a state dict."""
    state: dict[str, torch.Tensor] = {}
    for segment in artifact["segments"]:
        if segment["name"] in state:
            raise ValueError(f"duplicate segment {segment['name']!r}")
        state[segment["name"]] = decode_payload(
            artifact["q"],
            int(segment["offset"]),
            int(segment["length"]),
            float(segment["scale"]),
            list(segment["shape"]),
        )
    return state


def build_decoded_model(artifact: dict, device: str = "cpu") -> Tagger:
    """Rebuild a Tagger from the decoded wire artifact, not from the checkpoint."""
    model = Tagger(
        feature_rows=int(artifact["featureRows"]),
        rows_per_token=int(artifact["rowsPerToken"]),
        role_classes=int(artifact["roleClasses"]),
        schedule_kind_classes=int(artifact.get("scheduleKindClasses", 0)),
        speech_act_classes=int(artifact.get("speechActClasses", 0)),
        actionable_head=any(
            segment.get("name") == "actionable_head.weight"
            for segment in artifact.get("segments", [])
        ),
        hidden=int(artifact["hidden"]),
        quantization_bits=QUANTIZATION_BITS,
    ).to(device)
    model.load_state_dict(decode_artifact(artifact))
    model.qat = False
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: Tagger, dataset: Dataset, threshold: float, batch_size: int, device: str
) -> dict:
    """Role accuracy plus boundary precision/recall/F1 at the calibrated threshold."""
    model.eval()
    model.qat = False
    device = next(model.parameters()).device
    correct = 0
    total_tokens = 0
    exact_sequences = 0
    sequences = 0
    schedule_kind_correct = 0
    schedule_kind_total = 0
    true_positive = false_positive = false_negative = 0
    for indices in dataset.batches(batch_size):
        batch = dataset.batch(indices, device)
        labels = batch["labels"]
        mask = labels != -100
        roles, boundary, schedule_kind, _speech_act, _actionable = model(
            batch["rows"], batch["valid"], batch["neighbors"]
        )
        if schedule_kind.shape[-1] > 0:
            schedule_kind_correct += int(
                (schedule_kind.argmax(dim=-1) == batch["schedule_kinds"]).sum()
            )
            schedule_kind_total += len(indices)
        predictions = roles.argmax(dim=-1)
        correct += int(((predictions == labels) & mask).sum())
        total_tokens += int(mask.sum())

        predicted = (boundary >= threshold) & mask
        target = (batch["boundaries"] >= 0.5) & mask
        true_positive += int((predicted & target).sum())
        false_positive += int((predicted & ~target).sum())
        false_negative += int((~predicted & target).sum())

        label_ok = (predictions == labels) | ~mask
        boundary_ok = ((boundary >= threshold) == (batch["boundaries"] >= 0.5)) | ~mask
        exact_sequences += int((label_ok & boundary_ok).all(dim=1).sum())
        sequences += len(indices)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics = {
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
    if schedule_kind_total > 0:
        metrics["scheduleKindAccuracy"] = schedule_kind_correct / schedule_kind_total
    return metrics


@torch.no_grad()
def freeze_parity(
    model: Tagger,
    dataset: Dataset,
    fixtures_dir: Path,
    max_sequences: int = MAX_PARITY_SEQUENCES,
) -> dict:
    """Freeze held-out rows, offsets, and decoded-artifact logits for Phase 16.

    Sequences are encoded exactly as featurized (no padding), so the fixture is
    the reference the runtime must reproduce; a runtime padding test can pad a
    short fixture input and rely on padding neutrality.
    """
    model.eval()
    model.qat = False
    device = next(model.parameters()).device
    count = min(int(max_sequences), len(dataset))
    rows_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    boundaries_parts: list[np.ndarray] = []
    schedule_kind_parts: list[np.ndarray] = []
    speech_act_parts: list[np.ndarray] = []
    actionable_parts: list[np.ndarray] = []
    offsets = [0]
    for index in range(count):
        start = int(dataset.offsets[index])
        end = int(dataset.offsets[index + 1])
        rows = dataset.rows[start:end]
        neighbors = dataset.neighbors[start:end]
        row_tensor = torch.as_tensor(rows.astype(np.int64)).unsqueeze(0).to(device)
        valid = torch.ones(1, end - start, dtype=torch.bool, device=device)
        neighbor_tensor = (
            torch.as_tensor(neighbors.astype(np.int64)).unsqueeze(0).to(device)
        )
        roles, boundary, schedule_kind, speech_act, actionable = model(
            row_tensor, valid, neighbor_tensor
        )
        rows_parts.append(rows)
        logits_parts.append(roles[0].detach().cpu().numpy().astype(np.float32))
        boundaries_parts.append(boundary[0].detach().cpu().numpy().astype(np.float32))
        if schedule_kind.shape[-1] > 0:
            schedule_kind_parts.append(
                schedule_kind[0].detach().cpu().numpy().astype(np.float32)
            )
        if speech_act.shape[-1] > 0:
            speech_act_parts.append(
                speech_act[0].detach().cpu().numpy().astype(np.float32)
            )
        if actionable.shape[-1] > 0:
            actionable_parts.append(
                actionable[0].detach().cpu().numpy().astype(np.float32)
            )
        offsets.append(offsets[-1] + (end - start))

    rows_all = np.concatenate(rows_parts, axis=0).astype(np.uint16)
    logits_all = np.concatenate(logits_parts, axis=0).astype(np.float32)
    boundaries_all = np.concatenate(boundaries_parts, axis=0).astype(np.float32)
    offsets_all = np.asarray(offsets, dtype=np.uint32)

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "rows": ("parity.rows.bin", rows_all.tobytes()),
        "offsets": ("parity.offsets.bin", offsets_all.tobytes()),
        "logits": ("parity.logits.bin", logits_all.tobytes()),
        "boundaries": ("parity.boundaries.bin", boundaries_all.tobytes()),
    }
    if schedule_kind_parts:
        schedule_kinds_all = np.stack(schedule_kind_parts).astype(np.float32)
        files["scheduleKinds"] = (
            "parity.schedule-kinds.bin",
            schedule_kinds_all.tobytes(),
        )
    if speech_act_parts:
        speech_acts_all = np.stack(speech_act_parts).astype(np.float32)
        files["speechActs"] = (
            "parity.speech-acts.bin",
            speech_acts_all.tobytes(),
        )
    if actionable_parts:
        actionable_all = np.concatenate(actionable_parts).astype(np.float32)
        files["actionable"] = (
            "parity.actionable.bin",
            actionable_all.tobytes(),
        )
    summary = {
        "sequences": count,
        "tokens": int(offsets_all[-1]),
        "roleClasses": int(logits_all.shape[1]),
    }
    if schedule_kind_parts:
        summary["scheduleKindClasses"] = len(SCHEDULE_KIND_NAMES)
    summary["files"] = {}
    for key, (name, payload) in files.items():
        write_bytes_atomic(fixtures_dir / name, payload)
        summary["files"][key] = {
            "path": display_path(fixtures_dir / name),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    return summary


def render_module(artifact: dict, artifact_sha: str) -> str:
    payload = json.dumps(artifact, indent=2, ensure_ascii=True)
    return (
        "/**\n"
        " * Generated by training/cron/export.py. Do not edit by hand.\n"
        f" * artifactSha256: {artifact_sha}\n"
        " */\n"
        f"export const weights = {payload} as const;\n"
    )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def load_datasets(
    checkpoint: dict, data_dir: Path | None
) -> tuple[Dataset, Dataset, Path]:
    """Load validation and held-out shards, including external corpus runs."""
    run = checkpoint.get("config", {})
    external_prefix = run.get("data_prefix")
    if data_dir is None and external_prefix:
        base = resolve_path(external_prefix)
        validation_prefix = Path(f"{base}-validation")
        heldout_prefix = Path(f"{base}-heldout")
        for prefix in (validation_prefix, heldout_prefix):
            if not Path(f"{prefix}.json").exists():
                raise FileNotFoundError(f"external shard missing: {prefix}.json")
        return Dataset(validation_prefix), Dataset(heldout_prefix), base

    if data_dir is None:
        data_dir = Path(checkpoint.get("checkpointDirectory", ""))
        if not str(data_dir):
            raise ValueError("data_dir is required for a non-external checkpoint")
    validation_prefix = data_dir / "validation"
    heldout_prefix = data_dir / "heldout"
    if not Path(f"{heldout_prefix}.json").exists():
        prepare(
            "heldout",
            int(run.get("eval_samples", 500)),
            int(run.get("seed", 0)) + 2,
            data_dir,
        )
    if not Path(f"{validation_prefix}.json").exists():
        prepare(
            "validation",
            int(run.get("eval_samples", 500)),
            int(run.get("seed", 0)) + 1,
            data_dir,
            exclude=fingerprints_path(heldout_prefix),
        )
    return Dataset(validation_prefix), Dataset(heldout_prefix), data_dir


def build_artifact(
    segments: list[dict],
    q: str,
    labels: list[str],
    checkpoint: dict,
    hidden: int,
    threshold: float,
) -> dict:
    artifact = {
        "version": FORMAT_VERSION,
        "hidden": hidden,
        "featureRows": int(checkpoint["featureRows"]),
        "rowsPerToken": int(checkpoint["rowsPerToken"]),
        "storage": STORAGE,
        "roleClasses": int(checkpoint["roleClasses"]),
        "boundaryThreshold": float(threshold),
        "labels": labels,
        "q": q,
        "segments": segments,
    }
    schedule_kind_classes = int(checkpoint.get("scheduleKindClasses", 0))
    if schedule_kind_classes > 0:
        artifact["scheduleKindClasses"] = schedule_kind_classes
        artifact["scheduleKindLabels"] = list(checkpoint["scheduleKindLabels"])
    speech_act_classes = int(checkpoint.get("speechActClasses", 0))
    if speech_act_classes > 0:
        labels = list(checkpoint.get("speechActLabels", SPEECH_ACT_NAMES))
        if labels != list(SPEECH_ACT_NAMES):
            raise ValueError("speechActLabels must match SPEECH_ACT_NAMES")
        artifact["speechActClasses"] = speech_act_classes
        artifact["speechActLabels"] = labels
    return artifact


GATE_PACKAGE = "@syntaxless/cron-training"
GATE_SCRIPT = "gate"


def run_gate(
    candidate_path: Path,
    baseline_path: Path,
    dist_dir: Path,
    corpus_path: Path,
    train_split: Path,
    out_path: Path,
) -> tuple[int, dict | None, str]:
    """Invoke the Phase 19 gate from the repo root and read its report."""
    command = [
        "bun",
        "run",
        "--filter",
        GATE_PACKAGE,
        GATE_SCRIPT,
        "--",
        "--dist",
        str(dist_dir),
        "--candidate",
        str(candidate_path),
        "--baseline",
        str(baseline_path),
        "--corpus",
        str(corpus_path),
        "--train-split",
        str(train_split),
        "--out",
        str(out_path),
    ]
    completed = subprocess.run(
        command, cwd=repo_root(), capture_output=True, text=True
    )
    report: dict | None = None
    if out_path.exists():
        report = json.loads(out_path.read_text())
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode, report, output


def export(
    checkpoint_path: Path,
    candidate_path: Path,
    active_path: Path,
    candidate_report_path: Path,
    active_report_path: Path,
    fixtures_dir: Path,
    dist_dir: Path,
    corpus_path: Path,
    train_split: Path,
    data_dir: Path | None = None,
    batch_size: int = 64,
    device: str = "cpu",
    max_parity: int = MAX_PARITY_SEQUENCES,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    bits = int(checkpoint.get("quantizationBits", QUANTIZATION_BITS))
    if bits != QUANTIZATION_BITS:
        raise ValueError(f"checkpoint uses int{bits}; the exporter ships int6")

    state = checkpoint["model"]
    labels = load_labels()
    role_classes = int(checkpoint["roleClasses"])
    if len(labels) != role_classes:
        raise ValueError(
            f"labels.ts has {len(labels)} labels but the checkpoint has {role_classes}"
        )
    schedule_kind_classes = int(checkpoint.get("scheduleKindClasses", 0))
    schedule_kind_labels = list(checkpoint.get("scheduleKindLabels", []))
    if schedule_kind_classes > 0 and (
        schedule_kind_classes != len(SCHEDULE_KIND_NAMES)
        or schedule_kind_labels != list(SCHEDULE_KIND_NAMES)
    ):
        raise ValueError("checkpoint schedule-kind vocabulary does not match runtime")
    speech_act_classes = int(checkpoint.get("speechActClasses", 0))
    speech_act_labels = list(checkpoint.get("speechActLabels", []))
    if speech_act_classes > 0 and (
        speech_act_classes != len(SPEECH_ACT_NAMES)
        or speech_act_labels != list(SPEECH_ACT_NAMES)
    ):
        raise ValueError("checkpoint speech-act vocabulary does not match runtime")
    hidden = int(state["embedding.weight"].shape[1])

    segments, q = pack_state(state)
    artifact_sha = sha256_text(q)

    if data_dir is None and not checkpoint.get("config", {}).get("data_prefix"):
        data_dir = checkpoint_path.parent / "data"
    validation, heldout, resolved_data = load_datasets(checkpoint, data_dir)

    artifact = build_artifact(segments, q, labels, checkpoint, hidden, 0.0)
    decoded = build_decoded_model(artifact, device)

    # Validation is the calibration split. Held-out remains untouched until the
    # threshold is frozen, otherwise its reported score participates in model
    # selection and ceases to be held out.
    calibration: Calibration = calibrate(
        decoded, [validation], batch_size=batch_size, device=device
    )
    artifact["boundaryThreshold"] = calibration.threshold

    validation_metrics = evaluate(
        decoded, validation, calibration.threshold, batch_size, device
    )
    heldout_metrics = evaluate(
        decoded, heldout, calibration.threshold, batch_size, device
    )

    # Emit the candidate artifact. The active module is never written here.
    module_source = render_module(artifact, artifact_sha)
    write_text_atomic(candidate_path, module_source)

    gate_out = candidate_report_path.parent / "gate.json"
    returncode, gate_report, gate_output = run_gate(
        candidate_path, active_path, dist_dir, corpus_path, train_split, gate_out
    )
    accepted = returncode == 0 and bool(gate_report and gate_report.get("accepted"))

    # The frozen parity fixture is part of what ships, so only freeze it once the
    # candidate is accepted; a rejected export leaves the committed fixtures.
    parity = freeze_parity(decoded, heldout, fixtures_dir, max_parity) if accepted else None

    report = {
        "formatVersion": FORMAT_VERSION,
        "checkpoint": display_path(checkpoint_path),
        "checkpointSha256": sha256_file(checkpoint_path),
        "checkpointEpoch": checkpoint.get("epoch"),
        "checkpointCriterion": checkpoint.get("criterion"),
        "artifactSha256": artifact_sha,
        "parameters": parameter_counts(decoded),
        "quantizationBits": QUANTIZATION_BITS,
        "boundaryThreshold": calibration.threshold,
        "storage": STORAGE,
        "hidden": hidden,
        "featureRows": int(checkpoint["featureRows"]),
        "rowsPerToken": int(checkpoint["rowsPerToken"]),
        "roleClasses": role_classes,
        "labels": labels,
        "scheduleKindClasses": schedule_kind_classes,
        "scheduleKindLabels": schedule_kind_labels,
        "calibration": calibration.to_dict(),
        "segments": segments,
        "datasetDigests": {
            "validation": validation.digests(),
            "heldout": heldout.digests(),
        },
        "tokenCounts": {
            "validation": {
                "sequences": len(validation),
                "tokens": validation.token_count,
            },
            "heldout": {
                "sequences": len(heldout),
                "tokens": heldout.token_count,
            },
        },
        "evaluation": {
            "validation": validation_metrics,
            "heldout": heldout_metrics,
        },
        "parity": parity,
        "candidateModule": display_path(candidate_path),
        "activeModule": display_path(active_path),
        "module": display_path(candidate_path),
        "dataDirectory": display_path(resolved_data),
        "config": checkpoint.get("config"),
        "promotion": {
            "accepted": accepted,
            "gateReturnCode": returncode,
            "gateReport": gate_report,
            "gateOutput": gate_output,
        },
        "sourceHashes": source_hashes(),
        "gitRev": git_revision(),
    }
    write_text_atomic(candidate_report_path, json.dumps(report, indent=2) + "\n")

    if not accepted:
        # A rejected candidate is never promoted and never committed.
        raise SystemExit(
            "export rejected: "
            + (gate_output or "the promotion gate did not accept the candidate")
        )

    # Promote: copy the candidate module and publish the active report last.
    write_text_atomic(active_path, module_source)
    active_report = {
        **report,
        "module": display_path(active_path),
        "promotion": {**report["promotion"], "promoted": True},
    }
    write_text_atomic(active_report_path, json.dumps(active_report, indent=2) + "\n")
    return active_report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="export",
        description="Export a checkpoint as hash-verified int6 weights.",
    )
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--active", default=DEFAULT_ACTIVE_MODULE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--candidate-report", default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    parser.add_argument("--dist", default=DEFAULT_DIST)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory with validation/heldout shards (default: checkpoint sibling)",
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-parity", type=int, default=MAX_PARITY_SEQUENCES)
    args = parser.parse_args(argv)

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    data_dir = resolve_path(args.data_dir) if args.data_dir else None

    report = export(
        checkpoint_path=checkpoint_path,
        candidate_path=resolve_path(args.candidate),
        active_path=resolve_path(args.active),
        candidate_report_path=resolve_path(args.candidate_report),
        active_report_path=resolve_path(args.report),
        fixtures_dir=resolve_path(args.fixtures),
        dist_dir=resolve_path(args.dist),
        corpus_path=resolve_path(args.corpus),
        train_split=resolve_path(args.train_split),
        data_dir=data_dir,
        batch_size=args.batch,
        device=args.device,
        max_parity=args.max_parity,
    )
    print(f"artifact sha256: {report['artifactSha256']}")
    print(f"boundary threshold: {report['boundaryThreshold']}")
    print(f"parameters: {report['parameters']['total']}")
    print(f"candidate -> {report['candidateModule']}")
    print(f"active -> {report['activeModule']}")
    print(f"report -> {resolve_path(args.report)}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
