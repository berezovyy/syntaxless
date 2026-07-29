"""Export the selected checkpoint as int6 weights with complete provenance.

The single training workflow calls this module to write its run-local candidate.
It never promotes weights or changes the bundled runtime artifact.
It symmetrically quantizes every parameter, packs the integers into a compact
alphabet string, then *decodes that exact string back into a fresh PyTorch
module* and measures that module. An in-memory checkpoint is not the artifact
that ships, so calibration and the frozen parity fixture are both
computed from the decoded wire artifact.

The int6 range is `[-31, 31]`, so a symbol is `value + 31` in `[0, 62]`. One
segment per parameter records `name`, `offset`, `length`, `shape`, and `scale`.
`artifactSha256` identifies the complete calibrated wire artifact, including
layout, scales and vocabularies. `export.json` is written after the module and
parity fixtures and is the commit marker: an export without it is not valid.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from calibrate import Calibration, calibrate
from dataset import Dataset, repo_root
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger

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
#: Freeze at most this many validation sequences for runtime parity.
MAX_PARITY_SEQUENCES = 512


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_sha256(artifact: dict) -> str:
    """Hash the full artifact using the weights-hash.ts wire encoding.

    Type tags preserve JSON types; numeric bytes avoid Python/JS JSON spelling
    differences. UTF-8 byte counts delimit strings; item counts delimit arrays
    and dictionaries. Dictionary keys sort by UTF-8 bytes, array order remains.
    """
    digest = hashlib.sha256(b"syntaxless-weights-v1\n")

    def hash_value(value: object) -> None:
        if value is None:
            digest.update(b"z")
        elif isinstance(value, bool):
            digest.update(b"t" if value else b"f")
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError("Non-finite weight metadata")
            digest.update(b"n")
            digest.update(struct.pack(">d", 0.0 if value == 0 else float(value)))
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(f"s{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
        elif isinstance(value, list):
            digest.update(f"a{len(value)}:".encode("ascii"))
            for item in value:
                hash_value(item)
        elif isinstance(value, dict):
            digest.update(f"o{len(value)}:".encode("ascii"))
            for key in sorted(value, key=lambda key: key.encode("utf-8")):
                hash_value(key)
                hash_value(value[key])
        else:
            raise TypeError("Weight artifacts must contain only JSON values")

    hash_value(artifact)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Repo-relative where possible so a committed report stays portable."""
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def load_labels() -> list[str]:
    """Read the label vocabulary from the runtime TypeScript source."""
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
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("segment scale must be finite and positive")
    values = np.empty(length, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        scale32 = np.float32(scale)
        for index, symbol in enumerate(q[offset : offset + length]):
            position = SYMBOL_INDEX.get(symbol)
            if position is None:
                raise ValueError(f"unknown symbol {symbol!r} in packed payload")
            values[index] = np.float32(position - SYMBOL_BIAS) * scale32
    if not np.isfinite(values).all():
        raise ValueError(f"packed segment at offset {offset}: values overflow Float32")
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
    segments = artifact.get("segments", [])
    shapes = {segment.get("name"): segment.get("shape", []) for segment in segments}
    factorized = "word_lexical.weight" in shapes
    lexical_shape = shapes.get("word_lexical.weight") or [256]
    lexicon_shape = shapes.get("lexicon_class.weight") or [0]
    model = Tagger(
        feature_rows=int(artifact["featureRows"]),
        rows_per_token=int(artifact["rowsPerToken"]),
        role_classes=int(artifact["roleClasses"]),
        schedule_kind_classes=int(artifact.get("scheduleKindClasses", 0)),
        speech_act_classes=int(artifact.get("speechActClasses", 0)),
        actionable_head=any(
            segment.get("name") == "actionable_head.weight" for segment in segments
        ),
        hidden=int(artifact["hidden"]),
        quantization_bits=QUANTIZATION_BITS,
        factorized=factorized,
        lexical_buckets=int(lexical_shape[0]) if factorized else 256,
        lexicon_classes=int(lexicon_shape[0]) if factorized else 0,
    ).to(device)
    model.load_state_dict(decode_artifact(artifact))
    model.qat = False
    model.eval()
    return model


@torch.no_grad()
def freeze_parity(
    model: Tagger,
    dataset: Dataset,
    fixtures_dir: Path,
    max_sequences: int = MAX_PARITY_SEQUENCES,
) -> dict:
    """Freeze shard rows, offsets, and decoded-artifact logits for runtime parity.

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
        lexical_tensor = None
        lexicon_tensor = None
        if dataset.lexical is not None:
            lexical_tensor = (
                torch.as_tensor(np.asarray(dataset.lexical[start:end]).astype(np.int64))
                .unsqueeze(0)
                .to(device)
            )
        if dataset.lexicon is not None:
            lexicon_tensor = (
                torch.as_tensor(np.asarray(dataset.lexicon[start:end]).astype(np.int64))
                .unsqueeze(0)
                .to(device)
            )
        roles, boundary, schedule_kind, speech_act, actionable = model(
            row_tensor, valid, neighbor_tensor, lexical_tensor, lexicon_tensor
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


def render_module(artifact: dict) -> str:
    payload = json.dumps(artifact, indent=2, ensure_ascii=True)
    return (
        "/**\n"
        " * Generated by training/cron/export.py. Do not edit by hand.\n"
        f" * artifactSha256: {artifact_sha256(artifact)}\n"
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
        decision = ["request", "correction", "other"]
        if labels != list(SPEECH_ACT_NAMES) and labels != decision:
            raise ValueError("speechActLabels must match SPEECH_ACT_NAMES or the 3-way decision head")
        artifact["speechActClasses"] = speech_act_classes
        artifact["speechActLabels"] = labels
    return artifact


def calibrated_checkpoint(
    checkpoint_path: Path, validation: Dataset,
    *, device: str = "cpu", batch_size: int = 512,
) -> tuple[dict, Tagger, Calibration, dict]:
    """Build the exact int6 artifact and fit its thresholds only on validation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bits = int(checkpoint.get("quantizationBits", QUANTIZATION_BITS))
    if bits != QUANTIZATION_BITS:
        raise ValueError(f"checkpoint uses int{bits}; the exporter ships int6")
    state = checkpoint["model"]
    labels = load_labels()
    role_classes = int(checkpoint["roleClasses"])
    # The shipped typed vocabulary is the unchanged prefix before CONTEXT_NUMBER.
    if role_classes == 77:
        labels = labels[:77]
    if len(labels) != role_classes:
        raise ValueError(
            f"labels.ts has {len(labels)} labels but the checkpoint has {role_classes}"
        )
    if validation.role_classes != role_classes:
        raise ValueError("Checkpoint and calibration shard role vocabularies differ")
    hidden = int(state["embedding.weight"].shape[1])
    segments, q = pack_state(state)
    artifact = build_artifact(segments, q, labels, checkpoint, hidden, 0.0)
    decoded = build_decoded_model(artifact, device)
    calibration = calibrate(decoded, [validation], batch_size=batch_size, device=device)
    artifact["boundaryThreshold"] = calibration.threshold
    from abstain import fit_abstention

    energy_fit = fit_abstention(
        decoded, validation, batch_size=batch_size, device=device, max_sequences=4096
    )
    if energy_fit["energyThreshold"] is not None:
        artifact["energyThreshold"] = energy_fit["energyThreshold"]
    artifact["mahalanobisMean"] = energy_fit["mahalanobisMean"]
    artifact["mahalanobisVariance"] = energy_fit["mahalanobisVariance"]
    if energy_fit["mahalanobisThreshold"] is not None:
        artifact["mahalanobisThreshold"] = energy_fit["mahalanobisThreshold"]
    return artifact, decoded, calibration, energy_fit


def write_candidate_module(
    checkpoint_path: Path, candidate_path: Path, validation: Dataset,
    *, device: str = "cpu", batch_size: int = 512,
    expected_artifact_sha256: str | None = None,
) -> Path:
    """Export the selected artifact, refusing a change after decoded selection."""
    artifact, decoded, calibration, energy_fit = calibrated_checkpoint(
        checkpoint_path, validation, device=device, batch_size=batch_size,
    )
    artifact_sha = artifact_sha256(artifact)
    if expected_artifact_sha256 is not None and artifact_sha != expected_artifact_sha256:
        raise ValueError("Export differs from the calibrated artifact used for selection")
    parity = freeze_parity(decoded, validation, candidate_path.parent / "parity")
    write_text_atomic(candidate_path, render_module(artifact))
    write_text_atomic(
        candidate_path.parent / "export.json",
        json.dumps({
            "artifactSha256": artifact_sha,
            "checkpoint": str(checkpoint_path),
            "candidateModule": str(candidate_path),
            "calibrationSplit": str(validation.prefix),
            "calibration": calibration.to_dict(),
            "abstention": energy_fit,
            "parity": parity,
            "promoted": False,
        }, indent=2) + "\n",
    )
    return candidate_path
