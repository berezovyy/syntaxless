"""Export the selected checkpoint as hash-verified int6 weights.

The exporter writes candidate artifacts inside training/validation/runs only.
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

from calibrate import calibrate
from dataset import Dataset, repo_root
from model import SCHEDULE_KIND_NAMES, Tagger

#: The shipped format is int6 by contract.
QUANTIZATION_BITS = 6
QUANTIZATION_CEILING = (1 << (QUANTIZATION_BITS - 1)) - 1  # 31
#: `value + SYMBOL_BIAS` indexes the alphabet; 0..62 covers -31..31.
SYMBOL_BIAS = 31
#: Compact 64-symbol alphabet. One character per quantized parameter element.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(ALPHABET)}
FORMAT_VERSION = 2
STORAGE = "alphabet"
#: Fixtures freeze at most this many held-out sequences for Phase 16 parity.
MAX_PARITY_SEQUENCES = 512
#: Files hashed into the report provenance block.
EXPORT_SOURCES = (
    "calibrate.py",
    "dataset.py",
    "export.py",
    "model.py",
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
    source = repo_root() / "packages" / "validation" / "src" / "model" / "labels.ts"
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
        hidden=int(artifact["hidden"]),
        quantization_bits=QUANTIZATION_BITS,
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
        roles, boundary, schedule_kind = model(row_tensor, valid, neighbor_tensor)
        rows_parts.append(rows)
        logits_parts.append(roles[0].detach().cpu().numpy().astype(np.float32))
        boundaries_parts.append(boundary[0].detach().cpu().numpy().astype(np.float32))
        if schedule_kind.shape[-1] > 0:
            schedule_kind_parts.append(
                schedule_kind[0].detach().cpu().numpy().astype(np.float32)
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
        " * Generated by training/validation/export.py. Do not edit by hand.\n"
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
    return artifact


def export_candidate(checkpoint_path: Path, directory: Path) -> dict:
    """Export evidence to a run directory. This function never promotes weights."""
    directory = directory.resolve()
    runs = (repo_root() / "training/validation/runs").resolve()
    if not directory.is_relative_to(runs):
        raise ValueError("candidate output must be inside training/validation/runs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = load_labels()
    if checkpoint["roleClasses"] != len(labels):
        raise ValueError("checkpoint vocabulary differs from runtime")
    segments, q = pack_state(checkpoint["model"])
    hidden = int(checkpoint["model"]["embedding.weight"].shape[1])
    artifact = build_artifact(segments, q, labels, checkpoint, hidden, 0.0)
    artifact["confidenceFloor"] = 0.5
    model = build_decoded_model(artifact)
    validation = Dataset(Path(f"{checkpoint['dataPrefix']}-validation"))
    calibration = calibrate(model, [validation], batch_size=64, device="cpu")
    artifact["boundaryThreshold"] = calibration.threshold
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    digest = sha256_text(payload)
    write_text_atomic(directory / "weights.json", payload + "\n")
    write_text_atomic(directory / "weights.gen.ts", render_module(artifact, digest))
    parity = freeze_parity(model, validation, directory / "parity", 32)
    report = {"artifactSha256": digest, "qSha256": sha256_text(q),
              "checkpointSha256": sha256_file(checkpoint_path), "checkpointEpoch": checkpoint["epoch"],
              "calibration": calibration.to_dict(), "parity": parity,
              "sourceHashes": source_hashes(), "promotion": {"promoted": False, "gateRun": False}}
    write_text_atomic(directory / "export-report.json", json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export a candidate without promotion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(export_candidate(resolve_path(args.checkpoint), resolve_path(args.out))))
