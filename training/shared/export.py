"""Export a package checkpoint as an uncalibrated int6 candidate.

    python training/shared/export.py --package <name> --checkpoint <best.pt> --out <dir>

The candidate is written inside `training/<package>/runs` and is never promoted
here. The package's TypeScript `freeze.ts` calibrates the confidence floor and
its gate decides promotion. Validation uses `training/validation/export.py`,
which also calibrates the boundary threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from artifact import (
    build_artifact,
    build_decoded_model,
    freeze_parity,
    label_source,
    load_labels,
    pack_state,
    write_text_atomic,
)
from dataset import Dataset, repo_root

PACKAGES = ("filter", "sort", "retry", "retention", "format")
SOURCES = ("export.py", "artifact.py", "model.py", "dataset.py")


def export_package(package: str, checkpoint_path: Path, directory: Path) -> dict:
    root = repo_root()
    directory = directory.resolve()
    if not directory.is_relative_to(root / f"training/{package}/runs"):
        raise ValueError(f"candidate must be inside training/{package}/runs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = load_labels(package)
    if len(labels) != checkpoint["roleClasses"]:
        raise ValueError("vocabulary mismatch")
    segments, q = pack_state(checkpoint["model"])
    hidden = int(checkpoint["model"]["embedding.weight"].shape[1])
    artifact = build_artifact(segments, q, labels, checkpoint, hidden, 0.0)
    artifact["confidenceFloor"] = 0.5
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
    write_text_atomic(directory / "weights.json", payload)
    write_text_atomic(
        directory / "weights.gen.ts",
        f"/* Generated {package} candidate; no promotion. */\n"
        f"export const weights = {json.dumps(artifact)} as const;\n",
    )
    model = build_decoded_model(artifact)
    validation = Dataset(Path(checkpoint["dataPrefix"] + "-validation"))
    parity = freeze_parity(model, validation, directory / "parity", 32)
    sources = [label_source(package), *(root / "training/shared" / name for name in SOURCES)]
    report = {
        "artifactFileSha256": hashlib.sha256(payload.encode()).hexdigest(),
        "checkpointSha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpointEpoch": checkpoint["epoch"],
        "parity": parity,
        "sourceHashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "promotion": {"promoted": False},
    }
    write_text_atomic(directory / "export-report.json", json.dumps(report, indent=2) + "\n")
    return {"candidate": str(directory), "checkpointEpoch": checkpoint["epoch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a candidate without promotion")
    parser.add_argument("--package", required=True, choices=PACKAGES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(export_package(args.package, Path(args.checkpoint), Path(args.out))))


if __name__ == "__main__":
    main()
