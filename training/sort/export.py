"""Thin sort exporter using the already checked quantization/parity primitives."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "training/validation"))
from dataset import Dataset
from export import pack_state, build_artifact, build_decoded_model, freeze_parity, write_text_atomic

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    directory = Path(args.out).resolve()
    if not directory.is_relative_to(root / "training/sort/runs"):
        raise ValueError("candidate must be inside sort runs")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = root / "packages/sort/src/labels.ts"
    text = source.read_text()
    labels = re.findall(r'"([^"]+)"', text[text.index("["):text.index("] as const")])
    if len(labels) != checkpoint["roleClasses"]: raise ValueError("vocabulary mismatch")
    segments, q = pack_state(checkpoint["model"])
    artifact = build_artifact(segments, q, labels, checkpoint, int(checkpoint["model"]["embedding.weight"].shape[1]), 0.0)
    artifact["confidenceFloor"] = 0.5
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
    write_text_atomic(directory / "weights.json", payload)
    write_text_atomic(directory / "weights.gen.ts", "/* Generated sort candidate; no promotion. */\nexport const weights = " + json.dumps(artifact) + " as const;\n")
    model = build_decoded_model(artifact)
    parity = freeze_parity(model, Dataset(Path(checkpoint["dataPrefix"] + "-validation")), directory / "parity", 32)
    shared = root / "training/validation"
    sources = [Path(__file__), source, *[shared / name for name in ("export.py", "model.py", "dataset.py", "calibrate.py")]]
    report = {"artifactFileSha256": hashlib.sha256(payload.encode()).hexdigest(), "checkpointSha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "checkpointEpoch": checkpoint["epoch"], "parity": parity, "sourceHashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, "promotion": {"promoted": False}}
    write_text_atomic(directory / "export-report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate": str(directory), "checkpointEpoch": checkpoint["epoch"]}))

if __name__ == "__main__": main()
