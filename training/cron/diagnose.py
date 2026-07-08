"""Diagnose a checkpoint against one backward-corpus split.

Global token accuracy hides whether failures cluster in one linguistic frame.
This tool preserves corpus order and reports exact-sequence failures by family,
kind, and template, plus role confusions and bounded example sentences.
Generated reports are run artifacts and must not be committed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import torch

from dataset import Dataset, repo_root
from export import load_labels
from model import Tagger


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


class Metadata(NamedTuple):
    text: str
    family: str
    template: str


def records(path: Path, split: str) -> Iterator[Metadata]:
    with path.open(encoding="utf8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if record.get("split") == split:
                yield Metadata(
                    text=str(record.get("text", "")),
                    family=str(record.get("family", "<missing>")),
                    template=str(record.get("template", "<missing>")),
                )


def load_model(checkpoint_path: Path, device: str) -> Tagger:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model = Tagger(
        feature_rows=int(checkpoint["featureRows"]),
        rows_per_token=int(checkpoint["rowsPerToken"]),
        role_classes=int(checkpoint["roleClasses"]),
        schedule_kind_classes=int(checkpoint.get("scheduleKindClasses", 0)),
        hidden=int(state["embedding.weight"].shape[1]),
        quantization_bits=int(checkpoint["quantizationBits"]),
    )
    model.load_state_dict(state)
    model.qat = True
    return model.to(device).eval()


def fresh_stats() -> dict[str, int]:
    return {
        "sequences": 0,
        "exact": 0,
        "roleExact": 0,
        "tokenErrors": 0,
        "boundaryErrors": 0,
        "tokens": 0,
    }


def ranked(groups: dict[str, dict[str, int]]) -> list[dict]:
    result = []
    for name, stats in groups.items():
        sequences = stats["sequences"]
        tokens = stats["tokens"]
        result.append(
            {
                "name": name,
                **stats,
                "exactRate": stats["exact"] / max(sequences, 1),
                "roleExactRate": stats["roleExact"] / max(sequences, 1),
                "tokenAccuracy": 1 - stats["tokenErrors"] / max(tokens, 1),
                "boundaryAccuracy": 1 - stats["boundaryErrors"] / max(tokens, 1),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["exactRate"],
            row["tokenAccuracy"],
            -row["sequences"],
            row["name"],
        ),
    )


@torch.no_grad()
def diagnose(
    checkpoint_path: Path,
    data_prefix: Path,
    corpus_path: Path,
    split: str,
    batch_size: int,
    device: str,
    max_examples: int,
) -> dict:
    dataset = Dataset(Path(f"{data_prefix}-{split}"))
    model = load_model(checkpoint_path, device)
    labels = load_labels()
    # Batches must remain length-bucketed for practical GPU throughput. Keep
    # only the three short strings needed for diagnosis, indexed in shard order.
    metadata = list(records(corpus_path, split))
    if len(metadata) != len(dataset):
        raise RuntimeError(
            f"corpus has {len(metadata)} selected records but shard has "
            f"{len(dataset)} sequences; regenerate shards before diagnosing"
        )
    groups: dict[str, dict[str, dict[str, int]]] = {
        "family": defaultdict(fresh_stats),
        "kind": defaultdict(fresh_stats),
        "template": defaultdict(fresh_stats),
    }
    confusion: Counter[tuple[int, int]] = Counter()
    examples: list[dict] = []
    seen = 0
    overall = fresh_stats()

    for indices in dataset.batches(batch_size):
        batch = dataset.batch(indices, device)
        roles, boundary, _ = model(
            batch["rows"], batch["valid"], batch["neighbors"]
        )
        # One transfer per tensor avoids a GPU synchronization for every
        # sequence while the diagnostic aggregates Python dictionaries.
        predicted_roles = roles.argmax(dim=-1).cpu().numpy()
        predicted_boundaries = (torch.sigmoid(boundary) >= 0.5).cpu().numpy()
        target_roles = batch["labels"].cpu().numpy()
        target_boundaries = (batch["boundaries"] >= 0.5).cpu().numpy()
        mask = target_roles != -100

        for row in range(len(indices)):
            record = metadata[indices[row]]
            seen += 1
            row_mask = mask[row]
            role_ok = (predicted_roles[row] == target_roles[row]) | ~row_mask
            boundary_ok = (
                (predicted_boundaries[row] == target_boundaries[row]) | ~row_mask
            )
            exact = bool(np.all(role_ok & boundary_ok))
            role_exact = bool(np.all(role_ok))
            token_errors = int(
                np.sum((predicted_roles[row] != target_roles[row]) & row_mask)
            )
            boundary_errors = int(
                np.sum(
                    (predicted_boundaries[row] != target_boundaries[row]) & row_mask
                )
            )
            tokens = int(row_mask.sum())
            template = record.template
            kind = template.split("/", 1)[0]
            family = record.family
            for dimension, name in (
                ("family", family),
                ("kind", kind),
                ("template", template),
            ):
                stats = groups[dimension][name]
                stats["sequences"] += 1
                stats["exact"] += int(exact)
                stats["roleExact"] += int(role_exact)
                stats["tokenErrors"] += token_errors
                stats["boundaryErrors"] += boundary_errors
                stats["tokens"] += tokens
            overall["sequences"] += 1
            overall["exact"] += int(exact)
            overall["roleExact"] += int(role_exact)
            overall["tokenErrors"] += token_errors
            overall["boundaryErrors"] += boundary_errors
            overall["tokens"] += tokens

            mismatches = np.flatnonzero(
                (predicted_roles[row] != target_roles[row]) & row_mask
            )
            for position in mismatches:
                confusion[
                    (int(target_roles[row, position]), int(predicted_roles[row, position]))
                ] += 1

            if not exact and len(examples) < max_examples:
                examples.append(
                    {
                        "text": record.text,
                        "family": family,
                        "kind": kind,
                        "template": template,
                        "roleMismatches": [
                            {
                                "position": int(position),
                                "expected": labels[int(target_roles[row, position])],
                                "predicted": labels[int(predicted_roles[row, position])],
                            }
                            for position in mismatches.tolist()
                        ],
                        "boundaryMismatchPositions": np.flatnonzero(
                            (predicted_boundaries[row] != target_boundaries[row])
                            & row_mask
                        ).tolist(),
                    }
                )

    return {
        "checkpoint": str(checkpoint_path),
        "dataPrefix": str(data_prefix),
        "corpus": str(corpus_path),
        "split": split,
        "sequences": seen,
        "summary": {
            **overall,
            "exactRate": overall["exact"] / max(overall["sequences"], 1),
            "roleExactRate": overall["roleExact"]
            / max(overall["sequences"], 1),
            "tokenAccuracy": 1
            - overall["tokenErrors"] / max(overall["tokens"], 1),
            "boundaryAccuracy": 1
            - overall["boundaryErrors"] / max(overall["tokens"], 1),
        },
        "byFamily": ranked(groups["family"]),
        "byKind": ranked(groups["kind"]),
        "byTemplate": ranked(groups["template"]),
        "roleConfusions": [
            {
                "expected": labels[expected],
                "predicted": labels[predicted],
                "count": count,
            }
            for (expected, predicted), count in confusion.most_common()
        ],
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank checkpoint failures by backward-corpus language frame."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "heldout"), default="heldout"
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.max_examples < 0:
        raise ValueError("--batch must be positive and --max-examples non-negative")
    report = diagnose(
        resolve_path(args.checkpoint),
        resolve_path(args.data_prefix),
        resolve_path(args.corpus),
        args.split,
        args.batch,
        args.device,
        args.max_examples,
    )
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"diagnosed {report['sequences']:,} {args.split} sequences -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
