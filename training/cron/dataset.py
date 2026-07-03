"""Shard loader, length-bucketed batching, and split regeneration.

The shard contract is owned by Phase 13 (`packages/cron/training/src/featurize.ts`):
a `<prefix>.json` manifest plus six `<prefix>.<field>.bin` files. Nothing here
parses cron text: the loader reads labels and boundaries that Phase 13 recorded
from the generator, and the regenerator drives the committed TypeScript tools.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: dtype spellings used by the manifest's `format` block.
NUMPY_DTYPES: dict[str, type[np.generic]] = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "float32": np.float32,
}

#: Length buckets. A batch pads to the smallest bucket covering its longest
#: member, so short sequences never pay for a long neighbour.
BUCKETS: tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128)


def repo_root() -> Path:
    """The workspace root: `training/cron/` is two levels below it."""
    return Path(__file__).resolve().parents[2]


def length_bucket(length: int) -> int:
    for bucket in BUCKETS:
        if length <= bucket:
            return bucket
    return length


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ShardManifest:
    """The six-field manifest written by Phase 13."""

    fingerprints: list[str]
    role_classes: int
    rows_per_token: int
    feature_rows: int
    label_counts: dict[str, int]
    format: dict[str, dict[str, str]]

    @classmethod
    def load(cls, prefix: Path) -> "ShardManifest":
        parsed = json.loads(Path(f"{prefix}.json").read_text())
        return cls(
            fingerprints=list(parsed["fingerprints"]),
            role_classes=int(parsed["roleClasses"]),
            rows_per_token=int(parsed["rowsPerToken"]),
            feature_rows=int(parsed["featureRows"]),
            label_counts={key: int(value) for key, value in parsed["labelCounts"].items()},
            format=parsed["format"],
        )


class Dataset:
    """One featurized split, indexed by sequence."""

    def __init__(self, prefix: str | Path):
        self.prefix = Path(prefix)
        self.manifest = ShardManifest.load(self.prefix)
        self.role_classes = self.manifest.role_classes
        self.rows_per_token = self.manifest.rows_per_token
        self.feature_rows = self.manifest.feature_rows
        self.label_counts = self.manifest.label_counts
        self.fingerprints = self.manifest.fingerprints

        self.rows = self._load("rows")
        self.labels = self._load("labels")
        self.boundaries = self._load("boundaries")
        self.kinds = self._load("kinds")
        self.neighbors = self._load("neighbors")
        self.offsets = self._load("offsets")

        self.sequences = int(self.offsets.shape[0]) - 1
        self.token_count = int(self.offsets[-1])
        expected = self.token_count * self.rows_per_token
        if self.rows.size != expected:
            raise ValueError(
                f"{self.prefix}: rows has {self.rows.size} values, expected {expected}"
            )
        self.rows = self.rows.reshape(self.token_count, self.rows_per_token)
        self.neighbors = self.neighbors.reshape(self.token_count, 2)
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)

    def _load(self, key: str) -> np.ndarray:
        spec = self.manifest.format[key]
        dtype = NUMPY_DTYPES[spec["dtype"]]
        return np.memmap(f"{self.prefix}{spec['suffix']}", dtype=dtype, mode="r")

    def __len__(self) -> int:
        return self.sequences

    def digests(self) -> dict[str, str]:
        """SHA-256 of the manifest and every shard file for the report."""
        result = {"manifest": _sha256(Path(f"{self.prefix}.json"))}
        for key, spec in self.manifest.format.items():
            result[key] = _sha256(Path(f"{self.prefix}{spec['suffix']}"))
        return result

    def batches(self, batch_size: int, shuffle_seed: int | None = None):
        """Yield batches of sequence indices, grouped by similar length.

        Sorting by length before chunking means only the last batch of each
        length class carries meaningful padding.
        """
        tie_break = np.arange(self.sequences)
        if shuffle_seed is not None:
            permutation = np.random.default_rng(shuffle_seed).permutation(self.sequences)
            tie_break = np.empty(self.sequences, dtype=np.int64)
            tie_break[permutation] = np.arange(self.sequences)
        order = np.lexsort((tie_break, self.lengths))
        for start in range(0, self.sequences, batch_size):
            yield [int(index) for index in order[start : start + batch_size]]

    def batch(self, indices, device: str | None = None) -> dict:
        """Pad `indices` to the next length bucket and return model tensors.

        Unused rows hold the manifest's `featureRows` sentinel. Whitespace and
        any label at or above `roleClasses` (the Phase 13 padding sentinel) are
        set to `-100`, the cross-entropy ignore index, so they contribute no
        role or boundary loss.
        """
        import torch

        count = len(indices)
        longest = int(max(int(self.lengths[index]) for index in indices))
        width = length_bucket(longest)

        rows = np.full((count, width, self.rows_per_token), self.feature_rows, dtype=np.int64)
        labels = np.full((count, width), -100, dtype=np.int64)
        boundaries = np.zeros((count, width), dtype=np.float32)
        kinds = np.full((count, width), 3, dtype=np.int64)
        neighbors = np.full((count, width, 2), -1, dtype=np.int64)
        valid = np.zeros((count, width), dtype=bool)

        for row, index in enumerate(indices):
            start = int(self.offsets[index])
            end = int(self.offsets[index + 1])
            length = end - start

            rows[row, :length] = self.rows[start:end]
            sequence_labels = self.labels[start:end].astype(np.int64)
            sequence_kinds = self.kinds[start:end].astype(np.int64)
            unsupervised = (sequence_labels >= self.role_classes) | (sequence_kinds == 3)
            labels[row, :length] = np.where(unsupervised, -100, sequence_labels)
            boundaries[row, :length] = self.boundaries[start:end]
            kinds[row, :length] = sequence_kinds
            neighbors[row, :length] = self.neighbors[start:end]
            valid[row, :length] = True

        tensors = {
            "rows": torch.as_tensor(rows),
            "valid": torch.as_tensor(valid),
            "neighbors": torch.as_tensor(neighbors),
            "labels": torch.as_tensor(labels),
            "boundaries": torch.as_tensor(boundaries),
            "kinds": torch.as_tensor(kinds),
        }
        if device is not None:
            tensors = {key: value.to(device) for key, value in tensors.items()}
        return tensors


def fingerprints_path(prefix: str | Path) -> Path:
    """The sibling `<prefix>.jsonl.fingerprints.json` written by generate.ts."""
    return Path(f"{prefix}.jsonl.fingerprints.json")


def _normalize_generated(path: Path) -> None:
    """Tag raw generator records with the Phase 13 corpus `source` field.

    `generate.ts` emits its Phase 08 records with a `seed` field; the Phase 13
    featurizer consumes the merged corpus contract, which carries `source` and
    rejects unknown keys. This is the one bridge between the two committed
    contracts — no labels are invented here.
    """
    lines = path.read_text().splitlines()
    records: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        record.setdefault("source", "rule")
        record.pop("seed", None)
        records.append(json.dumps(record))
    path.write_text("\n".join(records) + ("\n" if records else ""))


def prepare(
    split: str,
    count: int,
    seed: int,
    directory: str | Path,
    exclude: str | Path | None = None,
) -> Path:
    """Regenerate a split through Phase 08 (`generate.ts`) and Phase 13.

    Runs the workspace tools from the repo root and returns the shard prefix.
    `exclude` is the held-out fingerprint file that non-held-out splits must
    exclude; heldout passes `None`.
    """
    repo = repo_root()
    target = Path(directory)
    if not target.is_absolute():
        target = repo / target
    target.mkdir(parents=True, exist_ok=True)
    prefix = target / split
    jsonl = Path(f"{prefix}.jsonl")

    generate = [
        "bun",
        "run",
        "packages/cron/training/src/generate.ts",
        "--split",
        split,
        "--count",
        str(count),
        "--seed",
        str(seed),
        "--out",
        str(jsonl),
    ]
    if exclude is not None:
        excluded = Path(exclude)
        if not excluded.is_absolute():
            excluded = repo / excluded
        generate += ["--exclude", str(excluded)]
    subprocess.run(generate, cwd=repo, check=True)

    _normalize_generated(jsonl)

    subprocess.run(
        [
            "bun",
            "run",
            "packages/cron/training/src/featurize.ts",
            str(jsonl),
            str(prefix),
        ],
        cwd=repo,
        check=True,
    )
    return prefix
