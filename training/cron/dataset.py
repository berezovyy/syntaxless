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

SCHEDULE_KIND_NAMES = (
    "none",
    "interval",
    "daily",
    "weekly",
    "monthly",
    "ordinal",
    "bounded",
    "exception",
    "ambiguous",
)
SCHEDULE_KIND_INDEX = {name: index for index, name in enumerate(SCHEDULE_KIND_NAMES)}


def schedule_kind_for_frame(frame_name: str) -> int:
    """Map a detailed generator kind onto the compact utterance-level head."""
    parts = frame_name.split("/")
    kind = parts[0]
    if kind == "negative":
        category = parts[1] if len(parts) > 1 else "ordinary"
        return SCHEDULE_KIND_INDEX[
            "ambiguous"
            if category in {"incomplete", "ambiguous", "conversational", "multiple"}
            else "none"
        ]
    if kind.startswith("bounds"):
        return SCHEDULE_KIND_INDEX["bounded"]
    if "exception" in kind:
        return SCHEDULE_KIND_INDEX["exception"]
    if kind.startswith("ordinal-weekday"):
        return SCHEDULE_KIND_INDEX["ordinal"]
    if kind.startswith("day-of-month") or kind.startswith("day-rule"):
        return SCHEDULE_KIND_INDEX["monthly"]
    if kind in {"unit-interval", "n-interval", "interval-at-time", "frequency-idiom"}:
        return SCHEDULE_KIND_INDEX["interval"]
    if kind in {"time-of-day", "day-at-time"}:
        return SCHEDULE_KIND_INDEX["daily"]
    return SCHEDULE_KIND_INDEX["weekly"]

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
    frame_names: list[str]
    frame_counts: list[int]

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
            frame_names=list(parsed.get("frameNames", [])),
            frame_counts=[int(value) for value in parsed.get("frameCounts", [])],
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
        self.frames = self._load("frames") if "frames" in self.manifest.format else None

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
        self.frame_members: list[np.ndarray] = []
        self.frame_schedule_kinds = np.asarray(
            [schedule_kind_for_frame(name) for name in self.manifest.frame_names],
            dtype=np.int64,
        )
        if self.frames is not None:
            if self.frames.size != self.sequences:
                raise ValueError(
                    f"{self.prefix}: frames has {self.frames.size} values, "
                    f"expected {self.sequences}"
                )
            frame_count = len(self.manifest.frame_names)
            if frame_count == 0 or int(self.frames.max(initial=0)) >= frame_count:
                raise ValueError(f"{self.prefix}: invalid frame ids or names")
            self.frame_members = [
                np.flatnonzero(self.frames == frame_id)
                for frame_id in range(frame_count)
            ]
            actual_counts = [int(members.size) for members in self.frame_members]
            if self.manifest.frame_counts and actual_counts != self.manifest.frame_counts:
                raise ValueError(f"{self.prefix}: frame counts do not match frames shard")

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

    def batches(
        self,
        batch_size: int,
        shuffle_seed: int | None = None,
        balance_frames: bool = False,
        balance_kinds: bool = False,
        balance_schedule_kinds: bool = False,
        frame_balance_power: float = 0.5,
    ):
        """Yield batches of sequence indices, grouped by similar length.

        Sorting by length before chunking means only the last batch of each
        length class carries meaningful padding.
        """
        rng = np.random.default_rng(shuffle_seed)
        selected = np.arange(self.sequences)
        if balance_frames or balance_kinds or balance_schedule_kinds:
            if not self.frame_members:
                raise ValueError(
                    f"{self.prefix}: balanced batching requires a frames shard"
                )
            counts = np.asarray(
                [members.size for members in self.frame_members], dtype=np.float64
            )
            masses = (
                np.power(counts, 1.0 - frame_balance_power)
                if balance_frames
                else counts.copy()
            )
            if balance_kinds:
                kind_frames: dict[str, list[int]] = {}
                for frame_id, frame_name in enumerate(self.manifest.frame_names):
                    kind_frames.setdefault(frame_name.split("/", 1)[0], []).append(
                        frame_id
                    )
                # Normalize within each kind. Every kind then contributes one
                # unit of total mass, while --balance-frames controls how that
                # unit is distributed among the kind's wording frames.
                for frame_ids in kind_frames.values():
                    kind_mass = float(masses[frame_ids].sum())
                    masses[frame_ids] /= kind_mass
            if balance_schedule_kinds:
                schedule_frames: dict[int, list[int]] = {}
                for frame_id, schedule_kind in enumerate(self.frame_schedule_kinds):
                    schedule_frames.setdefault(int(schedule_kind), []).append(frame_id)
                for frame_ids in schedule_frames.values():
                    class_mass = float(masses[frame_ids].sum())
                    masses[frame_ids] /= class_mass
            draw_counts = rng.multinomial(self.sequences, masses / masses.sum())
            draws: list[np.ndarray] = []
            for count, members in zip(draw_counts, self.frame_members, strict=True):
                if count:
                    draws.append(rng.choice(members, size=count, replace=True))
            selected = np.concatenate(draws)
            rng.shuffle(selected)

        tie_break = np.arange(selected.size)
        if shuffle_seed is not None:
            permutation = rng.permutation(selected.size)
            tie_break = np.empty(selected.size, dtype=np.int64)
            tie_break[permutation] = np.arange(selected.size)
        order = selected[np.lexsort((tie_break, self.lengths[selected]))]
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
            "schedule_kinds": torch.as_tensor(
                self.frame_schedule_kinds[np.asarray(self.frames[indices])]
                if self.frames is not None
                else np.zeros(count, dtype=np.int64)
            ),
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
