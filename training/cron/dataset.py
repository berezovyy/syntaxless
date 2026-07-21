"""Shard loader and one-pass, length-bucketed production batching.

The shard contract is owned by TypeScript (`packages/cron/training/src/featurize.ts`):
a `<prefix>.json` manifest plus `<prefix>.<field>.bin` files for rows, labels,
boundaries, kinds, neighbors, offsets, frames, speech, actionable, and populations.
The loader reads labels and discourse supervision that the generator recorded.
Corpus preparation belongs to the training coordinator.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: dtype spellings used by the manifest's `format` block.
#: The manifest `format` names the dtype for every required field.
NUMPY_DTYPES: dict[str, type[np.generic]] = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
    "uint64": np.uint64,
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

# Must match the production featurizer. Population is reporting metadata, not a
# model target or a sampling policy. Training visits every unique row once.
POPULATION_NAMES = ("R", "H", "E")

# Must match packages/cron/src/model/speech-acts.ts. Never reorder.
SPEECH_ACT_NAMES = (
    "none",
    "request",
    "descriptive",
    "question",
    "quoted",
    "correction",
    "one-off",
    "cancel",
    "incomplete",
)

# Must match packages/cron/src/model/labels.ts. Never reorder.
ROLE_LABELS = (
    "O",
    "RECUR",
    "GLUE",
    "EDGE",
    "COUNT",
    "WEEKDAY_MON",
    "WEEKDAY_TUE",
    "WEEKDAY_WED",
    "WEEKDAY_THU",
    "WEEKDAY_FRI",
    "WEEKDAY_SAT",
    "WEEKDAY_SUN",
    "DAYGROUP_WEEKDAY",
    "DAYGROUP_WEEKEND",
    "DAYGROUP_DAY",
    "HOUR_0",
    "HOUR_1",
    "HOUR_2",
    "HOUR_3",
    "HOUR_4",
    "HOUR_5",
    "HOUR_6",
    "HOUR_7",
    "HOUR_8",
    "HOUR_9",
    "HOUR_10",
    "HOUR_11",
    "HOUR_12",
    "HOUR_13",
    "HOUR_14",
    "HOUR_15",
    "HOUR_16",
    "HOUR_17",
    "HOUR_18",
    "HOUR_19",
    "HOUR_20",
    "HOUR_21",
    "HOUR_22",
    "HOUR_23",
    "HOUR_LIT",
    "MINUTE_0",
    "MINUTE_15",
    "MINUTE_30",
    "MINUTE_45",
    "MINUTE_LIT",
    "MERIDIEM_AM",
    "MERIDIEM_PM",
    "MERIDIEM_NONE",
    "ORDINAL_1",
    "ORDINAL_2",
    "ORDINAL_3",
    "ORDINAL_4",
    "ORDINAL_5",
    "ORDINAL_LAST",
    "BOUND_START",
    "BOUND_END_EXCL",
    "BOUND_END_INCL",
    "BOUND_BETWEEN",
    "EXCEPT",
    "UNIT_MIN",
    "UNIT_HOUR",
    "UNIT_DAY",
    "UNIT_WEEK",
    "UNIT_MONTH",
    "INTERVAL_2",
    "INTERVAL_3",
    "INTERVAL_4",
    "INTERVAL_5",
    "INTERVAL_6",
    "INTERVAL_7",
    "INTERVAL_8",
    "INTERVAL_9",
    "INTERVAL_10",
    "INTERVAL_11",
    "INTERVAL_12",
    "INTERVAL_LIT",
    "DOM_LIT",
    "CONTEXT_NUMBER",
)
STRUCTURAL_LABELS = {
    "RECUR",
    "GLUE",
    "EDGE",
    "BOUND_START",
    "BOUND_END_EXCL",
    "BOUND_END_INCL",
    "BOUND_BETWEEN",
    "EXCEPT",
}
CONTENT_LABELS = {
    name for name in ROLE_LABELS
    if name not in {"O", "CONTEXT_NUMBER"} and name not in STRUCTURAL_LABELS
}
CONTENT_LABEL_IDS = {
    index for index, name in enumerate(ROLE_LABELS) if name in CONTENT_LABELS
}

#: Bound working indices and memmap reads to nearby sequences. This is also
#: the largest per-sequence array any loader pass builds: memory scales with
#: this constant, never with the split size.
LOCALITY_BLOCK_SIZE = 262_144


class SequenceLengths:
    """Token count per sequence, read from the offsets memmap on demand.

    A materialized int64 array costs 8 bytes per sequence (1.6 GB at 200M
    rows). Indexing computes `offsets[i + 1] - offsets[i]` only for the
    requested sequences, so callers keep their `lengths[...]` code while the
    split stays on disk. Whole-split conversion (`np.asarray`, `list`, `for`)
    raises instead of silently materializing it.
    """

    __slots__ = ("_offsets", "_count")

    def __init__(self, offsets: np.ndarray):
        self._offsets = offsets
        self._count = int(offsets.shape[0]) - 1

    def __len__(self) -> int:
        return self._count

    def __array__(self, *args, **kwargs):
        raise TypeError("SequenceLengths would materialize a whole split; index a block")

    def __iter__(self):
        raise TypeError("SequenceLengths would materialize a whole split; index a block")

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._count)
            if step != 1:
                raise ValueError("SequenceLengths supports only unit-step slices")
            bounds = np.asarray(self._offsets[start : max(start, stop) + 1], dtype=np.int64)
            return bounds[1:] - bounds[:-1]
        index = np.asarray(key, dtype=np.int64)
        if index.size and (int(index.min()) < 0 or int(index.max()) >= self._count):
            raise IndexError("sequence index out of range")
        lengths = self._offsets[index + 1].astype(np.int64) - self._offsets[index].astype(np.int64)
        return int(lengths) if index.ndim == 0 else lengths


def schedule_kind_for_frame(frame_name: str) -> int:
    """Map a detailed generator kind onto the compact utterance-level head."""
    parts = frame_name.split("/")
    kind = parts[0]
    # Whole-utterance production sources name their semantic family, not the
    # legacy renderer kind. Never let ordinary negatives, clocks, bounds, or
    # corrections silently inherit the old catch-all "weekly" target.
    production_kinds = {
        "easy-no": "none",
        "minute-interval": "interval",
        "hour-interval": "interval",
        "day-interval": "interval",
        "week-interval": "interval",
        "month-interval": "interval",
        "daily-clock": "daily",
        "last-day-of-month": "monthly",
        "last-business-day": "monthly",
        "start-bound": "bounded",
        "exclusive-end-bound": "bounded",
        "inclusive-end-bound": "bounded",
        "date-window": "bounded",
        "correction-interval": "interval",
        "correction-clock": "daily",
        "correction-weekday": "weekly",
    }
    if kind in production_kinds:
        return SCHEDULE_KIND_INDEX[production_kinds[kind]]
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
    """The featurizer manifest for the published production supervision."""

    fingerprints: list[str]
    role_classes: int
    rows_per_token: int
    feature_rows: int
    label_counts: dict[str, int]
    format: dict[str, dict[str, str]]
    frame_names: list[str]
    frame_counts: list[int]
    speech_act_classes: int
    speech_act_names: list[str]
    actionable_counts: dict[str, int]
    population_counts: dict[str, int]
    sequence_metadata: dict | None = None

    @classmethod
    def load(cls, prefix: Path) -> "ShardManifest":
        parsed = json.loads(Path(f"{prefix}.json").read_text())
        return cls(
            sequence_metadata=parsed.get("sequenceMetadata"),
            fingerprints=list(parsed["fingerprints"]),
            role_classes=int(parsed["roleClasses"]),
            rows_per_token=int(parsed["rowsPerToken"]),
            feature_rows=int(parsed["featureRows"]),
            label_counts={key: int(value) for key, value in parsed["labelCounts"].items()},
            format=parsed["format"],
            frame_names=list(parsed.get("frameNames", [])),
            frame_counts=[int(value) for value in parsed.get("frameCounts", [])],
            speech_act_classes=int(parsed.get("speechActClasses", 0)),
            speech_act_names=list(parsed.get("speechActNames", [])),
            actionable_counts={
                str(key): int(value)
                for key, value in parsed.get("actionableCounts", {}).items()
            },
            population_counts={
                str(key): int(value)
                for key, value in parsed.get("populationCounts", {}).items()
            },
        )


class Dataset:
    """One featurized split, indexed by sequence.

    Every per-token and per-sequence field is a read-only memmap. Nothing here
    copies a whole field into process memory: batches gather their own rows,
    and whole-split checks walk `LOCALITY_BLOCK_SIZE` sequences at a time.
    Keep it that way (no `astype` over a full field, no full-split masks).
    """

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
        self.frames = self._load("frames")
        self.populations = self._load("populations")
        # Stored as uint8 memmaps; `batch` widens only the gathered rows.
        self.speech = self._load("speech")
        self.actionable = self._load("actionable")
        self.lexical = None
        self.lexicon = None

        self.sequences = int(self.offsets.shape[0]) - 1
        self.token_count = int(self.offsets[-1])
        expected = self.token_count * self.rows_per_token
        if self.rows.size != expected:
            raise ValueError(
                f"{self.prefix}: rows has {self.rows.size} values, expected {expected}"
            )
        self.rows = self.rows.reshape(self.token_count, self.rows_per_token)
        self.neighbors = self.neighbors.reshape(self.token_count, 2)
        self.lengths = SequenceLengths(self.offsets)
        self.frame_schedule_kinds = np.asarray(
            [schedule_kind_for_frame(name) for name in self.manifest.frame_names],
            dtype=np.int64,
        )
        for name, values, expected in (
            ("frames", self.frames, self.sequences),
            ("populations", self.populations, self.sequences),
            ("speech", self.speech, self.sequences),
            ("actionable", self.actionable, self.token_count),
        ):
            if values.size != expected:
                raise ValueError(
                    f"{self.prefix}: {name} has {values.size} values, expected {expected}"
                )
        for start in range(0, self.sequences, LOCALITY_BLOCK_SIZE):
            block = np.asarray(self.populations[start : start + LOCALITY_BLOCK_SIZE])
            invalid = np.flatnonzero(block >= len(POPULATION_NAMES))
            if invalid.size:
                value = int(block[invalid[0]])
                raise ValueError(f"{self.prefix}: invalid population id {value}")
        self._load_factor_channels()

    def population_counts(self) -> dict[str, int]:
        """Rows per population, counted one locality block at a time."""
        counts = np.zeros(len(POPULATION_NAMES), dtype=np.int64)
        for start in range(0, self.sequences, LOCALITY_BLOCK_SIZE):
            block = np.asarray(self.populations[start : start + LOCALITY_BLOCK_SIZE])
            counts += np.bincount(block, minlength=len(POPULATION_NAMES))
        return {name: int(counts[index]) for index, name in enumerate(POPULATION_NAMES)}

    def schedule_kind_blocks(self):
        """Per-sequence schedule-kind targets, one locality block at a time."""
        for start in range(0, self.sequences, LOCALITY_BLOCK_SIZE):
            frames = np.asarray(self.frames[start : start + LOCALITY_BLOCK_SIZE])
            yield self.frame_schedule_kinds[frames]

    def _load_factor_channels(self) -> None:
        """Optional 12-bit lexical and lexicon-class sidecars, one value per token."""
        lexical_path = Path(f"{self.prefix}.lexical.bin")
        class_path = Path(f"{self.prefix}.lexclass.bin")
        if lexical_path.exists():
            lexical = np.memmap(lexical_path, dtype=np.uint16, mode="r")
            if lexical.shape[0] != self.token_count:
                raise ValueError(
                    f"{lexical_path}: {lexical.shape[0]} tokens, expected {self.token_count}"
                )
            self.lexical = lexical
        if class_path.exists():
            lexicon = np.memmap(class_path, dtype=np.uint8, mode="r")
            if lexicon.shape[0] != self.token_count:
                raise ValueError(
                    f"{class_path}: {lexicon.shape[0]} tokens, expected {self.token_count}"
                )
            self.lexicon = lexicon

    def _load(self, key: str) -> np.ndarray:
        if key not in self.manifest.format:
            raise ValueError(f"{self.prefix}: missing required shard field '{key}'")
        spec = self.manifest.format[key]
        dtype = NUMPY_DTYPES[spec["dtype"]]
        return np.memmap(f"{self.prefix}{spec['suffix']}", dtype=dtype, mode="r")

    def __len__(self) -> int:
        return self.sequences

    def digests(self) -> dict[str, str]:
        """SHA-256 of the manifest and every shard file for the report.

        Hashing 20 GB of shards took 94 s of a 106 s run (S.1 profile). Digests
        are cached in `<prefix>.digests.json` keyed by file size and mtime, so a
        shard is re-hashed only when it changes.
        """
        cache_path = Path(f"{self.prefix}.digests.json")
        cache: dict[str, dict[str, object]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except (OSError, ValueError):
                cache = {}
        files = {"manifest": Path(f"{self.prefix}.json")}
        if self.manifest.sequence_metadata is not None:
            files["sequenceMetadata"] = Path(f"{self.prefix}{self.manifest.sequence_metadata['suffix']}")
        for key, spec in self.manifest.format.items():
            files[key] = Path(f"{self.prefix}{spec['suffix']}")
        result: dict[str, str] = {}
        dirty = False
        for key, path in files.items():
            stat = path.stat()
            entry = cache.get(key)
            if (
                isinstance(entry, dict)
                and entry.get("size") == stat.st_size
                and entry.get("mtime_ns") == stat.st_mtime_ns
                and isinstance(entry.get("sha256"), str)
            ):
                result[key] = str(entry["sha256"])
                continue
            result[key] = _sha256(path)
            cache[key] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": result[key],
            }
            dirty = True
        if dirty:
            try:
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
            except OSError:
                pass
        return result

    def batch_count(self, batch_size: int) -> int:
        """Exact full-pass optimizer steps, including locality/bucket tails."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        total = 0
        for start in range(0, self.sequences, LOCALITY_BLOCK_SIZE):
            lengths = self.lengths[start : start + LOCALITY_BLOCK_SIZE]
            buckets = np.searchsorted(BUCKETS, lengths, side="left")
            if np.any(buckets >= len(BUCKETS)):
                raise ValueError(f"{self.prefix}: sequence exceeds the maximum token length")
            counts = np.bincount(buckets, minlength=len(BUCKETS))
            total += int(((counts + batch_size - 1) // batch_size).sum())
        return total

    def batches(
        self,
        batch_size: int,
        shuffle_seed: int | None = None,
        max_sequences: int | None = None,
    ):
        """Visit rows once, using bounded locality blocks and length buckets.

        A cap is only used by diagnostic evaluation; it draws a fixed subset
        without replacement before bucketing. Full training never resamples.

        Memory: one block of indices at a time (`LOCALITY_BLOCK_SIZE`), plus
        the block order (one integer per block) and an optional subset of
        `max_sequences` indices. NumPy draws that subset with Floyd's
        algorithm once the split exceeds 50 * max_sequences, so it is never a
        permutation of the whole split. There is no global permutation: the
        corpus file is already in hash order (source-independent), blocks are
        visited in shuffled order, and rows are shuffled within each block.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_sequences is not None and max_sequences < 1:
            raise ValueError("max_sequences must be positive")
        rng = np.random.default_rng(shuffle_seed)
        subset = None
        if max_sequences is not None and max_sequences < self.sequences:
            subset = np.sort(rng.choice(self.sequences, max_sequences, replace=False))
        block_order = np.arange(math.ceil(self.sequences / LOCALITY_BLOCK_SIZE))
        if shuffle_seed is not None:
            rng.shuffle(block_order)
        for block in block_order:
            start = int(block) * LOCALITY_BLOCK_SIZE
            stop = min(start + LOCALITY_BLOCK_SIZE, self.sequences)
            if subset is None:
                selected = np.arange(start, stop, dtype=np.int64)
                lengths = self.lengths[start:stop]
            else:
                left, right = np.searchsorted(subset, (start, stop))
                selected = subset[left:right]
                lengths = self.lengths[selected]
            if selected.size == 0:
                continue
            buckets = np.searchsorted(BUCKETS, lengths, side="left")
            if np.any(buckets >= len(BUCKETS)):
                raise ValueError(f"{self.prefix}: sequence exceeds the maximum token length")
            groups = [selected[buckets == index] for index in range(len(BUCKETS))]
            descriptors = []
            for index, group in enumerate(groups):
                if shuffle_seed is not None:
                    rng.shuffle(group)
                descriptors.extend(
                    (index, offset, min(offset + batch_size, group.size))
                    for offset in range(0, group.size, batch_size)
                )
            if shuffle_seed is not None:
                rng.shuffle(descriptors)
            for index, offset, end in descriptors:
                yield np.sort(groups[index][offset:end])

    def batch(
        self,
        indices: np.ndarray | list[int],
        device: str | None = None,
        *,
        pin_memory: bool = False,
    ) -> dict:
        """Pad `indices` to the next length bucket and return model tensors.

        Length-bucket shards are contiguous arrays built once. A batch is a
        gather (or a handful of gathers when a chunk straddles two buckets),
        never a Python loop that slices the memmap per sequence.

        Unused rows hold the manifest's `featureRows` sentinel. Whitespace and
        any label at or above `roleClasses` (the Phase 13 padding sentinel) are
        set to `-100`, the cross-entropy ignore index, so they contribute no
        role or boundary loss.
        """
        import torch

        index_array = np.asarray(indices, dtype=np.int64)
        count = int(index_array.size)
        if count == 0:
            raise ValueError("batch requires at least one index")
        starts = self.offsets[index_array].astype(np.int64)
        lengths = self.lengths[index_array]
        width = int(length_bucket(int(lengths.max())))
        token_ix = np.arange(width, dtype=np.int64)
        valid = token_ix < lengths[:, None]
        gather = np.where(valid, starts[:, None] + token_ix, 0)

        rows = np.full(
            (count, width, self.rows_per_token), self.feature_rows, dtype=np.int64
        )
        rows[valid] = np.asarray(self.rows)[gather[valid]]

        raw_labels = np.asarray(self.labels)[gather].astype(np.int64, copy=False)
        raw_kinds = np.asarray(self.kinds)[gather].astype(np.int64, copy=False)
        unsupervised = (raw_labels >= self.role_classes) | (raw_kinds == 3)
        labels = np.full((count, width), -100, dtype=np.int64)
        labels[valid] = np.where(unsupervised[valid], -100, raw_labels[valid])

        boundaries = np.zeros((count, width), dtype=np.float32)
        boundaries[valid] = np.asarray(self.boundaries)[gather[valid]].astype(np.float32)

        kinds = np.full((count, width), 3, dtype=np.int64)
        kinds[valid] = raw_kinds[valid]

        neighbors = np.full((count, width, 2), -1, dtype=np.int64)
        neighbors[valid] = np.asarray(self.neighbors)[gather[valid]]

        actionable = np.zeros((count, width), dtype=np.float32)
        actionable[valid] = np.asarray(self.actionable)[gather[valid]].astype(
            np.float32
        )

        speech_acts = np.asarray(self.speech)[index_array].astype(np.int64, copy=False)
        schedule_kinds = self.frame_schedule_kinds[
            np.asarray(self.frames[index_array])
        ].astype(np.int64, copy=False)

        tensors = {
            "rows": torch.as_tensor(rows),
            "valid": torch.as_tensor(valid),
            "neighbors": torch.as_tensor(neighbors),
            "labels": torch.as_tensor(labels),
            "boundaries": torch.as_tensor(boundaries),
            "kinds": torch.as_tensor(kinds),
            "schedule_kinds": torch.as_tensor(schedule_kinds),
            "speech_acts": torch.as_tensor(speech_acts),
            "actionable": torch.as_tensor(actionable),
        }
        if self.lexical is not None:
            lexical = np.zeros((count, width), dtype=np.int64)
            lexical[valid] = np.asarray(self.lexical)[gather[valid]]
            tensors["lexical_ids"] = torch.as_tensor(lexical)
        if self.lexicon is not None:
            lexicon = np.zeros((count, width), dtype=np.int64)
            lexicon[valid] = np.asarray(self.lexicon)[gather[valid]]
            tensors["lexicon_ids"] = torch.as_tensor(lexicon)
        if pin_memory:
            tensors = {key: value.pin_memory() for key, value in tensors.items()}
        if device is not None:
            tensors = {
                key: value.to(device, non_blocking=pin_memory)
                for key, value in tensors.items()
            }
        return tensors
