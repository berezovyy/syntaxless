"""Shard loader, length-bucketed batching, and split regeneration.

The shard contract is owned by TypeScript (`packages/cron/training/src/featurize.ts`):
a `<prefix>.json` manifest plus `<prefix>.<field>.bin` files for rows, labels,
boundaries, kinds, neighbors, offsets, frames, speech, and actionable. Nothing
here parses cron text: the loader reads labels and discourse supervision that
the generator recorded, and the regenerator drives the committed TypeScript
tools.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: dtype spellings used by the manifest's `format` block.
#: Old shards stay readable: the manifest `format` names the dtype per field.
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

# Must match the production featurizer. This is corpus population, not a model
# target: it exists only so a complete, unique corpus can be mixed at training
# time without deleting rows while the corpus is built.
POPULATION_NAMES = ("R", "H", "E")
POPULATION_INDEX = {name: index for index, name in enumerate(POPULATION_NAMES)}

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
SPEECH_ACT_INDEX = {name: index for index, name in enumerate(SPEECH_ACT_NAMES)}

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
    name for name in ROLE_LABELS if name != "O" and name not in STRUCTURAL_LABELS
}
CONTENT_LABEL_IDS = {
    index for index, name in enumerate(ROLE_LABELS) if name in CONTENT_LABELS
}

#: Per-epoch validation cap (Track S.1). Full validation is scored at run end.
VAL_MAX_SEQUENCES = 20_000
#: `--fresh-each-epoch` draw from the pool.
FRESH_EPOCH_SIZE = 300_000


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


def group_frame_members(frames: np.ndarray, frame_count: int) -> list[np.ndarray]:
    """Group sequence indices by frame with one vectorized ordering pass.

    Scanning the complete frame shard once per frame is quadratic in practice:
    the schedule lattice has millions of rows and tens of thousands of wording
    frames. A stable argsort touches the shard once and keeps member indices in
    their original sequence order.
    """
    frame_ids = np.asarray(frames)
    if frame_ids.ndim != 1:
        raise ValueError("frames must be a one-dimensional array")
    if frame_count < 1 or int(frame_ids.max(initial=0)) >= frame_count:
        raise ValueError("invalid frame ids or names")
    counts = np.bincount(frame_ids, minlength=frame_count)
    order = np.argsort(frame_ids, kind="stable")
    stops = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), stops[:-1]))
    return [order[start:stop] for start, stop in zip(starts, stops, strict=True)]


def population_draw_counts(
    counts: np.ndarray,
    weights: np.ndarray,
    draw_total: int | None,
    *,
    cover_all: bool,
) -> np.ndarray:
    """Allocate an exact population mix, optionally retaining every member."""
    counts = np.asarray(counts, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if counts.shape != weights.shape or counts.ndim != 1:
        raise ValueError("population counts and weights must be matching vectors")
    if np.any(counts < 0) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("invalid population counts or weights")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("population mix must have positive total weight")
    weights = weights / total_weight
    if cover_all:
        if np.any((counts > 0) & (weights == 0)):
            raise ValueError("cover-all population mix cannot assign zero weight")
        total = max(
            int(counts.sum()),
            math.ceil(max(float(count) / float(weight) for count, weight in zip(counts, weights, strict=True) if count)),
        )
    else:
        if draw_total is None:
            draw_total = int(counts.sum())
        total = int(draw_total)
    while True:
        raw = weights * total
        allocated = np.floor(raw).astype(np.int64)
        remainder = total - int(allocated.sum())
        if remainder:
            order = np.argsort(-(raw - allocated), kind="stable")
            allocated[order[:remainder]] += 1
        if not cover_all or bool(np.all(allocated >= counts)):
            return allocated
        total += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ShardManifest:
    """The featurizer manifest: roles, frames, and optional discourse heads."""

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
        self.populations = (
            self._load("populations")
            if "populations" in self.manifest.format
            else None
        )
        self.speech = None
        self.actionable = None

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
        self.frame_members: list[np.ndarray] | None = None
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
        if self.populations is not None:
            if self.populations.size != self.sequences:
                raise ValueError(
                    f"{self.prefix}: populations has {self.populations.size} values, "
                    f"expected {self.sequences}"
                )
            invalid = np.asarray(self.populations) >= len(POPULATION_NAMES)
            if bool(np.any(invalid)):
                value = int(np.asarray(self.populations)[np.flatnonzero(invalid)[0]])
                raise ValueError(f"{self.prefix}: invalid population id {value}")
        self._load_discourse()

    def _load(self, key: str) -> np.ndarray:
        spec = self.manifest.format[key]
        dtype = NUMPY_DTYPES[spec["dtype"]]
        return np.memmap(f"{self.prefix}{spec['suffix']}", dtype=dtype, mode="r")

    def _load_discourse(self) -> None:
        """Load speech-act / actionable shards, or synthesize them for old prefixes."""
        if "speech" in self.manifest.format:
            speech = self._load("speech")
            if speech.size != self.sequences:
                raise ValueError(
                    f"{self.prefix}: speech has {speech.size} values, "
                    f"expected {self.sequences}"
                )
            self.speech = speech.astype(np.int64, copy=False)
        if "actionable" in self.manifest.format:
            actionable = self._load("actionable")
            if actionable.size != self.token_count:
                raise ValueError(
                    f"{self.prefix}: actionable has {actionable.size} values, "
                    f"expected {self.token_count}"
                )
            self.actionable = actionable.astype(np.uint8, copy=False)
        if self.speech is not None and self.actionable is not None:
            return
        speech = np.full(self.sequences, SPEECH_ACT_INDEX["none"], dtype=np.int64)
        actionable = np.zeros(self.token_count, dtype=np.uint8)
        for index in range(self.sequences):
            start = int(self.offsets[index])
            end = int(self.offsets[index + 1])
            labels = self.labels[start:end]
            kinds = self.kinds[start:end]
            has_content = bool(
                np.any(
                    (kinds != 3)
                    & np.isin(labels, list(CONTENT_LABEL_IDS))
                )
            )
            if has_content:
                speech[index] = SPEECH_ACT_INDEX["request"]
                content = (kinds != 3) & np.isin(labels, list(CONTENT_LABEL_IDS))
                actionable[start:end] = content.astype(np.uint8)
        if self.speech is None:
            self.speech = speech
        if self.actionable is None:
            self.actionable = actionable

    def __len__(self) -> int:
        return self.sequences

    def _frame_members(self) -> list[np.ndarray]:
        """Build the large frame index only when frame balancing requests it."""
        if self.frames is None:
            return []
        if self.frame_members is None:
            frame_count = len(self.manifest.frame_names)
            try:
                members = group_frame_members(self.frames, frame_count)
            except ValueError as error:
                raise ValueError(f"{self.prefix}: {error}") from error
            actual_counts = [int(group.size) for group in members]
            if self.manifest.frame_counts and actual_counts != self.manifest.frame_counts:
                raise ValueError(f"{self.prefix}: frame counts do not match frames shard")
            self.frame_members = members
        return self.frame_members

    def population_epoch_size(
        self, population_mix: tuple[float, float, float], *, cover_all: bool
    ) -> int:
        if self.populations is None:
            raise ValueError("population epoch size requires a populations shard")
        counts = np.bincount(
            np.asarray(self.populations), minlength=len(POPULATION_NAMES)
        )
        allocated = population_draw_counts(
            counts, np.asarray(population_mix), self.sequences, cover_all=cover_all
        )
        return int(allocated.sum())

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

    def batches(
        self,
        batch_size: int,
        shuffle_seed: int | None = None,
        balance_frames: bool = False,
        balance_kinds: bool = False,
        balance_schedule_kinds: bool = False,
        frame_balance_power: float = 0.5,
        population_mix: tuple[float, float, float] | None = None,
        population_cover_all: bool = False,
        max_sequences: int | None = None,
    ):
        """Yield batches of sequence indices, grouped by similar length.

        Sorting by length before chunking means only the last batch of each
        length class carries meaningful padding.
        """
        rng = np.random.default_rng(shuffle_seed)
        selected = np.arange(self.sequences)
        cover_all_extras: np.ndarray | None = None
        if population_mix is not None:
            if balance_frames or balance_kinds or balance_schedule_kinds:
                raise ValueError(
                    "population mixing cannot be combined with frame/kind balancing"
                )
            if self.populations is None:
                raise ValueError(
                    f"{self.prefix}: population mixing requires a populations shard"
                )
            weights = np.asarray(population_mix, dtype=np.float64)
            if weights.shape != (len(POPULATION_NAMES),):
                raise ValueError("population mix must have exactly R,H,E weights")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("population mix weights must be finite and non-negative")
            total_weight = float(weights.sum())
            if total_weight <= 0:
                raise ValueError("population mix must have positive total weight")
            counts = np.bincount(
                np.asarray(self.populations), minlength=len(POPULATION_NAMES)
            )
            draw_total = min(self.sequences, max_sequences) if max_sequences is not None else self.sequences
            draw_counts = population_draw_counts(
                counts,
                weights,
                draw_total,
                cover_all=population_cover_all,
            )
            population_values = np.asarray(self.populations)
            draws: list[np.ndarray] = []
            for population_id, count in enumerate(draw_counts):
                if count == 0:
                    continue
                members = np.flatnonzero(population_values == population_id)
                if members.size == 0:
                    raise ValueError(
                        f"{self.prefix}: population {POPULATION_NAMES[population_id]} "
                        "is empty but has non-zero requested weight"
                    )
                count = int(count)
                if population_cover_all:
                    extra = count - members.size
                    if extra:
                        draws.append(
                            rng.choice(
                                members, size=extra, replace=extra > members.size
                            )
                        )
                else:
                    draws.append(rng.choice(members, size=count, replace=count > members.size))
            if population_cover_all:
                cover_all_extras = (
                    np.concatenate(draws)
                    if draws
                    else np.empty(0, dtype=np.int64)
                )
            else:
                selected = np.concatenate(draws)
                rng.shuffle(selected)
            del draws, members
            # The cap was incorporated into the population allocation. Applying
            # it again would disturb the requested R/H/E proportions.
            max_sequences = None
        elif balance_frames or balance_kinds or balance_schedule_kinds:
            frame_members = self._frame_members()
            if not frame_members:
                raise ValueError(
                    f"{self.prefix}: balanced batching requires a frames shard"
                )
            counts = np.asarray(
                [members.size for members in frame_members], dtype=np.float64
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
            for count, members in zip(draw_counts, frame_members, strict=True):
                if count:
                    draws.append(rng.choice(members, size=count, replace=True))
            selected = np.concatenate(draws)
            rng.shuffle(selected)
            del draws, members

        if max_sequences is not None and selected.size > max_sequences:
            selected = rng.choice(selected, size=max_sequences, replace=False)

        if population_cover_all:
            # The 38 GiB rows shard is larger than RAM. Globally randomized
            # sequence ids turn every batch into thousands of NVMe page faults
            # and leave CUDA idle. Shuffle 262k-sequence locality blocks, then
            # bucket and shuffle batches inside each block. Every unique row is
            # still visited once, minority duplicates stay beside their source,
            # and each gather becomes predominantly sequential I/O.
            block_size = 262_144
            block_count = math.ceil(self.sequences / block_size)
            extras = (
                cover_all_extras
                if cover_all_extras is not None
                else np.empty(0, dtype=np.int64)
            )
            if extras.size:
                extra_blocks = extras // block_size
                extra_order = np.argsort(extra_blocks, kind="stable")
                extras = extras[extra_order]
                extra_counts = np.bincount(extra_blocks, minlength=block_count)
            else:
                extra_counts = np.zeros(block_count, dtype=np.int64)
            extra_stops = np.cumsum(extra_counts, dtype=np.int64)
            extra_starts = np.concatenate(
                (np.zeros(1, dtype=np.int64), extra_stops[:-1])
            )
            block_order = np.arange(block_count)
            if shuffle_seed is not None:
                rng.shuffle(block_order)
            for block in block_order:
                start = int(block) * block_size
                stop = min(start + block_size, self.sequences)
                block_selected = np.arange(start, stop, dtype=np.int64)
                extra_start = int(extra_starts[block])
                extra_stop = int(extra_stops[block])
                if extra_stop > extra_start:
                    block_selected = np.concatenate(
                        (block_selected, extras[extra_start:extra_stop])
                    )
                bucket_ids = np.searchsorted(
                    BUCKETS, self.lengths[block_selected], side="left"
                )
                groups = [
                    block_selected[bucket_ids == index]
                    for index in range(len(BUCKETS))
                ]
                descriptors: list[tuple[int, int, int]] = []
                for bucket_index, group in enumerate(groups):
                    if shuffle_seed is not None:
                        rng.shuffle(group)
                    descriptors.extend(
                        (bucket_index, offset, min(offset + batch_size, group.size))
                        for offset in range(0, group.size, batch_size)
                    )
                if shuffle_seed is not None:
                    rng.shuffle(descriptors)
                for bucket_index, offset, end in descriptors:
                    # Sorted ids preserve sequential memmap reads; order inside
                    # a batch cannot affect its averaged gradient.
                    yield np.sort(groups[bucket_index][offset:end])
            return

        # Group into the fixed padding buckets, then shuffle whole batches. This
        # keeps padding low without the old 20M-row global lexsort or a permanent
        # shortest-to-longest curriculum.
        bucket_ids = np.searchsorted(BUCKETS, self.lengths[selected], side="left")
        groups = [selected[bucket_ids == index] for index in range(len(BUCKETS))]
        del bucket_ids, selected
        descriptors = [
            (bucket_index, start, min(start + batch_size, group.size))
            for bucket_index, group in enumerate(groups)
            for start in range(0, group.size, batch_size)
        ]
        if shuffle_seed is not None:
            rng.shuffle(descriptors)
        for bucket_index, start, stop in descriptors:
            yield groups[bucket_index][start:stop]

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
        if self.actionable is not None:
            actionable[valid] = np.asarray(self.actionable)[gather[valid]].astype(
                np.float32
            )

        speech_acts = (
            np.asarray(self.speech)[index_array].astype(np.int64, copy=False)
            if self.speech is not None
            else np.zeros(count, dtype=np.int64)
        )
        if self.frames is not None:
            schedule_kinds = self.frame_schedule_kinds[
                np.asarray(self.frames[index_array])
            ].astype(np.int64, copy=False)
        else:
            schedule_kinds = np.zeros(count, dtype=np.int64)

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
        if pin_memory:
            tensors = {key: value.pin_memory() for key, value in tensors.items()}
        if device is not None:
            tensors = {
                key: value.to(device, non_blocking=pin_memory)
                for key, value in tensors.items()
            }
        return tensors


def fingerprints_path(prefix: str | Path) -> Path:
    """The sibling `<prefix>.jsonl.fingerprints.json` written by generate.ts."""
    return Path(f"{prefix}.jsonl.fingerprints.json")


def merge_fingerprint_excludes(paths: list[Path], out: Path) -> Path:
    """Union fingerprint JSON files into one generate.ts `--exclude` payload.

    Train must drop both held-out and validation wording; generate.ts accepts
    a single exclude file.
    """
    fingerprints: set[str] = set()
    for path in paths:
        parsed = json.loads(path.read_text())
        values = parsed.get("fingerprints")
        if not isinstance(values, list):
            raise TypeError(f"exclude file has no fingerprints array: {path}")
        fingerprints.update(str(value) for value in values)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fingerprints": sorted(fingerprints)}) + "\n")
    return out


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
