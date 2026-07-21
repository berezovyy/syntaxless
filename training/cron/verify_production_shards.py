"""Read every production shard and every training batch against source-derived hashes.

The expectation file must come from a separate TypeScript pass over the actual
JSONL, using featurizeRecord, not hashes copied from the output shard files.
This verifies storage/loader fidelity, not model quality or human provenance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from dataset import (
    Dataset, LOCALITY_BLOCK_SIZE, NUMPY_DTYPES, ROLE_LABELS, SPEECH_ACT_NAMES, length_bucket,
)


FIELDS = (
    "rows", "labels", "boundaries", "kinds", "neighbors", "offsets",
    "frames", "speech", "actionable",
    "populations",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def corpus_record_count(path: Path) -> int:
    """Read the complete canonical corpus, validating gzip CRC/truncation."""
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    if opener is gzip.open:
        stream = opener(path, "rb")
    else:
        stream = path.open("rb")
    with stream:
        return sum(1 for line in stream if line.strip())


def array_digest(values: np.ndarray, dtype: type[np.generic]) -> dict:
    """Hash loader values in their serialized dtype, 65,536 entries at a time."""
    return blocks_digest(
        (values[start:start + 65_536] for start in range(0, len(values), 65_536)), dtype
    )


def blocks_digest(blocks, dtype: type[np.generic]) -> dict:
    """Hash a stream of bounded blocks; never holds more than one block."""
    digest = hashlib.sha256()
    count = 0
    for values in blocks:
        block = np.ascontiguousarray(values, dtype=dtype)
        digest.update(memoryview(block).cast("B"))
        count += block.nbytes
    return {"bytes": count, "sha256": digest.hexdigest()}


def lengths_within(data: Dataset, low: int, high: int) -> bool:
    """Whether every sequence length is in [low, high], one block at a time."""
    for start in range(0, len(data), LOCALITY_BLOCK_SIZE):
        lengths = data.lengths[start:start + LOCALITY_BLOCK_SIZE]
        if lengths.size and (int(lengths.min()) < low or int(lengths.max()) > high):
            return False
    return True


def verify_batches(data: Dataset, batch_size: int = 512) -> int:
    """Exercise the actual model-input path for every sequence, without sampling."""
    require(batch_size > 0, "batch_size must be positive")
    checked = 0
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data)))
        batch = {name: tensor.numpy() for name, tensor in data.batch(indices).items()}
        width = batch["valid"].shape[1]
        lengths = data.lengths[indices]
        require(width == length_bucket(int(lengths.max())), f"batch {start}: truncated or wrong-width batch")
        valid = np.arange(width)[None, :] < lengths[:, None]
        require(np.array_equal(batch["valid"], valid), f"batch {start}: valid mask")
        gather = data.offsets[indices, None].astype(np.int64) + np.arange(width)
        raw_indices = gather[valid]

        def equal(name: str, expected: np.ndarray) -> None:
            require(np.array_equal(batch[name][valid], expected), f"batch {start}: {name}")

        equal("rows", data.rows[raw_indices])
        equal("neighbors", data.neighbors[raw_indices])
        equal("kinds", data.kinds[raw_indices])
        labels = data.labels[raw_indices].astype(np.int64)
        ignored = (labels >= data.role_classes) | (data.kinds[raw_indices] == 3)
        equal("labels", np.where(ignored, -100, labels))
        equal("boundaries", data.boundaries[raw_indices])
        equal("actionable", data.actionable[raw_indices])
        for name, sentinel in {
            "rows": data.feature_rows, "neighbors": -1, "labels": -100,
            "kinds": 3, "boundaries": 0, "actionable": 0,
        }.items():
            require(bool(np.all(batch[name][~valid] == sentinel)), f"batch {start}: {name} padding")
        require(np.array_equal(batch["speech_acts"], data.speech[indices]), f"batch {start}: speech")
        require(np.array_equal(batch["schedule_kinds"], data.frame_schedule_kinds[data.frames[indices]]), f"batch {start}: schedule kinds")
        checked += len(indices)
    return checked


def verify_production(corpus: Path, prefix: Path, expectations: Path) -> dict:
    expected_hash = file_sha256(expectations)
    expected = json.loads(expectations.read_text())
    require(expected.get("version") == 1, "Unsupported expectation version")
    require(expected.get("scope") == "source-derived-shard-expectations", "Expectations must be source-derived")
    require(file_sha256(corpus) == expected["corpus"]["sha256"], "Corpus hash does not match source-derived expectations")
    require(
        corpus_record_count(corpus) == expected["corpus"]["records"],
        "Corpus record count does not match source-derived expectations",
    )
    reports = {}
    total = 0
    for split in ("train", "validation", "heldout"):
        want = expected["splits"][split]
        manifest_path = Path(f"{prefix}-{split}.json")
        manifest_hash = file_sha256(manifest_path)
        data = Dataset(f"{prefix}-{split}")
        if "sequenceMetadata" in want:
            metadata = want["sequenceMetadata"]
            require(data.manifest.sequence_metadata == metadata, f"{split}: metadata manifest")
            metadata_path = Path(f"{prefix}-{split}{metadata['suffix']}")
            require(metadata_path.stat().st_size == metadata["bytes"], f"{split}: metadata size")
            require(file_sha256(metadata_path) == metadata["sha256"], f"{split}: metadata differs from source")
            require(metadata["rows"] == len(data), f"{split}: metadata rows")
        require(len(data) == want["sequences"] > 0, f"{split}: sequence count")
        require(data.token_count == want["tokens"], f"{split}: token count")
        require(data.manifest.frame_names == want["frameNames"], f"{split}: frame names")
        require(data.manifest.frame_counts == want["frameCounts"], f"{split}: frame counts")
        require(data.manifest.label_counts == want["labelCounts"], f"{split}: label counts")
        require(data.manifest.actionable_counts == want["actionableCounts"], f"{split}: actionability counts")
        require(data.population_counts() == want["populationCounts"], f"{split}: population counts")
        require(data.manifest.population_counts == want["populationCounts"], f"{split}: manifest population counts")
        require(data.role_classes == len(ROLE_LABELS), f"{split}: role vocabulary")
        require(data.manifest.speech_act_names == list(SPEECH_ACT_NAMES), f"{split}: speech vocabulary")
        require(data.rows_per_token == expected["rowsPerToken"], f"{split}: feature width")
        require(data.feature_rows == expected["featureRows"], f"{split}: feature vocabulary")
        require(int(data.offsets[0]) == 0, f"{split}: initial offset")
        require(lengths_within(data, 1, 128), f"{split}: token lengths")
        fields = {}
        paths = {}
        for field in FIELDS:
            values = getattr(data, field)
            require(values is not None, f"{split}: missing {field}")
            dtype = NUMPY_DTYPES[data.manifest.format[field]["dtype"]]
            actual = array_digest(values, dtype)
            require(actual == want["fields"][field], f"{split}: {field} differs from JSONL-derived supervision")
            path = Path(f"{prefix}-{split}{data.manifest.format[field]['suffix']}")
            require(path.stat().st_size == actual["bytes"] and file_sha256(path) == actual["sha256"], f"{split}: {field} file differs from loaded values")
            paths[field] = path
            fields[field] = actual
        require(
            blocks_digest(data.schedule_kind_blocks(), np.uint8) == want["scheduleKinds"],
            f"{split}: schedule-kind targets",
        )
        checked = verify_batches(data)
        require(checked == len(data), f"{split}: incomplete batch consumption")
        for field, path in paths.items():
            require(path.stat().st_size == fields[field]["bytes"] and file_sha256(path) == fields[field]["sha256"], f"{split}: {field} changed during verification")
        require(file_sha256(manifest_path) == manifest_hash, f"{split}: manifest changed during verification")
        reports[split] = {
            "sequences": len(data), "tokens": data.token_count,
            "batchesConsumedSequences": checked, "fields": fields,
        }
        total += len(data)
        print(f"verified {split}: {len(data):,} sequences, every batch", file=sys.stderr)
    require(total == expected["corpus"]["records"], "All-split record count")
    require(file_sha256(expectations) == expected_hash, "Expectations changed during verification")
    require(file_sha256(corpus) == expected["corpus"]["sha256"], "Corpus changed during verification")
    return {
        "version": 1, "scope": "complete-source-to-shards-python-consumption-not-model-acceptance",
        "passed": True, "records": total, "expectationsSha256": expected_hash,
        "corpusSha256": expected["corpus"]["sha256"], "splits": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("expectations", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    require(not args.report.exists(), "Refusing to overwrite verification report")
    report = verify_production(args.corpus, args.prefix, args.expectations)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
