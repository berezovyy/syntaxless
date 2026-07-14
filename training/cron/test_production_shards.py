"""Behavioral failure controls for the complete production consumption verifier."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from dataset import Dataset
from test_dataset import _write_tiny_shard
from verify_production_shards import (
    FIELDS, array_digest, corpus_record_count, file_sha256, verify_batches,
    verify_production,
)


def test_production_verifier_checks_every_field_and_rejects_valid_sized_corruption() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        populations = (0, 1, 2)
        original = _write_tiny_shard(root, populations)
        manifest = json.loads(Path(f"{original}.json").read_text())
        corpus = root / "corpus.jsonl"
        corpus.write_text("{}\n" * 9)  # Unit fixture; the verifier checks source identity, not semantics.
        expected = {
            "version": 1, "scope": "source-derived-shard-expectations",
            "rowsPerToken": 2, "featureRows": 16,
            "corpus": {"records": 9, "sha256": file_sha256(corpus)}, "splits": {},
        }
        for split in ("train", "validation", "heldout"):
            prefix = root / f"production-{split}"
            for path in root.glob("tiny.*"):
                shutil.copyfile(path, Path(str(prefix) + path.name[len("tiny"):]))
            expected["splits"][split] = {
                "sequences": 3, "tokens": 6, "frameNames": manifest["frameNames"],
                "frameCounts": manifest["frameCounts"], "labelCounts": manifest["labelCounts"],
                "actionableCounts": manifest["actionableCounts"],
                "populationCounts": manifest["populationCounts"],
                "fields": {
                    field: {
                        "bytes": Path(f"{prefix}.{field}.bin").stat().st_size,
                        "sha256": file_sha256(Path(f"{prefix}.{field}.bin")),
                    } for field in FIELDS
                },
                "scheduleKinds": array_digest(np.asarray([3, 2, 3]), np.uint8),
            }
        path = root / "expectations.json"
        path.write_text(json.dumps(expected))
        report = verify_production(corpus, root / "production", path)
        assert report["passed"] and report["records"] == 9
        assert all(s["batchesConsumedSequences"] == 3 for s in report["splits"].values())
        compressed = root / "corpus.jsonl.gz"
        with compressed.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(corpus.read_bytes())
        expected["corpus"]["sha256"] = file_sha256(compressed)
        path.write_text(json.dumps(expected))
        gzip_report = verify_production(compressed, root / "production", path)
        assert gzip_report["passed"] and gzip_report["corpusSha256"] == file_sha256(compressed)
        truncated = root / "truncated.jsonl.gz"
        truncated.write_bytes(compressed.read_bytes()[:-4])
        try:
            corpus_record_count(truncated)
        except (EOFError, OSError):
            pass
        else:
            raise AssertionError("truncated gzip corpus passed complete readback")
        labels = root / "production-heldout.labels.bin"
        damaged = bytearray(labels.read_bytes())
        damaged[-1] = 1  # Still a valid role ID and exactly the right byte count.
        labels.write_bytes(damaged)
        try:
            verify_production(compressed, root / "production", path)
        except ValueError as error:
            assert "heldout: labels differs" in str(error)
        else:
            raise AssertionError("same-size label corruption was not detected")


def test_complete_batch_verification_catches_a_late_sequence_and_padding_corruption() -> None:
    with tempfile.TemporaryDirectory() as directory:
        data = Dataset(_write_tiny_shard(Path(directory)))
        assert verify_batches(data, batch_size=1) == 3
        original = data.batch

        def corrupt_last(indices):
            result = original(indices)
            if int(indices[-1]) == 2:
                result["rows"][-1, 0, 0] += 1
            return result

        data.batch = corrupt_last
        try:
            verify_batches(data, batch_size=1)
        except ValueError as error:
            assert "batch 2: rows" in str(error)
        else:
            raise AssertionError("late sequence was not verified")

        def corrupt_padding(indices):
            result = original(indices)
            result["actionable"][0, -1] = 1
            return result

        data.batch = corrupt_padding
        try:
            verify_batches(data)
        except ValueError as error:
            assert "actionable padding" in str(error)
        else:
            raise AssertionError("padding corruption was not detected")

        def truncate_batch(indices):
            result = original(indices)
            for key in ("rows", "valid", "neighbors", "labels", "boundaries", "kinds", "actionable"):
                result[key] = result[key][:, :1]
            return result

        data.batch = truncate_batch
        try:
            verify_batches(data)
        except ValueError as error:
            assert "truncated or wrong-width" in str(error)
        else:
            raise AssertionError("truncated batch was certified as full consumption")
