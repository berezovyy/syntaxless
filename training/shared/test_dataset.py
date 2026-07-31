"""Shard loading and batching against a tiny hand-written shard.

Run with `python training/shared/test_dataset.py` in the training environment.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from dataset import Dataset

FORMAT = {
    "rows": {"suffix": ".rows.bin", "dtype": "uint16"},
    "labels": {"suffix": ".labels.bin", "dtype": "uint8"},
    "boundaries": {"suffix": ".boundaries.bin", "dtype": "uint8"},
    "kinds": {"suffix": ".kinds.bin", "dtype": "uint8"},
    "neighbors": {"suffix": ".neighbors.bin", "dtype": "int16"},
    "offsets": {"suffix": ".offsets.bin", "dtype": "uint32"},
    "frames": {"suffix": ".frames.bin", "dtype": "uint32"},
}
ROWS_PER_TOKEN = 2
FEATURE_ROWS = 10
ROLE_CLASSES = 3


def write_shard(directory: Path, frame_counts: list[int] | None = None) -> Path:
    """Two sequences: three tokens (middle one whitespace) and one token."""
    prefix = directory / "split"
    arrays = {
        "rows": np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16),
        "labels": np.asarray([1, 255, 2, 0], dtype=np.uint8),
        "boundaries": np.asarray([0, 0, 1, 1], dtype=np.uint8),
        "kinds": np.asarray([0, 3, 0, 1], dtype=np.uint8),
        "neighbors": np.asarray([-1, 2, -1, -1, 0, -1, -1, -1], dtype=np.int16),
        "offsets": np.asarray([0, 3, 4], dtype=np.uint32),
        "frames": np.asarray([0, 1], dtype=np.uint32),
    }
    for key, values in arrays.items():
        Path(f"{prefix}{FORMAT[key]['suffix']}").write_bytes(values.tobytes())
    manifest = {
        "fingerprints": ["a", "b"],
        "roleClasses": ROLE_CLASSES,
        "rowsPerToken": ROWS_PER_TOKEN,
        "featureRows": FEATURE_ROWS,
        "labelCounts": {"O": 1, "A": 1, "B": 1},
        "frameNames": ["first", "second"],
        "frameCounts": frame_counts if frame_counts is not None else [1, 1],
        "format": FORMAT,
    }
    Path(f"{prefix}.json").write_text(json.dumps(manifest))
    return prefix


def test_batch_pads_and_ignores_unsupervised_tokens() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(write_shard(Path(directory)))
        assert len(dataset) == 2
        assert dataset.lengths.tolist() == [3, 1]
        batch = dataset.batch([0, 1])
        assert tuple(batch["rows"].shape) == (2, 8, ROWS_PER_TOKEN)
        assert batch["rows"][0, :3].tolist() == [[1, 2], [3, 4], [5, 6]]
        assert int(batch["rows"][1, 1, 0]) == FEATURE_ROWS
        assert batch["labels"][0, :4].tolist() == [1, -100, 2, -100]
        assert batch["labels"][1, :2].tolist() == [0, -100]
        assert batch["valid"].sum(dim=1).tolist() == [3, 1]
        assert batch["neighbors"][0, 0].tolist() == [-1, 2]


def test_batches_group_by_length_and_seed_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(write_shard(Path(directory)))
        assert list(dataset.batches(1)) == [[1], [0]]
        assert list(dataset.batches(2, shuffle_seed=5)) == list(
            dataset.batches(2, shuffle_seed=5)
        )


def test_frame_count_mismatch_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        prefix = write_shard(Path(directory), frame_counts=[2, 0])
        try:
            Dataset(prefix)
        except ValueError as error:
            assert "frame counts" in str(error)
        else:
            raise AssertionError("mismatched frame counts were accepted")


if __name__ == "__main__":
    test_batch_pads_and_ignores_unsupervised_tokens()
    test_batches_group_by_length_and_seed_is_deterministic()
    test_frame_count_mismatch_is_rejected()
    print("test_dataset: all checks passed")
