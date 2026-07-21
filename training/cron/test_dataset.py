"""Dataset grouping, batching, and supervision compatibility checks."""
from __future__ import annotations

import numpy as np

import json
import tempfile
from pathlib import Path

from dataset import (
    CONTENT_LABEL_IDS,
    ROLE_LABELS,
    SPEECH_ACT_NAMES,
    Dataset,
    LOCALITY_BLOCK_SIZE,
)


def _write_tiny_shard(
    directory: Path, populations: tuple[int, int, int] = (0, 0, 2),
    *, name: str = "tiny",
) -> Path:
    """Three sequences, mixed lengths, discourse heads, manifest dtypes."""
    prefix = directory / name
    rows_per_token = 2
    feature_rows = 16
    # seq0 length 2, seq1 length 3, seq2 length 1 → buckets 8, 8, 8
    offsets = np.asarray([0, 2, 5, 6], dtype=np.uint32)
    token_count = 6
    rows = np.arange(token_count * rows_per_token, dtype=np.uint16).reshape(
        token_count, rows_per_token
    )
    labels = np.asarray(
        [
            ROLE_LABELS.index("RECUR"),
            ROLE_LABELS.index("WEEKDAY_MON"),
            ROLE_LABELS.index("HOUR_9"),
            ROLE_LABELS.index("MINUTE_30"),
            ROLE_LABELS.index("MERIDIEM_AM"),
            ROLE_LABELS.index("O"),
        ],
        dtype=np.uint8,
    )
    boundaries = np.asarray([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    kinds = np.asarray([0, 0, 0, 0, 0, 3], dtype=np.uint8)
    neighbors = np.asarray(
        [[-1, 1], [0, -1], [-1, 1], [0, 2], [1, -1], [-1, -1]], dtype=np.int16
    )
    frames = np.asarray([0, 1, 0], dtype=np.uint32)
    speech = np.asarray([1, 1, 0], dtype=np.uint8)
    actionable = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.uint8)
    files = {
        "rows": rows.reshape(-1),
        "labels": labels,
        "boundaries": boundaries,
        "kinds": kinds,
        "neighbors": neighbors.reshape(-1),
        "offsets": offsets,
        "frames": frames,
        "speech": speech,
        "actionable": actionable,
        "populations": np.asarray(populations, dtype=np.uint8),
    }
    fmt = {
        "rows": {"suffix": ".rows.bin", "dtype": "uint16"},
        "labels": {"suffix": ".labels.bin", "dtype": "uint8"},
        "boundaries": {"suffix": ".boundaries.bin", "dtype": "uint8"},
        "kinds": {"suffix": ".kinds.bin", "dtype": "uint8"},
        "neighbors": {"suffix": ".neighbors.bin", "dtype": "int16"},
        "offsets": {"suffix": ".offsets.bin", "dtype": "uint32"},
        "frames": {"suffix": ".frames.bin", "dtype": "uint32"},
        "speech": {"suffix": ".speech.bin", "dtype": "uint8"},
        "actionable": {"suffix": ".actionable.bin", "dtype": "uint8"},
        "populations": {"suffix": ".populations.bin", "dtype": "uint8"},
    }
    for key, array in files.items():
        array.tofile(f"{prefix}{fmt[key]['suffix']}")
    manifest = {
        "fingerprints": ["a", "b", "c"],
        "roleClasses": len(ROLE_LABELS),
        "rowsPerToken": rows_per_token,
        "featureRows": feature_rows,
        "labelCounts": {"WEEKDAY_MON": 1, "HOUR_9": 1},
        "format": fmt,
        "frameNames": ["single-weekday/on", "time-of-day/at"],
        "frameCounts": [2, 1],
        "speechActClasses": 9,
        "speechActNames": list(SPEECH_ACT_NAMES),
        "actionableCounts": {"0": 1, "1": 5},
        "populationCounts": {
            name: populations.count(index)
            for index, name in enumerate(("R", "H", "E"))
        },
    }
    Path(f"{prefix}.json").write_text(json.dumps(manifest))
    return prefix


def test_batch_preserves_requested_order_duplicates_and_padding() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        batch = dataset.batch([2, 0, 2, 1])
        pad = [16, 16]
        expected = {
            "rows": [
                [[10, 11]] + [pad] * 7,
                [[0, 1], [2, 3]] + [pad] * 6,
                [[10, 11]] + [pad] * 7,
                [[4, 5], [6, 7], [8, 9]] + [pad] * 5,
            ],
            "valid": [
                [True] + [False] * 7,
                [True] * 2 + [False] * 6,
                [True] + [False] * 7,
                [True] * 3 + [False] * 5,
            ],
            "labels": [
                [-100] * 8,
                [ROLE_LABELS.index("RECUR"), ROLE_LABELS.index("WEEKDAY_MON")]
                + [-100] * 6,
                [-100] * 8,
                [
                    ROLE_LABELS.index("HOUR_9"),
                    ROLE_LABELS.index("MINUTE_30"),
                    ROLE_LABELS.index("MERIDIEM_AM"),
                ]
                + [-100] * 5,
            ],
            "boundaries": [[0] * 8, [0, 1] + [0] * 6, [0] * 8, [1] + [0] * 7],
            "kinds": [[3] * 8, [0, 0] + [3] * 6, [3] * 8, [0, 0, 0] + [3] * 5],
            "neighbors": [
                [[-1, -1]] * 8,
                [[-1, 1], [0, -1]] + [[-1, -1]] * 6,
                [[-1, -1]] * 8,
                [[-1, 1], [0, 2], [1, -1]] + [[-1, -1]] * 5,
            ],
            "actionable": [[0] * 8, [1, 1] + [0] * 6, [0] * 8, [1, 1, 1] + [0] * 5],
            "speech_acts": [0, 1, 0, 1],
            "schedule_kinds": [3, 3, 3, 2],  # weekly, weekly, weekly, daily
        }
        assert batch.keys() == expected.keys()
        for field, values in expected.items():
            np.testing.assert_array_equal(batch[field].numpy(), values, err_msg=field)
        assert dataset.rows.dtype == np.uint16
        assert dataset.neighbors.dtype == np.int16


def test_shards_require_published_sequence_and_discourse_fields() -> None:
    for field in ("frames", "speech", "actionable", "populations"):
        with tempfile.TemporaryDirectory(prefix="cron-missing-target-") as directory:
            prefix = _write_tiny_shard(Path(directory), populations=(0, 0, 2))
            path = Path(f"{prefix}.json")
            manifest = json.loads(path.read_text())
            del manifest["format"][field]
            path.write_text(json.dumps(manifest))
            try:
                Dataset(prefix)
            except ValueError as error:
                assert str(prefix) in str(error)
                assert f"missing required shard field '{field}'" in str(error)
            else:
                raise AssertionError(f"Missing {field} was accepted as usable supervision")


def test_full_pass_visits_every_row_once_across_locality_blocks() -> None:
    dataset = Dataset.__new__(Dataset)
    dataset.prefix = Path("fixture")
    dataset.sequences = LOCALITY_BLOCK_SIZE + 13
    dataset.lengths = np.resize(np.asarray([1, 9, 33, 65, 128]), dataset.sequences)
    batches = list(dataset.batches(512, shuffle_seed=17))
    chosen = np.concatenate(batches)
    np.testing.assert_array_equal(np.sort(chosen), np.arange(dataset.sequences))
    assert len(batches) == dataset.batch_count(512)
    assert all(np.all(indices[:-1] <= indices[1:]) for indices in batches)
    assert all(indices[0] // LOCALITY_BLOCK_SIZE == indices[-1] // LOCALITY_BLOCK_SIZE for indices in batches)


def test_diagnostic_subset_is_seeded_without_replacement() -> None:
    dataset = Dataset.__new__(Dataset)
    dataset.prefix = Path("fixture")
    dataset.sequences = 100
    dataset.lengths = np.resize(np.asarray([1, 16, 48]), dataset.sequences)
    first = np.concatenate(list(dataset.batches(8, shuffle_seed=0, max_sequences=17)))
    second = np.concatenate(list(dataset.batches(8, shuffle_seed=0, max_sequences=17)))
    np.testing.assert_array_equal(first, second)
    assert first.size == np.unique(first).size == 17
    assert int(first.min()) >= 0 and int(first.max()) < dataset.sequences
