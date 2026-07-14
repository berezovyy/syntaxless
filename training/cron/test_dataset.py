"""Dataset grouping, batching, and supervision compatibility checks."""
from __future__ import annotations

import numpy as np

import json
import tempfile
from pathlib import Path

from dataset import (
    CONTENT_LABEL_IDS,
    POPULATION_INDEX,
    ROLE_LABELS,
    SPEECH_ACT_NAMES,
    Dataset,
    group_frame_members,
    merge_fingerprint_excludes,
    population_draw_counts,
)


def test_group_frame_members() -> None:
    frames = np.asarray([2, 0, 2, 1, 0, 2], dtype=np.uint16)
    members = group_frame_members(frames, 4)
    assert [item.tolist() for item in members] == [[1, 4], [3], [0, 2, 5], []]


def test_group_frame_members_rejects_invalid_ids() -> None:
    try:
        group_frame_members(np.asarray([0, 3], dtype=np.uint16), 3)
    except ValueError as error:
        assert "invalid frame" in str(error)
    else:
        raise AssertionError("out-of-range frame id was accepted")


def test_merge_fingerprint_excludes_unions_and_sorts():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        left = root / "a.json"
        right = root / "b.json"
        left.write_text('{"fingerprints": ["b", "a"]}\n')
        right.write_text('{"fingerprints": ["c", "a"]}\n')
        out = merge_fingerprint_excludes([left, right], root / "union.json")
        assert json.loads(out.read_text())["fingerprints"] == ["a", "b", "c"]


def test_speech_act_vocabulary_is_contiguous_and_frozen():
    assert SPEECH_ACT_NAMES == (
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
    assert ROLE_LABELS.index("INTERVAL_2") in CONTENT_LABEL_IDS
    assert ROLE_LABELS.index("WEEKDAY_MON") in CONTENT_LABEL_IDS
    assert ROLE_LABELS.index("HOUR_9") in CONTENT_LABEL_IDS
    assert 0 not in CONTENT_LABEL_IDS  # O
    assert ROLE_LABELS.index("GLUE") not in CONTENT_LABEL_IDS
    assert ROLE_LABELS.index("BOUND_START") not in CONTENT_LABEL_IDS


def _write_tiny_shard(
    directory: Path, populations: tuple[int, int, int] | None = None
) -> Path:
    """Three sequences, mixed lengths, discourse heads, manifest dtypes."""
    prefix = directory / "tiny"
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
    }
    if populations is not None:
        files["populations"] = np.asarray(populations, dtype=np.uint8)
        fmt["populations"] = {"suffix": ".populations.bin", "dtype": "uint8"}
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
    }
    if populations is not None:
        manifest["populationCounts"] = {
            name: populations.count(index)
            for index, name in enumerate(("R", "H", "E"))
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


def test_batch_rejects_empty_selection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        try:
            dataset.batch([])
        except ValueError as error:
            assert "at least one index" in str(error)
        else:
            raise AssertionError("empty batch was accepted")


def test_population_mix_draws_exact_seeded_training_ratio() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(
            _write_tiny_shard(
                Path(directory),
                (
                    POPULATION_INDEX["R"],
                    POPULATION_INDEX["H"],
                    POPULATION_INDEX["E"],
                ),
            )
        )
        kwargs = {
            "batch_size": 128,
            "shuffle_seed": 17,
            "population_mix": (0.68, 0.22, 0.10),
            "max_sequences": 100,
        }
        first = [index for batch in dataset.batches(**kwargs) for index in batch]
        second = [index for batch in dataset.batches(**kwargs) for index in batch]
        assert first == second
        assert len(first) == 3  # cap cannot exceed the number of unique corpus rows
        # With a larger complete corpus, max_sequences is the epoch draw size.
        dataset.sequences = 100
        dataset.lengths = np.resize(dataset.lengths, 100)
        dataset.populations = np.resize(np.asarray(dataset.populations), 100)
        selected = [index for batch in dataset.batches(**kwargs) for index in batch]
        counts = np.bincount(np.asarray(dataset.populations)[selected], minlength=3)
        assert counts.tolist() == [68, 22, 10]


def test_population_mix_requires_explicit_population_shard() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        try:
            list(dataset.batches(8, population_mix=(0.68, 0.22, 0.10)))
        except ValueError as error:
            assert "requires a populations shard" in str(error)
        else:
            raise AssertionError("population mixing inferred missing population labels")


def test_cover_all_population_mix_keeps_every_record() -> None:
    counts = np.asarray([14_997_846, 3_766_558, 1_225_170], dtype=np.int64)
    allocated = population_draw_counts(
        counts,
        np.asarray([0.68, 0.22, 0.10]),
        None,
        cover_all=True,
    )
    assert allocated.tolist() == [14_997_846, 4_852_244, 2_205_566]
    assert int(allocated.sum()) == 22_055_656
    assert np.all(allocated >= counts)


if __name__ == "__main__":
    test_group_frame_members()
    test_group_frame_members_rejects_invalid_ids()
    test_speech_act_vocabulary_is_contiguous_and_frozen()
    test_merge_fingerprint_excludes_unions_and_sorts()
    test_batch_preserves_requested_order_duplicates_and_padding()
    test_batch_rejects_empty_selection()
    test_population_mix_draws_exact_seeded_training_ratio()
    test_population_mix_requires_explicit_population_shard()
    print("test_dataset: all checks passed")
