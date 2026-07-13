"""Direct checks for scalable frame grouping used by balanced training."""
from __future__ import annotations

import numpy as np

import inspect
import json
import tempfile
from pathlib import Path

from dataset import (
    CONTENT_LABEL_IDS,
    ROLE_LABELS,
    SPEECH_ACT_NAMES,
    Dataset,
    group_frame_members,
    length_bucket,
    merge_fingerprint_excludes,
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


def _write_tiny_shard(directory: Path) -> Path:
    """Three sequences, mixed lengths, discourse heads, manifest dtypes."""
    prefix = directory / "tiny"
    rows_per_token = 2
    feature_rows = 8
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
    for key, array in files.items():
        array.tofile(f"{prefix}{fmt[key]['suffix']}")
    manifest = {
        "fingerprints": ["a", "b", "c"],
        "roleClasses": len(ROLE_LABELS),
        "rowsPerToken": rows_per_token,
        "featureRows": feature_rows,
        "labelCounts": {"WEEKDAY_MON": 1, "HOUR_9": 1},
        "format": fmt,
        "frameNames": ["weekly/on", "daily/at"],
        "frameCounts": [2, 1],
        "speechActClasses": 9,
        "speechActNames": list(SPEECH_ACT_NAMES),
        "actionableCounts": {"0": 1, "1": 5},
    }
    Path(f"{prefix}.json").write_text(json.dumps(manifest))
    return prefix


def test_batch_gathers_bucket_tensors() -> None:
    source = inspect.getsource(Dataset.batch)
    assert "for row, index in enumerate(indices)" not in source
    with tempfile.TemporaryDirectory() as directory:
        prefix = _write_tiny_shard(Path(directory))
        dataset = Dataset(prefix)
        indices = [0, 2, 1]
        first = dataset.batch(indices)
        second = dataset.batch(indices)
        width = length_bucket(3)
        assert first["rows"].shape == (3, width, 2)
        assert first["labels"].shape == (3, width)
        assert first["neighbors"].shape == (3, width, 2)
        assert first["valid"].shape == (3, width)
        assert first["speech_acts"].shape == (3,)
        # seq0 length 2: RECUR, WEEKDAY_MON then ignore
        assert int(first["labels"][0, 0]) == ROLE_LABELS.index("RECUR")
        assert int(first["labels"][0, 1]) == ROLE_LABELS.index("WEEKDAY_MON")
        assert int(first["labels"][0, 2]) == -100
        assert bool(first["valid"][0, 0]) is True
        assert bool(first["valid"][0, 2]) is False
        # padding feature sentinel
        assert int(first["rows"][0, 2, 0]) == dataset.feature_rows
        # seq2 length 1 is whitespace-kind 3 → ignore index
        assert int(first["labels"][1, 0]) == -100
        # seq1 length 3 keeps HOUR_9
        assert int(first["labels"][2, 0]) == ROLE_LABELS.index("HOUR_9")
        assert int(first["speech_acts"][0]) == 1
        assert int(first["speech_acts"][1]) == 0
        for key in first:
            assert (first[key] == second[key]).all(), key
        # manifest dtypes still drive the memmap
        assert dataset.manifest.format["rows"]["dtype"] == "uint16"
        assert dataset.manifest.format["neighbors"]["dtype"] == "int16"
        assert dataset.rows.dtype == np.uint16
        assert dataset.neighbors.dtype == np.int16


if __name__ == "__main__":
    test_group_frame_members()
    test_group_frame_members_rejects_invalid_ids()
    test_speech_act_vocabulary_is_contiguous_and_frozen()
    test_merge_fingerprint_excludes_unions_and_sorts()
    test_batch_gathers_bucket_tensors()
    print("test_dataset: all checks passed")
