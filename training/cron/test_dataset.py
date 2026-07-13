"""Direct checks for scalable frame grouping used by balanced training."""
from __future__ import annotations

import numpy as np

import json
import tempfile
from pathlib import Path

from dataset import (
    CONTENT_LABEL_IDS,
    SPEECH_ACT_NAMES,
    group_frame_members,
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
    assert 2 in CONTENT_LABEL_IDS  # INTERVAL
    assert 0 not in CONTENT_LABEL_IDS  # O


if __name__ == "__main__":
    test_group_frame_members()
    test_group_frame_members_rejects_invalid_ids()
    test_speech_act_vocabulary_is_contiguous_and_frozen()
    test_merge_fingerprint_excludes_unions_and_sorts()
    print("test_dataset: all checks passed")
