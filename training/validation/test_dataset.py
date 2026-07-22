"""Direct checks for scalable frame grouping used by balanced training."""
from __future__ import annotations

import numpy as np

from dataset import group_frame_members


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


if __name__ == "__main__":
    test_group_frame_members()
    test_group_frame_members_rejects_invalid_ids()
    print("test_dataset: all checks passed")
