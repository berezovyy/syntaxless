"""Checkpoint selection uses H when the per-epoch scorer returns a rate."""
from train import h_selection_score


def test_h_selection_score_uses_correct_over_total() -> None:
    assert h_selection_score({"ok": True, "correct": 10, "total": 20}) == 0.5
    assert h_selection_score({"ok": False, "correct": 10, "total": 20}) is None
    assert h_selection_score(None) is None
    assert h_selection_score({"ok": True, "correct": 0, "total": 0}) is None


if __name__ == "__main__":
    test_h_selection_score_uses_correct_over_total()
    print("test_h_select: all checks passed")
