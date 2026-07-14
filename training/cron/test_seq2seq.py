"""Linearize contract for the Track B 100k prototype."""
from seq2seq import linearize


def test_linearize_reject_and_daily() -> None:
    assert linearize(None) == ["(", "reject", ")"]
    assert linearize({"frequency": {"unit": "day", "interval": 1}}) == [
        "(",
        "every",
        "1",
        "day",
        ")",
    ]


if __name__ == "__main__":
    test_linearize_reject_and_daily()
    print("test_seq2seq: all checks passed")
