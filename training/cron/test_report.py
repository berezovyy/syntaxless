"""report.json must be strict JSON — NaN/Inf become null."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from report import write_report


def test_write_report_rejects_nan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        write_report(
            path,
            {"a": float("nan"), "b": float("inf"), "c": 1.5, "d": {"e": float("nan")}},
        )
        parsed = json.loads(path.read_text())
        assert parsed == {"a": None, "b": None, "c": 1.5, "d": {"e": None}}


if __name__ == "__main__":
    test_write_report_rejects_nan()
    print("test_report: all checks passed")
