"""Reports preserve numerical evidence and publish only a complete write."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from report import write_report


def test_report_preserves_valid_evidence_and_replaces_atomically() -> None:
    with tempfile.TemporaryDirectory(prefix="cron-report-") as directory:
        path = Path(directory) / "report.json"
        path.write_text('{"status":"trained"}\n')
        evidence = {
            "status": "exported",
            "config": {"epochs": 2, "amp": False},
            "history": [{"epoch": 0, "trainLoss": 0.0, "headLosses": (1.5, 2.5)}],
            "candidateModule": "weights.gen.ts",
            "optional": None,
        }
        write_report(path, evidence)
        assert path.read_text() == json.dumps(evidence, indent=2, allow_nan=False) + "\n"
        assert list(path.parent.iterdir()) == [path]


def test_nonfinite_report_does_not_replace_previous_evidence() -> None:
    with tempfile.TemporaryDirectory(prefix="cron-report-invalid-") as directory:
        path = Path(directory) / "report.json"
        previous = '{"status":"trained","trainLoss":0.25}\n'
        path.write_text(previous)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            try:
                write_report(path, {
                    "status": "exported",
                    "history": [{"trainLoss": invalid}],
                })
            except ValueError:
                pass
            else:
                raise AssertionError("Non-finite training evidence was published")
            assert path.read_text() == previous
            assert list(path.parent.iterdir()) == [path]


def test_partial_report_write_preserves_previous_evidence_and_cleans_up() -> None:
    with tempfile.TemporaryDirectory(prefix="cron-report-write-failure-") as directory:
        path = Path(directory) / "report.json"
        previous = '{"status":"trained"}\n'
        path.write_text(previous)
        create_temporary = tempfile.NamedTemporaryFile

        def disk_full(*args, **kwargs):
            handle = create_temporary(*args, **kwargs)

            def fail_write(payload):
                handle.file.write(payload[:8])
                handle.file.flush()
                raise OSError("fixture disk full")

            handle.write = fail_write
            return handle

        with patch("report.tempfile.NamedTemporaryFile", side_effect=disk_full):
            try:
                write_report(path, {"status": "exported"})
            except OSError as error:
                assert str(error) == "fixture disk full"
            else:
                raise AssertionError("Partial report write unexpectedly succeeded")
        assert path.read_text() == previous
        assert list(path.parent.iterdir()) == [path]
