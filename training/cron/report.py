"""Per-run report writer.

A report is evidence about one run, not an input to training. It is always the
last artifact written so a crash leaves no half-truth behind.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_report(path: str | Path, report: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination
