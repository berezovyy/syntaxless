"""Per-run report writer.

A report is evidence about one run, not an input to training. Atomic replacement
keeps intermediate and complete workflow statuses readable after interruption.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


def write_report(path: str | Path, report: dict) -> Path:
    destination = Path(path)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
