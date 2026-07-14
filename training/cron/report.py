"""Per-run report writer.

A report is evidence about one run, not an input to training. It is always the
last artifact written so a crash leaves no half-truth behind.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def write_report(path: str | Path, report: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    return destination
