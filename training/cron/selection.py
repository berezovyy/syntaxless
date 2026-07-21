"""Select calibrated checkpoints using the actual TypeScript intent resolver."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from dataset import Dataset, repo_root
from export import artifact_sha256, calibrated_checkpoint, render_module, write_text_atomic

REGRESSION_Z = 2.0


def significant_rate_drop(before: float, after: float, sequences: int) -> bool:
    if sequences <= 0 or after >= before:
        return False
    deviation = math.sqrt((before * (1 - before) + after * (1 - after)) / sequences)
    return before - after > max(REGRESSION_Z * deviation, 1 / sequences)


def significant_count_increase(before: int, after: int) -> bool:
    return after - before > max(REGRESSION_Z * math.sqrt(before + after), 1)


COUNT_FIELDS = (
    "total", "correct", "positiveTotal", "positiveExact", "positiveAccepted",
    "positiveWrongAccepted", "negativeTotal", "negativeFalseAccepted",
)


def validate_counts(metrics: dict) -> None:
    for key in COUNT_FIELDS:
        value = metrics[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Invalid decoded validation counts")
    if (
        metrics["positiveTotal"] + metrics["negativeTotal"] != metrics["total"]
        or metrics["positiveExact"] + metrics["positiveWrongAccepted"] != metrics["positiveAccepted"]
        or metrics["positiveAccepted"] > metrics["positiveTotal"]
        or metrics["negativeFalseAccepted"] > metrics["negativeTotal"]
        or metrics["correct"] != metrics["positiveExact"] + metrics["negativeTotal"] - metrics["negativeFalseAccepted"]
    ):
        raise ValueError("Inconsistent decoded validation counts")


def decoded_rank(metrics: dict) -> tuple[float, ...]:
    """Prioritize positive intent accuracy; refusal cannot earn a positive hit."""
    validate_counts(metrics)
    raw = metrics["rawPositiveExact"]
    if not isinstance(raw, int) or isinstance(raw, bool) or not 0 <= raw <= metrics["positiveTotal"]:
        raise ValueError("Invalid raw intent counts")
    for category in ("byFamily", "byKind", "bySpeechAct"):
        for count in metrics[category].values():
            validate_counts(count)
        if any(sum(count[key] for count in metrics[category].values()) != metrics[key] for key in COUNT_FIELDS):
            raise ValueError("Decoded group counts disagree with complete split counts")
    families = [count for count in metrics["byFamily"].values() if count["positiveTotal"]]
    macro = sum(count["positiveExact"] / count["positiveTotal"] for count in families) / max(len(families), 1)
    if not math.isclose(metrics["macroPositiveExact"], macro, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Decoded macro rate disagrees with family counts")
    values = (
        metrics["macroPositiveExact"], metrics["positiveExactRate"],
        -metrics["wrongAcceptedRateAllPositive"], -metrics["negativeFalseAcceptRate"],
        metrics["rawPositiveExactRate"],
    )
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in values):
        raise ValueError("Invalid decoded validation rates")
    if metrics["positiveTotal"] <= 0 or metrics["negativeTotal"] <= 0:
        raise ValueError("Decoded selection requires positive and negative support")
    for name, numerator, denominator in (
        ("positiveExactRate", "positiveExact", "positiveTotal"),
        ("wrongAcceptedRateAllPositive", "positiveWrongAccepted", "positiveTotal"),
        ("negativeFalseAcceptRate", "negativeFalseAccepted", "negativeTotal"),
        ("rawPositiveExactRate", "rawPositiveExact", "positiveTotal"),
    ):
        if not math.isclose(metrics[name], metrics[numerator] / metrics[denominator], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Decoded validation rate disagrees with counts")
    accepted = metrics["positiveAccepted"]
    wrong_rate = metrics["wrongAcceptedRateAcceptedPositive"]
    if (not accepted and wrong_rate is not None) or (
        accepted and (wrong_rate is None or not math.isclose(wrong_rate, metrics["positiveWrongAccepted"] / accepted, abs_tol=1e-12, rel_tol=0))
    ):
        raise ValueError("Decoded accepted-positive rate disagrees with counts")
    return values


def decoded_preserves(candidate: dict, current: dict) -> bool:
    """Reject significant safety or family losses despite a higher mean score."""
    decoded_rank(candidate)
    decoded_rank(current)
    for key in ("selectedRecordsSha256", "corpusSha256", "sourceHashes", "positiveTotal", "negativeTotal"):
        if candidate[key] != current[key]:
            raise ValueError("Decoded comparison requires identical data, code and support")
    for key in ("positiveWrongAccepted", "negativeFalseAccepted"):
        if significant_count_increase(current[key], candidate[key]):
            return False
    if significant_rate_drop(current["positiveExactRate"], candidate["positiveExactRate"], current["positiveTotal"]):
        return False
    for category in ("byFamily", "byKind", "bySpeechAct"):
        if candidate[category].keys() != current[category].keys():
            raise ValueError("Decoded comparison requires identical family coverage")
        for name, previous in current[category].items():
            following = candidate[category][name]
            if following["positiveTotal"] != previous["positiveTotal"] or following["negativeTotal"] != previous["negativeTotal"]:
                raise ValueError("Decoded comparison requires identical family support")
            if significant_rate_drop(
                previous["positiveExact"] / max(previous["positiveTotal"], 1),
                following["positiveExact"] / max(following["positiveTotal"], 1),
                previous["positiveTotal"],
            ):
                return False
            for key in ("positiveWrongAccepted", "negativeFalseAccepted"):
                if significant_count_increase(previous[key], following[key]):
                    return False
    return True


def selects_checkpoint(candidate: dict, current: dict | None) -> bool:
    """Prefer passing checkpoints, without letting an untrained refusal stop learning.

    Before a checkpoint passes the synthetic gate, rank improvements are development
    candidates only. After a passing checkpoint exists, enforce capability preservation.
    Fresh acceptance and the remaining promotion checks are separate requirements.
    """
    rank = decoded_rank(candidate)
    if current is None:
        return True
    preserves = decoded_preserves(candidate, current)
    if current["syntheticGatePassed"]:
        return candidate["syntheticGatePassed"] and preserves and rank > decoded_rank(current)
    return candidate["syntheticGatePassed"] or rank > decoded_rank(current)


def verify_preparation(prefix: Path) -> Path:
    """Do not allow a diagnostic corpus to authorize parameter training."""
    corpus = Path(f"{prefix}.jsonl")
    proof = json.loads((prefix.parent / "preparation.json").read_text())
    with corpus.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    ledger = repo_root() / "packages/cron/evaluation/consumed.json"
    ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    rows = proof.get("rows")
    if (
        proof.get("trainingEligible") is not True or proof.get("sha256") != digest
        or not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 10000
        or proof.get("consumption", {}).get("ledgerSha256") != ledger_hash
    ):
        raise ValueError("Training requires verified compact preparation, not diagnostic data")
    subprocess.run(
        [os.environ.get("SYNTAXLESS_BUN_BIN", "bun"), "--eval",
         'import { loadCompactProtection } from "./packages/cron/training/src/compact-protection.ts"; loadCompactProtection();'],
        cwd=repo_root(), check=True,
    )
    return corpus


def score_module(corpus: Path, module: Path, split: str, output: Path) -> dict:
    subprocess.run(
        [os.environ.get("SYNTAXLESS_BUN_BIN", "bun"), "run",
         "packages/cron/training/src/checkpoint.ts", str(corpus), str(module), split, str(output)],
        cwd=repo_root(), check=True,
    )
    result = json.loads(output.read_text())
    if result.get("split") != split or result.get("promotionEvidence") is not False:
        raise ValueError("Invalid checkpoint scoring scope")
    decoded_rank(result)
    return result


def score_checkpoint(checkpoint: Path, validation: Dataset, corpus: Path, *, device: str, batch_size: int) -> dict:
    artifact, _, _, _ = calibrated_checkpoint(checkpoint, validation, device=device, batch_size=batch_size)
    module = checkpoint.parent / "last-selection.gen.ts"
    write_text_atomic(module, render_module(artifact))
    metrics = score_module(corpus, module, "validation", checkpoint.parent / "last-selection.json")
    if metrics["artifactSha256"] != artifact_sha256(artifact):
        raise ValueError("TypeScript scored a different calibrated checkpoint")
    if metrics["total"] != len(validation):
        raise ValueError("Decoded selection didn't score the complete validation split")
    return metrics
