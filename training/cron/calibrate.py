"""Boundary-threshold calibration on development data only.

The boundary head emits a raw logit. The runtime applies
`boundaryLogit >= boundaryThreshold`, so the threshold lives in
logit space, where `0.0` is the uncalibrated `sigmoid(0.5)` boundary.

A single threshold has to serve two development distributions: the same-frame
validation split and the reordered-frame held-out split. We sweep a fixed grid
and pick the threshold that maximizes the *worst* boundary F1 across both, so a
threshold that looks good on one frame ordering cannot win by starving the
other. The held-out split is development data here; only fresh reserved
acceptance data can gate promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from abstain import conformal_threshold as energy_conformal_threshold

import numpy as np
import torch

#: Fixed, deterministic sweep in logit space. Coarse enough to be stable,
#: fine enough that the chosen value is meaningful.
THRESHOLD_GRID: tuple[float, ...] = tuple(
    round(float(value), 4) for value in np.linspace(-6.0, 6.0, 121)
)


@dataclass(slots=True)
class Calibration:
    """The chosen threshold, the grid searched, and the evidence for the pick."""

    threshold: float
    grid: list[float]
    worst_f1: float
    per_dataset: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "grid": list(self.grid),
            "worstF1": self.worst_f1,
            "perDataset": self.per_dataset,
        }


def boundary_f1(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> float:
    """F1 over supervised tokens with `sigmoid(logit) >= sigmoid(threshold)`."""
    predicted = (logits >= threshold) & mask
    truth = (targets >= 0.5) & mask
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


@dataclass(slots=True)
class BoundaryCounts:
    """Confusion counts for every grid threshold: O(grid), not O(tokens)."""

    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    tokens: int = 0
    positives: int = 0

    def f1(self, index: int) -> float:
        true_positive = int(self.true_positive[index])
        precision = true_positive / max(true_positive + int(self.false_positive[index]), 1)
        recall = true_positive / max(true_positive + int(self.false_negative[index]), 1)
        return 2 * precision * recall / max(precision + recall, 1e-12)


def count_boundaries(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: str,
    grid: Sequence[float],
) -> BoundaryCounts:
    """Run `model` once and accumulate per-threshold boundary counts.

    The mask matches the training harness: a token is supervised only when its
    label is not the cross-entropy ignore index, so whitespace and unsupervised
    tokens never influence the fit. Each batch is reduced to counts and
    dropped, so the validation split's logits are never all held at once.
    """
    model.eval()
    thresholds = np.asarray(grid, dtype=np.float64)
    counts = BoundaryCounts(
        true_positive=np.zeros(len(grid), dtype=np.int64),
        false_positive=np.zeros(len(grid), dtype=np.int64),
        false_negative=np.zeros(len(grid), dtype=np.int64),
    )
    with torch.no_grad():
        for indices in dataset.batches(batch_size):
            batch = dataset.batch(indices, device)
            _, boundary, _, _, _ = model(
                batch["rows"],
                batch["valid"],
                batch["neighbors"],
                batch.get("lexical_ids"),
                batch.get("lexicon_ids"),
            )
            mask = (batch["labels"] != -100).detach().cpu().numpy().reshape(-1)
            logits = boundary.detach().cpu().numpy().reshape(-1)[mask]
            truth = batch["boundaries"].detach().cpu().numpy().reshape(-1)[mask] >= 0.5
            # (grid, supervised tokens in this batch): bounded by batch size.
            # Compare in the logits' dtype, as `boundary_f1` does with a scalar.
            predicted = logits[None, :] >= thresholds.astype(logits.dtype)[:, None]
            counts.true_positive += (predicted & truth).sum(axis=1)
            counts.false_positive += (predicted & ~truth).sum(axis=1)
            counts.false_negative += (~predicted & truth).sum(axis=1)
            counts.tokens += int(mask.sum())
            counts.positives += int(truth.sum())
    return counts


def calibrate(
    model: torch.nn.Module,
    datasets: Sequence,
    batch_size: int = 64,
    device: str = "cpu",
    grid: Iterable[float] = THRESHOLD_GRID,
) -> Calibration:
    """Choose the threshold maximizing the worst boundary F1 across `datasets`.

    `datasets` contain validation/calibration data, never heldout data. A
    threshold is scored on every supplied development split; the split with
    the lowest F1 bounds the worst case. Ties break toward the threshold with
    the smallest magnitude, then the earliest in the grid, so the result is
    deterministic.
    """
    if not datasets:
        raise ValueError("calibrate requires at least one development dataset")

    grid = [float(value) for value in grid]
    collected = [
        count_boundaries(model, dataset, batch_size, device, grid) for dataset in datasets
    ]

    best_threshold = grid[0]
    best_worst = -1.0
    best_index = 0
    scores_by_threshold: list[list[float]] = []
    for index, threshold in enumerate(grid):
        scores = [counts.f1(index) for counts in collected]
        scores_by_threshold.append(scores)
        worst = min(scores)
        if worst > best_worst or (
            worst == best_worst and abs(threshold) < abs(best_threshold)
        ):
            best_threshold = threshold
            best_worst = worst
            best_index = index

    per_dataset = [
        {
            "threshold": best_threshold,
            "f1": scores_by_threshold[best_index][index],
            "tokens": counts.tokens,
            "positives": counts.positives,
        }
        for index, counts in enumerate(collected)
    ]
    return Calibration(
        threshold=best_threshold,
        grid=grid,
        worst_f1=best_worst,
        per_dataset=per_dataset,
    )
