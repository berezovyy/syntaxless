"""Boundary-threshold calibration on development data only.

The boundary head emits a raw logit. The runtime applies
`boundaryLogit >= boundaryThreshold` (Phase 16), so the threshold lives in
logit space, where `0.0` is the uncalibrated `sigmoid(0.5)` boundary.

A single threshold has to serve two development distributions: the same-frame
validation split and the reordered-frame held-out split. We sweep a fixed grid
and pick the threshold that maximizes the *worst* boundary F1 across both, so a
threshold that looks good on one frame ordering cannot win by starving the
other. The held-out split is development data here; the reserved-carrier test
set gates promotion only in Phase 19.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

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


def collect_logits(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `model` once and return flattened boundary logits, targets, and mask.

    The mask matches the training harness: a token is supervised only when its
    label is not the cross-entropy ignore index, so whitespace and unsupervised
    tokens never influence the fit.
    """
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for indices in dataset.batches(batch_size):
            batch = dataset.batch(indices, device)
            _, boundary, _, _, _ = model(
                batch["rows"], batch["valid"], batch["neighbors"]
            )
            logits.append(boundary.detach().cpu().numpy().reshape(-1))
            targets.append(batch["boundaries"].detach().cpu().numpy().reshape(-1))
            masks.append((batch["labels"] != -100).detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(logits),
        np.concatenate(targets),
        np.concatenate(masks),
    )


def calibrate(
    model: torch.nn.Module,
    datasets: Sequence,
    batch_size: int = 64,
    device: str = "cpu",
    grid: Iterable[float] = THRESHOLD_GRID,
) -> Calibration:
    """Choose the threshold maximizing the worst boundary F1 across `datasets`.

    `datasets` are development splits in a fixed order (validation first, then
    reordered held-out). A threshold is scored on every split; the split with
    the lowest F1 bounds the worst case. Ties break toward the threshold with
    the smallest magnitude, then the earliest in the grid, so the result is
    deterministic.
    """
    if not datasets:
        raise ValueError("calibrate requires at least one development dataset")

    collected = [
        collect_logits(model, dataset, batch_size, device) for dataset in datasets
    ]

    grid = [float(value) for value in grid]
    best_threshold = grid[0]
    best_worst = -1.0
    best_index = 0
    scores_by_threshold: list[list[float]] = []
    for threshold in grid:
        scores = [
            boundary_f1(logits, targets, mask, threshold)
            for logits, targets, mask in collected
        ]
        scores_by_threshold.append(scores)
        worst = min(scores)
        if worst > best_worst or (
            worst == best_worst and abs(threshold) < abs(best_threshold)
        ):
            best_threshold = threshold
            best_worst = worst
            best_index = len(scores_by_threshold) - 1

    per_dataset = [
        {
            "threshold": best_threshold,
            "f1": scores_by_threshold[best_index][index],
            "tokens": int(collected[index][2].sum()),
            "positives": int(
                ((collected[index][1] >= 0.5) & collected[index][2]).sum()
            ),
        }
        for index in range(len(datasets))
    ]
    return Calibration(
        threshold=best_threshold,
        grid=grid,
        worst_f1=best_worst,
        per_dataset=per_dataset,
    )
