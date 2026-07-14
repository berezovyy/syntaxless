"""Energy uses predicted roles; Mahalanobis is pooled-state, not a constant 0."""
from __future__ import annotations

import numpy as np

from abstain import (
    conformal_threshold,
    mahalanobis_diag,
    sequence_energy_predicted,
)


def test_energy_uses_predicted_roles_not_gold_labels() -> None:
    # Class 0 is O (not content). Gold marks both tokens content (id 5).
    logits = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        ],
        dtype=np.float64,
    )
    gold = np.array([5, 5], dtype=np.int64)
    energy = sequence_energy_predicted(logits, gold)
    only_second = sequence_energy_predicted(logits[1:], gold[1:])
    assert abs(energy - only_second) < 1e-6


def test_conformal_threshold_is_quantile_of_negatives() -> None:
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    threshold = conformal_threshold(scores, fa_target=0.1)
    assert threshold is not None
    accepted = float((scores <= threshold).mean())
    assert accepted <= 0.2
    assert conformal_threshold(np.asarray([])) is None


def test_mahalanobis_is_not_hardcoded_zero() -> None:
    mean = np.array([0.0, 0.0])
    var = np.array([1.0, 1.0])
    near = mahalanobis_diag(np.array([0.1, -0.1]), mean, var)
    far = mahalanobis_diag(np.array([5.0, -5.0]), mean, var)
    assert far > near
    assert far > 0.0


if __name__ == "__main__":
    test_energy_uses_predicted_roles_not_gold_labels()
    test_conformal_threshold_is_quantile_of_negatives()
    test_mahalanobis_is_not_hardcoded_zero()
    print("test_abstain: all checks passed")
