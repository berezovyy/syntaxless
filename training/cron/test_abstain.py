"""Energy uses predicted roles; Mahalanobis is pooled-state, not a constant 0."""
from __future__ import annotations

import numpy as np
import torch
import tempfile
from pathlib import Path

from abstain import (
    conformal_threshold,
    collect_scores,
    fit_abstention,
    mahalanobis_diag,
    sequence_energy_predicted,
)
from dataset import Dataset, ROLE_LABELS, SPEECH_ACT_NAMES
from model import Tagger
from test_dataset import _write_tiny_shard


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


def test_calibration_treats_only_requests_and_corrections_as_positive() -> None:
    model = Tagger(feature_rows=16, rows_per_token=2, role_classes=len(ROLE_LABELS), hidden=8)
    with tempfile.TemporaryDirectory() as directory:
        for start in range(0, len(SPEECH_ACT_NAMES), 3):
            prefix = _write_tiny_shard(Path(directory), name=f"speech-{start}")
            speech = np.arange(start, start + 3, dtype=np.uint8)
            speech.tofile(f"{prefix}.speech.bin")
            dataset = Dataset(prefix)
            _, _, negatives, positives, _ = collect_scores(model, dataset, 2, "cpu", 3)
            order = np.concatenate(list(dataset.batches(2, shuffle_seed=0, max_sequences=3)))
            expected = np.asarray([
                SPEECH_ACT_NAMES[int(speech[index])] in ("request", "correction")
                for index in order
            ])
            np.testing.assert_array_equal(positives, expected)
            np.testing.assert_array_equal(negatives, ~expected)
            fitted = fit_abstention(model, dataset, 2, "cpu", 3)
            assert fitted["positives"] == int(expected.sum())
            assert fitted["negatives"] == int((~expected).sum())


def test_score_calibration_uses_only_negatives_that_pass_the_speech_head() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        for classes, request, refusal in ((len(SPEECH_ACT_NAMES), 1, 2), (3, 0, 2)):
            model = Tagger(
                feature_rows=16, rows_per_token=2, role_classes=len(ROLE_LABELS),
                hidden=8, speech_act_classes=classes,
            )
            for predicted in (request, refusal):
                with torch.no_grad():
                    model.speech_act_head.weight.zero_()
                    model.speech_act_head.bias.zero_()
                    model.speech_act_head.bias[predicted] = 10
                fitted = fit_abstention(model, dataset, 2, "cpu", 3)
                assert fitted["positives"] == 2 and fitted["negatives"] == 1
                accepted = predicted == request
                assert fitted["decisionNegativeCandidates"] == int(accepted)
                assert (fitted["energyThreshold"] is not None) == accepted
                assert (fitted["mahalanobisThreshold"] is not None) == accepted


if __name__ == "__main__":
    test_energy_uses_predicted_roles_not_gold_labels()
    test_conformal_threshold_is_quantile_of_negatives()
    test_mahalanobis_is_not_hardcoded_zero()
    print("test_abstain: all checks passed")
