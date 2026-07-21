"""Capability and safety accounting at the real trainer's evaluation boundary."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from dataset import Dataset, ROLE_LABELS
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger
from test_dataset import _write_tiny_shard
from train import checkpoint_preserves_capabilities, evaluate


def test_evaluation_counts_all_heads_and_ignores_padding_and_whitespace() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        model = Tagger(
            feature_rows=16, rows_per_token=2, role_classes=len(ROLE_LABELS),
            hidden=64, quantization_bits=6,
            schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
            speech_act_classes=len(SPEECH_ACT_NAMES), actionable_head=True,
        )
        # Make genuine head outputs predictable while retaining the real trunk,
        # batching, quantization, masks, class mapping and metric accumulation.
        with torch.no_grad():
            for head in (
                model.role_head, model.boundary_head, model.schedule_kind_head,
                model.speech_act_head, model.actionable_head,
            ):
                head.weight.zero_()
                head.bias.zero_()
            model.role_head.bias[ROLE_LABELS.index("WEEKDAY_MON")] = 1
            model.boundary_head.bias.fill_(1)
            model.schedule_kind_head.bias[SCHEDULE_KIND_NAMES.index("weekly")] = 1
            model.speech_act_head.bias[SPEECH_ACT_NAMES.index("none")] = 1
            model.actionable_head.bias.fill_(1)
        metrics = evaluate(model, dataset, 2, "cpu")
        assert metrics["sequences"] == 3
        assert metrics["tokens"] == 5
        assert metrics["positiveSequences"] == 2
        assert metrics["negativeSequences"] == 1
        assert metrics["tokenAccuracy"] == 1 / 5
        assert metrics["scheduleKindAccuracy"] == 2 / 3
        assert metrics["macroScheduleKindAccuracy"] == 1 / 2
        assert metrics["speechActAccuracy"] == 1 / 3
        assert metrics["actionableF1"] == 1
        assert metrics["boundaryTruePositive"] == 2
        assert metrics["boundaryFalsePositive"] == 3
        assert metrics["boundaryFalseNegative"] == 0
        assert metrics["boundaryPrecision"] == 2 / 5
        assert metrics["boundaryRecall"] == 1
        assert math.isclose(metrics["boundaryF1"], 4 / 7)
        assert metrics["exactLabelAndBoundarySequence"] == 1 / 3
        assert metrics["jointHeadExactSequence"] == 1 / 3
        assert metrics["macroKindExactSequence"] == 1 / 4
        assert metrics["macroJointHeadExactSequence"] == 3 / 8
        assert metrics["positiveJointHeadExactSequence"] == 0
        assert metrics["requestDecisionFalseAccepts"] == 0
        assert metrics["requestDecisionFalseAcceptRate"] == 0
        assert metrics["byKind"]["single-weekday"] == {
            "sequences": 2, "exact": 1, "exactRate": 0.5, "jointExactRate": 0.5,
        }
        assert metrics["byKind"]["time-of-day"] == {
            "sequences": 1, "exact": 0, "exactRate": 0.0, "jointExactRate": 0.0,
        }
        assert metrics["bySpeechAct"]["none"] == {
            "sequences": 1, "jointExactRate": 1.0,
        }
        assert metrics["bySpeechAct"]["request"] == {
            "sequences": 2, "jointExactRate": 0.0,
        }


def _selection_metrics(
    *, false_accepts: int, positive: float, kind_rate: float, kind_sequences: int,
) -> dict:
    family = {"sequences": kind_sequences, "jointExactRate": kind_rate}
    return {
        "requestDecisionFalseAccepts": false_accepts,
        "positiveJointHeadExactSequence": positive,
        "positiveSequences": 10_000,
        "byScheduleKind": {"weekly": dict(family)},
        "bySpeechAct": {"request": dict(family)},
        "byKind": {"single-weekday": dict(family)},
    }


def test_checkpoint_selection_ignores_sampling_noise_but_not_regressions() -> None:
    current = _selection_metrics(
        false_accepts=100, positive=0.80, kind_rate=0.90, kind_sequences=10,
    )
    # One case lost in a ten-sequence family and a few extra false accepts are
    # within validation noise; the old zero-tolerance rule vetoed this epoch.
    noisy = _selection_metrics(
        false_accepts=110, positive=0.799, kind_rate=0.80, kind_sequences=10,
    )
    assert checkpoint_preserves_capabilities(noisy, current)
    more_false_accepts = _selection_metrics(
        false_accepts=140, positive=0.80, kind_rate=0.90, kind_sequences=10,
    )
    assert not checkpoint_preserves_capabilities(more_false_accepts, current)
    weaker_positives = _selection_metrics(
        false_accepts=100, positive=0.78, kind_rate=0.90, kind_sequences=10,
    )
    assert not checkpoint_preserves_capabilities(weaker_positives, current)
    large_family = _selection_metrics(
        false_accepts=100, positive=0.80, kind_rate=0.90, kind_sequences=5_000,
    )
    regressed_family = _selection_metrics(
        false_accepts=100, positive=0.80, kind_rate=0.87, kind_sequences=5_000,
    )
    assert not checkpoint_preserves_capabilities(regressed_family, large_family)
