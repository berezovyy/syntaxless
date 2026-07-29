"""Structural guards for the flat-memory rule: no whole-split materialization.

These tests cannot measure RSS. They pin the behavior that keeps memory
independent of corpus size: memmap-backed fields, on-demand lengths, block
scans, and per-batch metric accumulation.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from calibrate import THRESHOLD_GRID, boundary_f1, calibrate, count_boundaries
from dataset import ROLE_LABELS, SequenceLengths, Dataset
from model import SCHEDULE_KIND_NAMES, SPEECH_ACT_NAMES, Tagger
from test_dataset import _write_tiny_shard


def test_loader_fields_stay_memory_mapped() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        for name in ("rows", "labels", "boundaries", "kinds", "neighbors", "offsets",
                     "frames", "populations", "speech", "actionable"):
            value = getattr(dataset, name)
            assert isinstance(value, np.memmap), f"{name} was copied into memory"
        assert isinstance(dataset.lengths, SequenceLengths)


def test_block_scans_equal_whole_array_answers() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory), populations=(0, 2, 2)))
        assert dataset.population_counts() == {"R": 1, "H": 0, "E": 2}
        kinds = np.concatenate(list(dataset.schedule_kind_blocks()))
        np.testing.assert_array_equal(kinds, dataset.frame_schedule_kinds[np.asarray(dataset.frames)])


def _tiny_model() -> Tagger:
    torch.manual_seed(3)
    return Tagger(
        feature_rows=16, rows_per_token=2, role_classes=len(ROLE_LABELS), hidden=8,
        schedule_kind_classes=len(SCHEDULE_KIND_NAMES),
        speech_act_classes=len(SPEECH_ACT_NAMES), actionable_head=True,
    ).eval()


def test_streamed_calibration_counts_equal_the_collected_logit_reference() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Dataset(_write_tiny_shard(Path(directory)))
        model = _tiny_model()
        # Reference: whole-split collection, acceptable here because the
        # fixture has three sequences.
        logits, targets, masks = [], [], []
        with torch.no_grad():
            for indices in dataset.batches(2):
                batch = dataset.batch(indices, "cpu")
                _, boundary, _, _, _ = model(batch["rows"], batch["valid"], batch["neighbors"])
                logits.append(boundary.numpy().reshape(-1))
                targets.append(batch["boundaries"].numpy().reshape(-1))
                masks.append((batch["labels"] != -100).numpy().reshape(-1))
        logits, targets, masks = map(np.concatenate, (logits, targets, masks))
        # Batch size 1 proves the counts are accumulated across batches.
        counts = count_boundaries(model, dataset, 1, "cpu", THRESHOLD_GRID)
        for index, threshold in enumerate(THRESHOLD_GRID):
            assert math.isclose(
                counts.f1(index), boundary_f1(logits, targets, masks, threshold)
            ), threshold
        assert counts.tokens == int(masks.sum())
        assert counts.positives == int(((targets >= 0.5) & masks).sum())
        chosen = calibrate(model, [dataset], batch_size=1)
        assert chosen.per_dataset[0]["tokens"] == counts.tokens
