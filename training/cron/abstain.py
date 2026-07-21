"""Validation-only energy and pooled-state abstention calibration.

The training exporter calls this module on validation examples. Energy uses
predicted content-token roles, never gold labels as its content mask.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from dataset import CONTENT_LABEL_IDS, SPEECH_ACT_NAMES, Dataset
from model import Tagger

POSITIVE_SPEECH_ACT_IDS = frozenset(
    SPEECH_ACT_NAMES.index(name) for name in ("request", "correction")
)


def logsumexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + math.log(float(np.exp(values - peak).sum()))


def token_energy(logits: np.ndarray) -> float:
    return -logsumexp(logits)


def sequence_energy_predicted(role_logits: np.ndarray, pad_labels: np.ndarray) -> float:
    """Energy from *predicted* roles; pad_labels == -100 marks unsupervised tokens."""
    predicted = role_logits.argmax(axis=-1)
    energies: list[float] = []
    content = set(CONTENT_LABEL_IDS)
    for index, label in enumerate(predicted):
        if int(pad_labels[index]) < 0:
            continue
        if int(label) not in content:
            continue
        energies.append(token_energy(role_logits[index]))
    if not energies:
        return 0.0
    return float(sum(energies) / len(energies))


def conformal_threshold(
    negative_scores: np.ndarray, fa_target: float = 0.01
) -> float | None:
    """Use an empirical negative-score cutoff; reject when score > threshold.

    Ties and finite sample size can exceed the requested negative acceptance
    share. The quality gate checks actual acceptance by the complete parser.
    """
    if negative_scores.size == 0:
        return None
    ordered = np.sort(negative_scores)
    index = min(int(math.floor(fa_target * ordered.size)), ordered.size - 1)
    return float(ordered[max(index, 0)])


def mahalanobis_diag(vector: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    safe = np.maximum(variance, 1e-8)
    delta = vector - mean
    return float(np.sqrt(np.sum((delta * delta) / safe)))


def collect_scores(
    model: Tagger,
    dataset: Dataset,
    batch_size: int,
    device: str,
    max_sequences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy, pooled state, gold class masks and decision acceptance.

    Requests and corrections are positive; every other speech act is negative.
    Energy uses predicted-role content tokens.
    """
    model.eval()
    energies: list[float] = []
    pooled_rows: list[np.ndarray] = []
    negatives: list[bool] = []
    positives: list[bool] = []
    candidates: list[bool] = []
    with torch.no_grad():
        for indices in dataset.batches(
            batch_size, shuffle_seed=0, max_sequences=max_sequences
        ):
            batch = dataset.batch(indices, device)
            roles, _boundary, _sk, speech_logits, _act = model(
                batch["rows"],
                batch["valid"],
                batch["neighbors"],
                batch.get("lexical_ids"),
                batch.get("lexicon_ids"),
            )
            logits = roles.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            speech = batch["speech_acts"].detach().cpu().numpy()
            if speech_logits.shape[-1] == 0:
                decision_accepts = np.ones(len(indices), dtype=bool)
            else:
                predicted = speech_logits.argmax(dim=-1).detach().cpu().numpy()
                # Three-class artifacts use request, correction, other.
                positive_ids = (0, 1) if speech_logits.shape[-1] == 3 else tuple(POSITIVE_SPEECH_ACT_IDS)
                decision_accepts = np.isin(predicted, positive_ids)
            pooled = model.last_pooled
            if pooled is None:
                raise RuntimeError("tagger did not record last_pooled")
            pooled_np = pooled.detach().cpu().numpy()
            for row in range(logits.shape[0]):
                energies.append(sequence_energy_predicted(logits[row], labels[row]))
                pooled_rows.append(np.asarray(pooled_np[row], dtype=np.float64))
                positive = int(speech[row]) in POSITIVE_SPEECH_ACT_IDS
                negatives.append(not positive)
                positives.append(positive)
                candidates.append(bool(decision_accepts[row]))
    return (
        np.asarray(energies, dtype=np.float64),
        np.stack(pooled_rows) if pooled_rows else np.zeros((0, 0)),
        np.asarray(negatives, dtype=bool),
        np.asarray(positives, dtype=bool),
        np.asarray(candidates, dtype=bool),
    )


def fit_abstention(
    model: Tagger,
    dataset: Dataset,
    batch_size: int,
    device: str,
    max_sequences: int,
    fa_target: float = 0.01,
) -> dict:
    energies, pooled, is_neg, is_pos, candidates = collect_scores(
        model, dataset, batch_size, device, max_sequences
    )
    # The speech head already refuses non-requests. Fit the additional score
    # checks on negative examples it could accept, rather than demanding that
    # schedule-token energy distinguish identical positive/negative schedules.
    residual_negatives = is_neg & candidates
    neg_energy = energies[residual_negatives] if energies.size else np.asarray([], dtype=np.float64)
    pos_pooled = pooled[is_pos] if pooled.size and is_pos.any() else pooled
    if pos_pooled.size == 0:
        mean = np.zeros((model.hidden,), dtype=np.float64)
        var = np.ones((model.hidden,), dtype=np.float64)
    else:
        mean = pos_pooled.mean(axis=0)
        var = pos_pooled.var(axis=0)
        var = np.maximum(var, 1e-8)
    mahals = (
        np.asarray(
            [mahalanobis_diag(row, mean, var) for row in pooled],
            dtype=np.float64,
        )
        if pooled.size
        else np.asarray([], dtype=np.float64)
    )
    neg_mahal = mahals[residual_negatives] if mahals.size else np.asarray([], dtype=np.float64)
    energy_threshold = conformal_threshold(neg_energy, fa_target)
    mahalanobis_threshold = conformal_threshold(neg_mahal, fa_target)
    return {
        "count": int(energies.size),
        "negatives": int(is_neg.sum()) if is_neg.size else 0,
        "positives": int(is_pos.sum()) if is_pos.size else 0,
        "decisionNegativeCandidates": int(residual_negatives.sum()),
        "mean": float(energies.mean()) if energies.size else 0.0,
        "variance": float(energies.var()) if energies.size else 1.0,
        "energyThreshold": energy_threshold,
        "mahalanobisMean": mean.tolist(),
        "mahalanobisVariance": var.tolist(),
        "mahalanobisThreshold": mahalanobis_threshold,
        "predictedRoles": True,
    }
