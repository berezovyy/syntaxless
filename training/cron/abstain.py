"""Track F: energy (−logsumexp of predicted content-token role logits)
and Mahalanobis of pooled state. Thresholds are fit on real negatives vs
positives (speech_act none vs request), never on gold role labels as the
energy mask.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from dataset import CONTENT_LABEL_IDS, Dataset, repo_root
from model import Tagger


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
    """Split-conformal: reject when score > threshold.

    Threshold is the fa_target quantile of negatives so at most that share of
    negatives are accepted (score <= threshold).
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (energy, pooled, is_negative, is_positive) per sequence.

    Split uses speech_act (none = negative, else positive), not gold roles.
    Energy uses predicted-role content tokens.
    """
    model.eval()
    energies: list[float] = []
    pooled_rows: list[np.ndarray] = []
    negatives: list[bool] = []
    positives: list[bool] = []
    with torch.no_grad():
        for indices in dataset.batches(
            batch_size, shuffle_seed=0, max_sequences=max_sequences
        ):
            batch = dataset.batch(indices, device)
            roles, _boundary, _sk, _sa, _act = model(
                batch["rows"], batch["valid"], batch["neighbors"]
            )
            logits = roles.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            speech = batch["speech_acts"].detach().cpu().numpy()
            pooled = model.last_pooled
            if pooled is None:
                raise RuntimeError("tagger did not record last_pooled")
            pooled_np = pooled.detach().cpu().numpy()
            for row in range(logits.shape[0]):
                energies.append(sequence_energy_predicted(logits[row], labels[row]))
                pooled_rows.append(np.asarray(pooled_np[row], dtype=np.float64))
                negatives.append(int(speech[row]) == 0)
                positives.append(int(speech[row]) != 0)
    return (
        np.asarray(energies, dtype=np.float64),
        np.stack(pooled_rows) if pooled_rows else np.zeros((0, 0)),
        np.asarray(negatives, dtype=bool),
        np.asarray(positives, dtype=bool),
    )


def fit_abstention(
    model: Tagger,
    dataset: Dataset,
    batch_size: int,
    device: str,
    max_sequences: int,
    fa_target: float = 0.01,
) -> dict:
    energies, pooled, is_neg, is_pos = collect_scores(
        model, dataset, batch_size, device, max_sequences
    )
    neg_energy = energies[is_neg] if energies.size else np.asarray([], dtype=np.float64)
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
    neg_mahal = mahals[is_neg] if mahals.size else np.asarray([], dtype=np.float64)
    energy_threshold = conformal_threshold(neg_energy, fa_target)
    mahalanobis_threshold = conformal_threshold(neg_mahal, fa_target)
    return {
        "count": int(energies.size),
        "negatives": int(is_neg.sum()) if is_neg.size else 0,
        "positives": int(is_pos.sum()) if is_pos.size else 0,
        "mean": float(energies.mean()) if energies.size else 0.0,
        "variance": float(energies.var()) if energies.size else 1.0,
        "energyThreshold": energy_threshold,
        "mahalanobisMean": mean.tolist(),
        "mahalanobisVariance": var.tolist(),
        "mahalanobisThreshold": mahalanobis_threshold,
        "predictedRoles": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track F energy / conformal threshold")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-sequences", type=int, default=4096)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    repo = repo_root()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = repo / ckpt
    loaded = torch.load(ckpt, map_location=args.device)
    prefix = Path(args.data_prefix)
    if not prefix.is_absolute():
        prefix = repo / prefix
    heldout = Dataset(Path(f"{prefix}-heldout") if Path(f"{prefix}-heldout.json").exists() else prefix)
    model = Tagger(
        feature_rows=heldout.feature_rows,
        rows_per_token=heldout.rows_per_token,
        role_classes=heldout.role_classes,
        schedule_kind_classes=0,
        speech_act_classes=0,
        actionable_head="actionable" in heldout.manifest.format,
        hidden=int(loaded["model"]["embedding.weight"].shape[1]),
    ).to(args.device)
    model.load_state_dict(loaded["model"], strict=False)
    report = fit_abstention(
        model, heldout, args.batch, args.device, args.max_sequences
    )
    body = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(f"{body}\n")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
