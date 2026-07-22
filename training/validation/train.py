"""Bounded QAT training with selection through the production TypeScript decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset import Dataset, repo_root
from model import Tagger


@torch.no_grad()
def intent_metrics(model: Tagger, dataset: Dataset, corpus: Path, run: Path) -> dict:
    model.eval()
    model.qat = True
    predictions = run / "validation-predictions.jsonl"
    report = run / "validation-metrics.json"
    with predictions.open("w") as output:
        for indices in dataset.batches(64):
            batch = dataset.batch(indices, "cpu")
            logits, _, _ = model(batch["rows"], batch["valid"], batch["neighbors"])
            probabilities = torch.softmax(logits, dim=-1)
            top = probabilities.topk(2, dim=-1)
            roles = top.indices[..., 0].numpy()
            scores = (top.values[..., 0] - top.values[..., 1]).numpy()
            for i, index in enumerate(indices):
                length = int(dataset.lengths[index])
                output.write(json.dumps({"index": int(index), "labels": roles[i, :length].tolist(),
                                         "scores": scores[i, :length].tolist()}) + "\n")
    try:
        subprocess.run(["bun", "run", "packages/validation/training/src/evaluate.ts",
                        str(corpus), "validation", "-", str(report), str(predictions)],
                       cwd=repo_root(), check=True, capture_output=True, text=True)
        return json.loads(report.read_text())
    finally:
        predictions.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden not in (32, 64):
        raise ValueError("positive epoch count and hidden 32 or 64 required")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = repo_root()
    base = (root / args.data_prefix).resolve()
    corpus = (root / args.corpus).resolve()
    run = (root / "training/validation/runs" / args.run).resolve()
    if not run.is_relative_to(root / "training/validation/runs") or (run / "report.json").exists():
        raise ValueError("run must be new and inside training/validation/runs")
    run.mkdir(parents=True, exist_ok=True)
    train = Dataset(Path(f"{base}-train"))
    validation = Dataset(Path(f"{base}-validation"))
    heldout = Dataset(Path(f"{base}-heldout"))
    for left, right in ((train, validation), (train, heldout), (validation, heldout)):
        if set(left.fingerprints) & set(right.fingerprints):
            raise ValueError("structural split leak")
    model = Tagger(train.feature_rows, train.rows_per_token, train.role_classes,
                   schedule_kind_classes=0, hidden=args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / 64)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(1.0, (step + 1) / 50) * (0.5 + 0.5 * math.cos(math.pi * min(step / steps, 1))))
    history = []
    best = float("-inf")
    started = time.monotonic()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    for epoch in range(args.epochs):
        model.train()
        model.qat = epoch >= 2
        loss_total = 0.0
        token_total = 0
        for indices in train.batches(64, shuffle_seed=args.seed + epoch):
            batch = train.batch(indices, "cpu")
            roles, boundary, _ = model(batch["rows"], batch["valid"], batch["neighbors"])
            mask = batch["labels"] != -100
            loss = F.cross_entropy(roles.flatten(0, 1), batch["labels"].flatten(), ignore_index=-100)
            boundary_loss = F.binary_cross_entropy_with_logits(boundary, batch["boundaries"], reduction="none")
            loss += (boundary_loss * mask).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            token_total += int(mask.sum())
            loss_total += float(loss.detach()) * int(mask.sum())
        metrics = intent_metrics(model, validation, corpus, run)
        criterion = metrics["positiveExact"] - 5 * metrics["wrongAll"] - 5 * metrics["negativeFalseAcceptance"]
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "criterion": criterion,
                      "featureRows": train.feature_rows, "rowsPerToken": train.rows_per_token,
                      "roleClasses": train.role_classes, "quantizationBits": 6, "dataPrefix": str(base),
                      "config": vars(args), "sourceHashes": source_hashes}
        torch.save(checkpoint, run / "last.pt")
        if criterion > best:
            best = criterion
            torch.save(checkpoint, run / "best.pt")
            (run / "selected-validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        history.append({"epoch": epoch, "loss": loss_total / max(token_total, 1),
                        "positiveExact": metrics["positiveExact"], "wrongAll": metrics["wrongAll"],
                        "negativeFalseAcceptance": metrics["negativeFalseAcceptance"], "criterion": criterion})
        (run / "report.json").write_text(json.dumps({"config": vars(args), "sourceHashes": source_hashes,
            "corpusSha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "dataDigests": {"train": train.digests(), "validation": validation.digests(), "heldout": heldout.digests()},
            "history": history, "bestCriterion": best, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "elapsedSeconds": time.monotonic() - started}, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
