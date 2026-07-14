"""Track E.2: minimum-risk training, risk = 1 − EXEC.

On oracle-aligned gold shards, a sampled role sequence that matches gold
labels decodes to the gold intent (oracle 100 %), so EXEC=1. A mismatch is
EXEC=0. K=4 samples per example; REINFORCE with a batch baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.distributions import Categorical


K_SAMPLES = 4


def exec_from_labels(
    sample: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """EXEC=1 iff sampled roles match gold on every supervised token."""
    matched = torch.where(mask, sample == gold, torch.ones_like(mask, dtype=torch.bool))
    return matched.all(dim=-1).to(sample.dtype)


def exec_risk(
    sample: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return 1.0 - exec_from_labels(sample, gold, mask)


def mrt_loss(log_probs: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    """REINFORCE: mean[(risk − baseline) * log_prob] over K samples.

    log_probs, risks: (k, batch)
    """
    baseline = risks.mean(dim=0, keepdim=True)
    return ((risks - baseline).detach() * log_probs).mean()


def sample_k(
    logits: torch.Tensor,
    gold: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw k sequences from per-token softmax; copy gold on unsupervised steps."""
    dist = Categorical(logits=logits)
    samples: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    safe_gold = gold.clamp(min=0)
    for _ in range(k):
        drawn = dist.sample()
        drawn = torch.where(mask, drawn, safe_gold)
        token_lp = dist.log_prob(drawn)
        seq_lp = (token_lp * mask).sum(dim=-1)
        samples.append(drawn)
        log_probs.append(seq_lp)
    return torch.stack(samples, dim=0), torch.stack(log_probs, dim=0)


def train_from_checkpoint(
    checkpoint: Path,
    data_prefix: Path,
    device: str,
    samples: int,
    batch_size: int,
    k: int,
) -> dict[str, float]:
    from dataset import Dataset, repo_root
    from model import Tagger

    repo = repo_root()
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    if not data_prefix.is_absolute():
        data_prefix = repo / data_prefix
    loaded = torch.load(checkpoint, map_location=device)
    train = Dataset(Path(f"{data_prefix}-train"))
    tagger = Tagger(
        feature_rows=train.feature_rows,
        rows_per_token=train.rows_per_token,
        role_classes=train.role_classes,
        schedule_kind_classes=int(loaded.get("scheduleKindClasses", 0)),
        speech_act_classes=int(loaded.get("speechActClasses", 0)),
        actionable_head="actionable" in train.manifest.format,
        hidden=int(loaded["model"]["embedding.weight"].shape[1]),
    ).to(device)
    tagger.load_state_dict(loaded["model"], strict=False)
    optimizer = torch.optim.Adam(tagger.parameters(), lr=1e-4)
    last_loss = 0.0
    last_risk = 0.0
    seen = 0
    for indices in train.batches(batch_size, shuffle_seed=0, max_sequences=samples):
        batch = train.batch(indices, device)
        roles, _boundary, _sk, _sa, _act = tagger(
            batch["rows"], batch["valid"], batch["neighbors"]
        )
        gold = batch["labels"]
        mask = (gold != -100) & batch["valid"]
        sampled, log_probs = sample_k(roles, gold, mask, k)
        risks = torch.stack(
            [exec_risk(sampled[index], gold, mask) for index in range(k)],
            dim=0,
        )
        loss = mrt_loss(log_probs, risks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        last_risk = float(risks.mean().detach())
        seen += int(gold.size(0))
        if seen >= samples:
            break
    return {"loss": last_loss, "meanRisk": last_risk, "samples": float(seen), "k": float(k)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track E.2 MRT trainer (risk = 1 − EXEC, K samples)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--k", type=int, default=K_SAMPLES)
    args = parser.parse_args(argv)
    report = train_from_checkpoint(
        Path(args.checkpoint),
        Path(args.data_prefix),
        args.device,
        args.samples,
        args.batch,
        args.k,
    )
    print(
        f"mrt loss={report['loss']:.4f} meanRisk={report['meanRisk']:.4f} "
        f"samples={int(report['samples'])} k={int(report['k'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
