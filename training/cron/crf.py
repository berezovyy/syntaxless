"""Track E.1: globally normalised role decoding with a learned transition matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


class LinearChainCRF(torch.nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.transitions = torch.nn.Parameter(torch.zeros(num_tags, num_tags))

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """emissions: (batch, time, tags); tags/mask: (batch, time)."""
        gold = self._score(emissions, tags, mask)
        log_z = self._partition(emissions, mask)
        return (log_z - gold).mean()

    def _score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, time, _ = emissions.shape
        score = emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, time):
            prev = tags[:, t - 1]
            curr = tags[:, t]
            trans = self.transitions[prev, curr]
            emit = emissions[:, t].gather(1, curr.unsqueeze(1)).squeeze(1)
            score = score + (trans + emit) * mask[:, t]
        return score

    def _partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        alpha = emissions[:, 0]
        time = emissions.size(1)
        for t in range(1, time):
            # (batch, prev, next)
            step = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(step, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1).bool(), next_alpha, alpha)
        return torch.logsumexp(alpha, dim=1)


def one_batch(num_tags: int = 8, time: int = 4, batch: int = 2) -> float:
    model = LinearChainCRF(num_tags)
    emissions = torch.randn(batch, time, num_tags)
    tags = torch.randint(0, num_tags, (batch, time))
    mask = torch.ones(batch, time)
    loss = model.nll(emissions, tags, mask)
    loss.backward()
    return float(loss.detach())


def train_from_checkpoint(
    checkpoint: Path,
    data_prefix: Path,
    device: str,
    samples: int,
    batch_size: int,
) -> float:
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
    tagger.eval()
    crf = LinearChainCRF(train.role_classes).to(device)
    optimizer = torch.optim.Adam(crf.parameters(), lr=1e-2)
    last = 0.0
    seen = 0
    for indices in train.batches(batch_size, shuffle_seed=0, max_sequences=samples):
        batch = train.batch(indices, device)
        with torch.no_grad():
            roles, _b, _sk, _sa, _act = tagger(
                batch["rows"], batch["valid"], batch["neighbors"]
            )
        labels = batch["labels"].clamp(min=0)
        mask = (batch["labels"] != -100) & batch["valid"]
        loss = crf.nll(roles, labels, mask.to(roles.dtype))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        seen += int(mask.size(0))
        if seen >= samples:
            break
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny CRF trainer (Track E.1)")
    parser.add_argument("--one-batch", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-prefix", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    if args.checkpoint and args.data_prefix:
        loss = train_from_checkpoint(
            Path(args.checkpoint),
            Path(args.data_prefix),
            args.device,
            args.samples,
            args.batch,
        )
        print(f"crf from-checkpoint loss={loss:.4f}")
        return 0
    if args.one_batch:
        loss = one_batch()
        print(f"crf one-batch loss={loss:.4f}")
        return 0
    print("usage: crf.py --one-batch | --checkpoint PATH --data-prefix PREFIX")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
