"""Track D: masked-token pretrain proxy for the encoder."""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.rnn(self.embed(tokens))
        return self.out(encoded)


def one_epoch_proxy(vocab: int = 64, seq: int = 16, batch: int = 4) -> float:
    model = TinyEncoder(vocab=vocab)
    tokens = torch.randint(0, vocab, (batch, seq))
    masked = tokens.clone()
    masked[:, 0] = 0
    logits = model(masked)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), tokens.reshape(-1))
    loss.backward()
    return float(loss.detach())


def finetune_one_epoch(vocab: int = 64, classes: int = 16, seq: int = 16, batch: int = 8) -> float:
    """D.2: attach a linear tagger head to the pretrained encoder and take one step."""
    encoder = TinyEncoder(vocab=vocab)
    head = nn.Linear(32, classes)
    tokens = torch.randint(0, vocab, (batch, seq))
    labels = torch.randint(0, classes, (batch, seq))
    encoded, _ = encoder.rnn(encoder.embed(tokens))
    logits = head(encoded)
    loss = nn.functional.cross_entropy(logits.reshape(-1, classes), labels.reshape(-1))
    loss.backward()
    return float(loss.detach())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encoder pretrain proxy (Track D)")
    parser.add_argument("--one-epoch", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args(argv)
    if args.finetune:
        loss = finetune_one_epoch()
        print(f"pretrain finetune loss={loss:.4f}")
        return 0
    if args.one_epoch:
        loss = one_epoch_proxy()
        print(f"pretrain one-epoch-proxy loss={loss:.4f}")
        return 0
    print("usage: pretrain.py --one-epoch | --finetune")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
