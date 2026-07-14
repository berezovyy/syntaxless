"""Track B prototype: tiny encoder-decoder CE under a grammar mask."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


UNITS = ("minute", "hour", "day", "week", "month")
WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
TARGET_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "(",
    ")",
    "every",
    "at",
    "days",
    "reject",
    *UNITS,
    *WEEKDAYS,
    *[str(n) for n in range(0, 61)],
)
TARGET_INDEX = {token: index for index, token in enumerate(TARGET_TOKENS)}
PAD = TARGET_INDEX["<pad>"]
BOS = TARGET_INDEX["<bos>"]
EOS = TARGET_INDEX["<eos>"]
CHAR_VOCAB = 128
SRC_LEN = 32
TGT_LEN = 24


class TinySeq2Seq(nn.Module):
    def __init__(self, src_vocab: int = CHAR_VOCAB, tgt_vocab: int = len(TARGET_TOKENS), hidden: int = 64):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, hidden)
        self.tgt_embed = nn.Embedding(tgt_vocab, hidden)
        self.enc = nn.GRU(hidden, hidden, batch_first=True)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        encoded, hidden = self.enc(self.src_embed(src))
        decoded, _ = self.dec(self.tgt_embed(tgt), hidden)
        return self.out(decoded)


def linearize(intent: dict | None) -> list[str]:
    if intent is None:
        return ["(", "reject", ")"]
    tokens = ["(", "every", str(int(intent["frequency"]["interval"])), intent["frequency"]["unit"]]
    time_of_day = intent.get("time")
    if isinstance(time_of_day, dict):
        tokens += ["at", str(int(time_of_day.get("hour", 0))), str(int(time_of_day.get("minute", 0)))]
    weekdays = intent.get("weekdays")
    if isinstance(weekdays, list) and weekdays:
        tokens.append("days")
        tokens.extend(str(day) for day in weekdays)
    tokens.append(")")
    return tokens


def encode_text(text: str) -> list[int]:
    ids = [min(ord(char), CHAR_VOCAB - 1) for char in text[:SRC_LEN]]
    return ids + [0] * (SRC_LEN - len(ids))


def encode_target(tokens: list[str]) -> list[int]:
    ids = [BOS] + [TARGET_INDEX.get(token, PAD) for token in tokens] + [EOS]
    ids = ids[:TGT_LEN]
    return ids + [PAD] * (TGT_LEN - len(ids))


def load_rows(path: Path, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    src_rows: list[list[int]] = []
    tgt_rows: list[list[int]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            src_rows.append(encode_text(str(record.get("text", ""))))
            tgt_rows.append(encode_target(linearize(record.get("intent"))))
            if len(src_rows) >= limit:
                break
    return torch.tensor(src_rows, dtype=torch.long), torch.tensor(tgt_rows, dtype=torch.long)


def exact_rate(model: TinySeq2Seq, src: torch.Tensor, tgt: torch.Tensor, batch: int = 256) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, src.size(0), batch):
            chunk_src = src[start : start + batch]
            chunk_tgt = tgt[start : start + batch]
            decoder_in = chunk_tgt[:, :-1]
            logits = model(chunk_src, decoder_in)
            pred = logits.argmax(dim=-1)
            gold = chunk_tgt[:, 1:]
            pad = gold != PAD
            match = ((pred == gold) | ~pad).all(dim=1)
            correct += int(match.sum())
            total += match.numel()
    return correct / max(1, total)


def one_batch(vocab: int = 32) -> float:
    model = TinySeq2Seq(src_vocab=vocab, tgt_vocab=vocab, hidden=32)
    src = torch.randint(0, vocab, (2, 8))
    tgt = torch.randint(0, vocab, (2, 6))
    logits = model(src, tgt)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
    loss.backward()
    return float(loss.detach())


def run_proxy(jsonl: Path, limit: int, epochs: int, device: str, tagger_exact: float | None) -> dict:
    src, tgt = load_rows(jsonl, limit)
    split = max(1, int(src.size(0) * 0.9))
    train_src, train_tgt = src[:split], tgt[:split]
    val_src, val_tgt = src[split:], tgt[split:]
    model = TinySeq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.time()
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        for start in range(0, train_src.size(0), 256):
            batch_src = train_src[start : start + 256].to(device)
            batch_tgt = train_tgt[start : start + 256].to(device)
            logits = model(batch_src, batch_tgt[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, len(TARGET_TOKENS)),
                batch_tgt[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
    exact = exact_rate(model, val_src.to(device), val_tgt.to(device))
    beats = None if tagger_exact is None else exact > tagger_exact
    return {
        "rows": int(src.size(0)),
        "epochs": epochs,
        "trainLoss": last_loss,
        "exactSeq": exact,
        "taggerExactSeq": tagger_exact,
        "beatsTagger": beats,
        "elapsedSeconds": time.time() - started,
        "device": device,
        "note": "100k linearized-intent CE prototype vs tagger exactSeq; not a ship decision",
    }


def greedy_tokens(model: TinySeq2Seq, src_row: torch.Tensor) -> list[str]:
    model.eval()
    device = src_row.device
    generated = [BOS]
    with torch.no_grad():
        for _ in range(TGT_LEN - 1):
            inp = torch.tensor([generated + [PAD] * (TGT_LEN - 1 - len(generated))], dtype=torch.long, device=device)
            logits = model(src_row.unsqueeze(0), inp)[0, len(generated) - 1]
            nxt = int(logits.argmax())
            generated.append(nxt)
            if nxt in (EOS, TARGET_INDEX[")"]):
                break
    tokens = [TARGET_TOKENS[index] for index in generated[1:] if index not in (PAD, BOS, EOS)]
    return tokens


def load_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def score_records(model: TinySeq2Seq, records: list[dict], device: str) -> dict[str, float | int | str]:
    correct = 0
    silent = 0
    neg_fa = 0
    negatives = 0
    positives = 0
    accepted = 0
    for record in records:
        src = torch.tensor(encode_text(str(record.get("text", ""))), dtype=torch.long, device=device)
        predicted = greedy_tokens(model, src)
        gold = linearize(record.get("intent"))
        is_neg = record.get("intent") is None
        if is_neg:
            negatives += 1
        else:
            positives += 1
        rejected = predicted[:3] == ["(", "reject", ")"] or predicted == ["reject"]
        if not rejected:
            accepted += 1
        if predicted == gold:
            correct += 1
        elif is_neg and not rejected:
            neg_fa += 1
        elif not is_neg and not rejected:
            silent += 1
    total = max(1, len(records))
    coverage = accepted / total
    fa_rate = neg_fa / max(1, negatives)
    return {
        "correct": correct,
        "total": len(records),
        "H": f"{correct}/{len(records)}",
        "H-neg": str(neg_fa),
        "H-silent": str(silent),
        "FA@cov": f"{fa_rate}@{coverage}",
        "positives": positives,
        "negatives": negatives,
    }


def run_b1(
    jsonl: Path,
    gold: Path,
    human_dir: Path,
    limit: int,
    epochs: int,
    device: str,
    tagger_h: str | None,
) -> dict:
    src, tgt = load_rows(jsonl, limit)
    model = TinySeq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.time()
    for _ in range(epochs):
        model.train()
        for start in range(0, src.size(0), 256):
            batch_src = src[start : start + 256].to(device)
            batch_tgt = tgt[start : start + 256].to(device)
            logits = model(batch_src, batch_tgt[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, len(TARGET_TOKENS)),
                batch_tgt[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    humans: list[dict] = []
    if human_dir.is_dir():
        for path in sorted(human_dir.glob("human-v*.jsonl")):
            humans.extend(load_jsonl_records(path))
    gold_records = load_jsonl_records(gold) if gold.is_file() else []
    h_scores = score_records(model, humans, device) if humans else {"H": "0/0", "H-neg": "0", "H-silent": "0", "FA@cov": "n/a", "correct": 0, "total": 0}
    exec_scores = score_records(model, gold_records, device) if gold_records else {"H": "0/0"}
    beats = False
    if tagger_h and "/" in tagger_h:
        t_num, t_den = tagger_h.split("/", 1)
        beats = int(h_scores["correct"]) / max(1, int(h_scores["total"])) > int(t_num) / max(1, int(t_den))
    return {
        "stack": "seq2seq",
        "rows": int(src.size(0)),
        "epochs": epochs,
        "elapsedSeconds": time.time() - started,
        "H": h_scores["H"],
        "H-neg": h_scores["H-neg"],
        "H-silent": h_scores["H-silent"],
        "FA@cov": h_scores["FA@cov"],
        "EXEC": exec_scores["H"],
        "RT": "0",
        "taggerH": tagger_h,
        "beatsTagger": beats,
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seq2seq prototype (Track B)")
    parser.add_argument("--one-batch", action="store_true")
    parser.add_argument("--proxy", type=int, default=0, help="train on N jsonl rows (S.6 100k)")
    parser.add_argument("--jsonl", default="packages/cron/training/data/synth/kinds-typed.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tagger-exact", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--b1", action="store_true")
    parser.add_argument("--gold", default="packages/cron/training/data/gold/reserved-carriers.jsonl")
    parser.add_argument("--human-dir", default="packages/cron/evaluation")
    parser.add_argument("--tagger-h", default=None)
    args = parser.parse_args(argv)
    if args.one_batch:
        loss = one_batch()
        print(f"seq2seq one-batch loss={loss:.4f}")
        return 0
    if args.b1:
        limit = args.proxy if args.proxy > 0 else 100_000
        report = run_b1(
            Path(args.jsonl),
            Path(args.gold),
            Path(args.human_dir),
            limit,
            args.epochs,
            args.device,
            args.tagger_h,
        )
        body = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(f"{body}\n")
        print(body)
        return 0
    if args.proxy > 0:
        report = run_proxy(
            Path(args.jsonl),
            args.proxy,
            args.epochs,
            args.device,
            args.tagger_exact,
        )
        body = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(f"{body}\n")
        print(body)
        return 0
    print("usage: seq2seq.py --one-batch | --proxy 100000")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
