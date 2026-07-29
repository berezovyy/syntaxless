# Cron training

This package generates the synthetic training corpus for
[`@syntaxless/cron`](../README.md). It also coordinates training of the bundled
token tagger and scores candidates with the shipped parser and compiler.
Training never replaces the bundled weights: it writes a candidate into a run
directory. Promotion requires a separate fresh acceptance set (see the
[evaluation protocol](../evaluation/README.md)).

## Requirements

- Bun and the workspace dependencies (`bun install` at the repository root).
- Python 3.13 and [uv](https://docs.astral.sh/uv/). The coordinator runs
  `uv sync --project training/cron --locked` itself; see
  [`training/cron`](../../../training/cron/README.md) for the Python harness.
- A built runtime (`bun run build` in `packages/cron`). Human regression scoring
  imports `packages/cron/dist`.

## Running

From the repository root:

```sh
bun run train:cron
```

`bun run train` inside `packages/cron/training` does the same. The coordinator
takes no flags. Every run creates a fresh `training/cron/runs/compact-*/`
directory and never reuses or overwrites another run's output.

## Pipeline

1. **Protection.** `compact-protection.ts` loads every evaluation JSONL file,
   the archived historical acceptance partitions and
   `data/gold/reserved-carriers.jsonl`, and checks them against the consumption
   ledger. Generated rows whose wording matches protected text are rejected.
   Malformed evidence stops the run. Consumed partitions whose records are
   unavailable are reported as a limitation in `preparation.json`.
2. **Corpus.** `compact.ts` renders at most `COMPACT_RECIPE.maximumRows` rows
   from the sentence-kind catalog (`kinds.ts`). Each row is either a clean
   request, a request with a spelling or punctuation variant (`noise.ts`), a
   correction (`correction.ts`), the same schedule wrapped in a non-request
   speech act (description, question, quotation, one-off, cancellation,
   unfinished request), or ordinary prose (`background.ts`). A construction
   group (request, variants, negatives and corrections built from one schedule
   pattern) always stays in a single split. Before anything is written, every
   row must pass token alignment and gold decoding, and every split must
   contain each required schedule field and each variant type.
3. **Features.** `featurize-kinds.ts` writes binary shards per split. The shard
   contract is defined in `featurize.ts` and read by `training/cron/dataset.py`.
4. **Oracle gate.** `oracle.ts` decodes gold spans with the shipped decoder over
   the reserved carriers, catalog seeds and slot-bag wraps. Any disagreement stops the run
   before training.
5. **Training.** `training/cron/train.py` trains, selects a checkpoint and
   exports it (see below).
6. **Human regression.** `human-evaluate.ts` scores the exported candidate on
   the consumed `human-v*` evaluation corpora as regression diagnostics. These
   scores never select a checkpoint and never authorize promotion.

## Recipe

| Setting                | Value                                   | Defined in                          |
| ---------------------- | --------------------------------------- | ----------------------------------- |
| Corpus recipe version  | 4                                       | `src/compact.ts` (`COMPACT_RECIPE`) |
| Corpus seed            | 20261012                                | `src/compact.ts`                    |
| Bases per kind         | 120 (2,048 attempts)                    | `src/compact.ts`                    |
| Ordinary-prose rows    | 512                                     | `src/compact.ts`                    |
| Maximum rows           | 10,000                                  | `src/compact.ts`                    |
| Split rule             | construction-group hash                 | `src/compact.ts`                    |
| Hidden size            | 64                                      | `training/cron/config.py`           |
| Epochs / batch         | 24 / 128                                | `training/cron/config.py`           |
| Learning rate / warmup | 3e-3 / 10 steps                         | `training/cron/config.py`           |
| Quantization           | 6-bit, QAT from 2nd epoch               | `training/cron/config.py`           |
| Training seed          | 20261012                                | `training/cron/config.py`           |
| Device                 | CUDA if available, else CPU (2 threads) | `training/cron/train.py`            |

## Checkpoint selection

After each epoch, the checkpoint is packed into the same 6-bit artifact the
exporter writes. Boundary, energy and distance thresholds are fitted on the
validation split only. `checkpoint.ts` then parses every validation row with the
TypeScript engine. A positive row counts only when the complete expected intent
is returned; a refusal is never a positive hit.

The synthetic gate (`checkpoint.ts`) requires:

- positive exact rate at least 0.95, and at least 0.90 in every family;
- wrong accepted meanings at most 0.5% of all positives and of accepted
  positives;
- negative false accepts at most 1% overall and in every family.

Selection (`training/cron/selection.py`) prefers checkpoints that pass the gate.
Before any checkpoint passes, it ranks by mean family positive accuracy, then
overall positive accuracy, wrong accepted meanings, negative false accepts and
raw decoding accuracy. Once a checkpoint passes, a replacement must also pass.
It must also show no significant loss (two standard errors, at least one row)
overall or in any family, kind or speech act. Data, code and support identities
must match between compared epochs.

The exporter must reproduce the artifact identity used for selection before it
writes parity fixtures or the candidate module. The heldout split is scored
once, after selection. Every report from this pipeline is synthetic development
evidence, whether or not the gate passed.

## Outputs

Each `training/cron/runs/compact-*/` directory contains:

| File                                           | Contents                                              |
| ---------------------------------------------- | ----------------------------------------------------- |
| `corpus/corpus.jsonl`, `corpus/corpus-*.bin`   | Generated rows and feature shards per split           |
| `corpus/preparation.json`                      | Recipe, coverage, protected sources, input identities |
| `oracle.json`                                  | Oracle gate summary                                   |
| `best.pt`, `last.pt`                           | Selected and last checkpoints                         |
| `last-selection.json`, `last-selection.gen.ts` | Latest epoch's decoded validation score and artifact  |
| `weights.gen.ts`                               | Exported candidate module                             |
| `export.json`, `parity/`                       | Calibration, artifact identity and parity fixtures    |
| `heldout-runtime.json`                         | One-time heldout score of the selected artifact       |
| `report.json`                                  | Configuration, source hashes, per-epoch metrics       |
| `human-regression.json`                        | Consumed human-corpus regression diagnostics          |
| `workflow.json`                                | Status, error, elapsed time, storage per phase        |

A run must stay within 128 MiB of storage. The coordinator measures it after
each phase and fails the run when the budget is exceeded.

## Reproducibility

The corpus is deterministic for a given recipe, seed, catalog source and
protected-text set. `preparation.json` records the SHA-256 of every local
module the generator imports. It also records the lockfiles and the Python
shard loader (`src/corpus/source-hashes.ts`). `report.json` records the
Python source hashes and the Git revision. CUDA training can differ slightly
from CPU training.

## Layout

| Path                                                    | Purpose                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| `src/train.ts`                                          | Coordinator (`bun run train:cron`)                         |
| `src/compact*.ts`                                       | Corpus recipe, scope checks and protected-text loading     |
| `src/kinds.ts`, `src/slots/`                            | Sentence-kind catalog and carrier phrase bags              |
| `src/correction.ts`, `src/discourse.ts`, `src/noise.ts` | Corrections, speech-act wraps, typos                       |
| `src/featurize*.ts`                                     | Shard writer                                               |
| `src/oracle.ts`                                         | Gold decoding gate                                         |
| `src/checkpoint.ts`                                     | Decoded checkpoint scoring, called from Python             |
| `src/evaluate.ts`                                       | Gold-corpus contracts (token roles, raw decode, AST, cron) |
| `src/human-evaluate.ts`                                 | Human-style corpus scoring and promotion rules             |
| `src/corpus/`                                           | Record schema, JSONL I/O, protected text, source hashes    |
| `data/gold/reserved-carriers.jsonl`                     | Reserved carrier wording, protected and used by the oracle |
| `tests/`                                                | Bun tests                                                  |

## Testing

```sh
bun run --cwd packages/cron/training test       # TypeScript
bun run test:python                             # Python harness (repository root)
```

## Limitations

- The corpus is synthetic and AI-authored. Passing the synthetic gate does not
  establish accuracy on real user requests.
- Fresh reserved acceptance data is required for promotion; this pipeline only
  produces candidates.
- Some consumed historical evaluation partitions are unavailable, so the
  exclusion of their wording cannot be verified.
