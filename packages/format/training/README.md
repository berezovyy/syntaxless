# Format training

Private workspace that generates the format training corpus, featurizes it, trains
and exports a candidate model, and runs the release gate. Nothing here is
published.

## Requirements

- Bun 1.4 or newer.
- The cron Python environment (`uv sync --project training/cron --locked`). The
  coordinator runs `training/cron/.venv/bin/python` with the shared harness in
  [`training/shared`](../../../training/shared/README.md).

## Run

From the repository root:

```sh
bun run train:format
```

The coordinator (`src/train.ts`) takes no flags. It:

1. Generates a new timestamped corpus of 8000 rows under
   `packages/format/training/data/synth/`, excluding all reserved and consumed
   evaluation wording.
2. Featurizes the train, validation and heldout splits into binary shards.
3. Trains with `training/shared/train.py --package format` and exports the
   selected checkpoint with `training/shared/export.py --package format` into a
   new directory under
   `training/format/runs/`.
4. Calibrates the confidence floor on the validation split (`src/freeze.ts`) and
   writes the frozen candidate.
5. Verifies export parity (`src/parity.ts`) and evaluates the validation and
   heldout splits with the production runtime (`src/evaluate.ts`).

Generated data and runs are ignored by Git. The coordinator stops if
`packages/format` and the runs folder together exceed 512 MiB.

## Outputs

Each run directory contains `report.json` (training history and source hashes),
`best.pt`, `candidate/` (unpromoted export and parity references), `frozen/`
(calibrated `weights.json`, `weights.gen.ts` and `freeze.json`),
`validation-runtime.json`, `heldout-runtime.json` and `storage.json`. Training
never changes the bundled model.

## Recipe

| Setting     | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| Corpus      | 8000 rows, ten families, about one third negatives                   |
| Split       | Hash buckets 80/10/10 by construction fingerprint                    |
| Model       | Shared role tagger, hidden width 32, 25 role labels                  |
| Training    | 24 epochs, batch 64, AdamW lr 0.003, quantization-aware from epoch 2 |
| Selection   | Validation `positiveExact − 5·wrongAll − 5·negativeFalseAcceptance`  |
| Export      | Six-bit weights with frozen numerical and padding parity             |
| Calibration | Confidence floor chosen on validation only                           |

The model labels instruction roles. It does not produce output strings.
`src/resolve.ts` turns labeled lexemes into an intent and checks preserved
numeric, currency, style, locale and timezone literals. Gold intents come from
the corpus renderer and must round-trip through the resolver. Literal
substitutions share a construction fingerprint, and negative variants inherit
their base construction's split, so no construction crosses splits.

## Promotion

Training creates an inactive candidate. Promotion requires the full gate
described in [evaluation](../evaluation/README.md):

```sh
bun run packages/format/training/src/gate.ts prepare <run-directory>
bun run packages/format/evaluation/author.ts <run-directory>/scope-freeze.json \
  packages/format/evaluation/reserved/ai-v<N>.jsonl <revision>
bun run packages/format/training/src/gate.ts promote <run-directory> \
  packages/format/evaluation/reserved/ai-v<N>.jsonl
```

A passing gate replaces `src/model/weights.gen.ts` atomically and keeps the
previous module as `previous-weights.gen.ts` in the run directory.

## Reproducibility

The corpus seed and recipe are fixed in source. Corpus manifests record the
generator digest, and run reports record the digests of the Python sources,
corpus and binary shards. The freeze records the digests of every source file
that can affect the result. The JSONL, binary shard and reserved-wording helpers
are shared with the other packages through `packages/shared/training`.

## Tests

```sh
bun run --cwd packages/format/training test
```

The tests cover split stability across literal substitutions and paired
negatives, and the buffered binary shard writer.
