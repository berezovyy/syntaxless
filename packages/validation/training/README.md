# Validation training

This workspace generates the synthetic constraint corpus, trains and exports a
model candidate, and runs the evaluation and promotion gate for
`@syntaxless/validation`. Nothing here replaces the bundled model except
`gate.ts promote`.

## Requirements

- Bun (see the root `package.json`)
- The Python environment shared with cron at `training/cron/.venv`, created with
  `uv sync --project training/cron --locked`. The coordinator does not create
  or copy an environment.

## Running

From the repository root:

```sh
bun run train:validation
```

`src/train.ts` runs these steps and stops on the first failure:

1. Check free disk space and the generated-storage budget (512 MiB across
   `data/synth` and `training/validation/runs`).
2. Generate the corpus with `src/compact.ts`, excluding every reserved and
   consumed evaluation wording by hash.
3. Decode each row's supervision with the production TypeScript decoder and
   require every supported role in every split.
4. Featurize the train, validation, and heldout splits into binary shards.
5. Train with `training/shared/train.py --package validation`, selecting the checkpoint through
   the production decoder on the validation split.
6. Export an int6 candidate with `training/validation/export.py`, calibrated on
   validation only.
7. Freeze the candidate with its parity references, verify Python/TypeScript
   parity, and evaluate the validation and heldout splits.

It prints the run directory and whether the synthetic gate passed. It never
promotes.

## Outputs

| Path                                                                        | Content                                          |
| --------------------------------------------------------------------------- | ------------------------------------------------ |
| `packages/validation/training/data/synth/compact-v12-25k-<timestamp>.jsonl` | Corpus, manifest, and featurized shards          |
| `training/validation/runs/curve25k64-<timestamp>/`                          | Checkpoints, candidate, frozen artifact, reports |

Both locations are gitignored. Writers refuse to overwrite existing files.

## Recipe

`train.ts` passes the row count and hidden size; the other values are script
defaults. Standalone defaults differ where noted, so run `train.ts` rather than
the individual scripts.

| Setting       | Value                                                            | Standalone default      |
| ------------- | ---------------------------------------------------------------- | ----------------------- |
| Corpus rows   | 25,000 unique                                                    | `compact.ts`: 8,000     |
| Corpus seed   | 20261011                                                         | same                    |
| Hidden size   | 64                                                               | `train.py --hidden`: 32 |
| Epochs        | 24                                                               | same                    |
| Training seed | 20261011                                                         | same                    |
| Quantization  | six-bit QAT from epoch 3, int6 export                            | same                    |
| Optimizer     | AdamW, lr 0.003, weight decay 0.01, batch 64, warmup + cosine    | same                    |
| Selection     | exact − 5 × wrong − 5 × negative false acceptance, on validation | same                    |

Negative rows keep their field facts but are marked non-actionable. Nullable
rows keep an intent with an explicit unsupported compiler target. Numeric
variants and paired negative statements share structural split groups, so no
wording frame crosses splits. Token accuracy is not a release metric.

## Evaluation and promotion

After the synthetic gate passes, freeze the scope, author a new reserved set,
and run the gate. See [the evaluation protocol](../evaluation/README.md) for
commands, thresholds, and the dataset history.

Promotion replaces `src/model/weights.gen.ts` (model and calibration together)
and keeps the previous file in the run directory for rollback.

## Reproducibility

- Corpus generation is deterministic for a given seed, row count, and
  generator source; the manifest records the corpus sha256 and a generator
  source hash.
- The training report records source hashes, the corpus sha256, and shard
  digests. The gate's scope identity covers runtime, training, and Python
  sources, package configuration, and the evaluation README.
- The trainer has no resume path.

## Source map

| File                                                    | Role                                                                |
| ------------------------------------------------------- | ------------------------------------------------------------------- |
| `src/train.ts`                                          | Coordinator                                                         |
| `src/compact.ts`                                        | Training corpus generator                                           |
| `src/renderer.ts`, `src/sentence.ts`                    | Sentence rendering with labeled spans                               |
| `src/featurize.ts`                                      | Binary shard writer shared with the Python loader                   |
| `src/evaluate.ts`                                       | Intent metrics; its `interval` helper is used by other packages     |
| `src/freeze.ts`, `src/parity.ts`                        | Candidate freeze, calibration floor, numerical parity               |
| `src/gate.ts`                                           | Scope freeze, acceptance gate, promotion                            |
| `src/backward.ts`, `src/kinds.ts`, `src/fingerprint.ts` | Strict-grammar kinds catalog used as a renderer regression baseline |
| `src/generate.ts`                                       | Corpus record types                                                 |
| `src/rng.ts`                                            | Seeded random source                                                |

Run the workspace checks with `bun run --cwd packages/validation/training typecheck`
and `bun run --cwd packages/validation/training test`.
