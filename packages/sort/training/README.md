# Sort training

Corpus generation, model training, calibration and promotion for
`@syntaxless/sort`. Candidates stay inactive until the promotion gate passes.

## Run

From the repository root:

```sh
bun run train:sort
```

Training uses the Python environment at `training/cron/.venv` (create it with
`uv sync --project training/cron --locked`) and the shared Python harness in
[`training/shared`](../../../training/shared/README.md).

## Pipeline

1. **Corpus** (`src/corpus.ts`): 8,000 seeded rows (seed `20261011`) across
   the families `explicit-direction`, `ordering-alias`, `priority`,
   `null-first`, `null-last`, `null-update` and `caller-default`, plus
   negatives. Caller fields and ordering aliases become typed markers that
   keep their field and direction payloads. Every gold role sequence is checked
   through the production resolver. Rows are split 80/10/10 into
   train/validation/held-out by a hash of the language construction, assigned
   before negative wording or caller context changes, so questions and their
   base statements share a split. Text from reserved or consumed evaluation
   sets is excluded.
2. **Features** (`src/featurize.ts`): per-split binary shards using the runtime
   tokenizer; at most 128 tokens per row. Featurization rejects any language
   group that crosses splits.
3. **Train** (`training/shared/train.py --package sort`): hidden size 32, 24 epochs, AdamW
   (lr 0.003), six-bit quantization-aware training from epoch 3. Checkpoint
   selection decodes intents through the TypeScript resolver and penalizes
   wrong accepted meanings and negative false accepts.
4. **Export** (`training/shared/export.py --package sort`): packs the checkpoint into the wire
   artifact, decodes it back into PyTorch and freezes 32 parity sequences.
5. **Freeze** (`src/freeze.ts`): calibrates the confidence floor on the
   validation split only, choosing the passing floor with the best exact rate
   (ties go to the floor nearest 0.5).
6. **Parity** (`src/parity.ts`): the TypeScript engine must match PyTorch role
   and boundary logits within 1e-4, including padding invariance.
7. **Synthetic evaluation** (`src/evaluate.ts`): validation and held-out
   reports. The synthetic gate requires exact positives ≥ 95%, wrong meanings
   ≤ 0.5%, negative false acceptance ≤ 1% and every family ≥ 90%.

## Outputs

- `packages/sort/training/data/synth/corpus-8k-<timestamp>.jsonl` with a
  manifest and binary shards.
- `training/sort/runs/run-<timestamp>/`: checkpoints, `candidate/`,
  `frozen/`, runtime reports and `storage.json`.

Both locations are git-ignored. Corpus files and frozen candidates refuse to
overwrite existing files.

## Promotion

```sh
bun run packages/sort/training/src/gate.ts prepare <run-directory>
bun run packages/sort/evaluation/author.ts <run-directory>/scope-freeze.json packages/sort/evaluation/reserved/<set>.jsonl <revision>
bun run packages/sort/training/src/gate.ts promote <run-directory> packages/sort/evaluation/reserved/<set>.jsonl
```

`prepare` freezes the candidate identity. `promote` runs the checks in the
[evaluation protocol](../evaluation/README.md), consumes the reserved set, and
on success atomically replaces `src/model/weights.gen.ts`, keeping the previous
module in the run directory for rollback. Each reserved set can be consumed
once; later candidates need newly authored wording.

## Reproducibility

The corpus generator is deterministic for a given seed, row count and reserved
set. Training seeds Python, NumPy and PyTorch with the same seed. The training
coordinator checks that generated data stays under 512 MiB.

## Tests

```sh
bun run --cwd packages/sort/training typecheck
bun run --cwd packages/sort/training test
```
