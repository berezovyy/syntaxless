# Retention training

This workspace produces the bundled retention model. It generates a synthetic
corpus, trains a small role-labeling network, exports and calibrates an inactive
candidate, and promotes it only through the reserved acceptance gate.

## Run

```sh
bun run train:retention   # from the repository root
```

Requirements:

- Bun.
- The Python environment at `training/cron/.venv`, created with
  `uv sync --project training/cron --locked`.
- The shared Python harness in [`training/shared`](../../../training/shared/README.md)
  (`train.py --package retention` and `export.py --package retention`).

## Pipeline

1. **Corpus** (`src/corpus.ts`). The corpus has 8,000 rows from the `compact-v8-8k`
   recipe with seed `20261012`.
   - The rows vary bucket forms, representative-count wording, modal requests,
     calendar conversion, elapsed windows and clock durations.
   - Every gold policy is checked through the production resolver.
   - Structural fingerprints are grouped before splitting. Numeric substitutions
     and statement or question variants share their base construction's split.
   - Reserved evaluation wording is excluded by normalized text hash
     (`packages/shared/training/reserved.ts`), without reading evaluation labels.
2. **Features** (`src/featurize.ts`). Each split becomes a bounded binary file of
   at most 8192 tokens per frame, with role labels aligned to tokens.
3. **Training** (`training/shared/train.py --package retention`). Hidden width 32, 24 epochs.
   Checkpoints are selected by decoded intent accuracy and wrong accepts.
4. **Export and freeze** (`export.py`, `src/freeze.ts`).
   - Weights are quantized to six bits.
   - The confidence floor is calibrated on the validation split only.
   - Exported-wire parity is checked on 32 sequences, including padding invariance
     (`src/parity.ts`).
5. **Synthetic evaluation** (`src/evaluate.ts`). The validation and held-out gate
   requires:
   - exact positives ≥ 95%, and ≥ 90% per family;
   - wrong meanings ≤ 0.5%;
   - negative false acceptance ≤ 1%.

## Outputs (git-ignored)

- `packages/retention/training/data/synth/compact-v8-8k-<timestamp>.jsonl`, with a
  manifest and featurized splits.
- `training/retention/runs/hidden32-<timestamp>/` contains:
  - the checkpoint and `candidate/` export;
  - `frozen/` weights and parity data;
  - runtime reports and `storage.json`.

Generated retention data is limited to 512 MiB. The coordinator refuses to continue
past that limit. Existing corpus files are never overwritten.

## Promotion

Training never changes the bundled model. Promotion uses the one-shot gate
described in [evaluation](../evaluation/README.md):

```sh
bun run packages/retention/training/src/gate.ts prepare <run>
bun run packages/retention/evaluation/author.ts <run>/scope-freeze.json \
  packages/retention/evaluation/reserved/<set>.jsonl <revision>
bun run packages/retention/training/src/gate.ts promote <run> \
  packages/retention/evaluation/reserved/<set>.jsonl
```

`bun run replay` (in this workspace) replays every consumed set against the
bundled model. When a full set is git-ignored and absent, it replays that set's
archived critical rows.

## Reproducibility

- The corpus is deterministic for a given generator source, seed and reserved set
  inventory. The manifest records the generator hash and the excluded-text count.
- PyTorch training is CPU-bound but not guaranteed bit-identical across machines.
  Frozen candidates are identified by artifact hash, and the gate verifies parity
  against the exported reference.
- The trainer has no resume path. An interrupted run starts over.

## Scripts

`bun run test`, `bun run typecheck`, `bun run train`, `bun run replay`.
