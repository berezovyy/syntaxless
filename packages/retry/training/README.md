# Retry training

This workspace produces the bundled retry model. It generates a synthetic corpus,
trains a small role-labeling network, exports and calibrates an inactive candidate,
and promotes it only through the reserved acceptance gate.

## Run

```sh
bun run train:retry   # from the repository root
```

Requirements:

- Bun.
- The Python environment at `training/cron/.venv`, created with
  `uv sync --project training/cron --locked`.
- The shared Python harness in [`training/shared`](../../../training/shared/README.md)
  (`train.py --package retry` and `export.py --package retry`).

## Pipeline

1. **Corpus** (`src/corpus.ts`). The corpus has 8,000 rows from the
   `compact-v13-8k` recipe with seed `20261011`.
   - Every gold policy is checked through the production resolver.
   - Canonical constructions are assigned to `train`, `validation` or `heldout`
     before negative wording or context changes, so variants never cross splits.
   - Reserved evaluation wording is excluded by normalized text hash
     (`packages/shared/training/reserved.ts`), without reading evaluation labels.
2. **Features** (`src/featurize.ts`). Each split becomes a bounded binary file of
   at most 8192 tokens per frame, with role labels aligned to tokens.
3. **Training** (`training/shared/train.py --package retry`). Hidden width 32, 24 epochs.
   Checkpoints are selected by decoded intent accuracy, wrong accepted meanings and
   negative false acceptance.
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

- `packages/retry/training/data/synth/compact-v13-8k-<timestamp>.jsonl`, with a
  manifest and featurized splits.
- `training/retry/runs/hidden32-<timestamp>/` contains:
  - the checkpoint and `candidate/` export;
  - `frozen/` weights and parity data;
  - runtime reports and `storage.json`.

Generated retry data is limited to 512 MiB. The coordinator refuses to continue
past that limit. Existing corpus files are never overwritten.

## Promotion

Training never changes the bundled model. Promotion uses the one-shot gate
described in [evaluation](../evaluation/README.md):

```sh
bun run packages/retry/training/src/gate.ts prepare <run>
bun run packages/retry/evaluation/author.ts <run>/scope-freeze.json \
  packages/retry/evaluation/reserved/<set>.jsonl <revision>
bun run packages/retry/training/src/gate.ts promote <run> \
  packages/retry/evaluation/reserved/<set>.jsonl
```

`bun run replay` (in this workspace) replays every consumed set against the
bundled model.

## Reproducibility

- The corpus is deterministic for a given generator source, seed and reserved set
  inventory. The manifest records the generator hash and the excluded-text count.
- PyTorch training is CPU-bound but not guaranteed bit-identical across machines.
  Frozen candidates are identified by artifact hash, and the gate verifies parity
  against the exported reference.
- The trainer has no resume path. An interrupted run starts over.

## Scripts

`bun run test`, `bun run typecheck`, `bun run train`, `bun run replay`.
