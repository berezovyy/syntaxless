# Cron Python training harness

This directory holds the PyTorch side of cron training: the shard loader, the
tagger model, the training loop, calibration and int6 export. It has no command
line of its own. `bun run train` (see
[packages/cron/training](../../packages/cron/training/README.md)) prepares the
corpus and then runs `train.py` with absolute paths in environment variables.

The uv project here (`pyproject.toml`, `uv.lock`, Python 3.13, torch, numpy) is
also the shared Python environment for the other packages' training
coordinators. They run `training/cron/.venv/bin/python`.

## Modules

| File           | Purpose                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `config.py`    | The fixed training recipe (`Config`) and path/device resolution                                                          |
| `dataset.py`   | Memory-mapped shard loader, length-bucketed batching, label vocabularies                                                 |
| `model.py`     | Tagger: shared trunk with role, boundary, schedule-kind, speech-act and actionability heads; quantization-aware training |
| `train.py`     | Training loop, per-epoch selection, export and heldout scoring                                                           |
| `selection.py` | Decoded checkpoint ranking and regression checks (calls `checkpoint.ts`)                                                 |
| `calibrate.py` | Boundary-threshold calibration on validation                                                                             |
| `abstain.py`   | Energy and pooled-state abstention thresholds                                                                            |
| `export.py`    | int6 packing, decoded-artifact calibration, parity fixtures, `weights.gen.ts`                                            |
| `report.py`    | Atomic JSON report writer                                                                                                |
| `run_tests.py` | Test runner: collects `test_*` functions from `test_*.py` with `unittest`                                                |

The shard format is defined in TypeScript
(`packages/cron/training/src/featurize.ts`). `dataset.py` and `model.py` mirror
the label and speech-act order of `packages/cron/src/model`; these lists must
never be reordered independently.

## Environment variables

The coordinator sets these. `config.production_config()` refuses to start without the first two; `selection.py` falls back to `bun` when the third is unset.

| Variable                         | Meaning                                          |
| -------------------------------- | ------------------------------------------------ |
| `SYNTAXLESS_TRAIN_DATA_PREFIX`   | Absolute shard prefix written by the corpus step |
| `SYNTAXLESS_TRAIN_RUN_DIRECTORY` | Absolute run directory                           |
| `SYNTAXLESS_BUN_BIN`             | Bun executable used for decoded scoring          |

## Setup and tests

```sh
uv sync --project training/cron --locked
bun run test:python        # from the repository root
```

`bun run test:python` runs
`uv run --project training/cron --locked python -B training/cron/run_tests.py`.
Some tests run TypeScript through Bun, so workspace dependencies must be
installed.

## Outputs

`train.py` writes `best.pt`, `last.pt`, `report.json`, `last-selection.*`,
`heldout-runtime.json` and, through `export.py`, `weights.gen.ts`, `export.json`
and `parity/` into the run directory. It refuses to run if any of these already
exists. The bundled runtime weights are never modified.
