# Shared role-tagger harness

PyTorch training and export harness for the small quantization-aware token
tagger used by validation, filter, sort, retry, retention and format. Cron has
its own harness in [`training/cron`](../cron/README.md).

Each package's TypeScript coordinator (`bun run train:<package>`) prepares the
corpus and binary shards, then runs these scripts with the Python environment at
`training/cron/.venv`. Outputs go to `training/<package>/runs/`, which Git
ignores.

| File           | Role                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `train.py`     | `--package <name>`: seeded QAT training; selects checkpoints through the package's TypeScript evaluator |
| `export.py`    | `--package <name>`: writes an uncalibrated int6 candidate and parity fixtures; never promotes           |
| `artifact.py`  | Quantization, alphabet packing, wire decoding, parity fixtures and label loading                        |
| `model.py`     | Tagger with role and clause-boundary heads and six-bit fake quantization                                |
| `dataset.py`   | Loads the shards written by `packages/shared/training/binary.ts`; length-bucketed batches               |
| `calibrate.py` | Boundary-threshold calibration on development splits (used by validation)                               |

Validation exports with [`training/validation/export.py`](../validation/README.md),
which also calibrates the boundary threshold.

The promotion gates of all six packages hash these files into the frozen source
identity, so any edit here requires a new freeze before the next promotion.

## Tests

```sh
uv run --project training/cron --locked python -B training/cron/run_tests.py training/shared
```

The tests cover padding neutrality and finite QAT gradients, the int6 wire
round trip, the label vocabularies of every package, and shard loading and
batching. `bun run test:python` runs them together with the cron suite.
