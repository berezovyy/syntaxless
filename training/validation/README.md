# Validation model harness

The public training entry point is `bun run train:validation` from the repository
root. See [the corpus and coordinator documentation](../../packages/validation/training/README.md).

`train.py` loads shared shards, performs seeded six-bit QAT, and selects a
checkpoint through the production TypeScript intent decoder. `export.py` writes
only run-local candidates and parity fixtures. Neither phase promotes a model.
The runtime owns exact numeric comparisons and schema generation.

Reuse the existing cron Python environment. The domain dependency lock remains
available for reproducibility; do not duplicate its environment automatically.
The shared runner discovers all validation numerical and batching tests:

```sh
training/cron/.venv/bin/python -B training/cron/run_tests.py training/validation
```

Run `bun run test:python` from the root to verify both domains in separate processes.
