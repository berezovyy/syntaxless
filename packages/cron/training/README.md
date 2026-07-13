# @syntaxless/cron-training

Private, development-only workspace for generating the training corpus for
`@syntaxless/cron`. Nothing here is published; no training code runs on the
runtime path.

## Workflow

The corpus pipeline is:

1. **sample** — draw `ScheduleIntent` trees from the grammar using the AST in
   `packages/cron/src` as the ground truth.
2. **render** — turn each intent into natural-language surface forms.
3. **fingerprint** — compute a canonical, deduplicated identity for each pair so
   corpora stay unique across runs.
4. **featurize** — flatten the pairs into model-ready features.
5. **hand off to Python** — the Phase 14 Python harness under `training/cron/`
   consumes the featurized corpus for training.

A human miss is a **kind**, not a decoder patch. After a consumed acceptance
run, add an abstract wrap in `kinds.ts` and retrain; do not match a noun from
the failed row in `decode.ts`. See
[`packages/cron/evaluation/README.md`](../evaluation/README.md). Consumed
human JSONL never feeds `backward.ts`, `generate.ts`, `featurize.ts`, or the
Python trainer.

Sources import `../../src/**` directly; this package declares no runtime
dependencies.

Oracle decode (gold spans → shipped `decodeIntent`) is a CI gate. Train does
not start while it is red:

```sh
bun run --cwd packages/cron/training oracle:gate
```

## Generated data

Generated corpora live under `data/synth/` and are **untracked** (see the root
`.gitignore`). The venv for the Python harness lives at `training/cron/.venv/`
and is untracked as well. Regenerate data from source rather than committing it;
the pipeline is deterministic enough to reproduce a corpus from the committed
code.
