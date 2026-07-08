# Human-style evaluation

`human-v1.jsonl` is a versioned, evaluation-only behavioral corpus. It is not a
training corpus and must never be passed to `backward.ts`, `generate.ts`,
`featurize.ts`, the Python trainer, checkpoint selection, or calibration.

The partitions have separate purposes:

- `development`: diagnose capabilities and guide abstract fixes.
- `calibration`: choose confidence/abstention policy; never train parameters.
- `acceptance`: reserved final promotion evidence; never use its wording to
  change templates, features, decoding, confidence, or checkpoints.

Every record deliberately lacks renderer spans and training split/frame fields,
and carries `trainingEligible: false`. `human-evaluate.ts` fails closed if the
file is moved under `training/data`, becomes training-shaped, is referenced by
a generator, or contains wording copied verbatim into generator source.

Run development diagnostics from the repository root after building:

```sh
bun run --cwd packages/cron/training evaluate:human
```

Do not run the `acceptance` partition until the candidate is otherwise frozen.
