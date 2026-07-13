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

The acceptance partition in `human-v1.jsonl` was consumed on 2026-09-15 and is
now development evidence only. Its acceptance-partition hash is recorded in
`consumed.json`; the evaluator refuses to score those acceptance records even
if unrelated partitions change or the file is copied. Final promotion requires a newly and independently
authored corpus with a different hash. After its one permitted acceptance run,
copy `acceptancePartitionSha256` from the evaluation report into the ledger
before further development.

`human-v2.jsonl` was authored by the coding agent at the user's explicit
request, rather than by an independent human, and was consumed on 2026-09-15.
It is a broad 150-example behavioral set, but it is not independent evidence.
Its acceptance hash is also recorded in `consumed.json`, so it can now be used
only as development diagnostics and never as a fresh promotion gate.
The one valid frozen-candidate run scored 96/150 (64.0%), with 29 silent
meaning changes and 15 false accepts. After those failures were used for
decoder development, the same set reached 150/150; that later score is a
regression result, not promotion evidence.
After consumption, one gold-label typo (`half past three` had been labeled as
15:30 rather than 03:30) was corrected. Both the originally evaluated hash and
the corrected fixture hash are consumed; the correction cannot create a fresh
acceptance set.
Two later labels that incorrectly treated `nightly` as a PM marker were also
corrected to ordinary unqualified-clock semantics. The resulting v2 and v3
hashes are consumed as well.

`human-v3.jsonl` is a second, newly authored but still non-independent
100-example acceptance set. Its only frozen-candidate run scored 78/100
(78.0%), with 17 silent meaning changes and 12 false accepts. Its hash was
immediately consumed; all later use is development-only.

`human-v4.jsonl` is the third non-independent acceptance set. Its only frozen
run scored 82/100 (82.0%), with 13 silent meaning changes and 8 false accepts.
Its hash is consumed and it is development-only from this point onward.

`human-v5.jsonl` is the fourth non-independent acceptance set. Its only frozen
run scored 90/100 (90.0%), with 10 silent meaning changes and 7 false accepts.
Its hash is consumed and it is development-only from this point onward.
