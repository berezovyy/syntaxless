# Validation training

From the repository root:

```sh
bun run train:validation
```

The coordinator reuses `training/cron/.venv/bin/python`; it creates no second
PyTorch environment. The Python dependency specification matches that shared
environment. It checks supervision through the actual TypeScript decoder,
prepares structurally separated train/validation/heldout shards, trains, exports
an inactive candidate, calibrates on validation, freezes parity references, and
evaluates synthetic meaning and refusal metrics.

The current measured comparison uses 25,000 unique examples, hidden size 64,
seed 20261011, 24 epochs, and six-bit QAT. Smaller earlier comparisons had
repeated family failures, which motivated the bounded increase. Negatives preserve field facts
while marking non-actionable language. Nullable rows retain an intent and an
explicit unsupported compiler target. Numeric variants and paired negative
statements share structural split groups. Token accuracy is not a release gate.

Data lives in `data/synth`; runs live in `training/validation/runs`. Both paths
include a fresh timestamp, and corpus/shard writers refuse to overwrite an
existing file. No coordinator
or exporter writes active weights. Confidence calibration uses validation only;
reserved acceptance must never enter training, featurization, or calibration.

After synthetic checks pass, freeze source and scope, author new isolated
acceptance, then use the full gate:

```sh
bun run packages/validation/training/src/gate.ts prepare <run-directory>
bun run packages/validation/evaluation/author-reserved.ts <run-directory> <new-reserved.jsonl> 5
bun run packages/validation/training/src/gate.ts promote <run-directory> <new-reserved.jsonl>
```

The separate authoring writer imports neither training renderer nor model. Its
output is AI-authored by the same assistant; it is not independent human evidence.
The gate rejects overlap with training and consumed reserved wording, verifies
source and artifact identities, paired compiler gold, meaningful source checks,
Python/TypeScript parity, synthetic family thresholds, and prior critical
regressions before consuming acceptance exactly once. Fresh acceptance must
meet all declared overall, family, wrong-meaning, negative, and critical gates.
Actual tarball consumers run offline; all JavaScript chunks enter size accounting.
Promotion atomically replaces model and calibration together and retains rollback.

`backward.ts` is a small strict-grammar/renderer regression baseline. Development
fixtures and consumed acceptance remain evaluation-only. Their failures may
motivate general supervision changes; their wording must never be trained on.

Generation writes buffered JSONL, and file preparation reads JSONL incrementally.
The shared feature writer retains at most 8192 tokens, with 128 tokens per
sequence. Deduplication, frame metadata and split ownership use O(rows) memory,
bounded to 100000 rows. Storage and peak RSS are recorded. The generated-domain budget is 512 MiB. Keep one active corpus, the selected checkpoint, rollback and compact unique
reports. The trainer has no resume path, so a completed unselected last checkpoint
is not a resumable artifact. Remove retired binary shards by exact inventory only after provenance
checks; preserve unique source data and model evidence.

The preparation audit regenerated the complete active 25000-row corpus byte for
byte and matched all 21 binary shards and three manifests after buffering. No
model was retrained or promoted for the refactor. Shared preparation now also
serves filter and format, with their separate labels and boundary semantics.
Reserved wording is excluded before generation, with metadata/full-file hashes
and row counts checked. Missing consumed or pending wording fails preparation.
Future gate identities include the full training source, tests and configuration.
