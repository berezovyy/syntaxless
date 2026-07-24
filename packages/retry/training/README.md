# Retry training

Run `bun run train:retry` from the repository root. The coordinator prepares
8,000 seeded examples, checks exact gold policies through the production resolver,
assigns structurally disjoint splits, trains hidden 32 for 24 epochs, and exports
an inactive six-bit candidate. It uses the existing cron Python environment and
shared bounded numerical trainer, tokenizer, codec and inference math.

Numeric spelling and duration units normalize to preserved exact payloads.
The model predicts policy roles; it never supplies a number absent from the
request or caller defaults. Clause order varies in the compact renderer.
Checkpoint selection measures decoded intent accuracy, wrong accepted meanings
and negative false acceptance, not token accuracy. Confidence measures only
behaviorally significant spans, excluding ignored background and linking words.
Repeated nonnumeric role tokens in a span use their strongest evidence; every
numeric, unit, percentage and arithmetic-word occurrence remains separate.
Dropped numeric/unit/arithmetic facts refuse rather than being inferred.
Canonical base constructions receive their split before negative wording or
caller-context changes. Literal substitutions and non-actionable variants stay
in the same group. Preparation checks all split ownership before writing a
requested split.

The coordinator writes immutable timestamped corpora and excludes verified
reserved wording without reading gold labels. Generation streams serialized
records; the private shared binary writer retains at most 8192 tokens, with
retry's semantic boundary labels and whitespace padding unchanged. Bounded row
records, deduplication and frame metadata remain O(rows), rather than retaining
all token feature vectors.
Calibration uses validation only. Frozen exported-wire parity checks 32 sequences
and padding invariance.

```sh
bun run packages/retry/training/src/gate.ts prepare <run-directory>
bun run packages/retry/evaluation/author.ts <run-directory>/scope-freeze.json <new-reserved.jsonl> <revision>
bun run packages/retry/training/src/gate.ts promote <run-directory> <new-reserved.jsonl>
```

The one-shot gate checks every supported family, wrong meanings under both
positive denominators, negative false accepts, prior and fresh critical controls,
and independently authored mathematical schedule targets. Before fresh inference,
it checks every consumed row against the active public model and preserves each
previously correct positive and negative result. Consumed results are regression
evidence, never fresh promotion evidence. It verifies actual
packed offline consumers, public default handling, seeded schedules, real Chromium
lazy/concurrent/offline behavior and aggregate chunk sizes before consuming
reserved wording. Promotion atomically replaces weights plus calibration and
retains rollback. No request execution or publication is part of the gate.

Consumed acceptance is regression-only and must never enter training, selection
or calibration. A writer revision already consumed cannot provide fresh evidence
again. The same implementation assistant authored the bounded generated variations;
they are not independent human evidence or a population accuracy estimate.

Generated retry work is bounded to 512 MiB. Monitor every run and retire redundant
preparation binaries, inactive checkpoints and duplicate parity only by verified
exact paths. Keep selected active evidence, corpus, checkpoint and rollback.
The shared trainer has no resume path, so an unselected last checkpoint is not a
resumable artifact.
