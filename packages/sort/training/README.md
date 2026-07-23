# Sort training

Run `bun run train:sort` from the repository root. The coordinator writes a seeded
8,000-row corpus with structurally disjoint splits and role supervision checked
against the production resolver. It uses the existing cron Python environment,
validation's numerical model and dataset, and the shared bounded trainer. Hidden
size is 32 with six-bit quantization-aware training for 24 epochs. Checkpoint
selection measures decoded intents, wrong accepted meanings and false accepts.

Caller fields and ordering aliases become typed canonical markers with preserved
field/direction payloads. Exact priorities, null placement and tie policy resolve
in TypeScript. Training and runtime share preparation and tokenization. Gold
language constructions determine split assignment before adding negative wording
or changing caller context. Questions and their original statements remain in
one split, and caller IDs cannot separate identical language constructions.

Corpus serialization streams into a unique timestamped path and refuses to
overwrite evidence. Generation excludes verified reserved and consumed wording
without reading evaluation labels. Feature writing uses the private shared
8192-token buffer, with local field-only boundary labels and whitespace padding.
Bounded row records, deduplication, fingerprints and frame metadata remain
O(rows); feature vectors are not retained for the whole corpus. Preparation
checks group ownership across all splits even when writing one selected split.

Candidates stay run-local and inactive. Validation-only calibration freezes the
artifact and confidence floor; exported-wire parity uses 32 frozen sequences and
checks padding invariance. The full gate verifies fresh reserved wording, exact
intent accuracy, per-family coverage, both wrong-meaning denominators, negative
false acceptance and critical controls. Paired positive targets use independently
authored order over eight witnesses, checked in actual SQLite and JavaScript.

```sh
bun run packages/sort/training/src/gate.ts prepare <run-directory>
bun run packages/sort/evaluation/author.ts <run-directory>/scope-freeze.json <new-reserved.jsonl> <revision>
bun run packages/sort/training/src/gate.ts promote <run-directory> <new-reserved.jsonl>
```

Never reuse consumed wording as fresh evidence, training or calibration. Writer
revisions already consumed are regression-only and require new wording for later
candidates. The same implementation assistant authors these bounded generated
variations; no independent human evidence or population accuracy is claimed.
The gate checks actual packed declarations, offline public consumers and Chromium
lazy/concurrent/offline parsing before consumption. Successful promotion atomically
replaces model plus calibration and retains the previous module as rollback.

Generated sort work is bounded to 512 MiB. Keep the active corpus, chosen best and
latest checkpoint only while its run is active, rollback and compact unique evidence.
The bounded trainer has no resume command. Remove redundant
artifacts only through verified exact paths, and preserve pre-existing cron work.
