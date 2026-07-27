# Retention training

Run `bun run train:retention` from the repository root. The coordinator prepares
8,000 seeded constructions, checks exact policies through the production
resolver, groups structural fingerprints before splitting, and trains hidden
width 32 for 24 epochs. It uses the existing cron Python environment and shared
bounded trainer, tokenizer, codec and inference calculations.

Numbers, units, bucket words and calendar/elapsed markers remain preserved input
facts. The model predicts roles, and TypeScript resolves exact counts and units.
Base constructions receive their split before negative wording or context
changes. Numeric substitutions and statement/question variants remain in one
group. Preparation checks ownership across all splits before writing one split.

Generation streams serialized records into unique timestamped corpus paths and
refuses to overwrite evidence. Verified reserved wording remains excluded without
using evaluation labels. The private shared writer keeps at most 8192 tokens,
with retention's semantic boundaries and whitespace padding unchanged. Bounded
row records, deduplication and frame metadata remain O(rows), rather than retaining
all token feature vectors.
The corpus varies bucket forms, representative-count wording, modal requests,
calendar conversion and elapsed windows. Checkpoint selection measures decoded
intent accuracy and wrong accepts rather than token accuracy. Validation alone
selects the confidence floor, and export parity checks include padding behavior.

```sh
bun run packages/retention/training/src/gate.ts prepare <run-directory>
bun run packages/retention/evaluation/author.ts <run-directory>/scope-freeze.json <new-reserved.jsonl> <revision>
bun run packages/retention/training/src/gate.ts promote <run-directory> <new-reserved.jsonl>
```

The gate freezes the complete supported families and source/artifact identity
before authoring. It checks independently calculated UTC previews, hand-checked
zoned witnesses, every prior critical control, actual packed consumers, real
Chromium, lazy/offline inference and aggregate Brotli size. Reserved hashes are
checked against complete datasets or verified wording archives. Before fresh
inference, available full consumed sets must preserve every previously correct
public result; older compact archives preserve the prior critical controls.
Reserved hashes are consumed once, and promotion atomically replaces weights plus calibration while
retaining rollback. The package never executes removal recommendations.

Consumed rows cannot enter training, selection or calibration. The same
implementation assistant authored the evaluation constructions; numeric
variations are not independent human evidence or population accuracy. Failed
preflight checks occur before reserved inference, and an unconsumed set may be
bound to a later frozen candidate with new metadata.

Generated work is bounded to 512 MiB. Retire only verified exact inactive paths
and byte-identical copies, keeping active corpus/checkpoint, frozen exports and
unique evidence. The trainer has no resume path, so an unselected last checkpoint
is not a resumable artifact. Full corpora and runs remain ignored by Git.

The active recipe is `compact-v8-8k`, with construction grouping before negative
mutation and compositional policy requests. The complete gate promoted
`requests32-2026-10-12T00-44-10-723Z`, with 761/761 fresh positives and no negative
accepts. Its previous weight module preserves the earlier known-good ten-family
artifact and calibration.

Compact consumed archives preserve critical regressions and all wording hashes.
Historical full acceptance rows remain locally available but ignored. Archive
integrity and any present original file integrity are verified before reuse.
No consumed wording enters training, calibration or fresh evidence.
