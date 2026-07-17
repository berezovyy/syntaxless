# Production corpus inputs

Training uses `bun run train` from the repository root; see the
[training workflow](../../README.md). This directory contains versioned
**source specifications**, not a finished
corpus, an acceptance set, or evidence of model improvement.

## Whole-utterance constructions

`constructions-v1.json` contains 65 source groups and 431 complete forms. Every
form is AI-authored. The source has undergone AI-author self-review of the patterns and rendered
group representatives, with exhaustive automated validation recorded separately.
It has not been independently or human-reviewed.
Unlike carrier-prefix expansion, these forms place tasks between semantic
components, reverse component order, separate schedule fields across clauses,
ask for changes to two competing schedules, and contrast requests with reports,
questions, quotes, cancellations, single executions, and unfinished requests.

The source syntax is deliberately explicit:

- `{clock}`, `{selector}`, etc. insert role-annotated values supplied by a semantic
  case. Missing slots fail rather than becoming ordinary prose.
- `[UNIT_DAY:daily]` inserts the specified role on a fixed phrase.
- `[CONTEXT_NUMBER:17]` explicitly marks a background ID/count. It is numeric,
  non-actionable and cannot begin a schedule clause. Plain prose digits remain
  unknown operands; the renderer never guesses their relevance.
- `{!old}` inserts genuine but non-actionable discarded schedule evidence.
- `{#}` begins a new semantic clause.
- All remaining words are non-actionable prose. Full sentence normalization and
  span alignment occur after rendering.

`src/corpus/production/cases.ts` supplies finite semantic grids, independently of
model predictions. They include every hour/minute pair for daily clocks, all
seven weekdays, weekday pairs and ranges, all 31 monthly dates, all five ordinal
positions, leap/year/month boundary dates, exceptions, and changed values for
corrections. Unsupported word-valued literal roles are deliberately not emitted.

The group is the split unit: its value substitutions, wording variants and hard
negatives remain together. All groups have an explicit split. Separate identity
checks still reject cross-split text and fingerprint collisions; an invented
new group identifier is not permission to leak an old construction.

## Source review

The earlier exhaustive source audits rendered every form/value combination and
recorded supervision, coverage, isolation and repetition evidence. Their
standalone runners have been retired. Construction checks and discarded-clause
validation remain in the production renderer and its fixtures. `bun run train`
owns corpus preparation; earlier audit reports are historical evidence.
The latest refactoring checked all 2,592,141 authored rows with perfect labels,
including refused requests' inner schedules, with zero decoder mismatches.
This is semantic round-trip evidence, not complete corpus integrity or model
performance.

## Remaining production work

- Continue source language review as coverage expands. Bare factual negatives,
  punctuation joins, explicit recurrence labels, and per-form tags were reviewed
  in the first pass; that is not a claim of independent language validation.
- The complete union combines these constructions with systematic generator
  coverage and pinned sources. A source inventory alone is not the final corpus.
- Preserve the systematic catalog's full numeric and selector domains during
  integration, and review broader correction value pairings and range boundaries.
  Both grids now include all 42 directed weekday ranges, and the coverage matrix
  requires timed and untimed evidence for every endpoint pair in every split.
  The older exhaustive report predates this expansion; the strengthened matrix
  must pass on the actual selected production artifacts.
- The workflow samples public-English records at the specification's
  `sampling.public` rate and retains source attribution. Population counts are
  reported without oversampling.
- The workflow publishes the versioned corpus, manifest, and feature shards
  after resource checks and complete Python consumption verification.

Generator-heldout examples are development evidence, **not independent final
acceptance data**. Genuine consented scheduling requests are still absent.

## Synthetic source preparation

`bun run train` builds or reuses the manual synthetic source internally. Its
immutable directory is keyed by the fixed recipe, writer/publisher dependencies,
Tatoeba input files, and protected wording. Existing legacy snapshots are left
intact. No standalone manual-generation, mixture, or merge command is required.
Both union planning and emission replay the prepared artifact. Traversal rejects
missing preparation instead of running a second live generator.
The writer only creates the two canonical files in the coordinator's staging
directory. One directory publisher owns locks, atomic visibility and rollback.
Both source publication and resumable corpus phases hold the same native lock
protocol. A persistent sibling `.lock.sqlite` inode provides coordination;
process death releases ownership without PID-file recovery or unlinking.
Temporary fixtures verify this structure and artifact bytes; full production
preparation and model training remain unexecuted.

## Pinned public-English input

The workflow consumes the already published public-English `corpus.jsonl` and
its pinned `manifest.json`. Before use, preparation checks the manifest checksum
and the corpus bytes and checksum recorded in it. Records retain attribution,
source IDs and their existing split assignments.

The old source-import builders and downloader have been removed. The historical
`tatoeba-v2.json`, archives, reports and `source.sqlite` document how the input was
produced; training does not rebuild or consume the SQLite index. The separate
local sentence/manifest pin remains the required easy-negative input.

Missing or changed pinned inputs stop `bun run train`. Final corpus and shard
preparation still belongs to that command, with independent verification before
publication. Earlier source counts and reproduction evidence are preserved in
[PUBLIC-SOURCES.md](PUBLIC-SOURCES.md).

## Complete clean catalog inventory

`catalog.ts` can traverse every applicable clean template/value pair in the
existing systematic generator, without the carrier Cartesian product or the
former seed cap. Production samples (template, intent) pairs at
`sampling.systematic` before rendering, which keeps every template while
skipping most of each value grid. It emits **unassigned candidates**, not production-owned rows. All finite
numeric/selector domains and formerly held-out clean templates are inspected:
407,668 kind/intent combinations and 7,721,878 possible template/value attempts.

The earlier catalog audit used a temporary all-train assignment solely to check
supervision and identify conflicting labels. That historical report does **not**
certify split isolation. The production path includes every sampled, rendered
record. Global normalized-text deduplication removes overlap. Training
visits each row once per epoch; population metadata is retained for reporting.

## Integration and complete shard consumption

See [INTEGRATION.md](INTEGRATION.md) for the complete-union/publication path.
`complete-sources.ts` and `global-union.ts` provide canonical traversal and
bounded-memory deduplication without a candidate database.

The training workflow derives expectations from the emitted JSONL, then invokes
Python consumption verification before publication. Expectations are not copied
from shard checksums. The verifier checks all fields and every sequence/batch;
it does not train or promote a model. Existing evidence is not overwritten.
