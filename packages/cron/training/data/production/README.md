# Production corpus sources (in progress)

The authoritative goal is [`CORPUS-GOAL.md`](../../../../../CORPUS-GOAL.md).
This directory contains versioned **source specifications**, not a finished
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

## Validation commands

From the repository root, using a new report path:

```sh
bun packages/cron/training/src/corpus/production/source-audit.ts \
  packages/cron/training/results/whole-utterances-audit.json

bun test packages/cron/training/tests/production-constructions.test.ts
```

The source audit renders **every** form/value combination. It checks every
published candidate's supervision and split isolation, separately decodes the
discarded correction clauses, excludes protected wording without using that
wording to author examples, and reports source hashes, coverage, and repetition.
It does not write a final training corpus or enforce the final R/H/E mix.

## Remaining production work

- Continue source language review as coverage expands. Bare factual negatives,
  punctuation joins, explicit recurrence labels, and per-form tags were reviewed
  in the first pass; that is not a claim of independent language validation.
- Integrate these constructions with systematic generator coverage and actual
  sampling weights; do not publish this candidate inventory as the final mix.
- Preserve the systematic catalog's full numeric and selector domains during
  integration, and review broader correction value pairings and range boundaries.
  Both grids now include all 42 directed weekday ranges, and the coverage matrix
  requires timed and untimed evidence for every endpoint pair in every split.
  The older exhaustive report predates this expansion; the strengthened matrix
  must pass on the actual selected production artifacts.
- Select from the expanded, attributed public-English source pool with explicit
  concentration limits and no replacement; retain the full pool for traceability.
- Publish a complete versioned corpus, manifest, and all feature shards through
  a resource-checked, failure-safe path, then validate Python consumption.

Generator-heldout examples are development evidence, **not independent final
acceptance data**. Genuine consented scheduling requests are still absent.

## Complete public-English source builder

The importer uses `tatoeba-v2.json` to pin both detailed sentences and their
English-English links. From the repository root, with a new output directory:

```sh
bun packages/cron/training/src/corpus/production/public-source.ts \
  packages/cron/training/data/synth/production-sources/tatoeba-english-v2
```

Successful publication produces `corpus.jsonl`, `source.sqlite`, `validation.json`
and `manifest.json`. This is the full eligible source pool. The SQLite file is a
legacy importer/index artifact; the complete-corpus path reads `corpus.jsonl`
and does not use that database for candidate selection or training.

A build refuses changed archive checksums, changed source/configuration during
execution, malformed input, bad supervision, insufficient space, or an existing
output version. Failed runs never publish a partial final directory. The legacy
50k pin is not overwritten. Final corpus/shard publication remains part of the
production integration work.

## Complete clean catalog inventory

`catalog.ts` traverses every applicable clean template/value pair in the existing
systematic generator, without the carrier Cartesian product or the former seed
cap. It emits **unassigned candidates**, not production-owned rows. All finite
numeric/selector domains and formerly held-out clean templates are inspected:
407,668 kind/intent combinations and 7,721,878 possible template/value attempts.

```sh
bun packages/cron/training/src/corpus/production/catalog-audit.ts \
  packages/cron/training/results/systematic-catalog-audit.json
```

The audit uses a temporary all-train assignment solely to check every candidate's
supervision and identify conflicting labels. Its report explicitly does **not**
certify split isolation. The complete-corpus path includes every eligible unique
rendered record. Global normalized-text deduplication removes overlap; R/H/E
exposure is controlled by the training sampler instead of deleting rows here.

## Integration and complete shard consumption

See [INTEGRATION.md](INTEGRATION.md) for the complete-union/publication path.
`complete-sources.ts` and `global-union.ts` provide canonical traversal and
bounded-memory deduplication without a candidate database.

For a complete generated JSONL and its split-shard prefix, source-derived
expectations and Python consumption verification are available:

```sh
bun packages/cron/training/src/corpus/production/source-expectations.ts \
  CORPUS.jsonl NEW_EXPECTATIONS.json
training/cron/.venv/bin/python training/cron/verify_production_shards.py \
  CORPUS.jsonl SHARD_BASE NEW_EXPECTATIONS.json NEW_VERIFICATION.json
```

Expectations come from the actual JSONL, not from copying shard checksums. The
Python verifier checks all fields and every sequence/batch; it does not train or
promote a model. Use new output paths; existing evidence is not overwritten.
