# Budgeted-corpus integration design

The production corpus is the deterministic union of per-source budgets. Each
source contributes a configured amount, the union removes exact duplicates, and
the result is published in text-hash order. This document describes that path;
passing a source audit by itself is not evidence that the final corpus was
published.

## Canonical inputs

`complete-sources.ts` traverses, in a frozen order:

1. the manual mixture, generated to `manual.records` rows by the training
   workflow (schedule rows only; see below);
2. authored constructions, hash-sampled at `sampling.authored`;
3. the systematic catalog, hash-sampled per (template, intent) at
   `sampling.systematic` before rendering;
4. the pinned public-English JSONL, hash-sampled at `sampling.public`;
5. verified LLM records; and
6. accepted human records when honest per-record provenance exists.

Protected evaluation and calibration wording is excluded. Accepted real records
require explicit human provenance and model-training consent. The loader refuses
to invent either.

## Budgets instead of exhaustive grids

The corpus size used to be set by constants: a 20,000,000-row manual mixture
and every intent × template of the catalog's value grids (every minute of the
day × every weekday set). Those rows mostly repeat one supervision pattern with
another carrier phrase or literal value.
The specification now sets explicit budgets:

- `manual.records` rows of request (R) and non-actionable (H) wraps. The mixer
  no longer cycles its 50k-line easy-negative list: the public source contains
  every one of those sentences and won the union for all but 42 of 2M rows.
  Negatives come only from the public source.
- `manual.spellingNoise`: the share of manual rows replaced by one
  meaning-preserving typo (`src/noise.ts`; literal digits stay clean). Before
  this, typo robustness came only from 95 authored typo forms.
- `sampling.*`: keep rates for sources whose size is fixed by their inputs.
  A row is kept when the first 32 bits of SHA-256 of its source name and stable
  key fall below the rate. The same inputs always keep the same rows, splits
  are preserved, and sampling the catalog before rendering skips the work.

Sampling at equal rates preserves each source's share and the negative share.
The budgets are chosen from measured size-vs-quality evidence, not maximized.

## Union instead of selection

`global-union.ts` keys records by runtime-normalized text. Equal text with
conflicting supervision is an error. Equal text with equal supervision keeps
one deterministic, provenance-aware winner and preserves that record's split.

The union uses partitioned fixed-width hash records and an ascending file of
winning candidate ordinals, followed by one deterministic source replay.
Candidate JSON is never copied into scratch or retained in memory. This
replaces the former SQLite candidate index, ownership graph, coverage-winner
selection, public quota selection, and per-cell evidence artifact.

Every step keeps memory independent of the candidate count (the
[memory rule](../../README.md#memory-rule)). A partition file larger than the
32 MiB reduce window is split 16 ways on further hash bits, recursively, before
it is reduced; a text repeated past the window is reduced by streaming. Each
reduced window writes its winners as a sorted run of ordinals, and runs merge on
disk 64 at a time. Replay streams the merged file. Neither the partition count
nor the window changes the result.

Corpus generation never removes unique rows to force an R/H/E ratio. The
featurizer records population metadata for reporting. The single training
workflow visits each unique row once per epoch, without population oversampling.

Emission writes the union's winners in ascending text-hash order rather than
source traversal order. The trainer shuffles only inside bounded locality
blocks, so a source-ordered file trained on long single-source stretches and
its final epoch state depended on which source came last. The order comes from
an external merge sort (`hash-order.ts`): sorted runs of at most 16M characters
are written to disk and merged with a fixed fan-in, so emission memory does
not grow with the corpus.

The manual generator rejects a draw whose fingerprint or text it has already
emitted. That index is a SQLite file in the staging directory with a fixed page
cache (`mix/disk-hash-set.ts`), not an in-memory set, and is deleted before
publication.

From the repository root, `bun run train` performs all preparation phases and
training internally. There are no separate build-phase commands; budgets live
only in the fixed specification.
Synthetic source paths are derived from the fixed recipe and input/code identity;
users do not generate or choose that intermediate artifact separately.

## Publication and verification

- Preflight disk space and use same-filesystem staging with atomic publication.
- Never overwrite an existing corpus version or publish partial output.
- Write the unique JSONL (text-hash order) and train/validation/held-out shards.
- Record source, duplicate, population, split, provenance, exclusion, input-hash,
  and artifact-hash evidence in the manifest.
- Derive shard expectations from the emitted JSONL and verify complete Python
  loader consumption. Reproduce the artifact from unchanged inputs before
  declaring deterministic completion.

The public-source `source.sqlite` remains an importer/index artifact belonging
to the already published public pool. The complete corpus consumes that pool's
`corpus.jsonl`; SQLite is not part of the replacement union or model training.
Preparation uses Bun's built-in SQLite for an exclusive native lock on a
persistent sibling file and for the manual generator's temporary digest index.
Neither stores candidate or training records.
