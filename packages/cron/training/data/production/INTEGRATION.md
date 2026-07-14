# Complete-corpus integration design

The production corpus is the deterministic union of every eligible unique
record, not a quota-selected sample. This document describes the replacement
path; passing a source audit by itself is not evidence that the final corpus was
published.

## Canonical inputs

`complete-sources.ts` traverses, in a frozen order:

1. the normal/manual generator;
2. authored constructions;
3. the complete systematic catalog;
4. the complete pinned public-English JSONL;
5. verified LLM records; and
6. accepted human records when honest per-record provenance exists.

Protected evaluation and calibration wording is excluded. The current accepted
real-record loader deliberately refuses to invent human provenance, so that
source contributes zero rows until properly attributed records exist.

## Union instead of selection

`global-union.ts` keys records by runtime-normalized text. Equal text with
conflicting supervision is an error. Equal text with equal supervision keeps
one deterministic, provenance-aware winner and preserves that record's split.

The union uses partitioned fixed-width hash records and a compact selected-row
bitmap, followed by one deterministic source replay. Candidate JSON is never
copied into a database or retained in memory. This replaces the former SQLite
candidate index, ownership graph, coverage-winner selection, public quota
selection, and per-cell evidence artifact.

Corpus generation never removes unique rows to force an R/H/E ratio. The
featurizer records one population byte per sequence and the training sampler
controls exposure with `--population-mix`, for example:

```sh
--population-mix 0.68,0.22,0.10
```

## Publication and verification

- Preflight disk space and use same-filesystem staging with atomic publication.
- Never overwrite an existing corpus version or publish partial output.
- Write the complete unique JSONL and train/validation/held-out shards.
- Record source, duplicate, population, split, provenance, exclusion, input-hash,
  and artifact-hash evidence in the manifest.
- Derive shard expectations from the emitted JSONL and verify complete Python
  loader consumption. Reproduce the artifact from unchanged inputs before
  declaring deterministic completion.

The public-source `source.sqlite` remains an importer/index artifact belonging
to the already published public pool. The complete corpus consumes that pool's
`corpus.jsonl`; SQLite is not part of the replacement union or model training.
