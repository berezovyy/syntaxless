# Validation evaluation

This folder holds held-out, evaluation-only wording for `@syntaxless/validation`.
No row here may be used for training, featurization, checkpoint selection, or
confidence calibration. Corpus preparation excludes every reserved wording by
hash, and the promotion gate consumes each reserved set at most once, for one
frozen model candidate.

## Layout

```text
evaluation/
  README.md                  this protocol (hashed into the gate's scope identity)
  author.ts                  deterministic writer for reserved sets
  dev-negatives.jsonl        20 development negatives; regression only, no promotion claim
  reserved/<set>.jsonl       reserved acceptance rows
  reserved/<set>.jsonl.meta.json   authoring metadata bound to the dataset sha256
  consumed/<sha256>.json     one record per consumed set, named by dataset sha256
```

## Data format

Reserved rows (`reserved/*.jsonl`, one JSON object per line):

| Field                          | Meaning                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `id`, `text`                   | Row id and the sentence to parse                                              |
| `intent`                       | Expected `FieldIntent`, or `null` when the sentence must be refused           |
| `family`, `template`           | `constraint` with the field type, or `negative` with a refusal category       |
| `compileTarget`                | Expected JSON Schema, computed independently of the production compiler       |
| `critical`                     | Control row that must pass for promotion and in every later regression replay |
| `split`, `trainingEligible`    | Always `heldout` and `false`                                                  |
| `spans`, `fingerprint`, `seed` | Corpus-record compatibility fields                                            |

Metadata (`reserved/<set>.jsonl.meta.json`):

| Field                                                               | Meaning                                                     |
| ------------------------------------------------------------------- | ----------------------------------------------------------- |
| `sha256`, `rows`                                                    | Dataset digest and row count, checked by corpus preparation |
| `candidateSha256`, `scopeFreezeSha256`                              | Frozen artifact and scope freeze the set was authored for   |
| `authoringRevision`                                                 | `author.ts` revision argument (absent for revision 1)       |
| `authorSourceSha256`                                                | sha256 of the authoring script that wrote the set           |
| `createdAt`, `authorship`, `independenceLimits`, `trainingEligible` | Provenance                                                  |

Consumed records (`consumed/<sha256>.json`) contain `sha256`, `acceptance` (the
repository-relative dataset path), `candidateSha256`, `run` (the
repository-relative run directory, which is not tracked), `consumedAt`,
`authorship`, and `independenceLimits`.

## Acceptance policy

The gate (`../training/src/gate.ts`) first freezes the candidate and its scope:
runtime, training and Python sources, package configuration, and this README.
A new set is then checked before any model inference:

- metadata matches the dataset digest, frozen candidate, and scope freeze;
- at least 500 positives, 600 negatives, one critical row, and 50 positives for
  each of the seven field types;
- no wording overlaps the training corpus or any consumed set;
- every positive's paired `compileTarget` matches the compiler.

Then typecheck, unit tests, Python tests, numerical parity, the synthetic
heldout gate, and replay of every critical row from previously consumed sets
must pass. Only after that does the gate write `consumed/<sha256>.json`; from
that point the set can never again serve as fresh evidence.

A consumed set passes language acceptance when all of these hold:

| Metric                             | Threshold |
| ---------------------------------- | --------- |
| Positive exact intent match        | ≥ 90%     |
| Wrong meaning (all or accepted)    | ≤ 0.5%    |
| Negative false acceptance          | ≤ 1%      |
| Exact match within each field type | ≥ 85%     |
| Critical failures                  | 0         |

Package, browser, and size checks follow. `promote` then replaces the bundled
model and keeps the previous one for rollback.

`tests/evaluation.test.ts` replays every critical row of consumed sets and all
development negatives against the bundled model on each test run.

## Datasets

All five sets were written by `author.ts` with seed 733201 and are consumed.
Results come from each run's gate report. Runs are not tracked, so only the
current model's result can be reproduced from the repository.

| Set   | sha256       | Rows (pos / neg / critical) | Status   | Result                                                                   |
| ----- | ------------ | --------------------------- | -------- | ------------------------------------------------------------------------ |
| ai-v1 | ee2492c63f9b | 1356 (702 / 654 / 6)        | consumed | failed: 553/702 exact (78.8%), 9/654 false accepts                       |
| ai-v2 | ad7237d93dcd | 1356 (702 / 654 / 6)        | consumed | passed: 691/702 exact, 0 wrong, 0/654 false accepts; later superseded    |
| ai-v3 | e15d9badff82 | 1411 (702 / 709 / 11)       | consumed | failed: 695/702 exact, 1 critical failure                                |
| ai-v4 | b6e104ad23d3 | 1411 (702 / 709 / 11)       | consumed | failed: 702/702 exact, 1/709 false accepts, 1 critical failure           |
| ai-v5 | 0f0f5d3516c8 | 1411 (702 / 709 / 11)       | consumed | passed: 673/702 exact (95.9%), 0 wrong, 0/709 false accepts; **bundled** |

The bundled model is the ai-v5 candidate (`cbd9ec06…`, see the header of
`src/model/weights.gen.ts`). Negative categories in ai-v3 through ai-v5 are
description, question, quotation, negation, history, ambiguity, conflict,
incomplete, approximate bounds, unsafe numbers, inverted ranges, unsupported
units and patterns, and nested objects.

## Authoring a new set

```sh
bun run train:validation                                   # prints the new run directory
bun run packages/validation/training/src/gate.ts prepare <run>
bun run packages/validation/evaluation/author.ts <run> packages/validation/evaluation/reserved/<set>.jsonl <revision>
bun run packages/validation/training/src/gate.ts promote <run> packages/validation/evaluation/reserved/<set>.jsonl
```

`check` runs the same gate without replacing the bundled model. Both modes
consume the set, so run only one of them per set.

`author.ts` takes positional arguments: frozen run, output path, and an
authoring revision (1–5, default 1). It refuses to overwrite files. Each
revision selects different carrier sentences. Every existing revision produced
wording that is now consumed, so a new set needs a new revision with new
wording added to `author.ts` first. The set and its metadata must live in
`reserved/` so later candidates exclude and replay it.

## Limitations

- The same implementation assistant authored all sets with a separate writer
  that imports no training renderer, parser, or model. The sets are not
  independent human evidence and do not estimate population accuracy.
- Sets are generated variations of a limited number of constructions.
- The current `author.ts` reproduces ai-v3, ai-v4, and ai-v5 byte for byte.
  Its revisions 1 and 2 no longer reproduce ai-v1 and ai-v2; the recorded
  `authorSourceSha256` values identify the earlier script versions.
- `dev-negatives.jsonl` has no metadata and is not excluded from corpus
  generation by hash; it supports no promotion claim.
