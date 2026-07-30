# Sort evaluation

Held-out acceptance sets for `@syntaxless/sort`. Every row is evaluation-only
(`trainingEligible: false`). Reserved wording is never used for training,
checkpoint selection or calibration. Corpus generation also skips any text
whose normalized hash appears in a reserved or consumed set.

## Layout

```text
evaluation/
  README.md       this protocol
  author.ts       deterministic writer for reserved sets (no parser, compiler, model or training code)
  fixtures.ts     eight fixed witness rows for paired SQLite/JavaScript order checks
  reserved/       <set>.jsonl and <set>.jsonl.meta.json
  consumed/       <sha256>.json, one record per set used by a promotion gate
```

## Data format

Each line of a reserved set is one JSON row:

| Field              | Meaning                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `text`             | Request wording                                                          |
| `context`          | Caller catalog (`fields`, `defaultNulls`, `tieBreaker`)                  |
| `intent`           | Expected `SortIntent`, or `null` when the parser must refuse             |
| `family`           | Semantic family (for example `priority`, `null-update`, `contradiction`) |
| `critical`         | Hand-checked control that must pass exactly                              |
| `compileTarget`    | Expected witness `id` order for positives, `null` for negatives          |
| `split`            | Always `"heldout"`                                                       |
| `fingerprint`      | Language-construction hash                                               |
| `roles`            | Always `[]`; reserved rows carry no role supervision                     |
| `trainingEligible` | Always `false`                                                           |

`<set>.jsonl.meta.json` records `sha256` (of the dataset bytes), `rows`,
`candidateSha256` and `scopeFreezeSha256` (the frozen candidate it was authored
for), `createdAt`, `authorship`, `independenceLimits` and
`trainingEligible: false`.

`consumed/<sha256>.json` records `sha256`, `acceptance` (repository-relative
dataset path), `candidateSha256`, `consumedAt`, `authorship` and
`independenceLimits`. The file name is the dataset hash.

The dataset bytes are hash-bound: editing a reserved `.jsonl` file breaks
`training/src/gate.ts` and `packages/shared/training/reserved.ts`.

## Acceptance policy

`training/src/gate.ts promote` runs once per reserved set and frozen candidate:

1. The frozen identity (artifact, corpus, export report, TypeScript and Python
   sources, package configuration, this README and `fixtures.ts`) must match
   `scope-freeze.json`.
2. Metadata must name the same candidate and scope freeze, and must postdate it.
3. Coverage: at least 500 positives, 600 negatives, one critical row, and
   50 positives in each of the 7 families (`explicit-direction ordering-alias
priority null-first null-last null-update caller-default`).
4. No row may overlap training wording or any consumed set.
5. Every positive must compile, and actual SQLite `ORDER BY` and `sortRecords`
   must both produce `compileTarget` over the witness rows.
6. The synthetic held-out gate, runtime/PyTorch parity, prior critical
   regressions, the packed package check, the browser check and the size gate
   must pass. Only then is the set marked consumed.
7. Language acceptance requires exact positives ≥ 90%, wrong meanings ≤ 0.5%
   of positives and of accepted positives, negative false acceptance ≤ 1%,
   every family ≥ 85% exact, and zero critical failures.

A consumed set never authorizes another candidate. The gate replays its
critical rows as regressions for later candidates.

## Datasets

| Set                   | sha256       | Rows | Status                         | Result                                                                       |
| --------------------- | ------------ | ---- | ------------------------------ | ---------------------------------------------------------------------------- |
| `ai-v1.jsonl`         | 3540b8c2dfcd | 1185 | rejected (overlap), unconsumed | Not evaluated; one row overlapped training wording                           |
| `ai-v1-revised.jsonl` | 0987bacc9e60 | 1185 | consumed                       | 288/548 exact, 0 wrong, 180/637 FA, 9/9 critical; failed                     |
| `ai-v2.jsonl`         | 5e70ec6b7a66 | 1185 | consumed                       | 548/548 exact, 0 wrong, 0/637 FA, 9/9 critical; passed (superseded)          |
| `ai-v3.jsonl`         | 07716304f1ec | 1185 | consumed                       | 548/548 exact, 0 wrong, 0/637 FA, 9/9 critical; bundled model `297807645b8b` |

Each set has 548 positives (at least 78 per family), 637 negatives and 9
critical rows. The rejected set stays tracked because its wording is excluded
from training generation. `author.ts` revisions 1, 2 and 3 reproduce
`ai-v1-revised`, `ai-v2` and `ai-v3` byte for byte.

## Authoring a new set

Add a new revision branch to `author.ts` with wording that does not appear in
any existing set, then:

```sh
bun run packages/sort/training/src/gate.ts prepare <run-directory>
bun run packages/sort/evaluation/author.ts <run-directory>/scope-freeze.json packages/sort/evaluation/reserved/<set>.jsonl <revision>
bun run packages/sort/training/src/gate.ts promote <run-directory> packages/sort/evaluation/reserved/<set>.jsonl
```

The writer refuses to overwrite existing files.

## Limitations

- The implementation assistant authored these sets through a separate writer.
  Caller identifiers supply much of the variation. The sets are not independent
  human evidence or a population accuracy estimate.
- Passing critical controls alone is insufficient: `ai-v1-revised` passed all
  nine while failing coverage and false acceptance.
- Result figures come from the gate report written at consumption time; the
  consumed record binds only the dataset and candidate hashes.
- `ai-v1.jsonl` is not reproducible from the current `author.ts`.
