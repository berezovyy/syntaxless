# Filter evaluation

Held-out acceptance sets for `@syntaxless/filter`. Every row is evaluation-only
(`trainingEligible: false`). Reserved wording is never used for training,
checkpoint selection or calibration. Corpus generation also skips any text
whose normalized hash appears in a reserved or consumed set.

## Layout

```text
evaluation/
  README.md       this protocol
  author.ts       deterministic writer for reserved sets (no parser, model or training imports)
  fixtures.ts     eight fixed witness rows for paired SQLite/JavaScript truth checks
  reserved/       <set>.jsonl and <set>.jsonl.meta.json
  consumed/       <sha256>.json, one record per set used by a promotion gate
```

## Data format

Each line of a reserved set is one JSON row:

| Field              | Meaning                                                            |
| ------------------ | ------------------------------------------------------------------ |
| `text`             | Request wording                                                    |
| `context`          | Caller catalog (`fields` with ids, types, aliases, terms, columns) |
| `intent`           | Expected `FilterIntent`, or `null` when the parser must refuse     |
| `family`           | Semantic family (for example `range`, `exclusion`, `question`)     |
| `critical`         | Hand-checked control that must pass exactly                        |
| `compileTarget`    | Positives only: expected truth (`true`/`false`/`null`) per witness |
| `split`            | Always `"heldout"`                                                 |
| `fingerprint`      | Language-construction hash                                         |
| `roles`            | Always `[]`; reserved rows carry no role supervision               |
| `trainingEligible` | Always `false`                                                     |

`<set>.jsonl.meta.json` records `sha256` (of the dataset bytes), `rows`,
`candidateSha256` and `scopeFreezeSha256` (the frozen candidate it was authored
for), `createdAt`, `authorSourceSha256`, `fixtureSha256`, `authoringRevision`,
`authorship`, `independenceLimits` and `trainingEligible: false`.

`consumed/<sha256>.json` records `sha256`, `acceptance` (repository-relative
dataset path), `candidateSha256`, `consumedAt`, `authorship` and
`independenceLimits`. The file name is the dataset hash.

The dataset bytes are hash-bound: editing a reserved `.jsonl` file breaks
`training/src/gate.ts` and `packages/shared/training/reserved.ts`.

## Acceptance policy

`training/src/gate.ts promote` runs once per reserved set and frozen candidate:

1. The frozen identity (artifact, corpus, export report, source hashes including
   this README and `fixtures.ts`) must match `scope-freeze.json`.
2. Metadata must name the same candidate and scope freeze, and must postdate it.
3. Coverage: at least 500 positives, 600 negatives, one critical row, and
   50 positives in each of the 14 families (`eq ne lt le gt ge in range null and
or exclusion not group`).
4. No row may overlap training wording or any consumed set.
5. Every positive must compile, and actual SQLite and the JavaScript evaluator
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
| `ai-v1.jsonl`         | 52120e3ba31f | 1547 | consumed                       | 456/842 exact, 0 wrong, 0/705 FA, 1 critical failure; failed coverage        |
| `ai-v2.jsonl`         | a89913144a37 | 1547 | rejected (overlap), unconsumed | Not evaluated; wording overlapped `ai-v1`                                    |
| `ai-v2-revised.jsonl` | 8a8d6f2f915f | 1547 | consumed                       | 842/842 exact, 0 wrong, 0/705 FA, 7/7 critical; bundled model `d19f18105385` |

Each set has 842 positives, 705 negatives and 7 critical rows. The rejected set
stays tracked because its wording is excluded from training generation.
`author.ts` revisions 1 and 2 reproduce `ai-v1` and `ai-v2-revised` byte for
byte.

## Authoring a new set

Add a new revision branch to `author.ts` with wording that does not appear in
any existing set, then:

```sh
bun run packages/filter/training/src/gate.ts prepare <run-directory>
bun run packages/filter/evaluation/author.ts <run-directory> packages/filter/evaluation/reserved/<set>.jsonl <revision>
bun run packages/filter/training/src/gate.ts promote <run-directory> packages/filter/evaluation/reserved/<set>.jsonl
```

The writer refuses to overwrite existing files.

## Limitations

- The implementation assistant authored these sets through a separate writer.
  They are bounded generated variations, not independent human evidence or a
  population accuracy estimate.
- Result figures come from the gate report written at consumption time; the
  consumed record binds only the dataset and candidate hashes.
- `ai-v2.jsonl` is not reproducible from the current `author.ts`.
