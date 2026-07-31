# Format evaluation

Reserved acceptance sets measure whether a frozen format model candidate parses
held-out English instructions into the exact intended intent, and refuses
instructions it must not accept. Every row is evaluation-only
(`trainingEligible: false`). Reserved wording is never used for training,
checkpoint selection or calibration. Corpus generation also excludes reserved and
consumed wording by normalized-text hash.

## Layout

```
evaluation/
  README.md                 this protocol
  author.ts                 deterministic writer for new reserved sets
  fixtures.ts               fixed, hand-specified formatter witnesses
  reserved/<set>.jsonl      reserved rows (only unconsumed or retained sets are tracked)
  reserved/<set>.jsonl.meta.json   provenance for every set, tracked even when rows are not
  consumed/<sha256>.json           consumption record for one set
  consumed/<sha256>.archive.json   compact archive: critical rows and all wording hashes
```

## Data format

Each reserved row is one JSON object:

| Field                                      | Meaning                                                                        |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| `text`                                     | English instruction                                                            |
| `intent`                                   | Expected `FormatIntent`, or `null` when the parser must refuse                 |
| `family`                                   | One of ten positive families or a negative category                            |
| `critical`                                 | `true` for controls that must pass regardless of aggregate scores              |
| `compileOptions`                           | Caller context (`locale`, `timeZone`) for the paired compiler check            |
| `compileTarget`                            | Expected formatter options and probe outputs, or an expected refusal           |
| `context`, `roles`, `fingerprint`, `split` | Row-shape fields shared with the training corpus (`split` is always `heldout`) |
| `trainingEligible`                         | Always `false`                                                                 |

Formatter targets are calculated by `author.ts` with native `Intl`. The writer
does not import the production parser, compiler or training renderer.

`<set>.jsonl.meta.json` fields:

| Field                              | Meaning                                                       |
| ---------------------------------- | ------------------------------------------------------------- |
| `sha256`, `rows`                   | Digest and row count of the exact dataset bytes               |
| `candidateSha256`                  | Frozen candidate artifact the set was authored for            |
| `scopeFreezeSha256`                | Digest of that candidate's `scope-freeze.json`                |
| `authorSha256`                     | Digest of `author.ts` that wrote the set (newer sets only)    |
| `createdAt`                        | Authoring time; must follow the freeze                        |
| `authorship`, `independenceLimits` | Required provenance statements                                |
| `localeData`                       | Runtime versions (`process.versions`), including ICU          |
| `reboundFrom`, `reason`            | Present when unchanged wording was bound to a later candidate |

`consumed/<sha256>.json` fields: `sha256`, `acceptance` (repository-relative
dataset path), `candidateSha256`, `consumedAt`, `archiveSha256` (digest of the
archive file), `authorship` and `independenceLimits`. The archive holds `sha256`,
`rows`, every `critical` row and the sorted `textHashes` of all normalized
wording.

## Acceptance policy

`training/src/gate.ts promote` enforces the following, in order:

1. Source, scope, corpus and candidate identity match the freeze. The frozen
   identity includes this README and `fixtures.ts`.
2. Meta provenance matches: dataset digest, candidate, scope freeze, author
   digest, row count, and `createdAt` after the freeze.
3. Rows are unique and evaluation-only, with at least 500 positives, 600
   negatives, 50 positives per family and one critical row.
4. No wording overlaps the training corpus or any consumed set.
5. Every positive intent and every fixed witness compiles to its paired target.
   Archived critical rows from all consumed sets are re-checked too.
6. Typecheck and tests pass, export parity holds, and the synthetic heldout gate
   passes. The synthetic gate requires 95% exact positives, at most 0.5% wrong
   meanings, at most 1% negative false accepts and at least 90% per family.
7. Every archived critical row from earlier consumed sets is parsed correctly.
8. The packed tarball, offline Chromium check and size gate pass.

The set is **consumed** at this point: the archive and record are written before
inference, so a set evaluates exactly one candidate. The language gate then
requires all of the following:

| Metric                             | Threshold |
| ---------------------------------- | --------- |
| Positive exact intent              | ≥ 90%     |
| Wrong meanings, all positives      | ≤ 0.5%    |
| Wrong meanings, accepted positives | ≤ 0.5%    |
| Negative false acceptance          | ≤ 1%      |
| Exact intent per positive family   | ≥ 85%     |
| Critical failures                  | 0         |

A passing gate promotes the frozen weights atomically and rechecks the packed
API. Otherwise it restores the previous module. Consumed sets remain regression
evidence only: their archived critical rows are replayed by every later gate.

## Datasets

Each set has 10 positive families. Results come from the gate report of the
candidate the set was bound to.

| Set            | sha256       | Rows (+/−) | Critical | Status   | Result                                                          |
| -------------- | ------------ | ---------- | -------- | -------- | --------------------------------------------------------------- |
| ai-v1          | 2a64658a37cb | 765/640    | 25       | consumed | 607/765 exact, 0/640 FA, 0 wrong, 4 critical failures; rejected |
| ai-v2-bound-v4 | a6630ba8f2fa | 767/642    | 29       | consumed | 536/767 exact, 0/642 FA, 0 wrong; rejected (exact rate)         |
| ai-v3          | 6d27fe2ba36c | 767/642    | 29       | consumed | 767/767 exact, 0/642 FA, 0 wrong; promoted                      |
| ai-v4          | 49ea54201d3b | 767/642    | 29       | consumed | 767/767 exact, 0/642 FA, 0 wrong; promoted (bundled model)      |

- `ai-v2-bound-v4` is the unchanged `ai-v2` wording. `ai-v2` was authored for an
  earlier candidate that was never evaluated, then rebound to the next frozen
  candidate. Its first binding is recorded only in Git history.
- Full rows of `ai-v1`, `ai-v2-bound-v4` and `ai-v3` are kept outside Git. Their
  tracked archives preserve every critical row and wording hash, which is all
  that training exclusion and the gate need. `ai-v4` is tracked in full.

## Authoring a new set

```sh
# 1. Freeze the candidate before any reserved wording exists.
bun run packages/format/training/src/gate.ts prepare <run-directory>
# 2. Write a new set bound to that freeze.
bun run packages/format/evaluation/author.ts <run-directory>/scope-freeze.json \
  packages/format/evaluation/reserved/ai-v<N>.jsonl <revision>
# 3. Run the gate. This consumes the set.
bun run packages/format/training/src/gate.ts promote <run-directory> \
  packages/format/evaluation/reserved/ai-v<N>.jsonl
```

Use the next unused set number. Never edit a written set: a change of wording is
a new set, and consumed wording cannot be reused.

## Limitations

- All sets are AI-authored by the same implementation assistant that built the
  parser, using bounded constructions and numeric variation. They are not
  independent human evidence or a population accuracy estimate.
- Formatter targets use the runtime's own `Intl` locale data. Fixed witnesses and
  the browser check add evidence, but output can differ across engines and ICU
  versions.
- Gate reports live in ignored run directories. Results above can be checked
  against the consumption records but not recomputed from tracked files alone,
  except for `ai-v4`.
