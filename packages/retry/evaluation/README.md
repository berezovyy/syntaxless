# Retry evaluation

Reserved acceptance sets measure whether a frozen model candidate can be promoted.
Their wording is evaluation-only. It never enters training, checkpoint selection or
calibration, and corpus preparation excludes every reserved sentence by its
normalized text hash. Each set provides fresh evidence exactly once. After a set is
consumed, it serves only as regression evidence for later candidates.

## Layout

```text
evaluation/
  README.md        this protocol; its hash is part of each frozen candidate identity
  author.ts        evaluation-only writer; imports no parser, compiler, model or training renderer
  fixtures.ts      shared jitter seed used by paired schedule targets
  reserved/        <set>.jsonl datasets and <set>.jsonl.meta.json provenance
  consumed/        <sha256>.json consumption records (one per consumed dataset hash)
```

## Data format

Each line of a reserved dataset is one row:

| Field                             | Meaning                                                                    |
| --------------------------------- | -------------------------------------------------------------------------- |
| `text`                            | Request wording                                                            |
| `context`                         | Parse context, such as caller `defaults`                                   |
| `intent`                          | Expected `RetryIntent`, or `null` when the request must refuse             |
| `family`                          | Positive family (see `training/src/corpus.ts` `FAMILIES`) or negative kind |
| `critical`                        | Hand-checked control; any failure blocks promotion                         |
| `compileOptions`, `compileTarget` | Seed and independently calculated schedule (or refusal code) for `compile` |
| `split`, `trainingEligible`       | Always `"heldout"` and `false`                                             |
| `roles`, `fingerprint`            | Unused role list and construction fingerprint                              |

Metadata (`<set>.jsonl.meta.json`): `sha256` and `rows` of the dataset, the
`candidateSha256` and `scopeFreezeSha256` it is bound to, `createdAt`,
`trainingEligible: false`, `authorship` and `independenceLimits`.

Consumption record (`consumed/<sha256>.json`): `sha256`, the repository-relative
`acceptance` path, `candidateSha256`, `consumedAt`, `authorship`,
`independenceLimits`, and, for sets consumed by the current gate, `archiveSha256`.
An archive (`<sha256>.archive.json`) stores the critical rows and the hash of every
normalized row text, so wording exclusion and critical replay survive without the
full rows.

## Acceptance policy

`training/src/gate.ts promote` performs these checks in order:

1. **Provenance.** The candidate's source, artifact, corpus and evaluation identity
   must match its scope freeze. The metadata must match the dataset hash, candidate
   and freeze, and must be created after the freeze.
2. **Set validity.** Rows must be unique by text and context, held-out and
   evaluation-only. The set needs at least 500 positives, at least 600 negatives,
   at least 50 positives per family and at least one critical row. It must share no
   wording with the training corpus or any consumed set, and every paired schedule
   target must equal `compile`.
3. **Pre-consumption checks:**
   - typecheck and tests;
   - frozen parity;
   - the synthetic held-out gate (`training/src/evaluate.ts`);
   - regression replay of consumed sets: every positive the active model gets right
     and every negative it refuses must keep that result, and every consumed
     critical row must pass;
   - the packed package consumer, Chromium and size gates.
4. **Consumption.** The gate writes the consumption record before inference, so a
   set is never reused as fresh evidence.
5. **Language thresholds:**
   - exact positive intents ≥ 90%, and ≥ 85% in every family;
   - wrong meanings ≤ 0.5% of all positives and ≤ 0.5% of accepted positives;
   - negative false acceptance ≤ 1%;
   - zero critical failures.
6. **Promotion.** A pass atomically replaces `src/model/weights.gen.ts`, keeps the
   previous module as rollback, and reruns `check:package`.

## Datasets

All sets were consumed exactly once, except the rejected draft. Earlier metadata
for the same bytes bound to other frozen candidates (`ai-v1`, `ai-v2`, `ai-v3`,
`ai-v3-bound-v5`, `ai-v3-bound-v6`, `ai-v7`) has been removed. No code needs it,
because the wording remains excluded through the datasets below.

| Set               | sha256       | Rows (pos/neg) | Status   | Outcome at consumption                         |
| ----------------- | ------------ | -------------- | -------- | ---------------------------------------------- |
| `ai-v1-rebound`   | 684ee8b7ccf1 | 1243 (607/636) | consumed | Failed positive coverage and critical controls |
| `ai-v2-bound-v3`  | db68f3766666 | 1243 (607/636) | consumed | Failed positive coverage and one critical row  |
| `ai-v3-bound-v8`  | 796bc3f7c9bf | 1244 (607/637) | consumed | Failed critical controls                       |
| `ai-v4`           | 2b8ecf20abc5 | 1317 (680/637) | consumed | Failed positive coverage (neutral preambles)   |
| `ai-v5`           | 9ad52ae99464 | 1317 (680/637) | consumed | Failed negative false acceptance               |
| `ai-v6`           | 3edb170664a7 | 1317 (680/637) | rejected | Repeated consumed negatives; never inferred    |
| `ai-v6-distinct`  | 1ca445e47851 | 1317 (680/637) | consumed | Passed; promoted, later superseded             |
| `ai-v7-bound-v13` | 04bc8d5020d4 | 1317 (680/637) | consumed | Passed; promoted; **bundled model**            |

The rejected draft stays tracked because its wording was never consumed, and
preparation refuses to run when unconsumed reserved wording is missing. The
`candidateSha256` of `ai-v7-bound-v13` equals the hash of the bundled artifact
(`JSON.stringify(weights) + "\n"`, `fcc147db7f23…`).

Replaying the seven consumed sets against the bundled model gives these results:

- every positive intent is exact: 4541/4541;
- every negative is refused: 0/4457 accepted;
- every critical row passes: 88/88.

Run `bun run --cwd packages/retry/training replay` to reproduce them.

## Authoring a new set

```sh
bun run packages/retry/training/src/gate.ts prepare <run>
bun run packages/retry/evaluation/author.ts <run>/scope-freeze.json \
  packages/retry/evaluation/reserved/<set>.jsonl <revision>
bun run packages/retry/training/src/gate.ts promote <run> \
  packages/retry/evaluation/reserved/<set>.jsonl
```

To produce new wording, add a revision to `author.ts`. Existing revisions reproduce
the historical sets and must not change. An unconsumed set may be rebound to a later
frozen candidate by writing new metadata. A consumed set never provides fresh
evidence again.

## Limitations

- The rows were AI-authored by the same implementation assistant through bounded
  constructions and numeric variation. They are not independent human evidence or a
  population accuracy estimate.
- Schedule targets are independently calculated, and critical rows are hand-checked.
  Coverage is limited to the nine families in `FAMILIES`.
- Per-set metrics at consumption time are recorded in gate reports under
  `training/retry/runs/`, which are not tracked. The outcomes above summarize those
  reports. The replay numbers are reproducible from tracked files.
