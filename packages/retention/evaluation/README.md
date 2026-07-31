# Retention evaluation

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
  fixtures.ts      fixed backups, reference instant and hand-checked reference preview
  reserved/        <set>.jsonl datasets and <set>.jsonl.meta.json provenance
  consumed/        <sha256>.json consumption records and <sha256>.archive.json wording archives
```

## Data format

Each line of a reserved dataset is one row:

| Field                             | Meaning                                                                    |
| --------------------------------- | -------------------------------------------------------------------------- |
| `text`                            | Request wording                                                            |
| `context`                         | Always `{}`; parsing needs no context                                      |
| `intent`                          | Expected `RetentionIntent`, or `null` when the request must refuse         |
| `family`                          | Positive family (see `training/src/corpus.ts` `FAMILIES`) or negative kind |
| `critical`                        | Hand-checked control; any failure blocks promotion                         |
| `compileOptions`, `compileTarget` | Calendar context and independently calculated preview (or refusal code)    |
| `split`, `trainingEligible`       | Always `"heldout"` and `false`                                             |
| `roles`, `fingerprint`            | Unused role list and construction fingerprint                              |

Metadata (`<set>.jsonl.meta.json`): `sha256` and `rows` of the dataset, the
`candidateSha256` and `scopeFreezeSha256` it is bound to, `createdAt`,
`trainingEligible: false`, `authorship` and `independenceLimits`.

Consumption record (`consumed/<sha256>.json`): `sha256`, the repository-relative
`acceptance` path, `candidateSha256`, `consumedAt`, `archiveSha256`, `authorship`
and `independenceLimits`. The archive stores the critical rows and the sorted
hashes of every normalized row text. Its digest is verified before use.

### Archived sets

Full rows of four consumed sets (`ai-v1-distinct`, `ai-v2`, `ai-v3-bound-v4`,
`ai-v4`) are git-ignored to keep the repository small. Their metadata and
archives remain tracked. Corpus preparation and the gate accept an archive in
place of a missing full set: wording exclusion and critical-row replay keep
working. Full-row regression replay covers only the sets that are present
locally. Where a full file exists, its hash is still verified.

## Acceptance policy

`training/src/gate.ts promote` performs these checks in order:

1. **Provenance.** The candidate's source, artifact, corpus and evaluation identity
   must match its scope freeze. The metadata must match the dataset hash, candidate
   and freeze, and must be created after the freeze.
2. **Set validity.** Rows must be unique by text and context, held-out and
   evaluation-only. The set needs at least 500 positives, at least 600 negatives,
   at least 50 positives per family and at least one critical row. It must share no
   wording with the training corpus or any consumed set, and every paired preview
   target must equal `compile`.
3. **Pre-consumption checks:**
   - typecheck and tests;
   - frozen parity;
   - the synthetic held-out gate (`training/src/evaluate.ts`);
   - regression replay: every full consumed set must keep each result the active
     model gets right, and every consumed critical row must pass;
   - the packed package consumer, Chromium and size gates.
4. **Consumption.** The gate writes the archive and the consumption record before
   inference, so a set is never reused as fresh evidence.
5. **Language thresholds:**
   - exact positive intents ≥ 90%, and ≥ 85% in every family;
   - wrong meanings ≤ 0.5% of all positives and ≤ 0.5% of accepted positives;
   - negative false acceptance ≤ 1%;
   - zero critical failures.
6. **Promotion.** A pass atomically replaces `src/model/weights.gen.ts`, keeps the
   previous module as rollback, and reruns `check:package`.

## Datasets

Earlier metadata for the same bytes bound to other frozen candidates (`ai-v3`,
`ai-v6`) has been removed. No code needs it.

| Set              | sha256       | Rows (pos/neg) | Status             | Outcome at consumption                 |
| ---------------- | ------------ | -------------- | ------------------ | -------------------------------------- |
| `ai-v1`          | 365e01fa305f | 1321 (685/636) | rejected           | Repeated positive rows; never inferred |
| `ai-v1-distinct` | 91364a3ef48d | 1321 (685/636) | consumed, archived | Failed positive coverage               |
| `ai-v2`          | f44abb85c752 | 1321 (685/636) | consumed, archived | Failed one critical control            |
| `ai-v3-bound-v4` | 69e6a1b70435 | 1321 (685/636) | consumed, archived | Passed; promoted, later superseded     |
| `ai-v4`          | 8ab287b9b406 | 1398 (761/637) | consumed, archived | Failed negative false acceptance       |
| `ai-v5`          | 7fdc1ed8597d | 1398 (761/637) | consumed           | Passed; promoted, later superseded     |
| `ai-v6-bound-v8` | cd2632201460 | 1398 (761/637) | consumed           | Passed; promoted; **bundled model**    |

The rejected draft stays tracked because its wording was never consumed, and
preparation refuses to run when unconsumed reserved wording is missing. The
`candidateSha256` of `ai-v6-bound-v8` equals the hash of the bundled artifact
(`JSON.stringify(weights) + "\n"`, `5b0c49443491…`).

Replaying the six consumed sets against the bundled model, with all full rows
present, gives these results:

- every positive intent is exact: 4338/4338;
- every negative is refused: 0/3819 accepted;
- every critical row passes: 102/102.

Run `bun run --cwd packages/retention/training replay` to reproduce them. Without
the git-ignored files, the four archived sets replay their critical rows only.

## Authoring a new set

```sh
bun run packages/retention/training/src/gate.ts prepare <run>
bun run packages/retention/evaluation/author.ts <run>/scope-freeze.json \
  packages/retention/evaluation/reserved/<set>.jsonl <revision>
bun run packages/retention/training/src/gate.ts promote <run> \
  packages/retention/evaluation/reserved/<set>.jsonl
```

To produce new wording, add a revision to `author.ts`. Existing revisions reproduce
the historical sets and must not change. An unconsumed set may be rebound to a later
frozen candidate by writing new metadata. A consumed set never provides fresh
evidence again.

## Limitations

- The rows were AI-authored by the same implementation assistant through bounded
  constructions and numeric variation. They are not independent human evidence or a
  population accuracy estimate.
- Previews use independent UTC calculations, and the timezone witnesses (DST
  folds, Monday weeks, leap and month boundaries, exact cutoffs) are hand-checked.
  Coverage is limited to the ten families in `FAMILIES`.
- Per-set metrics at consumption time are recorded in gate reports under
  `training/retention/runs/`, which are not tracked. The outcomes above summarize
  those reports.
