# Cron evaluation

Held-out, evaluation-only corpora for `@syntaxless/cron`. Every record measures
the public parser and compiler end to end. None of this wording may be used for
training, checkpoint selection, threshold fitting or export calibration.
Training preparation actively excludes it (see [Isolation](#isolation)).

## Layout

```
evaluation/
  README.md
  human-v1.jsonl … human-v15.jsonl  # versioned corpora (v1 mixes partitions; v2–v15 are acceptance sets)
  human-calibration-v2.jsonl        # calibration corpus for confidence and abstention policy
  consumed.json                     # ledger of consumed acceptance-partition hashes
  historical-acceptance.json        # earlier consumed revisions whose wording is still protected
  compiler-contract-v2.json         # reviewed compiler-target corrections for existing records
```

File names and bytes are load-bearing: the training code reads this directory
by name pattern, and several records are bound by sha256. Don't rename, move
or reformat these files.

## Data format

Each corpus is JSONL with one record per line:

| Field              | Type                                           | Meaning                                                                                  |
| ------------------ | ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `version`          | `1`                                            | Record schema version                                                                    |
| `id`               | string                                         | Unique id, e.g. `v12-interval_clock-01`                                                  |
| `partition`        | `development` \| `calibration` \| `acceptance` | Permitted use (below)                                                                    |
| `family`           | string                                         | Balanced construction family used for per-family accuracy                                |
| `capabilities`     | string[]                                       | Phenomena exercised (e.g. `number-word`, `reported-speech`)                              |
| `critical`         | boolean                                        | Any failure on this record fails promotion                                               |
| `trainingEligible` | `false`                                        | Always false                                                                             |
| `text`             | string                                         | Input sentence                                                                           |
| `intent`           | `ScheduleIntent` \| `null`                     | Gold intent; `null` for a negative that must be refused                                  |
| `expected`         | object                                         | `{ parse: "reject" }`, `{ supported: true, expression }` or `{ supported: false, code }` |

Records must not contain training fields (`spans`, `split`, `frame`, `style`,
`template`). `parseHumanCorpus` in `training/src/human-evaluate.ts` enforces the
schema.

`consumed.json` contains `{ version: 1, acceptancePartitionSha256: string[] }`.
A partition hash is the sha256 of the partition's acceptance records, each
serialized with `JSON.stringify` and joined by `\n`, plus a trailing `\n`.
Every acceptance set is a single partition, so for v2–v15 the hash equals the
file's sha256.

`historical-acceptance.json` holds `{ version, sourceLedgerSha256, partitions }`.
Each partition lists its `partitionSha256`, its `source` path and its full
`records`, plus either the `gitBlob` it was read from or a `recovery` block. The
`recovery` block names the source commit and blob and lists the exact field
changes reversed to rebuild the consumed revision.

`compiler-contract-v2.json` lists 65 corrections of the form `{ id,
recordSha256, originalExpected, expected }`, together with the sha256 of every
corpus file. Each corrected record is a minute or hour interval that doesn't
divide 60 or 24. Its compiler target becomes `{ supported: false, code:
"non-unit-interval" }`. A correction applies only to the record whose canonical
JSON matches `recordSha256`. Text, intents and consumption hashes are unchanged.

## Acceptance policy

| Partition     | Use                                                                      |
| ------------- | ------------------------------------------------------------------------ |
| `development` | Diagnose capability gaps. Fixes must be general, never sentence-specific |
| `calibration` | Choose confidence and abstention policy. Never trains parameters         |
| `acceptance`  | One run on a frozen candidate, then consumed                             |

A candidate passes acceptance when all of these hold on the public parse
(`decideHumanPromotion` in `training/src/human-evaluate.ts`):

- overall accuracy is at least 0.95;
- accuracy is at least 0.90 in every family with at least 5 records;
- silent meaning changes (accepted with a wrong intent or expression) are at most 0.5%;
- false accepts are at most 1% of negatives;
- no `critical` record fails.

`evaluateHumanCorpus` refuses to score an acceptance partition whose hash is
already in `consumed.json`. Because the ledger is keyed by content, copying or
renaming a file doesn't make it fresh again. After the single acceptance run,
the partition hash is appended to the ledger. From then on the set is
regression evidence only. The training coordinator replays the union of all
`human-v*.jsonl` files after each export and writes `human-regression.json`.
That replay never selects checkpoints or certifies promotion.

When a fixture's gold label is corrected after consumption, the hashes of both
revisions stay in the ledger. A correction can never create a fresh set.

## Isolation

Training preparation adds every `*.jsonl` text in this directory, and the
records in `historical-acceptance.json`, to a protected-wording set. Generated
training examples that match it are excluded. Compact preparation also
recomputes each archived partition's digest and requires it to appear in the
ledger (`training/src/compact-protection.ts`). `assertHumanIsolation` fails if a
corpus is placed under `training/data`, or if a generator source references a
corpus or copies its wording.

The ledger holds 22 hashes. 15 match current files and 4 match archived
revisions. The remaining 3 belong to earlier revisions of v9 and v11 whose
records are no longer available, so their exclusion can't be verified.
Preparation reports these hashes as a limitation.

## Datasets

The result column is the single acceptance run of the frozen candidate at the
time of consumption. "Silent" counts silent meaning changes; "FA" counts false
accepts. Later 100% replays are regression results and are not listed.

| Set                  | sha256 (12)    | Rows | Negatives | Status      | Acceptance result                 |
| -------------------- | -------------- | ---: | --------: | ----------- | --------------------------------- |
| human-v1             | `44166078b7ec` |   96 |        29 | consumed    | 40 acceptance rows; not recorded  |
| human-v2             | `ed3914bb0cd6` |  150 |        60 | consumed    | 96/150, 29 silent, 15 FA          |
| human-v3             | `b1247ed94271` |  100 |        50 | consumed    | 78/100, 17 silent, 12 FA          |
| human-v4             | `5af1c32e7ff7` |  100 |        50 | consumed    | 82/100, 13 silent, 8 FA           |
| human-v5             | `989896ccaa11` |  100 |        50 | consumed    | 90/100, 10 silent, 7 FA           |
| human-v6             | `286f1e52a97d` |  100 |        50 | consumed    | 83/100, 11 silent, 6 FA           |
| human-v7             | `0643782939e9` |  100 |        50 | consumed    | 79/100, 17 silent, 12 FA          |
| human-v8             | `a4a7b969d91b` |  100 |        50 | consumed    | 82/100, 12 silent, 9 FA           |
| human-v9             | `2f89b818686a` |  100 |        51 | consumed    | 75/100, 19 silent, 9 FA           |
| human-v10            | `58a1380346c8` |  100 |        50 | consumed    | 88/100, 9 silent, 6 FA            |
| human-v11            | `5e8a2ce9fd01` |  120 |        60 | consumed    | 95/120, 17 silent, 11 FA          |
| human-v12            | `5f09f3c5681b` |  120 |        60 | consumed    | 83/120, 29 silent, 12 FA          |
| human-v13            | `f820004ba7aa` |  120 |        50 | consumed    | 94/120, 20 silent, 7 FA           |
| human-v14            | `3e82b1d1733d` |  120 |        50 | consumed    | 101/120, 17 silent, 10 FA         |
| human-v15            | `b5173887da94` |  120 |        50 | consumed    | 88/120, 22 silent, 12 FA          |
| human-calibration-v2 | `efe269e46330` |  100 |        60 | calibration | 85/100 before calibration changes |

No candidate has passed acceptance. Every acceptance set is consumed, so
promotion requires a new set.

Gold corrections made after consumption were:

- v2: `half past three` changed from 15:30 to 03:30.
- v2 and v3: two labels no longer read `nightly` as PM.
- v9: an inclusive-bound convention fixed and two ambiguous labels clarified.
- v11: `last_business_day` spelling fixed and the exclusive `between` end applied.
- v13: `nightly` no longer read as PM, and `last_day` changed to `last_day_of_month`.

Both revisions of every corrected set are in the ledger.

## Authoring a new set

1. Write `human-v<N>.jsonl` with all records in the `acceptance` partition.
   Balance the families and include refused negatives. Don't copy or adapt
   wording from earlier sets or from generator sources.
2. Freeze the candidate (code and weights) before reading any results.
3. Check the schema and the compiler-contract bindings without scoring:
   `bun test packages/cron/training/tests/evaluation-corpora.test.ts`.
4. Run the single acceptance evaluation with
   `evaluateHumanCorpus({ dist: "packages/cron/dist", partition: "acceptance", corpus: "packages/cron/evaluation/human-v<N>.jsonl" })`
   from `training/src/human-evaluate.ts`.
5. Append the reported `acceptancePartitionSha256` to `consumed.json`, then add
   a row to the table above. Changing the ledger invalidates existing compact
   preparations, which must then be regenerated.

## Limitations

- The implementation assistant wrote every corpus, at the maintainer's request.
  Independent people wrote none of them. The scores are behavioral evidence
  about this parser, not estimates of population accuracy.
- The sets were written after earlier failures were known, so later sets probe
  known weak spots more than ordinary usage does.
- Three consumed revisions (of v9 and v11) are unavailable, as described above.
- The acceptance results above come from the reports of each run. The runs
  themselves aren't kept in the repository.
