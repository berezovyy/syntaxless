# Cron training

Run the coordinator from the repository root.

```sh
bun run train
```

The coordinator takes no flags, and it prepares a compact corpus rather than
expanding the historical production mixture. Every run gets a unique
`training/cron/runs/compact-*/` directory with storage measurements. The run
budget is 128 MiB, and preparation requires another 2 GiB of free disk reserve.
The historical corpus and environment aren't deleted or rewritten.

## Current workflow

First, the coordinator validates available evaluation evidence and excludes its
wording before writing features or synchronizing dependencies. Current fixtures
cover 15 ledger partitions, and verified historical archives cover another four.
Three historical partitions lack their original records. Preparation records
their identities and the resulting exclusion limitation, but continues training.
The ledger stays unchanged, and fresh reserved acceptance is required for promotion.
Malformed available evidence still stops preparation.

Second, preparation reuses the sentence catalog and buffered shard writer.
Compact recipe 4 keeps the 10,000-row ceiling. Recipe 3 produced 9,940 distinct rows.
Negative rows account for 36.2 percent of the diagnostic corpus.
Construction groups keep requests, carriers, negative variants and corrections
in one split. Rare families receive construction groups in every split.
Every row passes token alignment and gold-label decoding checks. Numeric clock
labels use literal roles, so an unqualified hour retains its literal value.
Recipe 4 uses existing literal roles for every numeric hour, minute and interval
instead of asking the model to predict a separate class for each digit value.
Named times and word numbers keep their semantic labels.
Hourly minute anchors use a separate rendering with `MINUTE_LIT` and no hour.
Large minute/hour intervals are positive examples even when cron compilation is
unsupported. Construction reservation and pre-write checks cover every unit,
unit step, multiple interval, selector, ordinal, day rule, bound and exception
field in every split. The checks establish feature coverage rather than complete
language quality or exhaustive combinations of facts.

Controlled variants replace existing carrier rows rather than adding rows.
The generator reuses the spelling mutator for deletion, insertion, transposition,
repetition and keyboard-neighbor errors. Punctuation variants add a comma after
the carrier prefix. Positive and negative variants use the same altered schedule,
while literal values, numeric text and semantic annotations stay unchanged.
Clean bases, corrections and ordinary prose remain unchanged. Every split must
contain positive and negative rows for each variant type before any file is
written. Construction groups are calculated from the clean base before mutation.

Third, the configured candidate recipe uses hidden size 64, 24 epochs, batch size
128 and six-bit export. CPU training uses two threads. The coordinator evaluates
consumed human cases as regression evidence, and it never promotes a candidate.
Selection uses complete validation accuracy from actual calibrated TypeScript
parsing. Prediction-head
accuracy remains diagnostic, and a fresh acceptance gate remains required for
promotion.

## Checkpoint selection

First, each epoch is packed into the same six-bit artifact used by the exporter.
Boundary, energy and distance thresholds are fitted only on validation, using
the existing bounded calibration functions.
Requests and corrections supply the positive calibration population. All other
speech acts are negative. Additional energy and distance cutoffs use negative
examples that the speech head could accept, because refused descriptions can
contain the same schedule facts as a request. If the speech head refuses every
validation negative, no additional cutoff is fitted. Complete parser safety is
still checked on validation, heldout and fresh reserved acceptance.

Second, the source TypeScript engine parses every validation row with the overlay
disabled. A positive hit requires the complete expected intent, and a refusal
doesn't count as a positive hit. Wrong accepted meanings and accepted negative
rows have separate counts. The scorer records both denominators for wrong
accepted meanings, using all positive rows and accepted positive rows.

Third, selection prefers checkpoints that pass the synthetic gate. Before any
checkpoint passes, selection compares the mean positive accuracy across corpus
families, then overall positive accuracy, wrong accepted meanings, negative false
accepts and raw positive accuracy. A selected failing checkpoint remains a
development candidate and can't authorize promotion. Once a checkpoint passes,
replacement must also pass and preserve positive accuracy and safety overall and
within each family, generator kind and speech act. The comparison uses the existing
bound of two standard errors with a one-row minimum. Data, code and support
identities must match between epochs, including before the gate passes.

Finally, the exporter must reproduce the calibrated artifact identity used for
selection before writing parity fixtures or candidate files. The selected
artifact is then scored on heldout data once, with unchanged code and corpus
identities. Heldout results never select an epoch. Reports remain synthetic
development evidence, even when their synthetic gate passes.

Each epoch overwrites one private `last-selection.gen.ts` and one report rather
than retaining every model. TypeScript scoring keeps one row at a time, counters
per family, kind and speech act, and at most three failures per family/category.
Calibration retains one batch and the existing fixed 4,096-row sample; parity
fixtures are generated only for the final candidate. Memory depends on model,
batch and group counts rather than retained predictions for the complete split.
The diagnostic full validation pass measured 249016 KiB peak RSS.

The corrected selection run chose epoch 16 rather than the untrained epoch.
Validation achieved 628/894 exact positives with 37 wrong accepted meanings and
zero/489 negative false accepts. Heldout achieved 569/925 exact positives with
42 wrong accepted meanings and one/499 negative false accepts. The candidate
failed the synthetic gate and remains inactive. Validation diagnostics identified
numeric class confusion and errors in relative weekdays and bound operators.
Recipe 4 addresses numeric supervision without adding rows or changing runtime
decoding. Corrected speech-aware calibration produces 691/894 exact validation
positives and 657/925 exact heldout positives. Wrong accepted meanings remain
34 and 52 respectively, and both splits have zero negative false accepts.
The candidate fails the quality gate and remains inactive. The hidden size 64
comparison uses the same corpus, seed and 24 epochs. Its selected epoch 24 gives
736/894 exact validation positives with 31 wrong accepted meanings, and 737/925
exact heldout positives with 59 wrong accepted meanings. Both splits have zero
negative false accepts. The candidate also interprets the consumed reference
`every weekday at 9` as 09:00, but its complete quality gate still fails.
The bundled weights and acceptance ledger remain unchanged.

## Diagnostic preparation

Private diagnostic preparation writes corpora marked `trainingEligible: false`.
The flag distinguishes feature and split checks from verified training inputs.
Diagnostic checks don't establish language quality or authorize promotion.
Frozen models and the acceptance consumption ledger remain unchanged.

The current bounded generator retains at most 10,000 rows and a text/group index.
The existing eager sentence catalog also consumes memory. Full diagnostic
preparation measured 433688 KiB peak RSS for recipe 3. The shard writer keeps one fixed
buffer per binary column instead of retaining all feature arrays.
Preparation records the producer's local source and locked dependency identities.
Historical recipe 1 and recipe 2 corpora and proofs remain at their original paths;
their binary features and sequence metadata are preserved in verified archives.
The recipe 2 feature archive recovered 22564864 allocated bytes after checking
every archive member against the original checksum. The active recipe 3 diagnostic
uses 33935360 allocated bytes and remains ineligible for training.

## Historical production preparation

The remaining notes describe the preserved production preparation machinery,
which the current coordinator no longer invokes. They explain historical data
and tests rather than authorizing million-scale regeneration.

## Memory rule

**Peak RAM of every training-pipeline process must stay roughly the same however
large the corpus is.** A larger corpus may take more time, more disk, more
shards, more sequential I/O and more merge passes. It must not take more
memory. 2M rows and 200M rows should run with about the same working set.

Memory may depend on fixed quantities: the model, vocabulary and labels, the
configuration, the kernel pool and the code-defined catalog, the checked-in
gold, evaluation and Tatoeba files, one batch, one locality block, one reduce
window, one merge fan-in, and closed-vocabulary counters (per source, split,
population, label, schedule kind, speech act, frame or template). It must not
depend on the number of rows, corpus bytes, expectations, predictions, failures
or shard records.

Not allowed on any path whose input grows with the corpus:

- reading a corpus, shard, spool or source file whole (`readFileSync` or
  `read_text` followed by `split`), or decompressing one whole;
- turning a stream into an array (`[...it]`, `Array.from`, `list(...)`,
  `np.concatenate` of per-batch results, `torch.cat` over batches);
- sorting, deduplicating, shuffling or permuting all rows in memory,
  including an in-memory `Set`/`Map`/hash table with one entry per row;
- full-split NumPy temporaries: `astype` over a whole field, boolean masks over
  every row, fancy indexing by every row;
- collecting every prediction, metric input, expectation, error or result
  before reducing it. Reduce per batch to counters, and keep a capped sample
  (and a file on disk if every detail is needed);
- unbounded queues, prefetch or `Promise.all` over one promise per row.

What to do instead:

| Need                  | Bounded pattern used here                                                 |
| --------------------- | ------------------------------------------------------------------------- |
| Read JSONL / gzip     | `jsonlByteChunks` + `jsonlReader`, or `readLinesSync` for plain files     |
| Global dedup / winner | `global-union.ts`: hash partitions on disk, recursive split, ordinal runs |
| Global order          | `hash-order.ts`: external merge sort with a fixed run size and fan-in     |
| Online uniqueness     | `DiskHash256Set`: SQLite on disk with a fixed page cache (not tmpfs)      |
| Write large outputs   | one fixed buffer per file, or `writeGzip`'s backpressured pipeline        |
| Python training data  | memmap fields; `SequenceLengths`; gather one batch at a time              |
| Shuffle               | hash-ordered corpus, shuffled 262,144-row blocks, shuffle within a block  |
| Whole-split checks    | walk `LOCALITY_BLOCK_SIZE` sequences at a time                            |
| Metrics / calibration | per-batch counts (confusion counts per threshold), never stored logits    |

The training shuffle is deliberately not a global permutation: a permutation
is one integer per row. The emitted corpus is in SHA-256 text-hash order, which
is independent of source and generation order, so every locality block is
already a uniform mix; each epoch then shuffles block order and rows within a
block. Rows never repeat within an epoch.

Per stage (N = corpus rows):

| Stage                     | Working memory                                                             |
| ------------------------- | -------------------------------------------------------------------------- |
| Oracle gate               | O(gold files + catalog seeds); independent of N                            |
| Manual source generation  | O(kernel pool) + 2 x 16 MiB SQLite cache + 1 MiB output batch              |
| Source traversal          | one record at a time; constructions and catalog are code-sized             |
| Union plan                | O(partitions x 256 KiB) ingest; reduce <= 32 MiB window; merge 64 x 64 KiB |
| Emit (hash order)         | <= 16M-character run, then 64 x 64 KiB merge buffers + 1 MiB chunk         |
| Expectations, featurize   | one record + one 1 MiB buffer per output file                              |
| Python shard verification | one batch or one 262,144-sequence block                                    |
| Training, evaluation      | one batch (+ one prefetched batch) + model; counters per kind/head         |
| Calibration, abstention   | per-threshold counts; abstention uses a fixed 4,096-sequence sample        |
| Export, parity            | model size; parity uses at most 512 sequences                              |
| Human regression          | O(checked-in human corpus); independent of N                               |

Two quantities still grow with N, slowly and by design: the union's list of
winner-run file paths (about one per 32 MiB of scratch) and the shuffle's block
order (one integer per 262,144 rows). Both are negligible next to the bounds
above, even at billions of rows.

Measuring it (not yet done): build the corpus twice with only the budgets
scaled, for example `manual.records` 2M and 20M with the same sampling rates,
in fresh output directories. Record per stage the peak of each process
separately: the Bun coordinator, the Python verifier and the Python trainer.
Use `VmHWM`, `RssAnon` and `RssFile` from `/proc/<pid>/status`, or
`/usr/bin/time -v`. Judge on anonymous memory (`RssAnon`): memmapped shard
pages appear in `RssFile` and the OS page cache, which the kernel reclaims, and
they are not working memory. Bun heap (`bun:jsc` `heapStats`), Python heap
(`tracemalloc`) and CUDA (`torch.cuda.max_memory_allocated`) localize a
regression. Pass: anonymous peak at 20M is within a small fixed margin of 2M
(about 10% or 200 MiB) for every stage, not about 10 times larger.

Review rule: a change to corpus preparation, the loader, training, evaluation or
export must state its memory complexity in N. Anything that would grow with N
needs a disk-backed or streaming design before it is merged. RAM tests cannot
run in CI here, so the structural tests (`hash-order`, `production-global-union`,
`disk-hash-set`, `streaming-readers`, `training/cron/test_bounded_memory.py`)
guard the behaviors that keep memory flat. They do not replace a measurement.

## Inputs and outputs

The fixed corpus specification is `data/production/production-complete-v1.json`.
It sets the manual mixture's row count and spelling-noise share and the keep
rates for authored, systematic and public rows; see
[INTEGRATION.md](data/production/INTEGRATION.md).
The public-English corpus and pinned manifest, verified LLM records, and local
Tatoeba sentence/manifest files must be present. Missing or changed source artifacts
stop preparation. Synthetic source generation is internal; there is no separate
manual generation command or bootstrap fallback in production.
Corpus traversal requires that prepared artifact; it does not regenerate a
replacement when preparation is missing.
Published shards must contain frame, population, speech-act and actionability
columns. The loader does not invent supervision for incomplete shards.
Current training uses 78 roles, including explicit non-actionable context numbers.
Existing 77-label models remain supported; current preparation selects fresh
artifacts through code/source identity and does not rewrite previous shards.
Unresolved `COUNT` quantities refuse rather than silently becoming interval 1;
discarded correction clauses do not affect the replacement schedule.
Selected malformed literal values and unsupported fifth ordinals also refuse.
Selected scalar facts must agree; equivalent clocks and date bounds agree after
conversion, while contradictory selected values refuse.
Exception groups cannot change base monthly selectors. An ordinal rule requires
one distinct weekday; unsupported plural ordinals refuse.
Python 3.13, uv and Bun are required; workspace dependencies must be installed.

Historical production invocations wrote `training/cron/runs/production-*/` directories.

| File                           | Purpose                                                             |
| ------------------------------ | ------------------------------------------------------------------- |
| `best.pt`, `last.pt`           | Selected checkpoint and last epoch                                  |
| `weights.gen.ts`               | Packed weight export                                                |
| `report.json`                  | Configuration, capabilities, data digests, token/compute accounting |
| `export.json`, `parity/`       | Calibration evidence and decoded-weight reference fixtures          |
| `human-regression.json`        | Consumed-case diagnostics and miss attribution                      |
| `oracle.json`, `workflow.json` | Oracle result and workflow completion/failure                       |

The weights stay in the run directory. Human evaluation data never enters
parameter training; consumed cases provide diagnostics, and fresh acceptance is
required for release. See [evaluation policy](../evaluation/README.md).

Complete production preparation and training have not been run. Component
checks do not establish end-to-end training success or model performance.
