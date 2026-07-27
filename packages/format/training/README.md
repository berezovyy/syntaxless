# Format training

Run `bun run train:format` from the repository root. The coordinator creates a
new timestamped corpus and run, so another run cannot overwrite the active
corpus. It samples 8000 English instructions across ten numeric and date/time
families, with about one third negative examples. Hidden width is 32, with
25 role labels and 24 training epochs in the existing cron Python environment.

The model identifies instruction roles rather than calculating output strings.
The resolver checks preserved numeric, currency, style, locale and timezone
facts. Gold intents are authored by the renderer and checked against the
resolver. Literal substitutions share a construction fingerprint, and negative
variants inherit their base construction's split. Hash buckets assign roughly
80 percent to training and 10 percent each to validation and synthetic heldout.

Corpus generation writes buffered JSONL without retaining the serialized corpus.
The coordinator retains bounded row records for split preparation. Feature
buffers hold at most 8192 tokens and flush before another sequence would exceed
the limit. A sequence has at most 128 tokens.
Offsets, frame metadata and deduplication indexes use memory proportional to row
count. A byte comparison against the preceding full-array writer verified all
21 binary shards without changing model inputs. Preparation peak RSS was
185748 KiB at 8000 rows, and training peak RSS was 824688 KiB.

Checkpoint selection uses decoded intent accuracy and penalizes wrong accepts.
Validation alone selects calibration, and six-bit export has frozen numerical
and padding parity checks. Training creates an inactive candidate. The complete
gate is required for promotion.

```sh
bun run packages/format/training/src/gate.ts prepare <run-directory>
bun run packages/format/evaluation/author.ts <run-directory>/scope-freeze.json <new-reserved.jsonl> <revision>
bun run packages/format/training/src/gate.ts promote <run-directory> <new-reserved.jsonl>
```

First, freeze source, scope, corpus and candidate identity. Second, author new
reserved acceptance with independent Intl targets after the freeze. Third,
run the gate, including prior critical regressions, focused source checks,
synthetic thresholds, export parity, actual packed consumers, offline Chromium
and aggregate size. A passing gate atomically promotes weights and calibration,
retains the previous module, and rechecks the published API.

Both pending and consumed reserved wording are excluded from training. Consumed
hashes cannot be used again as fresh acceptance. Compact archives preserve every
critical row and every normalized wording hash. Archive and original-file hashes
are checked before reuse. A missing unconsumed dataset is an error.

The active run is `groups32-2026-10-11T23-25-13-726Z`, with a timestamped
`compact-v6-8k` corpus, selected epoch 4 and confidence floor 0.5. Its previous
module retains the passing v5 artifact for rollback. The earlier artifact used
weaker value-specific split fingerprints, so its synthetic result does not
prove the corrected structural split contract.

Keep active corpus, selected checkpoint, rollback artifact and compact evidence.
Remove only verified exact inactive paths or byte-identical copies. Historical
corpora remain locally available for the named split-comparison audit because
older generator versions are not reconstructible from the current source alone.
Generated work is bounded to 512 MiB, and no separate Python environment is used.

Private JSONL, binary writing and reserved-wording checks now share source with
validation and filter. The completed extraction preserved all active corpus
bytes, binary inputs and manifests, without retraining or modifying the model.
Each domain still defines its own role alignment and boundary semantics.
