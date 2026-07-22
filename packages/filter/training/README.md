# Filter training

Run `bun run train:filter` from the repository root. The coordinator prepares
8,000 seeded rows, checks their gold roles through the actual resolver, writes
structurally disjoint shards, and trains hidden 32 for 24 epochs with six-bit QAT.
Checkpoint selection measures decoded intents and penalizes wrong accepted
meanings and negative false accepts. It never selects on token accuracy.

The Python harness reuses validation's checked Tagger, Dataset, quantization and
parity primitives, and the existing cron Python environment. No environment is
copied. Runtime tensor helpers and tokenization were reviewed and reused from
validation; domain labels, catalog handling and predicate semantics stay local.
The checked tokenizer, tensor codec and inference math now share private source
with validation and sort. The bounded role-model trainer is also shared;
no general core package is published.

Exact field identifiers vary per example and are absent from other splits.
Caller aliases and typed literals become canonical markers; identifiers and
values remain payload facts and must be preserved by decoding. Quoted strings
are protected before alias matching. Models predict language roles, never SQL
or literal values. Training and runtime use the same preparation and tokenizer.

Each candidate is run-local and inactive. Calibration uses validation only,
preserves finite evidence and selects a passing confidence floor near 0.5.
Export decodes the exact packed wire artifact into PyTorch before freezing
32 parity sequences. Runtime checks compare roles and boundary logits and test
padding invariance. Synthetic evaluation reports refusal-inclusive exact intent,
both wrong-meaning denominators, per-family negative acceptance and uncertainty.

Major positive families are eq/ne/lt/le/gt/ge, membership, range, null, AND, OR,
exclusion, NOT and explicitly grouped expressions. Mixed ungrouped AND/OR needs
clarification. Unsupported types and features must refuse. Synthetic success
does not authorize promotion: fresh evaluation-only acceptance, critical controls,
paired real SQL/evaluator checks, packed/browser consumers and size gates remain
required. The separate authoring writer and frozen gate now implement these checks.

```sh
bun run packages/filter/training/src/gate.ts prepare <run-directory>
bun run packages/filter/evaluation/author.ts <run-directory> <new-reserved.jsonl> 2
bun run packages/filter/training/src/gate.ts promote <run-directory> <new-reserved.jsonl>
```

Authoring revision 2 has already been consumed for the promoted model. Add truly
new wording for the next candidate; replaying these commands with the same
wording is regression evaluation, not fresh acceptance. Browser checks need a
compatible installed Chromium; the session reused revision 1228 without a new
download. The gate checks infrastructure and actual packed/browser references
before consuming language acceptance. It atomically replaces the artifact and
calibration and retains rollback; a source-only smoke module is not known-good.

Generated filter work is bounded to 512 MiB; monitor before and after each run.
Keep the active corpus, chosen best and latest resumable checkpoint, rollback,
and compact unique evidence. Retire redundant shards by exact, verified paths.

The preparation audit confirms that numeric values and caller names were already
masked for split assignment. Every question variant recovers its statement's
fingerprint after removing the question marker. The active corpus has no
statement/question split mismatch. The earlier literal-value concern did not
apply to filter, so no retraining was needed for split grouping.

Generation now streams buffered JSONL into an immutable timestamped corpus.
The shared binary writer buffers at most 8192 tokens, while row records and
metadata remain bounded by the chosen corpus size. A byte comparison matched
the entire active 8000-row corpus, all 21 binary shards and three manifests.
Pending and consumed reserved wording is excluded before generation, and missing
or changed evaluation data fails preparation. Actual packed consumers still
pass offline in Bun and Node with the unchanged model and runtime.
