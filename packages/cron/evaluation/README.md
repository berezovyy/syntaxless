# Human-style evaluation

`human-v1.jsonl` is a versioned, evaluation-only behavioral corpus. It is not a
training corpus and must never be passed to `backward.ts`, `generate.ts`,
`featurize.ts`, the Python trainer, checkpoint selection, or calibration.

The partitions have separate purposes:

- `development`: diagnose capabilities and guide abstract fixes.
- `calibration`: choose confidence/abstention policy; never train parameters.
- `acceptance`: reserved final promotion evidence; never use its wording to
  change templates, features, decoding, confidence, or checkpoints.

Every record deliberately lacks renderer spans and training split/frame fields,
and carries `trainingEligible: false`. `human-evaluate.ts` fails closed if the
file is moved under `training/data`, becomes training-shaped, is referenced by
a generator, or contains wording copied verbatim into generator source.

Run development diagnostics from the repository root after building:

```sh
bun run --cwd packages/cron/training evaluate:human
```

Do not run the `acceptance` partition until the candidate is otherwise frozen.

The acceptance partition in `human-v1.jsonl` was consumed on 2026-09-15 and is
now development evidence only. Its acceptance-partition hash is recorded in
`consumed.json`; the evaluator refuses to score those acceptance records even
if unrelated partitions change or the file is copied. Final promotion requires a newly and independently
authored corpus with a different hash. After its one permitted acceptance run,
copy `acceptancePartitionSha256` from the evaluation report into the ledger
before further development.

`human-v2.jsonl` was authored by the coding agent at the user's explicit
request, rather than by an independent human, and was consumed on 2026-09-15.
It is a broad 150-example behavioral set, but it is not independent evidence.
Its acceptance hash is also recorded in `consumed.json`, so it can now be used
only as development diagnostics and never as a fresh promotion gate.
The one valid frozen-candidate run scored 96/150 (64.0%), with 29 silent
meaning changes and 15 false accepts. After those failures were used for
decoder development, the same set reached 150/150; that later score is a
regression result, not promotion evidence.
After consumption, one gold-label typo (`half past three` had been labeled as
15:30 rather than 03:30) was corrected. Both the originally evaluated hash and
the corrected fixture hash are consumed; the correction cannot create a fresh
acceptance set.
Two later labels that incorrectly treated `nightly` as a PM marker were also
corrected to ordinary unqualified-clock semantics. The resulting v2 and v3
hashes are consumed as well.

`human-v3.jsonl` is a second, newly authored but still non-independent
100-example acceptance set. Its only frozen-candidate run scored 78/100
(78.0%), with 17 silent meaning changes and 12 false accepts. Its hash was
immediately consumed; all later use is development-only.

`human-v4.jsonl` is the third non-independent acceptance set. Its only frozen
run scored 82/100 (82.0%), with 13 silent meaning changes and 8 false accepts.
Its hash is consumed and it is development-only from this point onward.

`human-v5.jsonl` is the fourth non-independent acceptance set. Its only frozen
run scored 90/100 (90.0%), with 10 silent meaning changes and 7 false accepts.
Its hash is consumed and it is development-only from this point onward.

`human-v6.jsonl` is a fifth coding-agent-authored, non-independent acceptance
set, created only after the prior candidate and decoder were frozen. It has 100
examples across ten equally supported broad families. Its one permitted fresh
run scored 83/100 (83.0%), with 11 silent meaning changes and 6 false accepts.
The failures exposed replacement-correction, reported-speech, cancellation,
vague-cadence, and clock-reconciliation gaps. Its hash was immediately recorded
in `consumed.json`; after those gaps were fixed it reached 100/100 only as a
development regression. That later result is not fresh promotion evidence, and
the original 83% result remains the honest acceptance score.

`human-v7.jsonl` is the sixth coding-agent-authored, non-independent acceptance
set. It was authored and gold-linted only after the v6 fixes and candidate were
frozen, and contains 100 new examples with ten examples in each of ten broad
families. Its one fresh run scored 79/100 (79.0%), with 17 silent meaning
changes and 12 false accepts, so promotion failed. The failures exposed dotted
clocks, night qualifiers, numeric relative clocks, weekday abbreviations,
alternate-week wording, further correction forms, prose/reporting contexts,
vague schedules, prohibitions, and incomplete ranges. The acceptance hash is
consumed. A later 100/100 result is development regression evidence only.

`human-v8.jsonl` is a seventh coding-agent-authored, non-independent set of 100
balanced examples. Its one fresh run scored 82/100, with 12 silent meaning
changes and 9 false accepts. Its consumed failures guided decoder and safety
changes; the later 100/100 score is regression evidence only.

`human-v9.jsonl` is an eighth coding-agent-authored, non-independent set of 100
balanced examples. The initially authored gold scored 75/100, with 19 silent
meaning changes and 9 false accepts. A subsequent gold audit corrected an
inclusive-bound convention error and two ambiguous author assumptions. Every
intermediate and final fixture hash remains consumed. The corrected set now
scores 100/100 only as development regression evidence.

`human-v10.jsonl` is a ninth coding-agent-authored, non-independent set of 100
balanced examples. Its only fresh run scored 88/100, with 9 silent meaning
changes and 6 false accepts, so promotion failed. Its later 100/100 result is
development regression evidence only; its wording and hash are permanently
consumed.

`human-calibration-v2.jsonl` is a separate coding-agent-authored calibration
set created with the user's explicit permission. It contains 100 deliberately
distributed examples: 40 positive recurring schedules and 60 difficult
negatives covering reported or quoted schedules, questions, ambiguity,
cancellations, incomplete requests, and one-off requests. It is not independent
acceptance evidence and must never enter training. The frozen candidate initially
scored 85/100. After
discourse-safety fixes and removal of the unsafe global-kind confidence boost,
the public parser scored 100/100. On raw decoding, the maximum safe-coverage
point is a 0.4 threshold (40% coverage, zero observed risk); the existing 0.5
public floor is therefore retained rather than tuned downward. This result is
calibration/development evidence only and does not authorize model promotion.

`human-v11.jsonl` is a tenth coding-agent-authored, non-independent acceptance
set, created after the calibration changes were frozen. It contains 120 new
examples across twelve equally supported families, with no normalized wording
overlap against the earlier human fixtures. Its one permitted acceptance run
scored 95/120 (79.2%), with 17 silent meaning changes and 11 false accepts, so
promotion failed. A gold audit corrected two authoring mistakes: the public
`last_business_day` spelling and the exclusive end convention for `between`.
Both the original and corrected fixture hashes are consumed. After using the
failures for safety and decoder development, the corrected gold moved the
audited baseline to 97/120 and ultimately reached 120/120 only as regression
evidence; those later results are not fresh promotion evidence.

`human-v12.jsonl` is an eleventh coding-agent-authored, non-independent set of
120 balanced examples across twelve families. It was gold-linted and checked
for normalized wording overlap before its single frozen-candidate acceptance
run. That fresh run scored 83/120 (69.2%), with 29 silent meaning changes and
12 false accepts, so promotion failed. Its hash was immediately consumed. The
failures then drove false-positive guards, correction scoping, typo-aware
evidence recovery, clock reconciliation, and additional wording support. The
later 120/120 score is development regression evidence only and does not alter
the original acceptance result or authorize promotion.

`human-v13.jsonl` is a twelfth coding-agent-authored, non-independent set of
120 new examples across twelve equally supported families. Before its one
fresh run it passed schema, compiler-gold, generator-isolation, and normalized
wording-overlap checks. The initially authored gold scored 94/120 (78.3%), with
20 silent meaning changes and 7 false accepts, so promotion failed. A gold
audit then corrected two author mistakes: `nightly` does not itself select PM,
and the public day-rule spelling is `last_day_of_month`. Both fixture hashes
are permanently consumed; the corrected audited baseline is 96/120. After the
failures drove safety and decoder work, the corrected set reached 120/120 only
as development regression evidence. It is not fresh promotion evidence.
