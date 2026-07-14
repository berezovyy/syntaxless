# `@syntaxless/cron` heads-only — experiment plan v2

Supersedes the cycle protocol in `GOAL.md` §"Cycle protocol". Everything else in `GOAL.md` (no overlay, no regex, no noun lists, no reserved-phrase copying, one change per train) still binds.

## Why a new plan

The v1 loop (`EXPERIMENTS.md` rows 0–7) measured the wrong thing and attributed every miss to the model. Probing candidate `8cfeae47…` on 2026-09-15 showed:

| Fact                                                                        | Evidence                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The 0/40 families were a **label-schema contradiction**, not a learning gap | `featurize.ts:defaultActionableBit` and `sentence.ts:defaultActionable` mark structural tokens (`GLUE`/`RECUR`/`EXCEPT`) non-actionable; `decodeIntent` filters to `actionable === true`, so `minutes`, `and`, `through`, `except` never reach the decoder. Same weights, filter disabled: reserved interval 0→40, exception 0→38, weekday-list 0→33, total 125→287/360. |
| The reserved-carrier gold is **ungrammatical concatenation**                | `"remind me to the cycle is 2 months"`, `"please add days are Sunday please put this in my diary for"`. The 21→147 climb came from adding the eval's own wrap templates to training.                                                                                                                                                                                     |
| Gold and decoder **disagree on semantics**; `evaluate.ts` patches over it   | Gold renders 13:30 as `1:30` (no meridiem) and treats `through` as exclusive; decoder does the opposite; `evaluate.ts` ~L527 and ~L571 special-case both.                                                                                                                                                                                                                |
| On **real English** the candidate is far from usable                        | Union of `human-v1..v15` (1646 rows, all `trainingEligible:false`): **44.5%** exact; 233 silent wrong intents; 234/760 negatives accepted. `interval` 0/67, `weekday` 11/81, `correction` 26/115.                                                                                                                                                                        |

Conclusion: fix the contract and the measurement first, then spend GPU. Every train before Phase 0 is green is wasted.

## Definitions (binding)

- **Hack (forbidden):** any code path that decides or overrides a token's label, actionable bit, speech act, or confidence from the token's _string_ (regex, word list, noun denylist, verb list, `if text includes`). Also forbidden: evaluator special-cases that make gold and decoder agree.
- **Decoder (allowed):** assembling `ScheduleIntent` from _labels_ the model emitted, including structural labels, using lexicon tables keyed by an already-labeled token (`INTERVAL`→number, `GLUE`→unit alias, `WEEKDAY`→day).
- **Schema (allowed, this plan's main lever):** what the renderer records as gold (`Span.label`, `Span.actionable`, `clauseStart`), what the featurizer writes, what the decoder consumes. A schema change is made at the renderer, propagated by regen + retrain, never patched in decode alone.
- **Oracle decode:** `decodeIntent(goldSpans → LabeledToken[])`. Measures the decoder in isolation from the model.
- **Attribution:** every miss is exactly one of `ROLE` (model labels ≠ gold labels on the schedule span), `ACT` (speech-act/actionable head wrong), `DECODE` (gold labels → wrong AST; oracle fails too), `CONF` (correct AST, rejected by margin), `GOLD` (gold itself violates the schema/semantics).

## Metrics (never summed)

| id           | set                                       | what                                             | current             | target                          |
| ------------ | ----------------------------------------- | ------------------------------------------------ | ------------------- | ------------------------------- |
| **H**        | `human-v*` union, re-tagged `regression`  | exact intent / correct reject                    | 733/1646            | promotion metric, see Phase 5   |
| **H-neg**    | `human-v*` negatives                      | false-accept                                     | 234/760             | 0                               |
| **H-silent** | `human-v*` positives                      | wrong intent emitted with `ok:true`              | 233                 | 0                               |
| **G2**       | new grammatical held-out (Phase 2)        | exact AST                                        | —                   | 100%                            |
| **O**        | oracle decode over full synth corpus + G2 | gold spans → gold intent                         | unknown (never run) | 100%, CI-gated                  |
| **INV**      | carrier-invariance pairs (Phase 3)        | label agreement on schedule span across carriers | —                   | ≥ 99%                           |
| R            | legacy reserved-carriers (word salad)     | exact AST                                        | 147/360             | diagnostic only; **not** a gate |

Ledger `EXPERIMENTS.md` gains columns: `H`, `H-neg`, `H-silent`, `G2`, `O`, `INV`, and an **attribution histogram** (`ROLE/ACT/DECODE/CONF/GOLD` counts on H+G2). A row without attribution is invalid.

---

## Phase 0 — Truth before training (no GPU)

Goal: make it impossible to blame the model for a decoder, gold, or evaluator bug.

**P0.1 Oracle round-trip test.** `packages/cron/training/src/oracle.ts` + test: for every record in `data/synth/kinds.jsonl` (sampled 200k), `data/gold/*.jsonl`, and every `KindDef` template × 50 seeds: build `LabeledToken[]` from gold spans (roles, `clauseStart`→boundary, `actionable`, score 1) and require `decodeIntent(...) == record.intent`. Report failures grouped by kind/template. This is a `bun test` and a CI gate; **no train starts while it is red**.

_Expected findings (already visible):_ actionable bit drops units/connectors; `through` inclusivity; 12h clock without meridiem; probably more.

**P0.2 Attribution report.** Extend `evaluate.ts` and `human-evaluate.ts` to emit per-record attribution (`ROLE/ACT/DECODE/CONF/GOLD`) by comparing predicted labels to gold spans where available and running oracle decode. Write `results/attribution.json`. Ledger rows must cite it.

**P0.3 Human regression harness.** All `human-v*` acceptance partitions are consumed; `human-evaluate.ts` refuses them. Add a `--partition regression` mode that scores consumed acceptance records as _development evidence_ (README already permits this), never as promotion. Baseline the candidate: H, H-neg, H-silent per family. Freeze as row **B0**.

**P0.4 Delete evaluator semantics patches** (`through` in bounds, twelve-hour clock rows) so the evaluator reports gold/decoder drift as `GOLD` instead of hiding it. Reserved score will drop; that is the true number.

Exit: P0.1 red list enumerated; B0 recorded; `EXPERIMENTS.md` has the new columns.

---

## Phase 1 — Contract repair (schema, then one retrain)

Each item is a schema decision made once, at the source, with the decoder consuming what the schema says. Retrain happens **once** at the end of the phase (E1), because these are contract fixes, not hypotheses about learning.

**P1.1 `actionable` is a span property.** Semantics: _1 iff the token belongs to the schedule the user is asking to run_. Structural tokens inside that schedule are actionable; tokens of quoted/reported/corrected-away/descriptive schedules are not. Implement at the source: `sentence.ts:defaultActionable(label) = label !== "O"`; `featurize.ts:defaultActionableBit` = same rule gated on speech act; delete `kinds.ts:markCarriedScheduleActionable` (now redundant). Decoder: keep the actionable filter but consume it at **clause granularity** — a clause is actionable if its content tokens are; structural tokens follow their clause. Never drop a structural token on its own bit. P0.1 must go green for the actionable class.

**P1.2 One semantics module.** `packages/cron/src/schedule-semantics.ts` (or extend `lexicon.ts`): `through`/`thru`/`until`/`ending` inclusivity, 12h↔24h rendering rules, `weekday` group membership. Renderer (`kinds.ts`, `renderer.ts`), decoder, and compiler import it. Gold generator must never render a clock the decoder cannot invert (no `1:30` for 13:30 without meridiem — render `13:30` or `1:30 pm`). P0.1 goes green for bounds/time.

**P1.3 Boundary semantics audit.** `clauseStart` marks the clause's first _content_ token; `splitClauses` moves trailing structural filler forward. Verify via oracle that every multi-clause kind (exception lead, bounds lead, correction) round-trips; fix at the renderer if not.

**P1.4 Regenerate + featurize full mix**, confirm `O = 100%`, then **E1**: retrain (same hyperparameters as `discourse-goal-7`) → export candidate → score H, G2 (when available), R, attribution.

Keep rule for E1: H up, H-neg not worse, `DECODE` attribution ≈ 0. Anything else means Phase 1 is incomplete; do not proceed to Phase 3 trains.

---

## Phase 2 — Measure the right thing

**P2.1 Grammatical held-out generator (G2).** Replace the reserved-carrier concatenator with a slot-typed grammar in `background.ts`/a new `heldout-grammar.ts`:

- Carrier types: `IMPERATIVE + SCHEDULE` (_"ping the endpoint every 5 minutes"_), `COPULAR + SCHEDULE` (_"the cadence is every 5 minutes"_), `PREPOSITIONAL + SCHEDULE` (_"set a reminder for 9am on weekdays"_), `SUBORDINATE` (_"while the dashboard is open, refresh every 19 minutes"_), `TRAILING` (schedule first, request after).
- Compose only type-compatible pairs. Never `at + copular`.
- Hold out **lexical items**, not fixed prefixes: a held-out verb set, noun set, connector synonyms (`thru`, `till`, `up to and including`), number spellings, and the eight old reserved prefixes. Training grammar uses disjoint lexicons; `assertDisjoint` enforces.
- 9 families × 60 rows, deterministic seed, spans recorded by `Sentence` so P0.1 covers it.

**P2.2 Retire R as a gate.** Keep the file; report it as diagnostic. `gate.ts` reads G2 + H.

**P2.3 Negative held-out.** G2 gets a negative half from the same grammar: descriptive (_"my sister attends yoga every Wednesday"_), reported, question, hypothetical, quoted, cancel, incomplete — the H-neg families where 234/760 are currently accepted.

---

## Phase 3 — Corpus: diversity from a generator, correctness from the oracle

Each experiment is one train. Keep/revert on H (primary), H-neg, H-silent, G2, INV.

**E2 — LLM-paraphrased carriers with label projection.** Use the existing `training/src/llm/` runner offline to paraphrase generated `(text, spans)` rows: prompt gives the schedule span verbatim and asks for N diverse _carriers_ around it (imperative, polite, subordinate, trailing, casual, typos). Project labels by locating the untouched schedule span (`align.ts`); reject if the span is altered or the oracle fails; store as a new kind source with `frame = llm-carrier:<angle>`. Hypothesis: real-English carrier diversity, which hand-written `POSITIVE_CARRIERS` cannot reach, is what H `ROLE` misses need. One change: add this source at ~10% of the mix.

**E3 — Negative diversity via the same path.** LLM generates descriptive/reported/question/hypothetical sentences that _contain_ a schedule span; labels: inner roles kept, `actionable = 0`, speech act from the angle. Oracle must reject (`unsafe-context`). Targets H-neg.

**E4 — Carrier-invariance pairs.** Generator emits K carriers per schedule instance sharing `pairId` (new shard `pairs.bin`). No loss change yet; this is the data for E6 and gives metric INV. Hypothesis: none for the model; keep if INV is measurable and no regression.

**E5 — Lexical holdout in training.** Split training lexicons (verbs, nouns, connector synonyms) 80/20 by seed so validation measures unseen words, matching G2's design. Expect in-distribution `exactSeq` to drop from 0.99 to something honest; keep if it correlates with H.

**Not allowed in Phase 3:** hand-adding wrap templates that mirror a failing eval row; copying human wording; gap-only training (already known poison).

---

## Phase 4 — Learning

**E6 — Carrier-invariance consistency loss.** Using pairs from E4: for each pair, KL between role distributions on the shared schedule span across carriers, weight 0.2. Directly optimizes the property the held-out prefix eval tests. Keep on INV↑ and H↑.

**E7 — Learned transition decoding.** Fit a label-bigram transition matrix from gold shards (not from text), export it in `weights.gen.ts`, run Viterbi over role logits in `inference.ts`. Removes sequences the schema never emits (`DAYGROUP through DAYGROUP`, `HOUR HOUR`). No string is consulted. Keep on `ROLE`↓ and H-silent↓.

**E8 — Teacher→student distillation.** Train a small transformer teacher (no QAT, hidden 128) on the same shards; distill soft role/actionable/speech-act targets into the 6-bit student (loss = CE(hard) + 0.5·KL(soft)). Hypothesis: the student's failures under carrier shift are optimization, not capacity. Keep on H↑ at equal model size.

**E9 — Speech-act head from a span-aware pooled representation.** Today speech act reads `pooled[:,0]` (mean over all tokens). Add attention pooling over tokens with actionable logit as the query so the discourse decision conditions on _where_ the schedule is. Targets H-neg (`ordinary`, `reported`, `question` families).

**E10 — Calibration under shift.** Re-fit `calibrate.py` thresholds on G2 + INV (carrier-shifted) instead of in-distribution validation; report H `CONF` count. Keep on `CONF`↓ with H-silent not worse.

**E11 — Capacity retest.** Only after E1–E7: hidden 48 and 64. v1 concluded capacity hurts; that conclusion was measured through a broken decoder and must be re-tested.

**E12 — Typo robustness.** Character-level noise is already in `noise.ts`; measure typo invariance on G2 and H `noisy` (13/50). One change: raise noise rate or add keyboard-adjacent substitutions. Keep on `noisy`↑.

Order is a default, not a law: after each row, the attribution histogram picks the next experiment (largest bucket first).

---

## Phase 5 — Promotion

Promote the candidate to `packages/cron/src/model/weights.gen.ts` when, on the **current** candidate, all hold:

- **O** = 100% (CI).
- **G2** = 100% exact AST, every family; negative half 0 false-accept.
- **FA@cov** ≤ 0.01 @ coverage ≥ 0.90, **H-silent** = 0 on the regression union (supersedes "H-neg = 0"; see `STACK-PLAN.md` §Metrics).
- **H-pos** ≥ 95% of positives (886) and ≥ 90% per family (real English has real ambiguity; 100% means the gold is wrong somewhere — investigate, don't chase). The composite H is retired: it scored a reject-all model 44%.
- **INV** ≥ 99%.
- Fresh, unconsumed `human-v16` acceptance run recorded once, hash into `consumed.json`.
- Size/privacy gates (`packages/cron/tests/browser.ts`, Brotli) green.

---

## Cycle protocol v2

1. `bun test` oracle (P0.1) green. If red: fix the schema at the source, regen, no train.
2. Read the last attribution histogram. Name the bucket you are attacking.
3. One change (E-row). State the metric that must move.
4. Regen/featurize full mix if corpus changed. Train with a unique `--run`. Export to candidate only.
5. Score: H (regression), G2, R (diagnostic), attribution. Append the ledger row with all columns.
6. Keep/revert on the stated metric; safety columns (H-neg, H-silent) may never get worse.

---

## Task checklist

- [x] P0.1 oracle round-trip test + CI gate
- [x] P0.2 attribution in `evaluate.ts` / `human-evaluate.ts`
- [x] P0.3 human regression mode; record B0
- [x] P0.4 remove evaluator semantics patches
- [x] P1.1 actionable-as-span (renderer, featurizer, clause-level decode)
- [x] P1.2 shared semantics module; gold clock/through rendering fixed
- [x] P1.3 boundary audit via oracle
- [x] P1.4 regen + E1 retrain + ledger row (E1 contaminated: kinds-sample oracle 90971/94849; not KEEP; E1b after noise fix)
- [ ] P2.1 grammatical held-out generator G2 (+ P2.3 negatives)
- [ ] P2.2 gate reads G2 + H
- [ ] E2 LLM carriers with label projection
- [ ] E3 LLM negatives
- [ ] E4 invariance pairs + INV metric
- [ ] E5 lexical holdout in training splits
- [ ] E6 consistency loss
- [ ] E7 learned transition decoding
- [ ] E8 distillation
- [ ] E9 span-aware speech-act pooling
- [ ] E10 calibration under shift
- [ ] E11 capacity retest
- [ ] E12 typo robustness
- [ ] Phase 5 promotion
