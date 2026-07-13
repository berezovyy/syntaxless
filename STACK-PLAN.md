# `@syntaxless/cron` — stack plan v3 (September 2026)

Supersedes Phases 2–4 of `EXPERIMENT-PLAN.md`. Phase 0/1 results (oracle, attribution, actionable-as-span, semantics module, B0/E1 rows) and the binding definitions in `EXPERIMENT-PLAN.md` §"Definitions" and `GOAL.md` (no overlay, no regex, no string-driven decisions, one change per train) still hold. The old E2–E12 rows are re-slotted into the tracks below; see §"Mapping".

## Why a new stack

The 2026-09-16 comparison with `~/Projects/gpu-time` established that `@syntaxless/cron` inherited its whole stack from that project: 32-dim summed hashed features, bidirectional gated linear recurrence, int6 QAT, role tagger + boundary head, deterministic decoder that reads values from token _strings_, generator-only supervision, "reserved carriers" as the held-out set, two-proportion promotion gate. gpu-time reports 982–993/1000 on its reserved carriers and **29.7 % on the only independent corpus it measures** (Microsoft Recognizers-Text). We are at 991/1000 in-distribution and **45 % on `human-v*`**. Same design, same ceiling.

Every attribution bucket on H maps to one of three inherited choices:

| Inherited choice                                                                                                                                                                                                      | Failure it causes                                                                                                                                 | Bucket on H (E1)            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Model emits a _role_, decoder reads the _value_ from the string (`weekdayOf`, `ORDINAL_ALIASES`, `BOUND_ALIASES`, `NAMED_HOURS`, `MINUTE_IDIOMS`, single-edit alias recovery — 20 lookup sites in `parser/decode.ts`) | Typos, morphology, synonyms fail after a correct label; `BOUND` direction and `through` inclusivity live in two places; oracle can be red on gold | ROLE (partly), DECODE, GOLD |
| Rejection = three pooled 9-way/1-bit heads (`schedule_kind`, `speech_act`, `actionable`) sharing a 32-dim state with the tagger                                                                                       | 357/760 false accepts; actionable-bit contradiction (P1.1)                                                                                        | ACT 357                     |
| Supervision entirely from `kinds.ts` (3 954 lines, 220 templates)                                                                                                                                                     | Model has never seen a sentence a human wrote; H stuck at 45 % regardless of corpus size (10 M rows did not help — `EXPERIMENTS.md` rows 3–5)     | ROLE 409                    |

The plan replaces each choice rather than tuning it. Tracks are ordered so that each one is useful on its own and none blocks the previous.

## Constraints dropped

These were gpu-time's product goals, not ours. Decide budget from the accuracy target, not the reverse.

- **50 KB Brotli** → new budget **≤ 2 MB** gzipped for the shipped model. Revisit downward only after H ≥ 95 %.
- **Zero-dependency WGSL runtime** → CPU typed-array inference in `inference.ts` stays; no GPU kernel work until accuracy is solved.
- **Hidden size ∈ {32, 48, 64}** (`training/cron/config.py`) → any size the budget allows.

Kept: browser-only, no network at inference, deterministic output, `ScheduleIntent` as the public contract, 5-field cron as the compiler target.

## Metrics

Unchanged from v2: **H**, **H-neg**, **H-silent**, **O**, **G2**, **INV**, **R** (diagnostic). Added:

| id         | what                                                                                             | why                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **EXEC**   | prediction and gold compile, and their next-N occurrences (N = 8, fixed reference) are set-equal | Semantically equivalent intents (`1-5` vs `mon..fri`, `12am` vs `0:00`) score correct; the `GOLD` bucket disappears as a category |
| **FA@cov** | false-accept rate on H-neg at the coverage the abstention threshold yields on H-pos              | Replaces "H-neg = 0" (unfalsifiable) with a stated bound: FA ≤ 1 % at coverage ≥ 90 %                                             |
| **RT**     | real-text share of the training mix (records from T-C / total)                                   | Exposes whether a gain came from real data or from more generator output                                                          |

Ledger rows in `EXPERIMENTS.md` gain `EXEC`, `FA@cov`, `RT`, and a `stack` column (`tagger-v2`, `tagger-typed`, `seq2seq`).

---

## Track A — Typed-value labels (decoder becomes a deserializer)

**Hypothesis.** If the label _is_ the value for every closed class, the decoder never reads a token string except literal digits; typo/morphology robustness becomes a model property; `BOUND` direction, inclusivity, and clock rendering cannot disagree between renderer and decoder because there is one symbol for each.

**Label space** (`packages/cron/src/model/labels.ts`, replaces the 15-label list):

```
O  RECUR  GLUE  EDGE  COUNT
WEEKDAY_{MON..SUN}                    7
DAYGROUP_{WEEKDAY,WEEKEND,DAY}        3
HOUR_{0..23}  HOUR_LIT                25   (HOUR_LIT = digits, read literally)
MINUTE_{0,15,30,45}  MINUTE_LIT        5
MERIDIEM_{AM,PM}  MERIDIEM_NONE        3
ORDINAL_{1..5,LAST}                   6
BOUND_{START,END_EXCL,END_INCL,BETWEEN} 4
EXCEPT
UNIT_{MIN,HOUR,DAY,WEEK,MONTH}        5
INTERVAL_{2..12}  INTERVAL_LIT        12
DOM_LIT                               1    (literal 1–31)
```

≈ 80 labels. Every `*_LIT` label means "value is the token's digits"; nothing else is read from text.

**Changes.**

1. `labels.ts`: new list + `valueOf(label)` table; `STRUCTURAL_LABELS`/`CONTENT_LABELS` regenerated.
2. `packages/cron/training/src/kinds.ts`, `renderer.ts`, `sentence.ts`: `Span.label` carries the typed label. Renderer already knows the value it rendered; this is a one-line change per `s.add(...)` site.
3. `parser/decode.ts`: delete `weekdayOf`, `WEEKDAY_GROUPS`, `ORDINAL_ALIASES`, `BOUND_ALIASES`, `NAMED_HOURS`, `MINUTE_IDIOMS`, `FREQUENCY_IDIOMS`, `ADVERB_FREQUENCIES`, `meridiemOf`, `UNIT_ALIASES`, single-edit alias recovery (`decode.ts:293-342`). `decodeClause` reads `valueOf(label)` or `parseInt(text)` for `_LIT`.
4. `schedule-semantics.ts` keeps only arithmetic (12h→24h, `END_INCL` +1 day), no word tables.
5. `training/cron/model.py`, `inference.ts`: `role_classes` from `LABEL_COUNT`; export/import unchanged otherwise.
6. `oracle.ts`: must be 100 % on gold, kind-seeds, **and** `--kinds-sample 200000` before train. With typed labels, an oracle failure can only be a renderer bug.
7. `noise.ts`: allowlist becomes _every_ label — noise no longer breaks decodability, because the decoder does not read the noised string. Only `_LIT` tokens stay clean.

**Train.** A1 = same recipe as E1 (`--data-prefix kinds`, 7 × 200 k, batch 64). Then A2 = A1 with gpu-time's recipe: `--fresh-each-epoch` (add to `dataset.py`/`train.py`: resample 300 k rows per epoch from the 13.7 M pool, seed + epoch), batch 1024, `--init` from A1 best. A2 tests the one recipe difference with evidence behind it (gpu-time's `MODEL_CARD`: cold start loses rare forms).

**Keep.** DECODE = 0 and GOLD = 0 on H by construction; ROLE on H not worse than E1; H ≥ E1. Expect `noisy` family and `weekday` family to move first.

**Cost.** ~3 KB weights. No GPU for steps 1–7. Two trains.

---

## Track C — Real-text supervision with executable verification

**Hypothesis.** H does not move because the training distribution is the generator's, not English. The fix is to label real sentences, and the only acceptable labeler is one whose output is checked by execution, not by opinion.

**Sources** (all licensed, hash-pinned in `packages/cron/training/data/real/manifest.json`):

- Tatoeba English (CC-BY 2.0 FR) — port gpu-time's `fetch-corpus.ts` filter; use as _negatives_ and as carriers.
- Microsoft Recognizers-Text DateTime specs (MIT) — recurrence-bearing subset; dev/reserved split by hashed masked frame as gpu-time does.
- Cron-domain text: titles/bodies of GitHub issues in cron-related repos, Stack Overflow questions tagged `cron` (CC-BY-SA; keep only the natural-language schedule phrase), calendar/reminder app reviews. Collected by `training/src/real/collect.ts`, deduplicated, PII-stripped.
- LLM paraphrases of `human-v*` (the originals stay `trainingEligible:false`; paraphrases must differ by ≥ 40 % token edit distance and are tagged `source: paraphrase-of-human`).

**Pipeline** (`training/src/real/`, reusing `training/src/llm/{client,runner,schema,verify}.ts`):

1. `label.ts`: frontier LLM, constrained JSON to `ScheduleIntent | null`, k = 3 samples (T = 0, 0.4, 0.8). Prompt contains the schema and 12 fixed exemplars, never the runtime parser's output.
2. `agree.ts`: canonicalise (`canonical-intent.ts`), keep only records where all 3 agree.
3. `execute.ts`: compile → cron → next 8 occurrences; reject if compile fails (`supported:false` allowed only for intents the compiler declares unsupported, and then the record is kept for the tagger, not for EXEC).
4. `roundtrip.ts`: show the LLM _only_ the 8 occurrences + unit, ask for the intent; keep if canonical-equal to step 2. This catches labels that are self-consistent but wrong.
5. `project.ts`: LLM also returns character spans for schedule tokens with typed labels (Track A); `align.ts` snaps to tokenizer boundaries; `oracle.ts` decode of the spans must equal the intent, else drop.
6. Output `data/real/<source>.jsonl` with `{text, intent, spans, provenance, agree:3, exec:true, roundtrip:true}`; `consumed.json`-style hash so a record never leaks into eval.

Rejected records are written to `data/real/rejected.jsonl` with the failing stage; that file is the coverage report for what the generator and compiler cannot express.

**Trains.** C1 = A2 mix + real data at RT ≈ 10 %. C2 = RT ≈ 30 %. C3 = real data only (`kinds` as validation). Same hyperparameters as A2.

**Keep.** H↑ with ROLE↓ as the primary; H-silent not worse. If C3 ≥ C2 the generator is demoted to a coverage backstop and `kinds.ts` stops growing.

**Forbidden.** Human-written eval text in training (paraphrases only). LLM labels without steps 3–4. Any hand edit of a label.

**Cost.** LLM tokens for ~50 k records × 5 calls; no GPU until C1.

---

## Track B — Seq2seq with grammar-constrained decoding

**Hypothesis.** The tagger + boundary + actionable + clause-grouping pipeline exists only because the model cannot emit structure. A decoder that emits the linearised intent under a CFG mask removes the label→AST gap as a category, and folds rejection into the same output (`REJECT` production).

**Target language** (`packages/cron/src/model/grammar.ts`, generated from `types/intent.ts`):

```
intent   := "(" freq time? days? dom? ord? bounds? except? ")" | "(reject" reason ")"
freq     := "(every" INT unit ")"
time     := "(at" INT INT ")"
days     := "(days" weekday+ ")"
...
```

Values are emitted as pointers to input tokens (copy head) or as closed-class symbols, never generated free-form. Grammar mask table is precomputed at export (`export.py`) and shipped as a small bitset; per-step masking is a lookup (parser-stack classification is unnecessary at this grammar size).

**Model** (`training/cron/seq2seq.py`): encoder = 2-layer transformer d = 128, or the existing recurrence at hidden 128; decoder = 2-layer transformer d = 128 with cross-attention + copy head. ≈ 800 k params, int8 → ≈ 800 KB. Output length ≤ 40. CPU inference in `inference-seq2seq.ts`, one step per output token, greedy under mask.

**Data.** Same records as Track A/C; `featurize-seq2seq.ts` linearises `intent` + spans into the target sequence. Boundary/actionable/speech-act shards are not needed.

**Train.** B1 = generator mix; B2 = C2 mix. Loss = CE on target tokens under the mask (masked tokens contribute nothing).

**Keep.** Head-to-head against the best Track A/C tagger on identical data: H, EXEC, FA@cov. Seq2seq wins if H higher **and** latency ≤ 20 ms p95 on CPU for 30-token input. Otherwise stays a research branch.

**Cost.** New Python model file, new TS inference path, new featurizer. Two trains. This is the largest item; it starts only after A2 and C1 have rows.

---

## Track D — Pretrained encoder, then distil

**Hypothesis.** Hashed FNV / vowel-stripped / char-n-gram features are 2019 feature engineering with collisions; a small pretrained byte- or SentencePiece-level encoder carries morphology and typo tolerance the generator cannot teach.

1. `training/cron/pretrain.py`: 2-layer d = 128 encoder, masked-token objective on Tatoeba + cron-domain text from Track C (unlabelled, ~5 M sentences). ~2 M params.
2. Fine-tune as the Track A tagger head (or Track B encoder).
3. If size matters later: distil into the 32/64-dim recurrence (old E8) — the teacher now has _language_ knowledge, which is what made E8 worth doing in the first place.

**Keep.** H↑ at RT held constant vs. the best hashed-feature model. Run after C1.

---

## Track E — Train against the metric

**Hypothesis.** Token-level CE is a proxy; token accuracy 0.99 with exact-AST 0.91 is the proxy gap.

1. Sequence-level: CRF / globally normalised decoding over role logits with a learned transition matrix (old E7 generalised to training time, not just inference).
2. Executable reward: minimum-risk training where risk = 1 − EXEC(pred, gold), sampling 4 decodes per example. Applies to Track A (via Viterbi samples) and Track B (via beam).

**Keep.** EXEC↑ and H-silent↓ at equal ROLE. Run after A2.

---

## Track F — Rejection as calibrated abstention

**Hypothesis.** A 9-way pooled speech-act classifier sharing 32 dims with the tagger cannot bound false accepts. Replace classification with a score + a threshold fitted to a guarantee.

1. Remove `schedule_kind_head` and `speech_act_head`; keep `actionable` only as the Track A span property (or drop when Track B lands).
2. `training/cron/abstain.py`: energy score `−logsumexp(role logits)` averaged over content tokens, plus Mahalanobis distance of pooled state to the training mean. Both exported.
3. `calibrate.py`: split-conformal threshold on a held-out set of **real** negatives (Tatoeba + Recognizers non-recurrence + Track C rejections) and real positives (Track C accepted). Target FA ≤ 1 % at coverage ≥ 90 %; report the achieved pair as `FA@cov`.
4. `parser/model.ts`: reject when score exceeds threshold; confidence bands derived from the same score.

**Keep.** FA@cov meets target on H-neg/H-pos with H not worse. Independent of all other tracks; can run on the A1 checkpoint.

---

## Sequencing

```
week 1-2   A.1-7  typed labels, decoder deletion, oracle 100% incl. kinds-sample   [no GPU]
           C.1-6  real-text pipeline, first 10k verified records                   [LLM only]
           F.1-3  abstention scoring + conformal threshold on A1                   [CPU]
week 2     A1, A2 trains                                                           [GPU]
week 3     C1, C2 trains; F applied to each                                        [GPU]
week 3-4   E.1 CRF on best of A2/C2; D.1 pretrain in parallel                      [GPU]
week 4-6   B grammar + model + TS inference; B1, B2 vs best tagger                 [GPU]
week 6     C3 (real only) if C2 ≥ C1; D.2 fine-tune; decision: tagger or seq2seq
then       promotion per EXPERIMENT-PLAN.md Phase 5, with EXEC and FA@cov added
```

Decision gates:

- After A2: if DECODE+GOLD ≠ 0, stop and fix the renderer; nothing else proceeds.
- After C2: if H did not move ≥ 5 points with RT = 30 %, the bottleneck is not data; prioritise B and D over C3.
- After B2: pick one stack. Two inference paths do not ship.

## Mapping from `EXPERIMENT-PLAN.md`

| old                                       | new                                                                |
| ----------------------------------------- | ------------------------------------------------------------------ |
| P2.1 G2 grammatical held-out              | kept as-is; G2 stays a synthetic sanity set, H/EXEC decide         |
| P2.2 gate reads G2 + H                    | `gate.ts` reads H, EXEC, FA@cov, O; G2 as floor                    |
| E2 LLM carriers with label projection     | subsumed by C (real text instead of paraphrased generator output)  |
| E3 LLM negatives                          | C sources + F calibration set                                      |
| E4 invariance pairs / E6 consistency loss | deferred; re-evaluate after C2 (real data may make it moot)        |
| E5 lexical holdout                        | replaced by real-text validation (`kinds` becomes the held-in set) |
| E7 transition decoding                    | E.1                                                                |
| E8 distillation                           | D.3                                                                |
| E9 span-aware speech-act pooling          | dropped; F replaces speech-act classification                      |
| E10 calibration under shift               | F.3                                                                |
| E11 capacity retest                       | budget constraint dropped; D and B cover it                        |
| E12 typo robustness                       | A.7 (noise on all labels) + D                                      |

## Rules specific to v3

- A `_LIT` label is the only place the decoder may read token text, and only via `parseInt`.
- No record enters training without `agree:3, exec:true, roundtrip:true` (Track C) or oracle-green spans (generator).
- A ledger row without `EXEC`, `FA@cov`, `RT`, `stack` is invalid.
- Human-written eval text never enters training; paraphrases must carry `paraphrase-of-human` provenance and are excluded from any set used to compute H.
- The promotion criterion is H/EXEC/FA@cov on real text. R and in-distribution `exactSeq` are diagnostics and may not be quoted as accuracy.

## Task checklist

- [ ] A.1 `labels.ts` typed label list + `valueOf`
- [ ] A.2 renderer/kinds/sentence emit typed labels
- [ ] A.3 delete lexicon lookups from `parser/decode.ts`; `_LIT` via `parseInt` only
- [ ] A.4 `schedule-semantics.ts` arithmetic-only
- [ ] A.5 model/inference `role_classes` from `LABEL_COUNT`
- [ ] A.6 oracle 100 % on gold + kind-seeds + `--kinds-sample 200000`; CI gate includes kinds-sample
- [ ] A.7 `noise.ts` allowlist = all non-`_LIT` labels
- [ ] A1 train (E1 recipe) + ledger row
- [ ] A2 train (`--fresh-each-epoch`, batch 1024, `--init` A1) + ledger row
- [ ] C.1 sources + `manifest.json` + `collect.ts`
- [ ] C.2 `label.ts` k=3 constrained
- [ ] C.3 `agree.ts`, C.4 `execute.ts`, C.5 `roundtrip.ts`, C.6 `project.ts`
- [ ] C.7 `data/real/*.jsonl` + `rejected.jsonl` coverage report
- [ ] C1 (RT 10 %), C2 (RT 30 %) trains + rows
- [ ] F.1 remove kind/speech-act heads, F.2 energy + Mahalanobis, F.3 conformal calibration, F.4 `model.ts` abstention
- [ ] E.1 CRF training on best tagger
- [ ] D.1 pretrain encoder, D.2 fine-tune
- [ ] B.1 `grammar.ts` + mask export, B.2 `seq2seq.py`, B.3 `featurize-seq2seq.ts`, B.4 `inference-seq2seq.ts`
- [ ] B1, B2 trains; head-to-head decision
- [ ] C3 (real only) if C2 ≥ C1
- [ ] E.2 executable-reward MRT on the chosen stack
- [ ] `gate.ts` reads H, EXEC, FA@cov, O
- [ ] Promotion (Phase 5 + EXEC + FA@cov)
