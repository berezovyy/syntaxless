# Goal: `@syntaxless/cron` heads-only — 100% reserved-carrier, keep/revert until it is

**Cycle protocol:** `EXPERIMENT-PLAN.md` supersedes the v1 loop below. Do not train while the oracle (`bun run --cwd packages/cron/training oracle:gate`) is red. Keep/revert on H / H-neg / H-silent / G2 / INV, not on reserved-carrier R.

Launch in Grok Build:

```
/goal GOAL.md
```

Goal kind: **code-change**. Workspace: `/home/berezovyy/Projects/syntaxless` (this repo). Not the validation worktree.

This goal is **not done when code exists**. It is done when the **public** `parse()` / `cron()` path is heads-only **and** reserved-carrier gold is **100%** on every listed contract. A rejected gate, a candidate file, in-distribution `speechActAccuracy`, or “we trained” is not done.

Time is not a stop. Prefer another real train over any regex restore, any decoder patch on failed wording, or any lowered bar. Budget is **up to 100 recorded train/export/eval cycles**, or months of wall time, whichever comes first **after** the bars below are met. Stop early **only** when those bars are met. Operator cancel is the other stop.

---

## One-line objective

Public `@syntaxless/cron` parse stays heads-only (overlay hacks already deleted — keep them gone). Improve generator kinds and the real QAT trainer **one change per cycle**, score overlay-off / heads-only on reserved-carrier gold, **keep what helps and revert what hurts**, and repeat until `tokenRoles`, `rawDecode`, `exactAst`, and `compiledExpression` are each 100%, every family is 100%, silent meaning-change is 0, negative false-accept is 0, and typo invariance is 100%. Then promote the candidate. Do not promote sooner.

---

## Binding documents

Read before every cycle:

- `tmp/INTENT.md` — parse/compile split, on-device, fail closed
- `tmp/cron-discourse/design.md` — why heads exist
- `tmp/cron-discourse/13-kind-driven-eval-loop.md` — a miss is a **kind**, not a noun regex
- `packages/cron/evaluation/README.md` — never train on consumed `human-v*.jsonl` wording
- `EXPERIMENTS.md` — keep/hurt ledger (append a row every cycle; never skip)
- `PROGRESS.md` — short status + next hypothesis

Phase docs `10–12` said retire overlay only after a frozen overlay-off gate. **This goal already deleted the hacks.** Do not restore them. Train until heads-only is 100%.

---

## Starting state (do not redo)

Already true — treat as regression constraints, not a work queue:

- Overlay classifiers are gone from `packages/cron/src/parser`: no `scopeActionableCorrection`, `unsafeScheduleContext`, `hasDirectScheduleRequest`, `unrequestedDeclarativeContext`.
- No `/bakery/`, `/flights/`, `/clock strikes/` under `packages/cron/src/parser`.
- `explicit*` does not overwrite neural roles or set `confidence = 1` from a raw-string hit.
- No post-parse mutation of `intent.frequency`.
- Public `parse()` / `cron()` is heads-only (`overlay` may exist as a no-op diagnostic flag; it must not change labels or confidence).
- Missing speech-act head is `model-error`, never a regex fallback.
- Lexicon tables and clock arithmetic from **already labeled** tokens remain (that is not a hack).
- Candidate weights live at `packages/cron/training/candidate/weights.gen.ts`. Active bundled weights at `packages/cron/src/model/weights.gen.ts` are **not** promoted until the 100% bars.
- Best reserved-carrier **exactAst** so far is **133/360** (discourse-goal-6). Interval / weekday-list / exception families are still **0/40**. `parse("every weekday at 9")` is still not gold on live candidate heads (conflicting calendar selectors / low-confidence). Not promoted.

Reserved gold includes request-looking **and** descriptive-looking **carriers** wrapping a schedule that gold still expects as a **successful parse** (`the train leaves at`, `our rehearsal begins at`, `note that the office opens at`, …). Heads-only must generalize to those held-out prefixes. Do not copy `RESERVED_PHRASES` into training.

---

## Known keep / hurt

Do not repeat a hurt. Full rows live in `EXPERIMENTS.md`. Summary:

| Change                                                                            | Reserved exactAst | Verdict                                       |
| --------------------------------------------------------------------------------- | ----------------- | --------------------------------------------- |
| Delete overlay, no retrain                                                        | 21/360            | baseline after hacks gone                     |
| QAT on full `kinds-heads` (~10M), hidden 32, ~50k samples/epoch                   | **75/360**        | **KEEP** vs 21                                |
| Same corpus, 200k samples/epoch                                                   | 71/360            | **REVERT** (more of the same hurt reserved)   |
| hidden 48, 200k, same corpus                                                      | 66/360            | **REVERT** (capacity without new kinds hurt)  |
| Extra disjoint request carriers, **gap-subset only** train                        | 20/360            | **REVERT** (gap-only destroys other families) |
| Mix extra carriers into the **full** 10M corpus                                   | **104/360**       | **KEEP** vs 75                                |
| More abstract kind wraps (cycle-is, ISO DOM, exception lead-*) + full mix         | **116/360**       | **KEEP**                                      |
| Preposition-tail carriers (`… is at`) + trailing O + exception lead-* on carriers | **133/360**       | **KEEP** (current best)                       |

What the zeros say: more capacity / more of the same 10M does **not** lift interval, weekday-list, or exception off 0/40. Extra **disjoint** request carriers mixed into the **full** corpus does. Gap-only training is poison. Next cycles must attack remaining zero families and `every weekday at 9` (DAYGROUP vs DOM/HOUR on `9`) with **new abstract kinds or mix**, then a full regen+train — not a gap-only run, not hidden-size bumps, not copying reserved prefixes.

---

## Acceptance criteria

1. Overlay hacks stay gone from the public `@syntaxless/cron` parse path. The identifiers and patterns in **Starting state** remain absent. TypeScript may still compute facts from already labeled tokens (clocks, lexicon aliases, compile/validate). Classifying English in `decode.ts` is forbidden.

2. Smoke on the **public** path (no overlay-on required): `parse("every weekday at 9")` yields week/1, Monday–Friday, 09:00, and `compile()` of that intent is `0 9 * * 1-5`. Bakery/flights descriptives reject via **speech-act** (`unsafe-context` or equivalent), not a noun regex.

3. Reserved-carrier gold, scored **overlay-off / heads-only** via the shipped `evaluate.ts` / `gate.ts` path against the **candidate** (then, only after this criterion, the promoted active weights):
   - `tokenRoles` 100%
   - `rawDecode` 100%
   - `exactAst` 100%
   - `compiledExpression` 100%
   - overall 100%
   - **every family** 100% (interval, weekday-list, weekday-range, ordinal, day-of-month, time-of-day, bounds, exception, terse — none left at 0/40)
   - silent meaning-change **0**
   - negative false-accept **0**
   - typo invariance **100%**

   Contracts are **never summed**. A 100% on a consumed `human-v*.jsonl` after a decoder patch is **not** this criterion. In-distribution `speechActAccuracy` alone is **not** this criterion.

4. `ScheduleIntent` and `compile()` are unchanged. No new runtime npm deps. User text never leaves the device. Training/eval must not copy `packages/cron/evaluation/*.jsonl` wording or `RESERVED_PHRASES` into kinds, templates, lexicon, or decode. Wrapped schedule words keep inner roles (`daily` stays `INTERVAL`, not `O`).

5. Every train/export/eval cycle in this goal has **exactly one** hypothesized change, a row in `EXPERIMENTS.md` with keep **or** revert, and the candidate on disk is the last **KEEP** (never a hurt run left in `packages/cron/training/candidate/weights.gen.ts`). Promotion copies candidate → `packages/cron/src/model/weights.gen.ts` **only** when criterion 3 is 100%. Export may write a candidate when gate exits 1; that is not promotion.

---

## Verification plan

1. **gating.** Search `packages/cron/src/parser` (and public parse entry) for `scopeActionableCorrection`, `unsafeScheduleContext`, `hasDirectScheduleRequest`, `unrequestedDeclarativeContext`, `/bakery/`, `/flights/`, `/clock strikes/`, post-parse `intent.frequency` mutation, and `confidence = 1` from `explicitScheduleEvidence` / raw-string hits. Observation: zero matches. Capture `{SCRATCH}/hack-search.txt`.

2. **gating.** Existing bun tests that call shipped `parse` / `cron` (`packages/cron/src/parser/decode.test.ts`, `model.test.ts`, `packages/cron/tests/model-backed-parse.test.ts` or equivalent): `parse("every weekday at 9")` equals week/1 Mon–Fri 09:00; a bakery/flights descriptive fails closed from speech-act; missing speech-act head is `model-error`. Run those tests; capture `{SCRATCH}/parse-smoke.log`. Tests must call shipped functions, not a reimplementation.

3. **gating.** From a fresh consumer (import `@syntaxless/cron` or the package `src`/`dist` entry, not a copy of parse), call `parse("every weekday at 9")` and assert the same intent fields. Capture `{SCRATCH}/library-parse.log`.

4. **gating.** Score heads-only on reserved-carrier gold:

   ```sh
   bun run --cwd packages/cron/training evaluate --weights packages/cron/training/candidate/weights.gen.ts
   bun run --cwd packages/cron/training gate
   ```

   Observation: each of `tokenRoles`, `rawDecode`, `exactAst`, `compiledExpression` is 100%; overall 100%; every family 100%; silent meaning-change 0; negative false-accept 0; typo invariance 100%. Copy the report to `{SCRATCH}/reserved-carriers.json`. Do not sum contracts. Do not use consumed human-eval wording as the gate.

5. **evidence.** `rg` shows no new decoder regex / noun denylist under `packages/cron/src/parser`. `EXPERIMENTS.md` has one row per cycle with keep or revert. Browser privacy test still asserts user text is not on the wire (`packages/cron/tests/browser.ts`) if Playwright can run; otherwise capture why it cannot. `compile()` of the weekday intent still yields `0 9 * * 1-5`. Cron package tests besides overlay-on-only cases stay green.

---

## Cycle protocol (this is the work)

Each cycle, **in order**. One hypothesized change. No mixing corpus mix + lr + hidden size in the same run.

1. Confirm parser still has no overlay hacks (`rg` as in verification 1). If any returned, **delete them before training**. Do not train on a regex scaffold.
2. Read `EXPERIMENTS.md` and `PROGRESS.md`. Name the miss class from the last reserved-carrier report (interval wrap, weekday-list under scene-setting prefix, exception lead, `9` as DOM vs HOUR, …).
3. Change **one** thing: an abstract generator kind / carrier / mix / noise / train schedule. Generator kinds only. **Never** copy reserved prefixes or human-eval wording. Train carriers (`POSITIVE_CARRIERS` in `packages/cron/training/src/kinds.ts`) must stay disjoint from `RESERVED_PHRASES` in `packages/cron/training/src/background.ts`.
4. If the corpus changed: regenerate and featurize the **full** mix (kinds + any extra carrier rows). **Do not** train on a gap-only subset. Commands:

   ```sh
   bun run --cwd packages/cron/training gen:kinds
   bun run --cwd packages/cron/training featurize:kinds
   # if extra disjoint carrier JSONL exists, merge then featurize the mix prefix
   ```

5. Train with the real harness (QAT, speech-act + actionable heads, CUDA `auto`):

   ```sh
   cd training/cron && uv run python train.py --run <id> --device auto --qat \
     --data-prefix <featurized mix> --samples <n> --epochs <n>
   ```

   Use a unique `--run` id. Do not assign `status=$?` in zsh (`status` is readonly). Launch GPU trains as long-running background tasks; do not busy-wait.

6. Export to **candidate only**:

   ```sh
   uv run python training/cron/export.py --checkpoint training/cron/runs/<id>/best.pt
   ```

   Do **not** copy to `packages/cron/src/model/weights.gen.ts` until criterion 3 is 100%.

7. Score evaluate + gate (verification 4). Write **one row** to `EXPERIMENTS.md` and a line to `PROGRESS.md`.
8. **Keep** if reserved exactAst (and the other contracts, and safety columns) did not get worse than the current best KEEP, and the targeted family moved the right way. Leave that candidate on disk.
9. **Revert** if reserved exactAst dropped, a previously non-zero family collapsed, silent meaning-change or false-accept got worse, or the change is gap-only / reserved-copy / regex. Restore the last KEEP candidate weights onto `packages/cron/training/candidate/weights.gen.ts`. Do not keep a hurt because the code is nicer.
10. If not 100%, next cycle from remaining error families. If 100%, promote the candidate, freeze size/privacy (Brotli gates, browser test), re-run evaluate/gate plus smoke, stop.

Cap: **100 recorded cycles**. Do not stop at 99.9%, a rejected gate, or “a candidate exists.” After two failed fixes for the **same** defect, restating the mechanism is required before a third.

---

## Hard rules

- Do not restore overlay regex, noun denylists, verb lists, or “if bakery then reject.”
- Do not copy `packages/cron/evaluation/*.jsonl` or `RESERVED_PHRASES` into kinds, templates, lexicon, or decode.
- Do not train on consumed `human-v*.jsonl` (`trainingEligible: false`).
- Do not label wrapped schedule content as all `O`. Inner roles stay; speech-act + actionable carry discourse.
- Do not use schedule-kind to boost confidence.
- Do not change `ScheduleIntent` or `compile()`.
- No new runtime npm deps. User text never leaves the device. No hosted inference.
- Do not promote a candidate that is not 100% on criterion 3.
- One change per training run. A mixed accidental run is invalid: log it as such and do not treat its score as a keep.
- Do not train gap-only. Do not bump hidden size as the next move (already hurt).
- `@syntaxless/validation` is a different worktree and a different goal. Do not edit it here.

---

## Non-goals

- `@syntaxless/validation` million-scale factory (other worktree).
- Changing `ScheduleIntent` or `compile()` semantics.
- Training on consumed `human-v*.jsonl` or treating 100% on those files as promotion.
- Replacing deleted overlay with new regex / verb lists / noun denylists.
- New runtime npm dependencies or hosted inference.
- Using schedule-kind to boost confidence.
- Labeling wrapped schedule content as all `O`.
- Copying `RESERVED_PHRASES` into `POSITIVE_CARRIERS` so reserved-carrier “generalization” is memorization.
- Authoring a new `human-vN.jsonl` as a promotion gate.

---

## Assumed scope

`packages/cron/training/` (`kinds.ts`, `discourse.ts`, `correction.ts`, renderer, `backward.ts`, `featurize-kinds.ts`, `evaluate.ts`, `gate.ts`, candidate weights), `training/cron/` (`train.py`, `export.py`, `dataset.py`, `model.py`), `packages/cron/src/model/` (speech-act heads, **promotion only at 100%**), `packages/cron/src/parser/` (stay-gone hack search only), `packages/cron/tests/`, `EXPERIMENTS.md`, `PROGRESS.md`. GPU / `uv` for real trains. RTX 2070 is sufficient.

Do not touch decode to paper over a family.

---

## Implementation approach

Stay on the last KEEP candidate. Attack remaining **zero families** (interval, weekday-list, exception) and `every weekday at 9` with abstract kinds and disjoint request-carrier wraps mixed into the **full** corpus, then real QAT, export candidate, score, keep or revert, log. Repeat. Isolate train I/O from decode so evaluate always loads shipped artifacts. Background the GPU job; when it finishes, export and score in the same cycle. Do not ask the operator whether to start the next cycle.

---

## Task checklist

- [ ] Confirm overlay hacks still gone (`rg` + `{SCRATCH}/hack-search.txt`). If any match, delete them before any train.
- [ ] Read `EXPERIMENTS.md`. Next hypothesis must target a still-zero family or weekday-at-9, and must not repeat a hurt row.
- [ ] One kind/carrier/mix change; keep `POSITIVE_CARRIERS` disjoint from `RESERVED_PHRASES`; do not copy evaluation JSONL.
- [ ] Full regen + featurize (never gap-only). Real `training/cron/train.py` QAT CUDA auto. Unique `--run` id.
- [ ] Export candidate only. Score `evaluate` + `gate`. Append `EXPERIMENTS.md` + `PROGRESS.md`. Keep or revert.
- [ ] Repeat until criterion 3 is 100% (up to 100 cycles). Then promote candidate, re-run evaluate/gate + smoke, capture `{SCRATCH}/reserved-carriers.json`.
- [ ] Do not stop at 99.9%, a rejected gate, or “candidate exists.” Stop only at 100% on the listed checks with hacks gone, or operator cancel.

---

## Risks / contradictions

- Reserved-carrier heads-only exactAst is **133/360** at best. Getting to 360/360 will take many GPU cycles. A single session, a compaction, or a CPU-only environment may not finish criterion 3. That is in-progress, **not** a reason to restore regex, copy reserved phrases, or lower 100% to the historical 0.95 / 0.5% SMC floors.
- 100% on every reserved-carrier contract plus 0 silent meaning-change is stricter than `gate.ts` constants (`MIN_OVERALL_ACCURACY = 0.95`, `MAX_SILENT_MEANING_CHANGE_RATE = 0.005`). This goal’s bars stay 100% / 0 as written. Do not “pass the gate” and promote.
- Export writes a candidate even when gate exits 1. Gate non-zero is expected until 100%. Do not treat a written candidate as a ship.
- zsh: `status` is readonly. Never `status=$?`; that aborts the rest of a `&&` chain (already lost an evaluate after export once).
- Descriptive-looking reserved carriers are still **positive gold**. Do not teach the speech-act head to refuse `the train leaves at …` as `descriptive` if gold expects a schedule.
