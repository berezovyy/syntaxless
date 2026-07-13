# Heads-only goal — progress

Protocol: `EXPERIMENT-PLAN.md`. Ledger: `EXPERIMENTS.md`.

## Status

- **Active:** E1b is a **REVERT**. Working candidate is still E1. Next is more “not a schedule” data (ACT / H-neg).
- **Hacks:** gone. Public `parse()` is heads-only. Missing speech-act is `model-error`.
- **B0:** H 733/1646, H-neg 234/760, H-silent 233.
- **E1** (`2d5b0712…`): H 748/1646, H-neg **357/760** (worse), H-silent 170, interval 30/67, weekday 26/81. Attribution ROLE 409 / ACT 357 / CONF 132. Candidate not promoted.
- **O:** `oracle:gate` **244849/244849** (gold + kind-seed × 50 + 200k kinds.jsonl). Green.
- **E1b** (`e00a552cd…`): **REVERT.** H 726, H-neg 446, H-silent 215. Worse than E1. Not shipped.

## Log

| When       | Cycle     | Result                                                                                               |
| ---------- | --------- | ---------------------------------------------------------------------------------------------------- |
| 2026-09-15 | P0        | Oracle harness + `oracle:gate`. Human `--partition regression`. B0 recorded.                         |
| 2026-09-15 | P1.1–P1.3 | Actionable-as-span; clause-level decode; `schedule-semantics.ts`.                                    |
| 2026-09-15 | P1b       | Gold/generator match decoder. Kind-seed oracle 100%.                                                 |
| 2026-09-16 | E1        | Trained on a corpus whose noise rows are unreachable gold. H 748, H-neg 357. Kept on disk, not KEEP. |
| 2026-09-16 | noise     | Default `spellingVariant` restricted to O/RECUR/EDGE. Dropped BOUND/EXCEPT target-noise.             |
