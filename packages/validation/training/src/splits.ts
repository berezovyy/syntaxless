/**
 * Split contract for the kinds catalog.
 *
 * Train, validation, and held-out share kinds but not frames. Frame assignment
 * lives in `backward.ts` (`assignSplits`): roughly 75% / 12.5% / 12.5% by a
 * stable hash of the structural pattern, then repaired so every kind has at
 * least one distinct frame in each split. Held-out wording is a different
 * frame of the same kind (operator alias, unit alias, or word order), not a
 * reserved-carrier overlay.
 */
export { SPLITS, type Split } from "./generate";
export type { ExampleSplit } from "./kinds";
