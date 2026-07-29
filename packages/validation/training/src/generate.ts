/**
 * Corpus record contract shared by the corpus generators (`compact.ts` for
 * training, `backward.ts` for the strict-grammar baseline), the featurizer,
 * evaluation, and the Python `dataset.py` loader.
 */
import type { FieldIntent } from "../../src/types/intent.js";
import type { Span } from "./sentence.js";

export type Split = "train" | "validation" | "heldout";

export type Family = "constraint" | "negative";

export interface CorpusRecord {
  id: string;
  split: Split;
  seed: number;
  family: Family;
  template: string;
  fingerprint: string;
  text: string;
  intent: FieldIntent | null;
  spans: Span[];
  /** Valid intent whose declared compiler target is intentionally unsupported. */
  compileFailure?: "nullable";
}

export const SPLITS: readonly Split[] = ["train", "validation", "heldout"];
