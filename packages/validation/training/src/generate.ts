/**
 * Corpus record contract shared by the backward generator and the featurizer.
 *
 * Sampling-family generate.ts is not the v0 path: kinds × templates in
 * `backward.ts` own the JSONL. This module only names the record shape so
 * Python `dataset.py` and featurize share one schema.
 */
import type { FieldIntent } from "../../src/types/intent";
import type { Span } from "./sentence";

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
