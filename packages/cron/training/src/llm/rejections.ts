/**
 * Rejection reasons and the rejections JSONL writer.
 *
 * A rejection is data, not an error: Phase 12's coverage report maps reason
 * codes to actions (`unknown-lexeme` grows the lexicon, `intent-text-mismatch`
 * hardens the prompt, `unsupported` is a compiler gap). The compiler's own
 * failure code is passed straight through so representability has exactly one
 * source of truth.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import type { CompileFailureCode } from "../../../src/compiler/index.js";

export type RejectionReason =
  | "schema-invalid"
  | "text-invalid"
  | "unsupported"
  | "negative-has-schedule"
  | "unknown-lexeme"
  | "unaligned"
  | "intent-text-mismatch"
  | CompileFailureCode;

/** One rejected candidate. `candidate` is the raw proposal, unmodified. */
export interface RejectedRecord {
  reason: RejectionReason;
  text: string;
  candidate: unknown;
  angleId: string;
  model: string;
  promptHash: string;
  runId: string;
  detail: string;
}

/** Write rejects to `<out>.rejections.jsonl`, one JSON object per line. */
export function writeRejections(path: string, records: readonly RejectedRecord[]): string {
  mkdirSync(dirname(path), { recursive: true });
  const body = records.map((record) => JSON.stringify(record)).join("\n");

  writeFileSync(path, body.length > 0 ? `${body}\n` : "");

  return path;
}
