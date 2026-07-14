/**
 * Track C project: keep spans only when oracle-decode of those spans equals
 * the agreed intent.
 */
import { isDeepStrictEqual } from "node:util";

import type { Label } from "../../../src/model/labels";
import type { ScheduleIntent } from "../../../src/types/intent";
import { canonicalizeIntent } from "../canonical-intent";
import { oracleDecode } from "../oracle";
import type { OracleSpan } from "../oracle";

export interface ProjectSpan {
  start: number;
  end: number;
  label: Label;
  clauseStart: boolean;
  actionable?: boolean;
}

export function projectOracleEquals(
  text: string,
  spans: readonly ProjectSpan[],
  intent: ScheduleIntent | null,
  speechAct: "request" | "none" = intent === null ? "none" : "request",
): boolean {
  const outcome = oracleDecode({
    id: "project",
    text,
    spans: spans as OracleSpan[],
    intent,
    speechAct,
    source: "gold",
  });

  if (intent === null) return !outcome.ok || outcome.intent === null;
  if (!outcome.ok) return false;

  return isDeepStrictEqual(canonicalizeIntent(outcome.intent), canonicalizeIntent(intent));
}

export function projectKeep<
  T extends {
    text: string;
    spans: readonly ProjectSpan[];
    intent: ScheduleIntent | null;
  },
>(records: readonly T[]): T[] {
  return records.filter((record) => projectOracleEquals(record.text, record.spans, record.intent));
}
