/**
 * EXEC: prediction and gold compile, and the compiled expressions match
 * (same 5-field cron ⇒ same next-N occurrences).
 */
import { compile } from "../../src/compiler/cron";
import type { ScheduleIntent } from "../../src/types/intent";
import { canonicalizeIntent } from "./canonical-intent";

export function execEqual(predicted: ScheduleIntent | null, gold: ScheduleIntent | null): boolean {
  if (predicted === null && gold === null) return true;
  if (predicted === null || gold === null) return false;
  const left = compile(canonicalizeIntent(predicted));
  const right = compile(canonicalizeIntent(gold));

  if (left.supported && right.supported) return left.expression === right.expression;
  if (!left.supported && !right.supported) return left.code === right.code;

  return false;
}

export function execRisk(predicted: ScheduleIntent | null, gold: ScheduleIntent | null): number {
  return execEqual(predicted, gold) ? 0 : 1;
}
