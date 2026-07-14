/**
 * Track C agree: keep a record only when k=3 constrained intents canonicalize equal.
 */
import { isDeepStrictEqual } from "node:util";

import type { ScheduleIntent } from "../../../src/types/intent";
import { canonicalizeIntent } from "../canonical-intent";

export interface IntentSample {
  intent: ScheduleIntent | null;
}

export function agreeIntents(samples: readonly IntentSample[]): boolean {
  if (samples.length < 3) return false;

  const [first, ...rest] = samples.map((sample) =>
    sample.intent === null ? null : canonicalizeIntent(sample.intent),
  );

  return rest.every((intent) => isDeepStrictEqual(intent, first));
}

export function agreeKeep<T extends { samples: readonly IntentSample[] }>(
  records: readonly T[],
): T[] {
  return records.filter((record) => agreeIntents(record.samples));
}
