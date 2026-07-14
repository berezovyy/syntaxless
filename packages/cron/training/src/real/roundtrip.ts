/**
 * Track C roundtrip: keep when the reconstructed intent is canonically equal
 * to the agreed intent. The live LLM sees only occurrences + unit; tests pass
 * the reconstructed intent directly.
 */
import { isDeepStrictEqual } from "node:util";

import type { ScheduleIntent } from "../../../src/types/intent";
import { canonicalizeIntent } from "../canonical-intent";

export function roundtripEqual(
  agreed: ScheduleIntent | null,
  reconstructed: ScheduleIntent | null,
): boolean {
  if (agreed === null || reconstructed === null) return agreed === reconstructed;

  return isDeepStrictEqual(canonicalizeIntent(agreed), canonicalizeIntent(reconstructed));
}

export function roundtripKeep<
  T extends { intent: ScheduleIntent | null; reconstructed: ScheduleIntent | null },
>(records: readonly T[]): T[] {
  return records.filter((record) => roundtripEqual(record.intent, record.reconstructed));
}
