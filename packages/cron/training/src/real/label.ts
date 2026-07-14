import type { ScheduleIntent } from "../../../src/types/intent";
/**
 * Track C label: k=3 constrained JSON proposals (T = 0, 0.4, 0.8).
 * The live LLM call is optional; this module owns the schema and k.
 */
import { scheduleIntentSchema } from "../llm/schema";

export const LABEL_TEMPERATURES = [0, 0.4, 0.8] as const;
export const LABEL_K = 3;

export interface LabelProposal {
  temperature: number;
  intent: ScheduleIntent | null;
}

export function parseConstrainedIntent(value: unknown): ScheduleIntent | null {
  if (value === null) return null;
  const parsed = scheduleIntentSchema.safeParse(value);

  if (!parsed.success) throw new TypeError(`label: schema-invalid: ${parsed.error.message}`);

  return parsed.data as ScheduleIntent;
}

export function assertLabelK(proposals: readonly LabelProposal[]): void {
  if (proposals.length !== LABEL_K) {
    throw new RangeError(`label: expected k=${LABEL_K}, got ${proposals.length}`);
  }

  const temps = proposals.map((item) => item.temperature);

  if (temps.some((temp, index) => temp !== LABEL_TEMPERATURES[index])) {
    throw new RangeError(`label: temperatures must be ${LABEL_TEMPERATURES.join(",")}`);
  }
}
