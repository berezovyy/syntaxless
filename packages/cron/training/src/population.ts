import { isScheduleLabel } from "../../src/model/labels";
/**
 * Training populations: request (R), hard negative (H) and easy negative (E).
 * A hard negative carries labeled schedule words inside a non-request speech
 * act; an easy negative has no schedule words at all.
 */
import type { SpeechAct } from "../../src/model/speech-acts";
import type { ScheduleIntent } from "../../src/types/intent";

export type Population = "R" | "H" | "E";

const REQUEST_ACTS = new Set<SpeechAct>(["request", "correction"]);

export interface PopulationInput {
  intent: ScheduleIntent | null;
  speechAct: SpeechAct;
  spans: readonly { label: string }[];
}

export function classifyPopulation(row: PopulationInput): Population {
  const hasHole = row.spans.some((span) => isScheduleLabel(span.label));

  if (!hasHole) {
    if (row.intent !== null) throw new Error("classifyPopulation: intent without a labeled hole");

    if (row.speechAct !== "none") {
      throw new Error(`classifyPopulation: no-hole row has speechAct ${row.speechAct}`);
    }

    return "E";
  }

  if (REQUEST_ACTS.has(row.speechAct)) return "R";

  return "H";
}
