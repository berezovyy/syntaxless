/**
 * Per-miss attribution. Every miss is exactly one of ROLE / ACT / DECODE /
 * CONF / GOLD. No string-matching hacks on the source text beyond the gold
 * schema checks that name GOLD (uninvertible 12h clocks, through-as-shown).
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import type { Label } from "../../src/model/labels.js";
import type { SpeechAct } from "../../src/model/speech-acts.js";
import { hourWithoutMeridiemIsInvertible } from "../../src/schedule-semantics.js";
import type { ScheduleIntent } from "../../src/types/intent";
import type { OracleOutcome, OracleSpan } from "./oracle";
import { oracleAgrees } from "./oracle";

export const ATTRIBUTION_BUCKETS = ["ROLE", "ACT", "DECODE", "CONF", "GOLD"] as const;
export type AttributionBucket = (typeof ATTRIBUTION_BUCKETS)[number];

const REFUSE_ACTS = new Set<SpeechAct>([
  "descriptive",
  "question",
  "quoted",
  "one-off",
  "cancel",
  "incomplete",
  "none",
]);

export interface AttributionHistogram {
  ROLE: number;
  ACT: number;
  DECODE: number;
  CONF: number;
  GOLD: number;
}

export interface AttributionRecord {
  id: string;
  source: string;
  family?: string;
  bucket: AttributionBucket;
}

export interface AttributionReport extends AttributionHistogram {
  records: AttributionRecord[];
}

export interface AttributionInput {
  text: string;
  expectedIntent: ScheduleIntent | null;
  accepted: boolean;
  actualIntent: ScheduleIntent | null;
  goldSpans?: readonly OracleSpan[];
  goldSpeechAct?: SpeechAct;
  goldActionable?: readonly boolean[];
  predictedLabels?: readonly Label[];
  goldLabels?: readonly Label[];
  predictedSpeechAct?: SpeechAct;
  predictedActionable?: readonly boolean[];
  goldOracle?: OracleOutcome;
  predictedOracle?: OracleOutcome;
}

export function emptyHistogram(): AttributionHistogram {
  return { ROLE: 0, ACT: 0, DECODE: 0, CONF: 0, GOLD: 0 };
}

export function emptyAttributionReport(): AttributionReport {
  return { ...emptyHistogram(), records: [] };
}

export function addAttribution(report: AttributionReport, record: AttributionRecord): void {
  report[record.bucket] += 1;
  report.records.push(record);
}

export function histogramOf(report: AttributionReport): AttributionHistogram {
  return {
    ROLE: report.ROLE,
    ACT: report.ACT,
    DECODE: report.DECODE,
    CONF: report.CONF,
    GOLD: report.GOLD,
  };
}

export async function writeAttributionFile(
  path: string,
  source: string,
  report: AttributionReport,
): Promise<void> {
  let existing: {
    ROLE?: number;
    ACT?: number;
    DECODE?: number;
    CONF?: number;
    GOLD?: number;
    sources?: Record<string, AttributionHistogram>;
  } = {};

  try {
    existing = JSON.parse(await readFile(path, "utf8")) as typeof existing;
  } catch {
    existing = {};
  }

  const sources = { ...(existing.sources ?? {}) };

  sources[source] = histogramOf(report);
  const combined = emptyHistogram();

  for (const histogram of Object.values(sources)) {
    for (const bucket of ATTRIBUTION_BUCKETS) combined[bucket] += histogram[bucket];
  }

  await mkdir(dirname(path), { recursive: true });

  await writeFile(
    path,
    `${JSON.stringify(
      {
        ...combined,
        sources,
        records: report.records,
      },
      null,
      2,
    )}\n`,
  );
}

/** Gold that the decoder cannot invert even with perfect labels. */
export function goldViolatesSemantics(input: AttributionInput): boolean {
  const intent = input.expectedIntent;
  const spans = input.goldSpans;

  if (intent?.time !== undefined && spans !== undefined) {
    const hourSpan = spans.find((span) => span.label.startsWith("HOUR_"));
    const hasMeridiem = spans.some((span) => span.label.startsWith("MERIDIEM_"));

    if (hourSpan !== undefined && !hasMeridiem) {
      const shown = Number(input.text.slice(hourSpan.start, hourSpan.end));

      if (!hourWithoutMeridiemIsInvertible(shown, intent.time.hour)) return true;
    }
  }

  if (intent?.bounds?.end !== undefined && spans !== undefined) {
    const bound = spans.find((span) => span.label.startsWith("BOUND_"));

    if (bound !== undefined) {
      const connector = input.text.slice(bound.start, bound.end).toLowerCase();

      if (connector === "through" || connector === "thru" || bound.label === "BOUND_END_INCL") {
        const dateSpan = spans.find((span) => span.label === "DOM_LIT" && span.start >= bound.end);

        if (
          dateSpan !== undefined &&
          input.text.slice(dateSpan.start, dateSpan.end) === intent.bounds.end
        ) {
          return true;
        }
      }
    }
  }

  return false;
}

function rolesDiffer(input: AttributionInput): boolean {
  if (input.goldLabels === undefined || input.predictedLabels === undefined) return false;
  const n = Math.min(input.goldLabels.length, input.predictedLabels.length);

  if (input.goldLabels.length !== input.predictedLabels.length) return true;

  for (let index = 0; index < n; index += 1) {
    if (input.goldLabels[index] !== input.predictedLabels[index]) return true;
  }

  return false;
}

function actDiffers(input: AttributionInput): boolean {
  if (
    input.goldSpeechAct !== undefined &&
    input.predictedSpeechAct !== undefined &&
    input.goldSpeechAct !== input.predictedSpeechAct
  ) {
    return true;
  }

  if (input.goldActionable !== undefined && input.predictedActionable !== undefined) {
    const n = Math.min(input.goldActionable.length, input.predictedActionable.length);

    if (input.goldActionable.length !== input.predictedActionable.length) return true;

    for (let index = 0; index < n; index += 1) {
      if (input.goldActionable[index] !== input.predictedActionable[index]) return true;
    }
  }

  return false;
}

/**
 * Classify one miss. Callers must only invoke this when the scored record is
 * wrong; a correct record has no bucket.
 */
export function attributeMiss(input: AttributionInput): AttributionBucket {
  if (input.goldSpans !== undefined) {
    if (goldViolatesSemantics(input)) return "GOLD";

    if (input.goldOracle !== undefined && !oracleAgrees(input.goldOracle, input.expectedIntent)) {
      return "DECODE";
    }

    if (rolesDiffer(input)) return "ROLE";
    if (actDiffers(input)) return "ACT";

    if (
      !input.accepted &&
      ((input.predictedOracle !== undefined &&
        oracleAgrees(input.predictedOracle, input.expectedIntent)) ||
        (input.goldOracle !== undefined && oracleAgrees(input.goldOracle, input.expectedIntent)))
    ) {
      return "CONF";
    }

    return "DECODE";
  }

  if (input.expectedIntent === null && input.accepted) return "ACT";

  if (
    input.expectedIntent !== null &&
    !input.accepted &&
    input.predictedSpeechAct !== undefined &&
    REFUSE_ACTS.has(input.predictedSpeechAct)
  ) {
    return "ACT";
  }

  if (
    !input.accepted &&
    input.predictedOracle !== undefined &&
    oracleAgrees(input.predictedOracle, input.expectedIntent)
  ) {
    return "CONF";
  }

  return "ROLE";
}
