import { isScheduleLabel } from "../../../src/model/labels";
/**
 * Train-mix quotas: request (R), hard-no (H), easy-no (E).
 * Shares are of sequences, not unique strings.
 */
import type { SpeechAct } from "../../../src/model/speech-acts";
import type { ScheduleIntent } from "../../../src/types/intent";

export const POPULATIONS = ["R", "H", "E"] as const;
export type Population = (typeof POPULATIONS)[number];

/** Start inside the bands. 13% E (v1) was reject-heavy; retry uses 10% E. */
export const TARGET_SHARE: Readonly<Record<Population, number>> = {
  R: 0.68,
  H: 0.22,
  E: 0.1,
};

export const SHARE_BAND: Readonly<Record<Population, readonly [number, number]>> = {
  R: [0.6, 0.7],
  H: [0.2, 0.25],
  E: [0.1, 0.15],
};

const REQUEST_ACTS = new Set<SpeechAct>(["request", "correction"]);

export interface MixClassifiable {
  intent: ScheduleIntent | null;
  speechAct: SpeechAct;
  spans: readonly { label: string }[];
}

export function classifyPopulation(row: MixClassifiable): Population {
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

export function sharesOf(counts: Readonly<Record<Population, number>>): Record<Population, number> {
  const total = counts.R + counts.H + counts.E;

  if (total === 0) return { R: 0, H: 0, E: 0 };

  return {
    R: counts.R / total,
    H: counts.H / total,
    E: counts.E / total,
  };
}

export function sharesInBand(shares: Readonly<Record<Population, number>>): boolean {
  for (const population of POPULATIONS) {
    const [lo, hi] = SHARE_BAND[population];
    const share = shares[population];

    if (share < lo || share > hi) return false;
  }

  return true;
}

export function bandFailures(shares: Readonly<Record<Population, number>>): string[] {
  const failures: string[] = [];

  for (const population of POPULATIONS) {
    const [lo, hi] = SHARE_BAND[population];
    const share = shares[population];

    if (share < lo || share > hi) {
      failures.push(`${population} share ${share.toFixed(4)} outside [${lo}, ${hi}]`);
    }
  }

  return failures;
}

/**
 * Schedule-only targets (R and H at their TARGET_SHARE ratio) for sources
 * whose negatives come from elsewhere. Exact integers that sum to `n`.
 */
export function scheduleTargetCounts(n: number): Record<Population, number> {
  if (n < 40) throw new RangeError(`scheduleTargetCounts(${n}): need enough rows to hit all bands`);
  const h = Math.round((n * TARGET_SHARE.H) / (TARGET_SHARE.R + TARGET_SHARE.H));

  return { R: n - h, H: h, E: 0 };
}

/** Exact integer targets that sum to `n` and sit in band for n >= 40. */
export function targetCounts(n: number): Record<Population, number> {
  if (n < 40) throw new RangeError(`targetCounts(${n}): need enough rows to hit all bands`);
  const h = Math.round(n * TARGET_SHARE.H);
  const e = Math.round(n * TARGET_SHARE.E);
  const r = n - h - e;
  const counts = { R: r, H: h, E: e };
  const shares = sharesOf(counts);

  if (!sharesInBand(shares)) {
    throw new Error(`targetCounts(${n}) missed a band: ${JSON.stringify(shares)}`);
  }

  return counts;
}
