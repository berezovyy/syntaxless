/**
 * Anti-cousin fingerprint: outer slots only. Hole text is ignored.
 */
import { EMPTY_ID, type SlotDraw } from "./types";

export const FINGERPRINT_MAX_SHARE = 0.02;

export function slotFingerprint(draw: SlotDraw): string {
  return [
    draw.speech.act,
    draw.position,
    draw.scene.id,
    draw.reason.id,
    draw.hedge.id,
    draw.tail.id,
  ].join("|");
}

export interface FingerprintBucket {
  fingerprint: string;
  n: number;
  share: number;
}

export interface FingerprintReport {
  n: number;
  maxShare: number;
  top: FingerprintBucket[];
  emptyShare: number;
}

export function fingerprintReport(fingerprints: readonly string[], topK = 8): FingerprintReport {
  const counts = new Map<string, number>();

  for (const fingerprint of fingerprints) {
    counts.set(fingerprint, (counts.get(fingerprint) ?? 0) + 1);
  }

  const n = fingerprints.length;

  const buckets: FingerprintBucket[] = [...counts.entries()].map(([fingerprint, count]) => ({
    fingerprint,
    n: count,
    share: n === 0 ? 0 : count / n,
  }));

  buckets.sort((left, right) => right.n - left.n);
  const emptyKey = `request|right|${EMPTY_ID}|${EMPTY_ID}|${EMPTY_ID}|${EMPTY_ID}`;
  const emptyN = counts.get(emptyKey) ?? 0;

  return {
    n,
    maxShare: buckets[0]?.share ?? 0,
    top: buckets.slice(0, topK),
    emptyShare: n === 0 ? 0 : emptyN / n,
  };
}

export function exceedsFingerprintCap(
  fingerprints: readonly string[],
  cap = FINGERPRINT_MAX_SHARE,
): boolean {
  return fingerprintReport(fingerprints).maxShare >= cap;
}
