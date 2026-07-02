/**
 * Confidence bands for the model-backed parser.
 *
 * The Phase 15 weight artifact ships a calibrated boundary threshold but no
 * margin bands, so the high/medium thresholds live here as constants. They are
 * pure inputs to {@link classifyConfidence}; changing them never changes an
 * intent, only whether a parse is reported as high, medium, or low confidence.
 */
export type ConfidenceBand = "high" | "medium" | "low";

export interface ConfidenceThresholds {
  /** At or above this margin the parse is clean. */
  high: number;
  /** At or above this margin the parse is kept but carries a suggestion. */
  medium: number;
}

export const CONFIDENCE_THRESHOLDS: ConfidenceThresholds = {
  high: 0.9,
  medium: 0.5,
};

/**
 * Classify a decoded margin score into a band. The score is the weakest
 * per-token margin from Phase 17, in `[0, 1]`.
 */
export function classifyConfidence(
  score: number,
  thresholds: ConfidenceThresholds = CONFIDENCE_THRESHOLDS,
): ConfidenceBand {
  if (!Number.isFinite(score)) return "low";
  if (score >= thresholds.high) return "high";
  if (score >= thresholds.medium) return "medium";

  return "low";
}
