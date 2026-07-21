/**
 * Confidence bands for the model-backed parser.
 *
 * Changing these never changes an intent, only whether a parse is reported as
 * high, medium, or low confidence. Low band fails closed with no intent.
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
 * per-token margin, in `[0, 1]`.
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
