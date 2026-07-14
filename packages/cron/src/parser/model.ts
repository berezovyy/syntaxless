/**
 * Model-backed parser: tokenize (Phase 07) -> infer (Phase 16) ->
 * `decodeIntent` (Phase 17) -> the additive {@link ParseResult}.
 *
 * The model predicts roles, boundaries, and the actionable span. Track F
 * drops pooled schedule-kind / speech-act heads: refuse is calibrated energy,
 * not a 9-way classifier. A role id outside the label vocabulary is a hard
 * {@link UnknownRoleError}. Overlay-on is not a regex scaffold.
 */
import { getEngine } from "../model/engine.js";
import type { EngineConfig } from "../model/engine.js";
import type { Predictions } from "../model/inference.js";
import { isStructural, labelFromId } from "../model/labels.js";
import type { Label } from "../model/labels.js";
import type { ScheduleKind } from "../model/schedule-kinds.js";
import type { SpeechAct } from "../model/speech-acts.js";
import { tokenize } from "../model/tokenizer.js";
import type { RawToken } from "../model/tokenizer.js";
import { mahalanobisDiag, rejectOnAbstainScore, sequenceEnergy } from "./abstain.js";
import { classifyConfidence } from "./confidence.js";
import { decodeIntent, UnknownRoleError, UnsafeScheduleContextError } from "./decode.js";
import type { LabeledToken, OverlayMode } from "./decode.js";
import { normalize } from "./normalize.js";
import type { Diagnostic, ParseResult } from "./parse.js";

export type { OverlayMode };

/** Public parser options. `source` defaults to `"model"`. Overlay defaults to `"off"`. */
export interface ModelParserOptions {
  source?: "model" | "rules";
  /** Self-hosted weight bytes; overrides the bundled artifact. */
  weights?: Uint8Array;
  /**
   * Diagnostic flag only. Public parse is heads-only either way: `"on"` is
   * not a discourse/correction regex scaffold.
   */
  overlay?: OverlayMode;
  /** Track F conformal energy threshold; reject when sequence energy exceeds it. */
  energyThreshold?: number;
}

/**
 * Build the decoder's input from tokens and one forward pass. Pure: the
 * predictions are supplied by the caller, so this module never imports the
 * inference engine.
 */
export function labelTokensFromPredictions(
  tokens: readonly RawToken[],
  predictions: Predictions,
): LabeledToken[] {
  return tokens.map((token, index) => ({
    token,
    role: predictions.labels[index]!,
    boundary: predictions.boundaries[index]! === 1,
    score: predictions.scores[index]!,
    boundaryScore: predictions.boundaryScores[index]!,
    ...(predictions.actionable === undefined
      ? {}
      : { actionable: predictions.actionable[index] === 1 }),
  }));
}

/**
 * Map labeled tokens onto the extended result. Split out from
 * {@link parseWithModel} so the role-validation and confidence mapping are pure
 * and testable without running the engine.
 */
/** Speech-act from span labels, not the pooled classifier (Track F.1). */
export function speechActFromSpanLabels(labeled: readonly LabeledToken[]): SpeechAct {
  for (const item of labeled) {
    if (item.token.kind === 3) continue;
    let label: Label;

    try {
      label = labelFromId(item.role);
    } catch {
      continue;
    }

    if (label === "O" || isStructural(label)) continue;
    if (item.actionable !== false) return "request";
  }

  return "none";
}

export function energyFromPredictions(predictions: Predictions): number {
  const logits = predictions.logits;

  if (logits === undefined || predictions.labels.length === 0) return 0;
  const width = predictions.labels.length;
  const classes = Math.trunc(logits.length / width);

  if (classes < 1) return 0;
  const rows: number[][] = [];
  const roles: number[] = [];

  for (let index = 0; index < width; index++) {
    rows.push(Array.from(logits.subarray(index * classes, (index + 1) * classes)));
    roles.push(predictions.labels[index]!);
  }

  return sequenceEnergy(rows, roles);
}

export function mahalanobisFromPredictions(
  predictions: Predictions,
  mean: readonly number[] | undefined,
  variance: readonly number[] | undefined,
): number | undefined {
  if (predictions.pooled === undefined || mean === undefined || variance === undefined) {
    return undefined;
  }

  return mahalanobisDiag(predictions.pooled, mean, variance);
}

/** Shipped reject path for Track F: calibrated energy, not the speech-act head. */
export function rejectWithCalibratedEnergy(
  energy: number,
  threshold: number | undefined,
  mahalanobis?: number,
  mahalanobisThreshold?: number,
): ParseResult | undefined {
  if (
    !rejectOnAbstainScore({
      energy,
      energyThreshold: threshold,
      ...(mahalanobis === undefined ? {} : { mahalanobis }),
      ...(mahalanobisThreshold === undefined ? {} : { mahalanobisThreshold }),
    })
  ) {
    return undefined;
  }

  return {
    ok: false,
    reason: `abstain energy ${energy.toFixed(4)} exceeds ${threshold}`,
    code: "unsafe-context",
    confidence: 0,
    diagnostics: [
      {
        code: "calibrated-abstain",
        severity: "warning",
        message: "sequence energy exceeded the conformal threshold",
      },
    ],
    source: "model",
  };
}

export function resultFromLabeledTokens(
  labeled: readonly LabeledToken[],
  scheduleKind?: { kind: ScheduleKind; score: number },
  sourceText?: string,
  overlay: OverlayMode = "off",
  speechAct?: SpeechAct,
): ParseResult {
  let decoded;

  try {
    decoded = decodeIntent(labeled, scheduleKind, sourceText, overlay, speechAct);
  } catch (error) {
    if (error instanceof UnsafeScheduleContextError) {
      return {
        ok: false,
        reason: error.message,
        code: "unsafe-context",
        confidence: 0,
        diagnostics: [
          {
            code: error.contextReason,
            severity: "warning",
            message: error.message,
          },
        ],
        source: "model",
      };
    }

    if (error instanceof UnknownRoleError) {
      return {
        ok: false,
        reason: error.message,
        code: "model-error",
        confidence: 0,
        diagnostics: [{ code: "unknown-role", severity: "error", message: error.message }],
        source: "model",
      };
    }

    if (error instanceof RangeError) {
      return {
        ok: false,
        reason: error.message,
        code: "model-error",
        confidence: 0,
        diagnostics: [{ code: "decode-error", severity: "error", message: error.message }],
        source: "model",
      };
    }

    throw error;
  }

  const band = classifyConfidence(decoded.confidence);

  if (band === "low") {
    return {
      ok: false,
      reason: `low confidence (${decoded.confidence.toFixed(3)})`,
      code: "low-confidence",
      confidence: decoded.confidence,
      diagnostics: [
        {
          code: "low-confidence",
          severity: "warning",
          message: "model margin is below the confidence floor; no intent emitted",
        },
      ],
      source: "model",
    };
  }

  const diagnostics: Diagnostic[] =
    band === "medium"
      ? [
          {
            code: "low-margin",
            severity: "suggestion",
            message: "model margin is above the floor but below the clean band",
          },
        ]
      : [];

  return {
    ok: true,
    intent: decoded.intent,
    confidence: decoded.confidence,
    diagnostics,
    source: "model",
  };
}

/** Run the bundled model over normalized text and resolve the intent. */
export async function parseWithModel(
  text: string,
  options?: ModelParserOptions,
): Promise<ParseResult> {
  const overlay = options?.overlay ?? "off";
  const normalized = normalize(text);

  if (normalized.length === 0) {
    return {
      ok: false,
      reason: "empty input",
      code: "empty",
      confidence: 0,
      diagnostics: [],
      source: "model",
    };
  }

  const config: EngineConfig | undefined =
    options?.weights === undefined ? undefined : { weights: options.weights };

  const engine = await getEngine(config);
  const tokens = tokenize(normalized);
  const predictions = engine.inferTokens(tokens, true);
  const labeled = labelTokensFromPredictions(tokens, predictions);
  const inference = await import("../model/inference.js");
  const threshold = options?.energyThreshold ?? inference.ENERGY_THRESHOLD;

  const abstain = rejectWithCalibratedEnergy(
    energyFromPredictions(predictions),
    threshold,
    mahalanobisFromPredictions(
      predictions,
      inference.MAHALANOBIS_MEAN,
      inference.MAHALANOBIS_VARIANCE,
    ),
    inference.MAHALANOBIS_THRESHOLD,
  );

  if (abstain !== undefined) return abstain;

  if (options?.weights === undefined && !inference.WEIGHTS_MATCH_LABELS) {
    return {
      ok: false,
      reason: "bundled weights do not match the typed label vocabulary",
      code: "model-error",
      confidence: 0,
      diagnostics: [
        {
          code: "stale-weights",
          severity: "error",
          message: "bundled weights do not match the typed label vocabulary",
        },
      ],
      source: "model",
    };
  }

  return resultFromLabeledTokens(
    labeled,
    undefined,
    text,
    overlay,
    speechActFromSpanLabels(labeled),
  );
}
