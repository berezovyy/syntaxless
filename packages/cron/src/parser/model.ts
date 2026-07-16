/** Model-backed parsing: tokenize, infer, apply calibration, decode the intent. */
import { getEngine } from "../model/engine.js";
import type { EngineConfig, InferenceEngine } from "../model/engine.js";
import type { Predictions } from "../model/inference.js";
import { isScheduleLabel, isStructural, labelFromId } from "../model/labels.js";
import type { Label } from "../model/labels.js";
import type { ScheduleKind } from "../model/schedule-kinds.js";
import type { SpeechAct } from "../model/speech-acts.js";
import { tokenize } from "../model/tokenizer.js";
import type { RawToken } from "../model/tokenizer.js";
import { mahalanobisDiag, sequenceEnergy } from "./abstain.js";
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
 * predictions are supplied by the caller; this operation never initializes or
 * runs the inference engine.
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

/** Legacy artifacts derive request/refusal from actionable content labels. */
export function speechActFromSpanLabels(labeled: readonly LabeledToken[]): SpeechAct {
  for (const item of labeled) {
    if (item.token.kind === 3) continue;
    let label: Label;

    try {
      label = labelFromId(item.role);
    } catch {
      continue;
    }

    if (!isScheduleLabel(label) || isStructural(label)) continue;
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

  return sequenceEnergy(logits, predictions.labels, classes);
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

/** Apply artifact-owned energy and Mahalanobis rejection thresholds. */
export function calibratedRejection(
  energy: number,
  threshold: number | undefined,
  mahalanobis?: number,
  mahalanobisThreshold?: number,
): ParseResult | undefined {
  let metric = "energy";
  let value = energy;
  let cutoff = threshold;

  if (cutoff === undefined || !(value > cutoff)) {
    if (
      mahalanobis === undefined ||
      mahalanobisThreshold === undefined ||
      !(mahalanobis > mahalanobisThreshold)
    ) {
      return undefined;
    }

    metric = "Mahalanobis distance";
    value = mahalanobis;
    cutoff = mahalanobisThreshold;
  }

  return {
    ok: false,
    reason: `abstain ${metric} ${value.toFixed(4)} exceeds ${cutoff}`,
    code: "unsafe-context",
    confidence: 0,
    diagnostics: [
      {
        code: "calibrated-abstain",
        severity: "warning",
        message: `${metric === "energy" ? "sequence energy" : metric} exceeded the conformal threshold`,
      },
    ],
    source: "model",
  };
}

/** Resolve labeled tokens without initializing or running an inference engine. */
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

  return parsePrediction(normalized, engine, predictions, labeled, options);
}

/** Interpret one prediction using the owning engine's calibration and speech policy. */
export function parsePrediction(
  normalized: string,
  engine: InferenceEngine,
  predictions: Predictions,
  labeled: readonly LabeledToken[],
  options?: Pick<ModelParserOptions, "overlay" | "energyThreshold">,
): ParseResult {
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

  const overlay = options?.overlay ?? "off";

  const speechAct =
    engine.decisionHead && predictions.speechAct !== undefined
      ? engine.predictedSpeechAct(predictions.speechAct)
      : speechActFromSpanLabels(labeled);

  const calibration = engine.calibration;
  const threshold = options?.energyThreshold ?? calibration.energyThreshold;

  const abstain = calibratedRejection(
    energyFromPredictions(predictions),
    threshold,
    mahalanobisFromPredictions(
      predictions,
      calibration.mahalanobisMean,
      calibration.mahalanobisVariance,
    ),
    calibration.mahalanobisThreshold,
  );

  if (abstain !== undefined) return abstain;

  if (!engine.weightsMatchLabels) {
    return {
      ok: false,
      reason: "weights do not match the typed label vocabulary",
      code: "model-error",
      confidence: 0,
      diagnostics: [
        {
          code: "stale-weights",
          severity: "error",
          message: "weights do not match the typed label vocabulary",
        },
      ],
      source: "model",
    };
  }

  return resultFromLabeledTokens(labeled, undefined, normalized, overlay, speechAct);
}
