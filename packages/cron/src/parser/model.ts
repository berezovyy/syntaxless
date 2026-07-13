/**
 * Model-backed parser: tokenize (Phase 07) -> infer (Phase 16) ->
 * `decodeIntent` (Phase 17) -> the additive {@link ParseResult}.
 *
 * The model predicts roles, speech-act, and the actionable span; TypeScript
 * resolves every fact from those labels. A role id outside the label
 * vocabulary is a hard {@link UnknownRoleError} and becomes `model-error`.
 * A decoded intent that cannot be recognized is also a `model-error`, never
 * a guessed intent. Nothing here calls the rule parser.
 *
 * Public parse is heads-only. Overlay-on is not a regex scaffold: a missing
 * speech-act head is `model-error`, never a discourse fallback. Parse options
 * are `{ source?: "model" | "rules", weights?: Uint8Array, overlay?: "on" | "off" }`.
 * `parseWithModel` resolves the engine through the lazy Phase 20 lifecycle.
 */
import { getEngine } from "../model/engine.js";
import type { EngineConfig } from "../model/engine.js";
import type { Predictions } from "../model/inference.js";
import { SCHEDULE_KINDS } from "../model/schedule-kinds.js";
import type { ScheduleKind } from "../model/schedule-kinds.js";
import { speechActFromId } from "../model/speech-acts.js";
import type { SpeechAct } from "../model/speech-acts.js";
import { tokenize } from "../model/tokenizer.js";
import type { RawToken } from "../model/tokenizer.js";
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

  if (options?.weights === undefined) {
    const { WEIGHTS_MATCH_LABELS } = await import("../model/inference.js");

    if (!WEIGHTS_MATCH_LABELS) {
      return {
        ok: false,
        reason: "bundled weights do not match the typed label vocabulary",
        code: "model-error",
        confidence: 0,
        diagnostics: [
          {
            code: "stale-weights",
            severity: "error",
            message: "typed labels require an A1 re-export; bundled 15-role weights are not used",
          },
        ],
        source: "model",
      };
    }
  }

  const tokens = tokenize(normalized);
  const predictions = engine.inferTokens(tokens);

  if (predictions.speechAct === undefined) {
    return {
      ok: false,
      reason: "heads-only parse requires a speech-act head",
      code: "model-error",
      confidence: 0,
      diagnostics: [
        {
          code: "missing-speech-act-head",
          severity: "error",
          message: "weights do not include a speech-act head; overlay regex is not a fallback",
        },
      ],
      source: "model",
    };
  }

  let speechAct: SpeechAct;

  try {
    speechAct = speechActFromId(predictions.speechAct);
  } catch (error) {
    if (error instanceof RangeError) {
      return {
        ok: false,
        reason: error.message,
        code: "model-error",
        confidence: 0,
        diagnostics: [
          {
            code: "unknown-speech-act",
            severity: "error",
            message: error.message,
          },
        ],
        source: "model",
      };
    }

    throw error;
  }

  const scheduleKind =
    predictions.scheduleKind === undefined || predictions.scheduleKindScore === undefined
      ? undefined
      : {
          kind: SCHEDULE_KINDS[predictions.scheduleKind]!,
          score: predictions.scheduleKindScore,
        };

  return resultFromLabeledTokens(
    labelTokensFromPredictions(tokens, predictions),
    scheduleKind,
    text,
    overlay,
    speechAct,
  );
}
