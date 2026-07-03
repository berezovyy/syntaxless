/**
 * Model-backed parser: tokenize (Phase 07) -> infer (Phase 16) ->
 * `decodeIntent` (Phase 17) -> the additive {@link ParseResult}.
 *
 * The model predicts roles and boundary flags; TypeScript resolves every fact.
 * A role id outside the label vocabulary is a hard {@link UnknownRoleError} and
 * becomes `model-error`. A decoded intent that cannot be recognized is also a
 * `model-error`, never a guessed intent. Nothing here calls the rule parser.
 *
 * This module deliberately exposes no injectable engine: the parse options are
 * `{ source?: "model" | "rules", weights?: Uint8Array }`, and `parseWithModel`
 * resolves the engine through the lazy Phase 20 lifecycle. The inference module
 * is imported on demand, so importing this module initializes nothing.
 */
import { getEngine } from "../model/engine.js";
import type { EngineConfig } from "../model/engine.js";
import type { Predictions } from "../model/inference.js";
import { tokenize } from "../model/tokenizer.js";
import type { RawToken } from "../model/tokenizer.js";
import { classifyConfidence } from "./confidence.js";
import { decodeIntent, UnknownRoleError } from "./decode.js";
import type { LabeledToken } from "./decode.js";
import { normalize } from "./normalize.js";
import type { Diagnostic, ParseResult } from "./parse.js";

/** Public parser options. `source` defaults to `"model"`. */
export interface ModelParserOptions {
  source?: "model" | "rules";
  /** Self-hosted weight bytes; overrides the bundled artifact. */
  weights?: Uint8Array;
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
  }));
}

/**
 * Map labeled tokens onto the extended result. Split out from
 * {@link parseWithModel} so the role-validation and confidence mapping are pure
 * and testable without running the engine.
 */
export function resultFromLabeledTokens(labeled: readonly LabeledToken[]): ParseResult {
  let decoded;

  try {
    decoded = decodeIntent(labeled);
  } catch (error) {
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
  const predictions = engine.inferTokens(tokens);

  return resultFromLabeledTokens(labelTokensFromPredictions(tokens, predictions));
}
