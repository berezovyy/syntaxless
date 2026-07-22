/**
 * Model-backed parser: tokenize → infer → thin `decodeIntent`.
 *
 * The model predicts roles; TypeScript resolves numbers, inclusive flags, and
 * slots. A role id outside the label vocabulary is `model-error`. Low margin
 * is `low-confidence` with no intent. Nothing here calls the rule parser, and
 * there is no silent fallback.
 */
import { getEngine } from "../model/engine.js";
import type { EngineConfig } from "../model/engine.js";
import type { Predictions } from "../model/inference.js";
import { tokenize } from "../model/tokenizer.js";
import type { RawToken } from "../model/tokenizer.js";
import { classifyConfidence } from "./confidence.js";
import { decodeIntent, InputError, UnknownRoleError, UnsupportedInputError } from "./decode.js";
import type { LabeledToken } from "./decode.js";
import { normalize } from "./normalize.js";
import type { Diagnostic, ParseResult } from "./parse.js";

export interface ModelParserOptions {
  source?: "model" | "rules";
  /** Self-hosted weight bytes; overrides the bundled artifact. */
  weights?: Uint8Array;
}

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
  }));
}

export function resultFromLabeledTokens(
  labeled: readonly LabeledToken[],
  sourceText: string,
  confidenceFloor = 0.5,
): ParseResult {
  let decoded;

  try {
    decoded = decodeIntent(labeled, sourceText);
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
        code: error instanceof InputError ? error.code : "model-error",
        confidence: 0,
        diagnostics: [
          {
            code: error instanceof UnsupportedInputError ? "unsupported-input" : "decode-error",
            severity: "error",
            message: error.message,
          },
        ],
        source: "model",
      };
    }

    throw error;
  }

  const band = classifyConfidence(decoded.confidence, {
    high: Math.max(0.9, confidenceFloor),
    medium: confidenceFloor,
  });

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

export async function parseWithModel(
  text: string,
  options?: ModelParserOptions,
): Promise<ParseResult> {
  if (typeof text !== "string") {
    return {
      ok: false,
      reason: "unrecognized input",
      code: "unrecognized",
      confidence: 0,
      diagnostics: [],
      source: "model",
    };
  }

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

  let engine;

  try {
    engine = await getEngine(config);
  } catch (error) {
    return {
      ok: false,
      code:
        error instanceof TypeError ||
        error instanceof RangeError ||
        (error instanceof Error && error.name === "unsupported-weights-version")
          ? "incompatible-model"
          : "model-error",
      reason: error instanceof Error ? error.message : "model initialization failed",
      confidence: 0,
      diagnostics: [],
      source: "model",
    };
  }

  const tokens = tokenize(normalized);
  const predictions = engine.inferTokens(tokens);

  return resultFromLabeledTokens(
    labelTokensFromPredictions(tokens, predictions),
    normalized,
    engine.confidenceFloor,
  );
}
