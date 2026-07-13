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
import { SCHEDULE_KINDS } from "../model/schedule-kinds.js";
import type { ScheduleKind } from "../model/schedule-kinds.js";
import { tokenize } from "../model/tokenizer.js";
import type { RawToken } from "../model/tokenizer.js";
import { classifyConfidence } from "./confidence.js";
import { decodeIntent, UnknownRoleError, UnsafeScheduleContextError } from "./decode.js";
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
 * Reduce an explicit replacement correction to the clause the user actually
 * selected. The patterns require both a correction marker and an actionable
 * replacement; cancellations and unresolved alternatives remain untouched and
 * are rejected later by the discourse-safety decoder.
 */
export function scopeActionableCorrection(text: string): string {
  const value = text
    .normalize("NFKC")
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\s+/g, " ")
    .trim();

  const patterns: Array<[RegExp, (match: RegExpExecArray) => string]> = [
    [
      /^actually,?\s+forget\b.+?\band\s+((?:poll|run|schedule|send|set|trigger)\b.+?)\s+instead[.!]?$/i,
      (match) => match[1]!,
    ],
    [/^actually,?\s+make that\s+(.+?),\s+not\b/i, (match) => match[1]!],
    [/^(?:apologies|sorry)\s*[-,:]\s*make it\s+(.+?),\s+not\b/i, (match) => `make ${match[1]!}`],
    [/^make that\s+(.+?),\s+instead of\b/i, (match) => `make ${match[1]!}`],
    [/^(?:sorry,?\s+)?i meant\s+(.+?)\s+rather than\b/i, (match) => match[1]!],
    [/^(?:sorry,?\s+)?i meant\s+(.+?),\s+not\b/i, (match) => match[1]!],
    [/^i said\b.+?,\s+but i meant\s+(.+)$/i, (match) => match[1]!],
    [
      /^i (?:gave|sent|told)\b.+?\bby mistake\.\s*((?:please\s+)?(?:make|run|schedule|send|set|use)\b.+)$/i,
      (match) => match[1]!,
    ],
    [/^change\b[^:]*:\s*(.+?),\s*not\b/i, (match) => match[1]!],
    [/^no,?\s+use\s+(.+?)\s+instead of\b/i, (match) => `use ${match[1]!}`],
    [/^(?:sorry,?\s+)?use\s+(.+?)\s+instead of\b/i, (match) => `use ${match[1]!}`],
    [/^(?:sorry|apologies)\s*[-,:]\s*use\s+(.+?),\s+not\b/i, (match) => `use ${match[1]!}`],
    [/^no,?\s+i need\s+(.+?)\s+rather than\b/i, (match) => match[1]!],
    [/^no,?\s+(.+?)\s+instead of\b/i, (match) => match[1]!],
    [/^no\s*-\s*(.+?),\s+rather than\b/i, (match) => match[1]!],
    [/^correction\s*[-:]\s*(.+?),\s*not\b/i, (match) => match[1]!],
    [/^correction\s*[-:]\s*(.+?)\s+rather than\b/i, (match) => match[1]!],
    [/^forget\b[^;]*;\s*(.+)$/i, (match) => match[1]!],
    [
      /^(?:discard|scratch)\b[^;]*;\s*((?:make|run|schedule|send|set|trigger)\b.+)$/i,
      (match) => match[1]!,
    ],
    [/^i said\b.*?\bby mistake\.\s*(.+)$/i, (match) => match[1]!],
    [
      /^i said\b.*?\b(?:in error|was wrong)\.\s*((?:make|run|schedule|send|set|use)\b.+)$/i,
      (match) => match[1]!,
    ],
    [/^(use\s+.+?)\s*-\s*ignore\b/i, (match) => match[1]!],
    [/^(use\s+.+?)\s+instead\s*-\s*(?:forget|ignore)\b/i, (match) => match[1]!],
    [/^let me revise that:\s*(.+)$/i, (match) => match[1]!],
    [/^not\b[^;]*;\s*(.+)$/i, (match) => match[1]!],
    [/^not\b.+?\s*-\s*((?:make|run|schedule|send|set|trigger|use)\b.+)$/i, (match) => match[1]!],
    [/^move\b.*\bfrom\b.*?\bto\s+(.+)$/i, (match) => `schedule ${match[1]!}`],
    [/^change\b.*\bfrom\b.*?\bto\s+(.+)$/i, (match) => `schedule ${match[1]!}`],
    [
      /^no,?\s+(?:i need it|make it|schedule it|set it|use)\s+(.+?),\s+not\b/i,
      (match) => match[1]!,
    ],
    [/^the\b[^;]*\bwas wrong;\s*(.+)$/i, (match) => match[1]!],
    [/^the\b[^;]*\bwas a mistake;\s*(.+)$/i, (match) => match[1]!],
    [/^[^.]+\bwon't work\.\s*((?:make|run|schedule|send|set|use)\b.+)$/i, (match) => match[1]!],
    [/^revise that to\s+(.+?),\s*not\b/i, (match) => match[1]!],
    [/^revise\b.*?\bto\s+(.+?),\s*rather than\b/i, (match) => match[1]!],
    [/^(?:please\s+)?revise\b.*?\bto\s+(.+?)\s+instead of\b/i, (match) => match[1]!],
    [/^instead of\b[^,]*,\s*(.+)$/i, (match) => match[1]!],
  ];

  for (const [pattern, select] of patterns) {
    const match = pattern.exec(value);

    if (match !== null) return select(match).trim();
  }

  return text;
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
): ParseResult {
  let decoded;

  try {
    decoded = decodeIntent(labeled, scheduleKind, sourceText);
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
  const scopedText = scopeActionableCorrection(text);
  const normalized = normalize(scopedText);

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

  const scheduleKind =
    predictions.scheduleKind === undefined || predictions.scheduleKindScore === undefined
      ? undefined
      : {
          kind: SCHEDULE_KINDS[predictions.scheduleKind]!,
          score: predictions.scheduleKindScore,
        };

  const result = resultFromLabeledTokens(
    labelTokensFromPredictions(tokens, predictions),
    scheduleKind,
    scopedText,
  );

  // A spelled monthly calendar day is a unit cadence unless the user also
  // supplied an explicit multi-month interval. Keep this final public-result
  // invariant independent of whatever interval role the model assigned to a
  // component of a compound ordinal such as "twenty-seventh".
  if (
    result.ok &&
    result.intent.dayOfMonth !== undefined &&
    /\b(?:monthly|every\s+month|each\s+month|once\s+per\s+month)\b/.test(normalized) &&
    !/\bevery\s+(?:\d+|second|third|fourth)\s+months?\b/.test(normalized)
  ) {
    result.intent.frequency.unit = "month";
    result.intent.frequency.interval = 1;
  }

  return result;
}
