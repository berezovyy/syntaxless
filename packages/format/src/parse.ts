import { LABELS } from "./labels.js";
import { prepare } from "./prepare.js";
import { resolveRoles } from "./resolve.js";
import { InputError } from "./types.js";
import type { FailureCode, FormatIntent } from "./types.js";

export type ParseResult =
  | { ok: true; intent: FormatIntent; confidence: number; source: "model" }
  | {
      ok: false;
      code: FailureCode | "invalid-input" | "low-confidence" | "incompatible-model" | "model-error";
      reason: string;
      source: "model";
    };

export interface ParseOptions {
  weights?: Uint8Array;
}

export async function parse(text: string, options: ParseOptions = {}): Promise<ParseResult> {
  try {
    const prepared = prepare(text);
    const { getEngine } = await import("./model/engine.js");
    const { tokenize } = await import("./model/tokenizer.js");
    let engine;

    try {
      engine = await getEngine(
        options.weights === undefined ? undefined : { weights: options.weights },
      );
    } catch (error) {
      return {
        ok: false,
        code: "incompatible-model",
        reason: error instanceof Error ? error.message : "model could not initialize",
        source: "model",
      };
    }

    const tokens = tokenize(prepared.text);
    const predictions = engine.inferTokens(tokens);

    const labeled = prepared.lexemes.map((lexeme) => {
      const index = tokens.findIndex(
        (token) => token.start === lexeme.start && token.end === lexeme.end,
      );

      if (index < 0) throw new Error("canonical lexeme did not tokenize atomically");
      const role = LABELS[predictions.labels[index]!];

      if (role === undefined) throw new Error("model emitted an unknown role");

      return { role, score: predictions.scores[index]! };
    });

    const decoded = resolveRoles(prepared, labeled);

    if (decoded.confidence < engine.confidenceFloor) {
      return {
        ok: false,
        code: "low-confidence",
        reason: "model confidence is below its calibrated floor",
        source: "model",
      };
    }

    return { ok: true, ...decoded, source: "model" };
  } catch (error) {
    return {
      ok: false,
      code: error instanceof InputError ? error.code : "model-error",
      reason: error instanceof Error ? error.message : "model error",
      source: "model",
    };
  }
}
