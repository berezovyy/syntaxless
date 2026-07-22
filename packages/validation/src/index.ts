import { compile } from "./compiler/index.js";
import type { CompileFailureCode, JsonSchema } from "./compiler/index.js";
import { parse } from "./parser/parse.js";
import type { ParseFailureCode } from "./parser/parse.js";
import type { FieldIntent } from "./types/intent.js";

export type {
  Bound,
  ComparisonIntent,
  FieldIntent,
  FieldType,
  NumberLikeType,
  StringLikeType,
} from "./types/intent.js";

export { compile, compileObject } from "./compiler/index.js";

export type {
  CompileFailureCode,
  CompileResult,
  JsonSchema,
  ObjectSchema,
} from "./compiler/index.js";

export { parse } from "./parser/parse.js";

export type {
  Diagnostic,
  ParseFailureCode,
  ParseOptions,
  ParseResult,
  ParseSource,
} from "./parser/index.js";

export type { ModelParserOptions } from "./parser/index.js";
export { getEngine, preload, dispose } from "./model/engine.js";
export type { EngineConfig, InferenceEngine } from "./model/engine.js";

export type SchemaFailureCode = ParseFailureCode | CompileFailureCode;

export type SchemaResult =
  | { supported: true; schema: JsonSchema; intent: FieldIntent }
  | {
      supported: false;
      reason: string;
      code: SchemaFailureCode;
      intent: FieldIntent | null;
    };

export async function schema(
  text: string,
  options?: import("./parser/parse.js").ParseOptions,
): Promise<SchemaResult> {
  const parsed = await parse(text, options);

  if (!parsed.ok) {
    return { supported: false, reason: parsed.reason, code: parsed.code, intent: null };
  }

  const result = compile(parsed.intent);

  return result.supported
    ? { supported: true, schema: result.schema, intent: parsed.intent }
    : {
        supported: false,
        reason: result.reason,
        code: result.code,
        intent: parsed.intent,
      };
}

export { explain } from "./explain.js";
export { validateField } from "./compiler/validate.js";
