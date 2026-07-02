export type {
  DaySpec,
  FrequencyUnit,
  Ordinal,
  OrdinalWeekday,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "./types/intent.js";

export { compile, compileWeekdays, WEEKDAY_NUMBER } from "./compiler/index.js";
export type { CompileFailureCode, CompileResult } from "./compiler/index.js";

export { parse } from "./parser/index.js";

export type {
  Diagnostic,
  ModelParserOptions,
  ParseFailureCode,
  ParseResult,
  ParseSource,
} from "./parser/index.js";

import { compile } from "./compiler/index.js";
import type { CompileFailureCode } from "./compiler/index.js";
import { parse } from "./parser/index.js";
import type { ParseFailureCode } from "./parser/index.js";
import type { ScheduleIntent } from "./types/intent.js";

export type CronFailureCode = ParseFailureCode | CompileFailureCode;

export type CronResult =
  | { supported: true; expression: string; intent: ScheduleIntent }
  | { supported: false; reason: string; code: CronFailureCode; intent: ScheduleIntent | null };

export async function cron(text: string): Promise<CronResult> {
  const parsed = await parse(text);

  if (!parsed.ok) {
    return { supported: false, reason: parsed.reason, code: parsed.code, intent: null };
  }

  const result = compile(parsed.intent);

  return result.supported
    ? { supported: true, expression: result.expression, intent: parsed.intent }
    : { supported: false, reason: result.reason, code: result.code, intent: parsed.intent };
}
