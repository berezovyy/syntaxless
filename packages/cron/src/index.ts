export type {
  DaySpec,
  FrequencyUnit,
  MinuteOfHour,
  Ordinal,
  OrdinalWeekday,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "./types/intent.js";

export { validate } from "./validate.js";
export { explain } from "./explain.js";
export { clarify, resolveClarification } from "./clarify.js";
export type { ScheduleClarification, ClarificationAnswer } from "./clarify.js";
export { IntentError } from "./types/errors.js";
export type { IntentFailureCode } from "./types/errors.js";

export { recurrences } from "./recurrence/index.js";

export type {
  Occurrence,
  RecurrenceOptions,
  RecurrencePolicies,
  RecurrenceResult,
} from "./recurrence/index.js";

export { RecurrenceError } from "./recurrence/calendar.js";

export type {
  LocalDateTime,
  RecurrenceCalendar,
  RecurrenceFailureCode,
} from "./recurrence/calendar.js";

export { compile, compileWeekdays, WEEKDAY_NUMBER } from "./compiler/index.js";
export type { CompileFailureCode, CompileResult } from "./compiler/index.js";

export { parse } from "./parser/parse.js";

export type {
  Diagnostic,
  ModelParserOptions,
  OverlayMode,
  ParseFailureCode,
  ParseResult,
  ParseSource,
} from "./parser/index.js";

export { getEngine, preload, dispose } from "./model/engine.js";
export type { EngineConfig, InferenceEngine } from "./model/engine.js";

import { clarify } from "./clarify.js";
import type { ScheduleClarification } from "./clarify.js";
import { compile } from "./compiler/index.js";
import type { CompileFailureCode } from "./compiler/index.js";
import type { ParseFailureCode } from "./parser/index.js";
import { parse } from "./parser/parse.js";
import type { ScheduleIntent } from "./types/intent.js";

export type CronFailureCode = ParseFailureCode | CompileFailureCode;

export type CronResult =
  | { supported: true; expression: string; intent: ScheduleIntent }
  | {
      supported: false;
      reason: string;
      code: CronFailureCode;
      intent: ScheduleIntent | null;
      clarification?: ScheduleClarification;
    };

export async function cron(text: string): Promise<CronResult> {
  const parsed = await parse(text);

  if (!parsed.ok) {
    return { supported: false, reason: parsed.reason, code: parsed.code, intent: null };
  }

  const result = compile(parsed.intent);
  const clarification = result.supported ? null : clarify(parsed.intent);

  return result.supported
    ? { supported: true, expression: result.expression, intent: parsed.intent }
    : {
        supported: false,
        reason: result.reason,
        code: result.code,
        intent: parsed.intent,
        ...(clarification ? { clarification } : {}),
      };
}
