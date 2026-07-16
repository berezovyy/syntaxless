import { compile } from "./compiler/cron.js";
import { IntentError } from "./types/errors.js";
import type { ScheduleIntent, Weekday } from "./types/intent.js";
import { validate } from "./validate.js";

/** A missing fact needed for cron compilation, rather than a guessed interpretation. */
export type ScheduleClarification =
  | {
      code: "missing-anchor";
      field: "weekdays";
      question: string;
      allowed: Weekday[];
    }
  | {
      code: "missing-anchor";
      field: "dayOfMonth";
      question: string;
      minimum: 1;
      maximum: 31;
    };

export type ClarificationAnswer = { weekdays: Weekday[] } | { dayOfMonth: number };

/** Ask only when supplying a day can make the current intent expressible in cron. */
export function clarify(intent: ScheduleIntent): ScheduleClarification | null {
  validate(intent);
  const result = compile(intent);

  if (result.supported || result.code !== "missing-anchor") return null;

  if (intent.frequency.unit === "week") {
    return {
      code: "missing-anchor",
      field: "weekdays",
      question: "Which weekdays should the weekly schedule use?",
      allowed: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    };
  }

  if (intent.frequency.unit === "month") {
    return {
      code: "missing-anchor",
      field: "dayOfMonth",
      question: "Which day of the month should the monthly schedule use?",
      minimum: 1,
      maximum: 31,
    };
  }

  return null;
}

/** Add the answer to a fresh intent without discarding or sharing existing facts. */
export function resolveClarification(
  intent: ScheduleIntent,
  answer: ClarificationAnswer,
): ScheduleIntent {
  const clarification = clarify(intent);

  if (!clarification) {
    throw new IntentError("invalid-intent", "the intent has no missing cron anchor to clarify");
  }

  if (
    !answer ||
    typeof answer !== "object" ||
    Array.isArray(answer) ||
    Reflect.ownKeys(answer).length !== 1 ||
    !Object.hasOwn(answer, clarification.field)
  ) {
    throw new IntentError("invalid-intent", `the answer must contain only ${clarification.field}`);
  }

  const combined = { ...intent, ...answer };

  validate(combined);
  const resolved = structuredClone(combined);

  validate(resolved);

  return resolved;
}
