export type IntentFailureCode =
  | "invalid-intent"
  | "invalid-interval"
  | "invalid-time"
  | "invalid-weekday"
  | "invalid-day-of-month"
  | "invalid-ordinal"
  | "invalid-day"
  | "invalid-bounds"
  | "invalid-exceptions"
  | "conflicting-selectors"
  | "conflicting-time"
  | "missing-anchor";

export interface IntentFailure {
  supported: false;
  code: IntentFailureCode;
  reason: string;
}

/** A malformed intent is a caller error, not an unsupported cron expression. */
export class IntentError extends TypeError {
  constructor(
    public readonly code: IntentFailureCode,
    message: string,
  ) {
    super(message);
    this.name = "IntentError";
  }
}
