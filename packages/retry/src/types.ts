export type Jitter =
  | { kind: "none" }
  | { kind: "full" }
  | { kind: "equal" }
  | { kind: "proportional"; fraction: number };

export interface RetryPolicy {
  /** Retries after the initial attempt, not total attempts. */
  retries: number;
  strategy: "constant" | "exponential";
  initialDelayMs: number;
  multiplier?: number;
  maxDelayMs?: number;
  jitter?: Jitter;
}

export type PolicyField = keyof RetryPolicy;

export interface RetryIntent extends RetryPolicy {
  defaults: readonly PolicyField[];
}

export interface Context {
  defaults?: Partial<RetryPolicy>;
}

export type FailureCode =
  | "invalid-input"
  | "invalid-intent"
  | "invalid-defaults"
  | "conflict"
  | "missing-count"
  | "missing-strategy"
  | "missing-initial-delay"
  | "missing-multiplier"
  | "missing-seed"
  | "overflow"
  | "unsupported";

export type Result<T> =
  ({ supported: true } & T) | { supported: false; code: FailureCode; reason: string };

export class InputError extends Error {
  constructor(
    readonly code: FailureCode,
    message: string,
  ) {
    super(message);
  }
}
