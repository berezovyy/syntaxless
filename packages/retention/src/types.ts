export const BUCKETS = ["day", "week", "month", "year"] as const;
export type Bucket = (typeof BUCKETS)[number];

export type Window =
  { kind: "calendar"; unit: Bucket; count: number } | { kind: "elapsed"; milliseconds: number };

export interface Tier {
  bucket: Bucket;
  window: Window;
}

export interface RetentionIntent {
  tiers: readonly Tier[];
}

export interface Context {
  timeZone: string;
  referenceInstant: number;
}

export interface Backup {
  id: string;
  timestamp: number;
}

export interface CompileOptions extends Context {
  backups: readonly Backup[];
}

export interface Preview {
  keep: readonly { id: string; tierIndexes: readonly number[] }[];
  remove: readonly string[];
}

export type FailureCode =
  | "invalid-input"
  | "invalid-intent"
  | "missing-context"
  | "invalid-context"
  | "invalid-backups"
  | "conflict"
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
