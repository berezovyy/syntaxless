import type { Role } from "./labels.js";
import type { Prepared } from "./prepare.js";
import { InputError } from "./types.js";
import type { Bucket, RetentionIntent, Tier } from "./types.js";
import { calendarWindow, elapsedMs } from "./units.js";
import { validate } from "./validate.js";

const bucketRoles: Partial<Record<Role, Bucket>> = {
  DAY: "day",
  WEEK: "week",
  MONTH: "month",
  YEAR: "year",
};

export function resolveRoles(
  prepared: Prepared,
  labels: readonly { role: Role; score: number }[],
  _context: unknown = {},
): { intent: RetentionIntent; confidence: number } {
  if (labels.length !== prepared.lexemes.length) {
    throw Error("prediction length does not match input");
  }

  const tokens: { role: Role; lexeme: Prepared["lexemes"][number]; score: number }[] = [];
  let confidence = 1;

  for (const [i, lexeme] of prepared.lexemes.entries()) {
    const { role, score } = labels[i]!;

    if (!Number.isFinite(score) || score < 0 || score > 1) {
      throw Error("invalid prediction confidence");
    }

    if (role === "UNSUPPORTED") {
      throw new InputError("unsupported", "unsupported or non-actionable retention request");
    }

    if (
      (lexeme.numeric !== undefined &&
        role !== "CALENDAR" &&
        role !== "ELAPSED" &&
        role !== "ONE") ||
      (lexeme.unit !== undefined && role !== "UNIT" && bucketRoles[role] !== lexeme.unit) ||
      (lexeme.bucket !== undefined && bucketRoles[role] !== lexeme.bucket) ||
      (lexeme.rolling && role !== "ROLLING") ||
      (lexeme.calendar && role !== "PERIODS")
    ) {
      throw new InputError(
        "invalid-input",
        "model did not preserve a number, unit or calendar fact",
      );
    }

    if (role === "O" || role === "GLUE") continue;
    const prior = tokens.at(-1);

    if (prior?.role === role && bucketRoles[role]) prior.score = Math.max(prior.score, score);
    else tokens.push({ role, lexeme, score });
  }

  for (const t of tokens) confidence = Math.min(confidence, t.score);
  const tiers: Tier[] = [];

  let current:
    | {
        bucket: Bucket;
        window?: Tier["window"];
        rolling: boolean;
        calendar: boolean;
        clock: boolean;
      }
    | undefined;

  const flush = () => {
    if (!current) return;

    if (!current.window) {
      throw new InputError("invalid-input", "specify a window for each retention tier");
    }

    if (current.window.kind === "elapsed" && !current.rolling && !current.clock) {
      throw new InputError("invalid-input", "elapsed windows require an explicit rolling marker");
    }

    if (
      (current.rolling && current.window.kind !== "elapsed") ||
      (current.calendar && current.window.kind !== "calendar")
    ) {
      throw new InputError("conflict", "calendar and elapsed window instructions conflict");
    }

    tiers.push({ bucket: current.bucket, window: current.window });
  };

  for (let position = 0; position < tokens.length; position++) {
    const { role, lexeme } = tokens[position]!;
    const bucket = bucketRoles[role];

    if (bucket) {
      if (lexeme.bucket !== bucket && lexeme.unit !== bucket) {
        throw new InputError("invalid-input", "bucket roles require a preserved calendar word");
      }

      flush();
      current = { bucket, rolling: false, calendar: false, clock: false };
      continue;
    }

    if (role === "ONE") {
      if (
        lexeme.numeric === undefined ||
        calendarWindow(lexeme.numeric, "day", "day").count !== 1
      ) {
        throw new InputError(
          "unsupported",
          "only one newest representative per bucket is supported",
        );
      }

      continue;
    }

    if (!current) throw new InputError("invalid-input", "provide a bucket before its window");

    if (role === "ROLLING") {
      if (!lexeme.rolling) {
        throw new InputError("invalid-input", "elapsed marker requires an explicit input word");
      }

      current.rolling = true;
      continue;
    }

    if (role === "PERIODS") {
      if (!lexeme.calendar) {
        throw new InputError("invalid-input", "calendar marker requires an explicit input word");
      }

      current.calendar = true;
      continue;
    }

    if ((role !== "CALENDAR" && role !== "ELAPSED") || lexeme.numeric === undefined) {
      throw new InputError("invalid-input", "window requires an exact number and supported unit");
    }

    if (current.window) throw new InputError("conflict", "each bucket must have one window");

    while (tokens[position + 1]?.role === "PERIODS") {
      if (!tokens[++position]!.lexeme.calendar) {
        throw new InputError("invalid-input", "calendar marker requires an explicit input word");
      }

      current.calendar = true;
    }

    const unit = tokens[++position];

    if (unit?.role !== "UNIT" || !unit.lexeme.unit) {
      throw new InputError("invalid-input", "provide a supported window unit");
    }

    current.clock = ["millisecond", "second", "minute", "hour"].includes(unit.lexeme.unit);

    current.window =
      role === "CALENDAR" && !current.clock
        ? calendarWindow(lexeme.numeric, unit.lexeme.unit, current.bucket)
        : {
            kind: "elapsed",
            milliseconds: elapsedMs(lexeme.numeric, unit.lexeme.unit),
          };
  }

  flush();
  const intent = { tiers };

  validate(intent);

  return { intent, confidence };
}
