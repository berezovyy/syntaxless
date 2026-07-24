/** Numeric roles determine policy fields; exact values and units remain input facts. */
import { decimal, exactNumber, integer, numberDecimal } from "./exact.js";
import type { Role } from "./labels.js";
import type { Lexeme, Prepared } from "./prepare.js";
import { InputError } from "./types.js";
import type { Context, PolicyField, RetryIntent, RetryPolicy } from "./types.js";
import { FIELDS, validate, validateContext } from "./validate.js";

const numericRoles: readonly Role[] = ["RETRIES", "INITIAL", "CAP", "FACTOR", "JITTER_PERCENT"];

export function resolveRoles(
  prepared: Prepared,
  labels: readonly { role: Role; score: number }[],
  context: Context,
): { intent: RetryIntent; confidence: number } {
  validateContext(context);

  if (labels.length !== prepared.lexemes.length) {
    throw new Error("prediction length does not match input");
  }

  const tokens: { role: Role; lexeme: Lexeme; score: number }[] = [];
  let confidence = 1;

  for (const [i, lexeme] of prepared.lexemes.entries()) {
    const { role, score } = labels[i]!;

    if (!Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error("invalid prediction confidence");
    }

    if (role === "UNSUPPORTED") {
      throw new InputError("unsupported", "unsupported or non-actionable request");
    }

    if (
      (lexeme.numeric !== undefined && !numericRoles.includes(role)) ||
      (lexeme.unitMs !== undefined && role !== "UNIT") ||
      (lexeme.percent !== undefined && role !== "PERCENT") ||
      (lexeme.multiplier !== undefined && role !== (lexeme.multiplier === 2 ? "DOUBLE" : "TRIPLE"))
    ) {
      throw new InputError("invalid-input", "model did not preserve a number or unit");
    }

    if (role === "O" || role === "GLUE") continue;
    const previous = tokens.at(-1);

    if (
      previous?.role === role &&
      ![...numericRoles, "UNIT", "PERCENT", "DOUBLE", "TRIPLE"].includes(role)
    ) {
      previous.score = Math.max(previous.score, score);
    } else {
      tokens.push({ role, lexeme, score });
    }
  }

  for (const token of tokens) confidence = Math.min(confidence, token.score);

  if (!tokens.length) {
    throw new InputError("invalid-input", "request must contain an explicit retry policy fact");
  }

  const policy: Partial<RetryPolicy> = {};

  function set<K extends PolicyField>(field: K, value: RetryPolicy[K]): void {
    if (Object.hasOwn(policy, field)) {
      throw new InputError("conflict", `duplicate or conflicting ${field}`);
    }

    policy[field] = value;
  }

  let position = 0;

  function duration(numeric: string): number {
    const value = decimal(numeric);
    const unit = tokens[position++];

    if (unit?.role !== "UNIT" || unit.lexeme.unitMs === undefined) {
      throw new InputError("invalid-input", "provide an explicit supported duration unit");
    }

    return integer({
      numerator: value.numerator * BigInt(unit.lexeme.unitMs),
      denominator: value.denominator,
    });
  }

  while (position < tokens.length) {
    const { role, lexeme } = tokens[position++]!;

    if (numericRoles.includes(role) && lexeme.numeric === undefined) {
      throw new InputError("invalid-input", "numeric policy roles require an exact input number");
    }

    if (role === "RETRIES") {
      set("retries", integer(decimal(lexeme.numeric!)));
    } else if (role === "INITIAL") {
      set("initialDelayMs", duration(lexeme.numeric!));
    } else if (role === "CAP") {
      set("maxDelayMs", duration(lexeme.numeric!));
    } else if (role === "FACTOR") {
      set("multiplier", exactNumber(lexeme.numeric!));
    } else if (role === "DOUBLE" || role === "TRIPLE") {
      if (lexeme.multiplier !== (role === "DOUBLE" ? 2 : 3)) {
        throw new InputError(
          "invalid-input",
          "multiplier roles require a preserved arithmetic word",
        );
      }

      set("multiplier", lexeme.multiplier);
    } else if (role === "CONSTANT" || role === "EXPONENTIAL") {
      set("strategy", role === "CONSTANT" ? "constant" : "exponential");
    } else if (role === "JITTER_FULL" || role === "JITTER_EQUAL") {
      set("jitter", { kind: role === "JITTER_FULL" ? "full" : "equal" });
    } else if (role === "JITTER_NONE") {
      set("jitter", { kind: "none" });
    } else if (role === "JITTER_PERCENT") {
      const value = decimal(lexeme.numeric!);
      const percentage = tokens[position++];

      if (percentage?.role !== "PERCENT" || !percentage.lexeme.percent) {
        throw new InputError("invalid-input", "provide an explicit jitter percentage");
      }

      const fraction = Number(value.numerator) / Number(value.denominator * 100n);
      const rendered = numberDecimal(fraction);

      if (
        rendered.numerator * value.denominator * 100n !==
        value.numerator * rendered.denominator
      ) {
        throw new InputError("invalid-input", "jitter percentage cannot be preserved exactly");
      }

      set("jitter", { kind: "proportional", fraction });
    } else {
      throw new InputError("invalid-input", "incomplete or unsupported policy clause");
    }
  }

  if (policy.multiplier !== undefined && policy.strategy === undefined) {
    policy.strategy = "exponential";
  }

  const defaults: PolicyField[] = [];

  for (const field of FIELDS) {
    if (field === "multiplier" && policy.strategy === "constant") continue;

    if (!Object.hasOwn(policy, field) && Object.hasOwn(context.defaults ?? {}, field)) {
      Object.assign(policy, { [field]: context.defaults![field] });
      defaults.push(field);
    }
  }

  if (policy.retries === undefined) {
    throw new InputError("missing-count", "specify retries after the initial attempt");
  }

  if (policy.initialDelayMs === undefined) {
    throw new InputError("missing-initial-delay", "provide an initial delay or a caller default");
  }

  if (policy.strategy === undefined) {
    throw new InputError(
      "missing-strategy",
      "specify constant or exponential delays, or a caller default",
    );
  }

  const intent = { ...policy, defaults } as RetryIntent;

  validate(intent);

  return { intent, confidence };
}
