/**
 * Track B: linearised intent CFG. Values are copy-head pointers or closed
 * class symbols, never free-form strings.
 */
export const GRAMMAR_TOKENS = [
  "(",
  ")",
  "every",
  "at",
  "days",
  "dom",
  "ord",
  "bounds",
  "except",
  "reject",
  "COPY",
  "UNIT_MIN",
  "UNIT_HOUR",
  "UNIT_DAY",
  "UNIT_WEEK",
  "UNIT_MONTH",
] as const;

export type GrammarToken = (typeof GRAMMAR_TOKENS)[number];

const INDEX = Object.fromEntries(GRAMMAR_TOKENS.map((token, index) => [token, index])) as Record<
  GrammarToken,
  number
>;

/** Allowed next tokens from a tiny stack-free prefix classifier. */
export function grammarMask(prefix: readonly GrammarToken[]): boolean[] {
  const mask = GRAMMAR_TOKENS.map(() => false);

  const allow = (...tokens: GrammarToken[]) => {
    for (const token of tokens) mask[INDEX[token]] = true;
  };

  const last = prefix[prefix.length - 1];

  if (prefix.length === 0) {
    allow("(");

    return mask;
  }

  if (last === "(") {
    allow("every", "reject");

    return mask;
  }

  if (last === "every") {
    allow("COPY");

    return mask;
  }

  if (last === "COPY") {
    allow("UNIT_MIN", "UNIT_HOUR", "UNIT_DAY", "UNIT_WEEK", "UNIT_MONTH", "at", "days", ")");

    return mask;
  }

  if (last === "reject") {
    allow("COPY", ")");

    return mask;
  }

  allow(")", "at", "days", "dom", "ord", "bounds", "except");

  return mask;
}

const UNIT_FROM_TOKEN: Record<string, "minute" | "hour" | "day" | "week" | "month"> = {
  UNIT_MIN: "minute",
  UNIT_HOUR: "hour",
  UNIT_DAY: "day",
  UNIT_WEEK: "week",
  UNIT_MONTH: "month",
};

/** Closed-class deserialize. COPY after `every` is interval 1 (copy-head later). */
export function intentFromGrammar(
  tokens: readonly GrammarToken[],
): { frequency: { unit: "minute" | "hour" | "day" | "week" | "month"; interval: number } } | null {
  if (tokens.includes("reject")) return null;
  let unit: "minute" | "hour" | "day" | "week" | "month" = "day";
  let interval = 1;

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;
    const mapped = UNIT_FROM_TOKEN[token];

    if (mapped !== undefined) unit = mapped;
    if (token === "every" && tokens[index + 1] === "COPY") interval = 1;
  }

  return { frequency: { unit, interval } };
}

export function greedyStep(
  prefix: readonly GrammarToken[],
  logits: readonly number[],
): GrammarToken {
  const mask = grammarMask(prefix);
  let best = -Infinity;
  let index = 0;

  for (let i = 0; i < logits.length && i < mask.length; i++) {
    if (!mask[i]) continue;

    if (logits[i]! > best) {
      best = logits[i]!;
      index = i;
    }
  }

  return GRAMMAR_TOKENS[index]!;
}
