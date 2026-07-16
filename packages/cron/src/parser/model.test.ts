import { expect, test } from "bun:test";

import { LABEL_COUNT, labelId } from "../model/labels";
import { tokenize } from "../model/tokenizer";
import type { LabeledToken } from "./decode";
import { resultFromLabeledTokens } from "./model";

function daily(score = 1): LabeledToken[] {
  return [
    {
      token: tokenize("daily")[0]!,
      role: labelId.UNIT_DAY,
      boundary: true,
      score,
      actionable: true,
    },
  ];
}

test("a score below 0.5 suppresses the decoded intent", () => {
  const result = resultFromLabeledTokens(daily(0.499), undefined, "daily", "off", "request");

  expect(result).toMatchObject({
    ok: false,
    source: "model",
    code: "low-confidence",
    confidence: 0.499,
    diagnostics: [expect.objectContaining({ code: "low-confidence", severity: "warning" })],
  });

  expect(result).not.toHaveProperty("intent");
});

test.each([
  { score: 0.5, diagnostic: "low-margin" },
  { score: 0.9, diagnostic: null },
])(
  "confidence $score preserves the intent with the correct diagnostic",
  ({ score, diagnostic }) => {
    const result = resultFromLabeledTokens(daily(score), undefined, "daily", "off", "request");

    expect(result).toMatchObject({
      ok: true,
      source: "model",
      confidence: score,
      intent: { frequency: { unit: "day", interval: 1 } },
    });

    expect(result.diagnostics).toEqual(
      diagnostic === null
        ? []
        : [expect.objectContaining({ code: diagnostic, severity: "suggestion" })],
    );
  },
);

test.each(["descriptive"] as const)(
  "%s rejects even a high-confidence actionable schedule",
  (speechAct) => {
    const result = resultFromLabeledTokens(
      daily(),
      undefined,
      "Please run this daily",
      "off",
      speechAct,
    );

    expect(result).toMatchObject({
      ok: false,
      code: "unsafe-context",
      source: "model",
    });

    expect(result).not.toHaveProperty("intent");
  },
);

test.each([
  {
    name: "unknown role",
    tokens: [{ ...daily()[0]!, role: LABEL_COUNT }],
    act: "request" as const,
    diagnostic: "unknown-role",
  },
  {
    name: "missing speech act",
    tokens: daily(),
    act: undefined,
    diagnostic: "decode-error",
  },
])(
  "$name becomes an explicit model error rather than a guessed intent",
  ({ tokens, act, diagnostic }) => {
    const result = resultFromLabeledTokens(tokens, undefined, "daily", "off", act);

    expect(result).toMatchObject({
      ok: false,
      code: "model-error",
      diagnostics: [expect.objectContaining({ code: diagnostic, severity: "error" })],
    });

    expect(result).not.toHaveProperty("intent");
  },
);

test.each(["thursday friday"])(
  "correction selects the actionable replacement, not the first weekday: %s",
  (text) => {
    const tokens = tokenize(text).filter((token) => token.kind !== 3);

    const labeled = tokens.map((token, index) => ({
      token,
      role: token.text === "friday" ? labelId.WEEKDAY_FRI : labelId.WEEKDAY_THU,
      boundary: index === 0,
      score: 1,
      actionable: token.text === "friday",
    }));

    expect(resultFromLabeledTokens(labeled, undefined, text, "off", "correction")).toMatchObject({
      ok: true,
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: ["friday"],
      },
    });

    const empty = resultFromLabeledTokens(
      labeled.map((token) => ({ ...token, actionable: false })),
      undefined,
      text,
      "off",
      "correction",
    );

    expect(empty).toMatchObject({ ok: false, code: "unsafe-context" });
    expect(empty).not.toHaveProperty("intent");
  },
);

test("raw schedule wording cannot rescue all-O labels", () => {
  const result = resultFromLabeledTokens(
    [{ ...daily()[0]!, role: labelId.O }],
    undefined,
    "Please run this daily",
    "on",
    "request",
  );

  expect(result).toMatchObject({
    ok: false,
    code: "unsafe-context",
    diagnostics: [expect.objectContaining({ code: "no-actionable-span" })],
  });

  expect(result).not.toHaveProperty("intent");
});
