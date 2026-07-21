import { expect, test } from "bun:test";

import { LABEL_COUNT, labelId } from "../src/model/labels.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import { decodeIntent, UnknownRoleError } from "../src/parser/decode.ts";
import type { LabeledToken } from "../src/parser/decode.ts";
import { resultFromLabeledTokens } from "../src/parser/model.ts";
import { normalize } from "../src/parser/normalize.ts";
import { renderField } from "../training/src/renderer.ts";

function goldLabels(
  text: string,
  spans: { start: number; end: number; label: keyof typeof labelId }[],
): LabeledToken[] {
  const tokens = tokenize(text);
  let spanIndex = 0;

  return tokens.map((token) => {
    while (spanIndex < spans.length && spans[spanIndex]!.end <= token.start) spanIndex++;
    const span = spans[spanIndex];

    const role =
      token.kind === 3 || span === undefined || token.start < span.start || token.end > span.end
        ? labelId.O
        : labelId[span.label];

    return { token, role, boundary: false, score: 1, boundaryScore: 1 };
  });
}

test("operator roles own inclusivity rather than re-parsing a predicted phrase", () => {
  const text = normalize("required integer more than 10");

  const labeled = goldLabels(text, [
    { start: 0, end: 8, label: "REQUIRED" },
    { start: 9, end: 16, label: "TYPE_INTEGER" },
    { start: 17, end: 26, label: "GTE" },
    { start: 27, end: 29, label: "NUMBER" },
  ]);

  const decoded = decodeIntent(labeled, text);

  expect(decoded.intent.minimum).toEqual({ value: 10, inclusive: true });
});

test("unknown role becomes model-error", () => {
  const text = normalize("required email");
  const tokens = tokenize(text);

  const labeled: LabeledToken[] = tokens.map((token) => ({
    token,
    role: token.kind === 3 ? labelId.O : LABEL_COUNT + 3,
    boundary: false,
    score: 1,
    boundaryScore: 1,
  }));

  expect(() => decodeIntent(labeled, text)).toThrow(UnknownRoleError);
  const result = resultFromLabeledTokens(labeled, text);

  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.code).toBe("model-error");
});

test("a predicted unsupported constraint refuses the entire intent explicitly", () => {
  const text = "required string matching";

  const labeled = goldLabels(text, [
    { start: 0, end: 8, label: "REQUIRED" },
    { start: 9, end: 15, label: "TYPE_STRING" },
    { start: 16, end: 24, label: "UNSUPPORTED" },
  ]);

  expect(resultFromLabeledTokens(labeled, text)).toMatchObject({
    ok: false,
    code: "unrecognized",
    source: "model",
  });
});

test("low margin fails closed with no intent", () => {
  const rendered = renderField({ type: "email", required: true, maxLength: 100 }, undefined, {
    order: "presence-type-glue-comparison",
    typeSurface: "email",
    presenceSurface: "required",
    maxSurface: "max",
    unitSurface: "characters",
    glueSurface: "with",
    numberStyle: "digits",
  });

  const labeled = goldLabels(rendered.text, rendered.spans).map((item) => ({
    ...item,
    score: 0.1,
  }));

  const result = resultFromLabeledTokens(labeled, rendered.text);

  expect(result.ok).toBe(false);

  if (!result.ok) {
    expect(result.code).toBe("low-confidence");
    expect(result.source).toBe("model");
  }
});
