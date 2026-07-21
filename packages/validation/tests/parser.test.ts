import { expect, test } from "bun:test";

import { parse } from "../src/index.ts";
import type { FieldIntent, ParseFailureCode } from "../src/index.ts";

const positives: Array<{ phrase: string; intent: FieldIntent }> = [
  {
    phrase: "optional url",
    intent: { type: "url", required: false },
  },
  {
    phrase: "required boolean",
    intent: { type: "boolean", required: true },
  },
  {
    phrase: "required string between 5 and 20 characters",
    intent: { type: "string", required: true, minLength: 5, maxLength: 20 },
  },
];

for (const { phrase, intent } of positives) {
  test(`parse(${JSON.stringify(phrase)})`, async () => {
    const result = await parse(phrase, { source: "rules" });

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.intent).toEqual(intent);
      expect(result.source).toBe("rules");
    }
  });
}

test("more than 10 characters on string widens to inclusive minLength 11", async () => {
  const result = await parse("required string more than 10 characters");

  expect(result.ok).toBe(true);

  if (result.ok) {
    expect(result.intent).toEqual({ type: "string", required: true, minLength: 11 });
  }
});

const negatives: Array<{ phrase: string; code: ParseFailureCode }> = [
  { phrase: "The docs say required email max 100.", code: "unrecognized" },
  { phrase: "required max 100 characters", code: "missing-type" },
  { phrase: "email max 100 characters", code: "missing-presence" },
  { phrase: "required email min 10", code: "unrecognized-slot" },
  { phrase: "required optional email", code: "conflicting-presence" },
];

for (const { phrase, code } of negatives) {
  test(`parse(${JSON.stringify(phrase)}) fails as ${code}`, async () => {
    const result = await parse(phrase, { source: "rules" });

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.code).toBe(code);
  });
}
