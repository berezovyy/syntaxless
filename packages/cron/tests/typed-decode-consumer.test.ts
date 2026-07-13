/**
 * Fresh consumer of the shipped decode/labels path (not an internal decoder test).
 * WEEKDAY_MON + HOUR_9 must deserialize via valueOf, not token text.
 */
import { expect, test } from "bun:test";

import { labelId, valueOf } from "../src/model/labels.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import { decodeIntent } from "../src/parser/decode.ts";

test("consumer decodeIntent reads valueOf(WEEKDAY_MON) and HOUR_9", () => {
  expect(valueOf("WEEKDAY_MON")).toBe("monday");
  expect(valueOf("HOUR_9")).toBe(9);
  const tokens = tokenize("xyzzy 9").filter((token) => token.kind !== 3);

  const labeled = tokens.map((token) => ({
    token,
    role: token.text === "9" ? labelId.HOUR_9 : labelId.WEEKDAY_MON,
    boundary: token.text !== "9",
    score: 1,
    actionable: true,
  }));

  const decoded = decodeIntent(labeled, undefined, "xyzzy 9", "off", "request");

  expect(decoded.intent.weekdays).toEqual(["monday"]);
  expect(decoded.intent.time).toEqual({ hour: 9, minute: 0 });
});

test("consumer HOUR_LIT on digits uses parseInt", () => {
  const tokens = tokenize("14 daily").filter((token) => token.kind !== 3);

  const labeled = tokens.map((token) => ({
    token,
    role: token.text === "14" ? labelId.HOUR_LIT : labelId.UNIT_DAY,
    boundary: token.text === "14",
    score: 1,
    actionable: true,
  }));

  const decoded = decodeIntent(labeled, undefined, "14 daily", "off", "request");

  expect(decoded.intent.time).toEqual({ hour: 14, minute: 0 });
  expect(decoded.intent.frequency).toEqual({ unit: "day", interval: 1 });
});
