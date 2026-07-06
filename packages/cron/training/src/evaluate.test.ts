import { describe, expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import type { GoldRecord } from "./evaluate";
import {
  canonicalizeIntent,
  lintGoldRecords,
  repairInclusiveBoundGold,
  repairObservableGold,
} from "./evaluate";

describe("canonicalizeIntent", () => {
  test("normalizes semantic sets without mutating the input", () => {
    const intent = {
      frequency: { unit: "week" as const, interval: 1 },
      weekdays: ["friday" as const, "monday" as const],
      exceptions: {
        weekdays: ["sunday" as const, "tuesday" as const],
        dates: ["2026-12-02", "2026-01-01", "2026-12-02"],
      },
    };

    expect(canonicalizeIntent(intent)).toEqual({
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday", "friday"],
      exceptions: {
        weekdays: ["tuesday", "sunday"],
        dates: ["2026-01-01", "2026-12-02"],
      },
    });

    expect(intent.weekdays).toEqual(["friday", "monday"]);
  });
});

describe("repairObservableGold", () => {
  test("makes a latent PM clock observable and keeps spans aligned", () => {
    const item: GoldRecord = {
      id: "clock",
      family: "time-of-day",
      text: "remind me 1:30 daily",
      spans: [
        { start: 0, end: 9, label: "O", clauseStart: false },
        { start: 10, end: 11, label: "HOUR", clauseStart: true },
        { start: 11, end: 12, label: "GLUE", clauseStart: false },
        { start: 12, end: 14, label: "MINUTE", clauseStart: false },
        { start: 15, end: 20, label: "RECUR", clauseStart: false },
      ],
      intent: {
        frequency: { unit: "day", interval: 1 },
        time: { hour: 13, minute: 30 },
      },
      expected: { supported: true, expression: "30 13 * * *" },
    };

    const repaired = repairObservableGold(item);

    expect(repaired.text).toBe("remind me 1:30 pm daily");

    expect(repaired.spans.find((span) => span.label === "MERIDIEM")).toEqual({
      start: 15,
      end: 17,
      label: "MERIDIEM",
      clauseStart: false,
    });

    for (const token of tokenize(repaired.text).filter((value) => value.kind !== 3)) {
      expect(
        repaired.spans.some((span) => token.start >= span.start && token.end <= span.end),
      ).toBe(true);
    }
  });

  test("leaves already explicit and non-clock records byte-equivalent", () => {
    const explicit: GoldRecord = {
      id: "explicit",
      family: "time-of-day",
      text: "1:30 pm daily",
      spans: [],
      intent: {
        frequency: { unit: "day", interval: 1 },
        time: { hour: 13, minute: 30 },
      },
      expected: { supported: true, expression: "30 13 * * *" },
    };

    expect(repairObservableGold(explicit)).toBe(explicit);
  });
});

describe("lintGoldRecords", () => {
  test("rejects duplicate ids and overlapping spans", () => {
    const record: GoldRecord = {
      id: "same",
      family: "negative",
      text: "ordinary prose",
      spans: [{ start: 0, end: 8, label: "O", clauseStart: false }],
      intent: null,
      expected: { supported: false, expression: null },
    };

    expect(() => lintGoldRecords([record, record])).toThrow(/duplicate/);

    expect(() =>
      lintGoldRecords([
        {
          ...record,
          spans: [
            { start: 0, end: 8, label: "O", clauseStart: false },
            { start: 7, end: 13, label: "O", clauseStart: false },
          ],
        },
      ]),
    ).toThrow(/span/);
  });
});

describe("repairInclusiveBoundGold", () => {
  test("turns a legacy through date into an exclusive end", () => {
    const item: GoldRecord = {
      id: "bounds",
      family: "bounds",
      text: "from 2026-01-01 through 2026-01-31, every day",
      spans: [],
      intent: {
        frequency: { unit: "day", interval: 1 },
        bounds: { start: "2026-01-01", end: "2026-01-31" },
      },
      expected: { supported: false, expression: null },
    };

    expect(repairInclusiveBoundGold(item).intent?.bounds).toEqual({
      start: "2026-01-01",
      end: "2026-02-01",
    });
  });
});
