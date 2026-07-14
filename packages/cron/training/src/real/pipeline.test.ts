import { expect, test } from "bun:test";

import { weekdayLabel } from "../../../src/model/labels";
import { tokenize } from "../../../src/model/tokenizer";
import type { ScheduleIntent } from "../../../src/types/intent";
import { agreeKeep } from "./agree";
import { executeKeep } from "./execute";
import { projectKeep } from "./project";
import { roundtripKeep } from "./roundtrip";

const monday: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday"],
};

const friday: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["friday"],
};

test("agree drops disagreement among k=3 samples", () => {
  const kept = agreeKeep([
    {
      id: "ok",
      samples: [{ intent: monday }, { intent: monday }, { intent: monday }],
    },
    {
      id: "disagree",
      samples: [{ intent: monday }, { intent: friday }, { intent: monday }],
    },
  ]);

  expect(kept.map((row) => row.id)).toEqual(["ok"]);
});

test("execute keeps compiler-unsupported intents and drops thrown compile-fail", () => {
  const ordinal: ScheduleIntent = {
    frequency: { unit: "month", interval: 1 },
    ordinalWeekday: { ordinal: "first", weekday: "monday" },
  };

  const kept = executeKeep([
    { id: "daily", intent: { frequency: { unit: "day", interval: 1 } } },
    { id: "ordinal", intent: ordinal },
    { id: "crash", intent: null },
  ]);

  expect(kept.map((row) => row.id)).toEqual(["daily", "ordinal", "crash"]);

  const dropped = executeKeep([
    {
      id: "bad",
      intent: { frequency: { unit: "day" } } as unknown as ScheduleIntent,
    },
  ]);

  expect(dropped.map((row) => row.id)).toEqual([]);
});

test("roundtrip drops canonical mismatch", () => {
  const kept = roundtripKeep([
    { id: "ok", intent: monday, reconstructed: monday },
    { id: "bad", intent: monday, reconstructed: friday },
  ]);

  expect(kept.map((row) => row.id)).toEqual(["ok"]);
});

test("project drops spans whose oracle-decode is not the intent", () => {
  const text = "every monday";
  const tokens = tokenize(text).filter((token) => token.kind !== 3);

  const spans = tokens.map((token) => ({
    start: token.start,
    end: token.end,
    label:
      token.text === "monday"
        ? weekdayLabel("monday")
        : token.text === "every"
          ? ("RECUR" as const)
          : ("GLUE" as const),
    clauseStart: token.text === "every",
    actionable: true,
  }));

  const kept = projectKeep([
    { id: "ok", text, spans, intent: monday },
    {
      id: "bad",
      text,
      spans: spans.map((span) =>
        span.label === weekdayLabel("monday") ? { ...span, label: weekdayLabel("friday") } : span,
      ),
      intent: monday,
    },
  ]);

  expect(kept.map((row) => row.id)).toEqual(["ok"]);
});
