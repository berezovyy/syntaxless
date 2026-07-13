import { describe, expect, test } from "bun:test";

import { attributeMiss, goldViolatesSemantics } from "./attribution";
import { oracleDecode, type OracleRecord } from "./oracle";

const hourly: OracleRecord = {
  id: "hourly",
  source: "kind-seed",
  family: "interval",
  kind: "unit-interval",
  template: "adverb",
  text: "hourly",
  spans: [{ start: 0, end: 6, label: "INTERVAL", clauseStart: true, actionable: true }],
  intent: { frequency: { unit: "hour", interval: 1 } },
  speechAct: "request",
};

const unknownWeekday: OracleRecord = {
  id: "unknown-weekday",
  source: "gold",
  family: "weekday-list",
  kind: "single-weekday",
  template: "bare",
  text: "xyzzy",
  spans: [{ start: 0, end: 5, label: "WEEKDAY", clauseStart: true, actionable: true }],
  intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
  speechAct: "request",
};

const twelveHour: OracleRecord = {
  id: "clock-130",
  source: "gold",
  family: "time-of-day",
  kind: "time-of-day",
  template: "digits",
  text: "1:30 daily",
  spans: [
    { start: 0, end: 1, label: "HOUR", clauseStart: true, actionable: true },
    { start: 1, end: 2, label: "GLUE", clauseStart: false, actionable: false },
    { start: 2, end: 4, label: "MINUTE", clauseStart: false, actionable: true },
    { start: 5, end: 10, label: "INTERVAL", clauseStart: false, actionable: true },
  ],
  intent: {
    frequency: { unit: "day", interval: 1 },
    time: { hour: 13, minute: 30 },
  },
  speechAct: "request",
};

describe("attributeMiss", () => {
  test("GOLD: uninvertible 12h clock without meridiem", () => {
    expect(
      goldViolatesSemantics({
        text: twelveHour.text,
        expectedIntent: twelveHour.intent,
        accepted: true,
        actualIntent: { frequency: { unit: "day", interval: 1 }, time: { hour: 1, minute: 30 } },
        goldSpans: twelveHour.spans,
        goldOracle: oracleDecode(twelveHour),
      }),
    ).toBe(true);

    expect(
      attributeMiss({
        text: twelveHour.text,
        expectedIntent: twelveHour.intent,
        accepted: true,
        actualIntent: { frequency: { unit: "day", interval: 1 }, time: { hour: 1, minute: 30 } },
        goldSpans: twelveHour.spans,
        goldOracle: oracleDecode(twelveHour),
        goldLabels: ["HOUR", "GLUE", "MINUTE", "INTERVAL"],
        predictedLabels: ["HOUR", "GLUE", "MINUTE", "INTERVAL"],
      }),
    ).toBe("GOLD");
  });

  test("DECODE: gold labels do not round-trip through shipped decodeIntent", () => {
    const goldOracle = oracleDecode(unknownWeekday);

    expect(goldOracle.ok).toBe(false);

    expect(
      attributeMiss({
        text: unknownWeekday.text,
        expectedIntent: unknownWeekday.intent,
        accepted: false,
        actualIntent: null,
        goldSpans: unknownWeekday.spans,
        goldOracle,
        goldLabels: ["WEEKDAY"],
        predictedLabels: ["WEEKDAY"],
        goldSpeechAct: "request",
        predictedSpeechAct: "request",
      }),
    ).toBe("DECODE");
  });

  test("ROLE: predicted labels differ from gold on a round-trippable record", () => {
    const goldOracle = oracleDecode(hourly);

    expect(goldOracle.ok).toBe(true);

    expect(
      attributeMiss({
        text: hourly.text,
        expectedIntent: hourly.intent,
        accepted: false,
        actualIntent: null,
        goldSpans: hourly.spans,
        goldOracle,
        goldLabels: ["INTERVAL"],
        predictedLabels: ["O"],
        goldSpeechAct: "request",
        predictedSpeechAct: "request",
      }),
    ).toBe("ROLE");
  });

  test("ACT: speech-act head disagrees on a round-trippable record", () => {
    const goldOracle = oracleDecode(hourly);

    expect(goldOracle.ok).toBe(true);

    expect(
      attributeMiss({
        text: hourly.text,
        expectedIntent: hourly.intent,
        accepted: false,
        actualIntent: null,
        goldSpans: hourly.spans,
        goldOracle,
        goldLabels: ["INTERVAL"],
        predictedLabels: ["INTERVAL"],
        goldSpeechAct: "request",
        predictedSpeechAct: "descriptive",
        goldActionable: [true],
        predictedActionable: [true],
      }),
    ).toBe("ACT");
  });

  test("CONF: gold and predicted oracles match, public parse rejected", () => {
    const goldOracle = oracleDecode(hourly);

    expect(goldOracle.ok).toBe(true);

    expect(
      attributeMiss({
        text: hourly.text,
        expectedIntent: hourly.intent,
        accepted: false,
        actualIntent: null,
        goldSpans: hourly.spans,
        goldOracle,
        predictedOracle: goldOracle,
        goldLabels: ["INTERVAL"],
        predictedLabels: ["INTERVAL"],
        goldSpeechAct: "request",
        predictedSpeechAct: "request",
        goldActionable: [true],
        predictedActionable: [true],
      }),
    ).toBe("CONF");
  });
});
