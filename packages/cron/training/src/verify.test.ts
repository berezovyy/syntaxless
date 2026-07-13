import { expect, test } from "bun:test";

import type { ScheduleIntent } from "../../src/types/intent";
import { verifyCandidate } from "./llm/verify";
import type { VerifyContext } from "./llm/verify";

const CONTEXT: VerifyContext = {
  angleId: "ops/weekday-list/terse",
  model: "test-model",
  promptHash: "f".repeat(64),
  runId: "run-test",
};

function schedule(text: string, intent: ScheduleIntent) {
  return { text, intent, output: "schedule" as const };
}

test("a correct schedule is accepted with exact spans", async () => {
  const result = await verifyCandidate(
    schedule("every Monday", {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday"],
    }),
    CONTEXT,
  );

  expect(result.ok).toBe(true);
  if (!result.ok) return;
  const { record } = result;

  expect(record.source).toBe("llm");
  expect(record.angleId).toBe(CONTEXT.angleId);
  expect(record.model).toBe(CONTEXT.model);
  expect(record.promptHash).toBe(CONTEXT.promptHash);
  expect(record.runId).toBe(CONTEXT.runId);

  expect(record.spans).toEqual([
    { start: 0, end: 5, label: "RECUR", clauseStart: false, actionable: true },
    {
      start: 6,
      end: 12,
      label: "WEEKDAY_MON",
      clauseStart: true,
      value: "monday",
      actionable: true,
    },
  ]);

  expect(record.fingerprint).toMatch(/^[0-9a-f]{64}$/);

  expect(record.intent).toEqual({
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
  });
});

test("a wrong weekday is intent-text-mismatch", async () => {
  const result = await verifyCandidate(
    schedule("every Wednesday", {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["tuesday"],
    }),
    CONTEXT,
  );

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.rejection.reason).toBe("intent-text-mismatch");
});

test("a schedule candidate without an intent is unsupported", async () => {
  const result = await verifyCandidate(
    { text: "do the thing", intent: null, output: "schedule" },
    CONTEXT,
  );

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.rejection.reason).toBe("unsupported");
});

test("the compiler's failure code is passed straight through", async () => {
  const result = await verifyCandidate(
    schedule("the first Monday of every month", {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    }),
    CONTEXT,
  );

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.rejection.reason).toBe("ordinal-weekday");
});

test("a sentence with a schedule marked negative is negative-has-schedule", async () => {
  const result = await verifyCandidate(
    { text: "every Monday", intent: null, output: "negative" },
    CONTEXT,
  );

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.rejection.reason).toBe("negative-has-schedule");
});

test("a new schedule word is unknown-lexeme", async () => {
  const result = await verifyCandidate(
    schedule("every Monday fortnightly", {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday"],
    }),
    CONTEXT,
  );

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.rejection.reason).toBe("unknown-lexeme");
  expect(result.rejection.text).toBe("every Monday fortnightly");
  expect(result.rejection.model).toBe(CONTEXT.model);
});

test("ordinary prose is accepted as a negative with all-O spans", async () => {
  const text = "Please print the report.";
  const result = await verifyCandidate({ text, intent: null, output: "negative" }, CONTEXT);

  expect(result.ok).toBe(true);
  if (!result.ok) return;
  expect(result.record.intent).toBeNull();
  expect(result.record.spans.every((span) => span.label === "O")).toBe(true);
});

test("text policy rejects empty and control characters as text-invalid", async () => {
  for (const text of ["   ", "bad\u0001text"]) {
    const result = await verifyCandidate({ text, intent: null, output: "negative" }, CONTEXT);

    expect(result.ok).toBe(false);
    if (result.ok) continue;
    expect(result.rejection.reason).toBe("text-invalid");
  }
});

test("model-proposed slots are schema-invalid and never reach alignment", async () => {
  const result = await verifyCandidate(
    {
      text: "every Monday",
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday"],
      },
      output: "schedule",
      slots: [{ start: 0, end: 12, label: "WEEKDAY_MON" }],
    },
    CONTEXT,
  );

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.rejection.reason).toBe("schema-invalid");
});

test("verification is deterministic for ambiguous text", async () => {
  const candidate = schedule("every Monday at 9", {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
    time: { hour: 9, minute: 0 },
  });

  const first = await verifyCandidate(candidate, CONTEXT);
  const second = await verifyCandidate(candidate, CONTEXT);

  expect(JSON.stringify(first)).toBe(JSON.stringify(second));
});
