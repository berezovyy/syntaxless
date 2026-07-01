import { expect, test } from "bun:test";

import { ANGLES } from "./llm/angles";
import { isTransient, matchesAngle, parseArgs, selectAngles } from "./llm/runner";
import type { CandidateMeta } from "./llm/workflow";
import { parseCandidates, renderPrompt } from "./llm/workflow";

const META: CandidateMeta = {
  angleId: "ops/weekday-list/terse",
  promptId: "schedule-basic",
  promptVersion: 1,
  promptHash: "f".repeat(64),
  model: "test-model",
  runId: "run-test",
};

const SCHEDULE_LINE =
  '{"text":"every Monday at 9am","intent":{"frequency":{"unit":"week","interval":1},"weekdays":["monday"]},"output":"schedule"}';

const NEGATIVE_LINE = '{"text":"just a note for later","intent":null,"output":"negative"}';

test("parses one JSON candidate per line and attaches provenance", () => {
  const result = parseCandidates(`${SCHEDULE_LINE}\n${NEGATIVE_LINE}\n`, META);

  expect(result.rejected).toBe(0);
  expect(result.candidates.length).toBe(2);

  const first = result.candidates[0]!;

  expect(first.text).toBe("every Monday at 9am");
  expect(first.output).toBe("schedule");
  expect(first.intent).not.toBeNull();
  expect(first.angleId).toBe(META.angleId);
  expect(first.promptId).toBe(META.promptId);
  expect(first.promptVersion).toBe(META.promptVersion);
  expect(first.promptHash).toBe(META.promptHash);
  expect(first.model).toBe(META.model);
  expect(first.runId).toBe(META.runId);
  expect(first.rawHash).toMatch(/^[0-9a-f]{64}$/);

  const second = result.candidates[1]!;

  expect(second.intent).toBeNull();
  expect(second.output).toBe("negative");
  expect(second.rawHash).not.toBe(first.rawHash);
});

test("malformed lines are candidate-level rejects, not crashes", () => {
  const raw = [
    NEGATIVE_LINE,
    "not json at all",
    '{"text":"missing output","intent":null}',
    '["an","array"]',
    '{"text":42,"intent":null,"output":"negative"}',
    SCHEDULE_LINE,
    "   ",
  ].join("\n");

  const result = parseCandidates(raw, META);

  expect(result.candidates.length).toBe(2);
  expect(result.rejected).toBe(4);
});

test("an empty response yields no candidates and no rejects", () => {
  const result = parseCandidates("\n  \n", META);

  expect(result.candidates).toEqual([]);
  expect(result.rejected).toBe(0);
});

test("a raw candidate has exactly the provenance-bearing shape", () => {
  const [candidate] = parseCandidates(NEGATIVE_LINE, META).candidates;

  expect(Object.keys(candidate!).sort()).toEqual([
    "angleId",
    "intent",
    "model",
    "output",
    "promptHash",
    "promptId",
    "promptVersion",
    "rawHash",
    "runId",
    "text",
  ]);
});

test("renderPrompt substitutes every placeholder", () => {
  const body =
    "family={{family}} count={{count}} schema={{schema}} " +
    "lexicon={{lexicon}} seed={{seedIntent}}";

  const rendered = renderPrompt(body, {
    lexicon: "WEEKDAY: Monday",
    family: "weekday-list",
    seedIntent: "null",
    count: 5,
    schema: "{}",
  });

  expect(rendered).toContain("family=weekday-list");
  expect(rendered).toContain("count=5");
  expect(rendered).toContain("schema={}");
  expect(rendered).toContain("lexicon=WEEKDAY: Monday");
  expect(rendered).toContain("seed=null");
  expect(rendered).not.toContain("{{");
});

test("angle patterns select by prefix and wildcard", () => {
  expect(matchesAngle("ops/weekday-list", "ops/weekday-list/terse")).toBe(true);
  expect(matchesAngle("ops/*", "ops/bounds/terse")).toBe(true);
  expect(matchesAngle("standup", "ops/weekday-list/terse")).toBe(false);

  const selected = selectAngles(["ops/weekday-list"]);

  expect(selected.length).toBe(2);
  expect(selected.every((angle) => angle.id.startsWith("ops/weekday-list/"))).toBe(true);
});

test("run args require angles, a positive count, and an output", () => {
  expect(parseArgs(["--angles", "ops/*", "--count", "5", "--out", "/tmp/x.jsonl"])).toEqual({
    patterns: ["ops/*"],
    count: 5,
    out: "/tmp/x.jsonl",
  });

  expect(() => parseArgs(["--count", "5", "--out", "x"])).toThrow();
  expect(() => parseArgs(["--angles", "ops/*", "--count", "0", "--out", "x"])).toThrow();
  expect(() => parseArgs(["--angles", "ops/*", "--count", "5"])).toThrow();
});

test("only 429/5xx, socket codes, and fetch errors are transient", () => {
  expect(isTransient({ status: 500 })).toBe(true);
  expect(isTransient({ status: 429 })).toBe(true);
  expect(isTransient({ status: 400 })).toBe(false);
  expect(isTransient({ code: "ECONNRESET" })).toBe(true);
  expect(isTransient(new Error("fetch failed"))).toBe(true);
  expect(isTransient(new Error("bad model"))).toBe(false);
});

test("the registry is non-empty for selection smoke coverage", () => {
  expect(selectAngles(["*"])).toHaveLength(ANGLES.length);
});
