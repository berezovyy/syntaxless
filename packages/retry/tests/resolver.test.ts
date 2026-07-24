import { expect, test } from "bun:test";

import { compile } from "../src/compile.js";
import { explain } from "../src/explain.js";
import type { Role } from "../src/labels.js";
import { prepare } from "../src/prepare.js";
import { resolveRoles } from "../src/resolve.js";
import type { Context } from "../src/types.js";

function resolve(text: string, roles: Role[], context: Context = {}) {
  return resolveRoles(
    prepare(text, context),
    roles.map((role) => ({ role, score: 1 })),
    context,
  ).intent;
}

const reference = "retry three times, double the delay, cap at one minute";

const referenceRoles: Role[] = [
  "GLUE",
  "RETRIES",
  "GLUE",
  "GLUE",
  "DOUBLE",
  "GLUE",
  "GLUE",
  "GLUE",
  "GLUE",
  "GLUE",
  "CAP",
  "UNIT",
];

test(
  "reference requires initial delay, " + "preserves caller defaults and produces exact delays",
  () => {
    expect(() => resolve(reference, referenceRoles)).toThrow("initial delay");

    const intent = resolve(reference, referenceRoles, {
      defaults: { initialDelayMs: 250 },
    });

    expect(intent).toEqual({
      retries: 3,
      multiplier: 2,
      maxDelayMs: 60000,
      strategy: "exponential",
      initialDelayMs: 250,
      defaults: ["initialDelayMs"],
    });

    expect(compile(intent)).toEqual({ supported: true, delaysMs: [250, 500, 1000] });
    expect(explain(intent)).toContain("250 ms (caller default)");

    expect(
      resolve("three retries at constant delay of 1.001 seconds", [
        "RETRIES",
        "GLUE",
        "GLUE",
        "CONSTANT",
        "GLUE",
        "GLUE",
        "INITIAL",
        "UNIT",
      ]),
    ).toMatchObject({
      retries: 3,
      initialDelayMs: 1001,
      strategy: "constant",
      defaults: [],
    });

    expect(
      resolve("retry twenty-one times at constant delay of 0.1 seconds", [
        "GLUE",
        "RETRIES",
        "GLUE",
        "GLUE",
        "CONSTANT",
        "GLUE",
        "GLUE",
        "INITIAL",
        "UNIT",
      ]),
    ).toMatchObject({ retries: 21, initialDelayMs: 100 });
  },
);

test("lost numeric facts, unsupported precision, conflicts and unspecified jitter refuse", () => {
  expect(() =>
    resolve(
      reference,
      referenceRoles.map((role) => (role === "DOUBLE" ? "GLUE" : role)),
      { defaults: { initialDelayMs: 250 } },
    ),
  ).toThrow("preserve");

  expect(() =>
    resolve(
      reference,
      referenceRoles.map((r, i) => (i === 1 ? "O" : r)),
      { defaults: { initialDelayMs: 1 } },
    ),
  ).toThrow("preserve");

  expect(() =>
    resolve("retry 3 times constant delay 0.0001 seconds", [
      "GLUE",
      "RETRIES",
      "GLUE",
      "CONSTANT",
      "GLUE",
      "INITIAL",
      "UNIT",
    ]),
  ).toThrow("whole numbers");

  expect(() =>
    resolve("retry 3 times constant delay 1 seconds double", [
      "GLUE",
      "RETRIES",
      "GLUE",
      "CONSTANT",
      "GLUE",
      "INITIAL",
      "UNIT",
      "DOUBLE",
    ]),
  ).toThrow("constant delays");

  expect(() =>
    resolve("retry 3 times constant delay 1 seconds jitter 150%", [
      "GLUE",
      "RETRIES",
      "GLUE",
      "CONSTANT",
      "GLUE",
      "INITIAL",
      "UNIT",
      "GLUE",
      "JITTER_PERCENT",
      "PERCENT",
    ]),
  ).toThrow("zero and one");

  expect(() =>
    resolve("hello", ["O"], {
      defaults: { retries: 3, strategy: "constant", initialDelayMs: 1 },
    }),
  ).toThrow("explicit retry policy fact");

  expect(() =>
    resolve("retry 3 attempts", ["GLUE", "RETRIES", "UNSUPPORTED"], {
      defaults: { strategy: "constant", initialDelayMs: 1 },
    }),
  ).toThrow("non-actionable");
});

test("explicit no jitter overrides caller jitter without requiring a seed", () => {
  const intent = resolve(
    "retry twice with constant delay of 25 ms without jitter",
    [
      "GLUE",
      "RETRIES",
      "GLUE",
      "CONSTANT",
      "GLUE",
      "GLUE",
      "INITIAL",
      "UNIT",
      "JITTER_NONE",
      "GLUE",
    ],
    { defaults: { jitter: { kind: "full" } } },
  );

  expect(intent.jitter).toEqual({ kind: "none" });
  expect(intent.defaults).toEqual([]);
  expect(compile(intent)).toEqual({ supported: true, delaysMs: [25, 25] });
  expect(explain(intent)).toContain("No jitter is applied.");

  expect(() =>
    resolve("retry twice constant delay 25 ms full jitter without jitter", [
      "GLUE",
      "RETRIES",
      "CONSTANT",
      "GLUE",
      "INITIAL",
      "UNIT",
      "JITTER_FULL",
      "GLUE",
      "JITTER_NONE",
      "GLUE",
    ]),
  ).toThrow("conflicting jitter");
});
