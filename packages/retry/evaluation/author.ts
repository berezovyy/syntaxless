/** Evaluation-only writer with independent mathematical schedule targets. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

import type { Context, RetryIntent } from "../src/types.js";
import type { Row } from "../training/src/corpus.js";
import { SEED } from "./fixtures.js";

export interface Acceptance extends Row {
  critical: boolean;
  compileOptions: { seed?: number };
  compileTarget:
    { supported: true; delaysMs: readonly number[] } | { supported: false; code: string } | null;
}

const families = [
  "constant",
  "exponential",
  "cap",
  "duration-units",
  "caller-default",
  "full-jitter",
  "equal-jitter",
  "percentage-jitter",
  "no-jitter",
];

const hash = (b: string | Uint8Array) => createHash("sha256").update(b).digest("hex");

function targets(intent: RetryIntent, seed: number): Acceptance["compileTarget"] {
  let state = seed;

  const draw = () => {
    state = (state * 1664525 + 1013904223) >>> 0;

    return BigInt(state);
  };

  const out: number[] = [];
  const factor = intent.multiplier ?? 1;

  // Evaluation uses small exact factors (1, 1.5, 2, 3); compute powers from the initial delay.
  for (let i = 0; i < intent.retries; i++) {
    let delay = Math.ceil(
      intent.initialDelayMs * (intent.strategy === "constant" ? 1 : factor ** i),
    );

    if (intent.maxDelayMs !== undefined) delay = Math.min(delay, intent.maxDelayMs);
    if (!Number.isSafeInteger(delay)) return { supported: false, code: "overflow" };

    if (intent.jitter && intent.jitter.kind !== "none") {
      const low =
        intent.jitter.kind === "full"
          ? 0
          : intent.jitter.kind === "equal"
            ? Math.ceil(delay / 2)
            : Math.ceil(delay * (1 - intent.jitter.fraction));

      let high =
        intent.jitter.kind === "proportional"
          ? Math.floor(delay * (1 + intent.jitter.fraction))
          : delay;

      if (intent.maxDelayMs !== undefined) high = Math.min(high, intent.maxDelayMs);
      const sample = draw() * 2097152n + draw() / 2048n;

      delay = low + Number((sample * BigInt(high - low + 1)) / 9007199254740992n);
    }

    out.push(delay);
  }

  return { supported: true, delaysMs: out };
}

export function author(revision = 1): Acceptance[] {
  if (![1, 2, 3, 4, 5, 6, 7].includes(revision)) throw new Error("unsupported authoring revision");
  const rows: Acceptance[] = [];

  function add(
    text: string,
    intent: RetryIntent | null,
    context: Context,
    family: string,
    critical = false,
    expected?: Acceptance["compileTarget"],
  ) {
    rows.push({
      text,
      intent,
      context,
      family,
      critical,
      compileOptions: { seed: SEED },
      compileTarget: expected ?? (intent ? targets(intent, SEED) : null),
      roles: [],
      split: "heldout",
      fingerprint: hash(text),
      trainingEligible: false,
    });
  }

  const prefixes =
    revision === 1
      ? [
          "Please configure retries so we",
          "For the client request,",
          "I would like to",
          "For failed calls,",
          "Set the request policy to",
          "Please arrange to",
        ]
      : revision === 2
        ? [
            "For requests,",
            "I need to",
            "Please configure the client to",
            "For the operation,",
            "Use a retry policy to",
            "Please set up the operation to",
          ]
        : revision === 3
          ? [
              "For the application request,",
              "Please set retries to",
              "I want the client to",
              "For service calls,",
              "Configure the operation to",
              "For each request,",
            ]
          : revision === 4
            ? [
                "For the next outbound call,",
                "Please configure the operation so it can",
                "In the client settings,",
                "I need the request to",
                "For a failed service call,",
                "Set the application to",
              ]
            : revision === 5
              ? [
                  "For outbound operations,",
                  "Please make the caller",
                  "In our request policy,",
                  "I want retries to",
                  "For the request configuration,",
                  "The client should",
                ]
              : revision === 6
                ? [
                    "For our outgoing requests,",
                    "In the service options,",
                    "For repeated calls,",
                    "Please let the application",
                    "The request should",
                    "Set up retry settings to",
                  ]
                : [
                    "For our service calls,",
                    "Please configure the request to",
                    "In the operation settings,",
                    "I need the application to",
                    "The service should",
                    "Set our retry policy to",
                  ];

  for (const [f, family] of families
    .filter((family) => revision >= 4 || family !== "no-jitter")
    .entries()) {
    for (let i = 0; i < 75; i++) {
      const retries = 1 + (i % 7);
      const initial = [125, 3125, 8125, 14125, 125, 30125, 45125][revision - 1]! + 25 * i + f * 2;
      const ctx: Context = {};

      const exponential = family === "exponential" || (family === "cap" && i % 2 === 0);

      const intent: RetryIntent = {
        retries,
        strategy: exponential ? "exponential" : "constant",
        initialDelayMs: initial,
        defaults: [],
      };

      let text = `${prefixes[i % prefixes.length]} retry ${retries} times`;

      if (exponential) {
        intent.multiplier = i % 2 === 0 ? 2 : 1.5;
        text += ` with exponential backoff, multiply each delay by ${intent.multiplier}`;
      } else {
        text += ` at a constant delay`;
      }

      if (family === "caller-default") {
        ctx.defaults = { initialDelayMs: initial };
        intent.defaults = ["initialDelayMs"];
        text += ", using the caller initial delay";
      } else if (family === "duration-units") {
        // Exact thousandths of a second are integral milliseconds.
        const amount = (initial / 1000).toFixed(3);

        text += `, starting at ${amount} seconds`;
      } else {
        text += `, with initial delay ${initial} ms`;
      }

      if (family === "cap") {
        intent.maxDelayMs = initial + 200;
        text += `, cap the delay at ${intent.maxDelayMs} milliseconds`;
      }

      if (family === "full-jitter" || family === "equal-jitter") {
        const kind = family === "full-jitter" ? "full" : "equal";

        intent.jitter = { kind };
        text += `, with ${kind} jitter`;
      }

      if (family === "percentage-jitter") {
        const percent = [10, 20, 25, 50, 75][i % 5]!;

        intent.jitter = { kind: "proportional", fraction: percent / 100 };
        text += `, apply ${percent}% jitter`;
      }

      if (family === "no-jitter") {
        intent.jitter = { kind: "none" };
        text += i % 2 ? ", without jitter" : ", with no jitter";
        ctx.defaults = { jitter: { kind: i % 2 ? "full" : "equal" } };
      }

      text += ".";
      add(text, intent, ctx, family);
    }
  }

  for (let i = 0; i < 630; i++) {
    const initial = [777, 5777, 9777, 15777, 24777, 33777, 48777][revision - 1]! + i;
    const retries = 1 + (i % 8);
    const core = `retry ${retries} times with constant delay, start at ${initial} milliseconds`;
    const mode = i % 9;

    const text = [
      `Can we ${core}?`,
      revision >= 6
        ? `The saved settings specified to ${core}.`
        : `The previous configuration said to ${core}.`,
      `Perhaps ${core}.`,
      `Use ${retries} attempts in total, with a fixed ${initial} ms delay.`,
      `Retry ${retries} times with constant delay, initially wait ${initial} frames.`,
      `${core}, then retry ${retries + 1} times.`,
      `Retry ${retries + 0.5} times with constant delay, start at ${initial} milliseconds.`,
      `Retry ${retries} times with constant delay and start at 0.000${initial} milliseconds.`,
      `${core} with unspecified jitter.`,
    ][mode]!;

    add(
      text,
      null,
      {},
      [
        "question",
        "description",
        "uncertainty",
        "total-attempts",
        "unknown-unit",
        "duplicate-count",
        "fractional-count",
        "fractional-milliseconds",
        "unspecified-jitter",
      ][mode]!,
    );
  }

  if (revision === 1) {
    const reference = "retry three times, double the delay, cap at one minute";

    add(reference, null, {}, "missing-initial", true);

    add(
      reference,
      {
        retries: 3,
        strategy: "exponential",
        initialDelayMs: 250,
        multiplier: 2,
        maxDelayMs: 60000,
        defaults: ["initialDelayMs"],
      },
      { defaults: { initialDelayMs: 250 } },
      "caller-default",
      true,
      { supported: true, delaysMs: [250, 500, 1000] },
    );

    add(
      "retry five times with exponential backoff, start at 1 ms and multiply by 1.5",
      {
        retries: 5,
        strategy: "exponential",
        initialDelayMs: 1,
        multiplier: 1.5,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: true, delaysMs: [1, 2, 3, 4, 6] },
    );

    add(
      "retry three times with constant delay, initially wait 1.001 seconds",
      { retries: 3, strategy: "constant", initialDelayMs: 1001, defaults: [] },
      {},
      "duration-units",
      true,
      { supported: true, delaysMs: [1001, 1001, 1001] },
    );

    add(
      "retry three times, double the delay, initially wait 2 seconds, cap at one second",
      {
        retries: 3,
        strategy: "exponential",
        initialDelayMs: 2000,
        multiplier: 2,
        maxDelayMs: 1000,
        defaults: [],
      },
      {},
      "cap",
      true,
      { supported: true, delaysMs: [1000, 1000, 1000] },
    );

    add(
      "retry twice with constant delay of ten milliseconds",
      { retries: 2, strategy: "constant", initialDelayMs: 10, defaults: [] },
      {},
      "constant",
      true,
      { supported: true, delaysMs: [10, 10] },
    );

    add(
      "retry zero times with constant delay, initially wait 1 millisecond",
      { retries: 0, strategy: "constant", initialDelayMs: 1, defaults: [] },
      {},
      "constant",
      true,
      { supported: true, delaysMs: [] },
    );

    add(
      "retry two times with exponential backoff, " +
        "start with 9007199254740991 milliseconds and multiply by 2",
      {
        retries: 2,
        strategy: "exponential",
        initialDelayMs: 9007199254740991,
        multiplier: 2,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: false, code: "overflow" },
    );

    add(
      "retry three times with constant delay of 0.0001 ms",
      null,
      {},
      "fractional-milliseconds",
      true,
    );

    add(
      "retry three times with constant delay of 9007199254740991.1 ms",
      null,
      {},
      "unsafe-duration",
      true,
    );

    add("retry three times with constant delay of 1 second?", null, {}, "question", true);

    add(
      "retry three times, double and triple the delay, initially wait 1 second",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "try three attempts including the first, with constant delay of 1 second",
      null,
      {},
      "total-attempts",
      true,
    );
  } else if (revision === 2) {
    const reference = "please retry four times, double the delay, cap the delay at thirty seconds";

    add(reference, null, {}, "missing-initial", true);

    add(
      reference,
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 500,
        multiplier: 2,
        maxDelayMs: 30000,
        defaults: ["initialDelayMs"],
      },
      { defaults: { initialDelayMs: 500 } },
      "caller-default",
      true,
      { supported: true, delaysMs: [500, 1000, 2000, 4000] },
    );

    add(
      "retry six times, multiply the delay by 1.5, start with 2 milliseconds",
      {
        retries: 6,
        strategy: "exponential",
        initialDelayMs: 2,
        multiplier: 1.5,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: true, delaysMs: [2, 3, 5, 7, 11, 16] },
    );

    add(
      "start at 0.025 seconds, constant delay, retry four times",
      { retries: 4, strategy: "constant", initialDelayMs: 25, defaults: [] },
      {},
      "duration-units",
      true,
      { supported: true, delaysMs: [25, 25, 25, 25] },
    );

    add(
      "cap the delay at 250 ms, initially wait 2 seconds, retry four times and double the delay",
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 2000,
        multiplier: 2,
        maxDelayMs: 250,
        defaults: [],
      },
      {},
      "cap",
      true,
      { supported: true, delaysMs: [250, 250, 250, 250] },
    );

    add(
      "retry once with constant delay, initially wait twenty milliseconds",
      { retries: 1, strategy: "constant", initialDelayMs: 20, defaults: [] },
      {},
      "constant",
      true,
      { supported: true, delaysMs: [20] },
    );

    add(
      "with constant delay, start at ten milliseconds, retry zero times",
      { retries: 0, strategy: "constant", initialDelayMs: 10, defaults: [] },
      {},
      "constant",
      true,
      { supported: true, delaysMs: [] },
    );

    add(
      "start at 9007199254740991 ms, multiply by 3, retry two times",
      {
        retries: 2,
        strategy: "exponential",
        initialDelayMs: 9007199254740991,
        multiplier: 3,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: false, code: "overflow" },
    );

    add(
      "retry two times with constant delay of 0.0005 seconds",
      null,
      {},
      "fractional-milliseconds",
      true,
    );

    add(
      "retry two times with constant delay of 9007199254740992 milliseconds",
      null,
      {},
      "unsafe-duration",
      true,
    );

    add("try again three times at fixed interval starting with 25 ms?", null, {}, "question", true);

    add(
      "retry four times with constant delay of 25 ms, triple the delay",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry two times including the initial attempt at fixed delay of 20 ms",
      null,
      {},
      "total-attempts",
      true,
    );
  }

  if (revision === 3) {
    add(
      "please retry five times, triple the delay and cap it at 20 seconds",
      null,
      {},
      "missing-initial",
      true,
    );

    add(
      "please retry five times, triple the delay and cap it at 20 seconds",
      {
        retries: 5,
        strategy: "exponential",
        initialDelayMs: 200,
        multiplier: 3,
        maxDelayMs: 20000,
        defaults: ["initialDelayMs"],
      },
      { defaults: { initialDelayMs: 200 } },
      "caller-default",
      true,
      { supported: true, delaysMs: [200, 600, 1800, 5400, 16200] },
    );

    add(
      "retry four times and multiply by 1.5, with initial delay 3 ms",
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 3,
        multiplier: 1.5,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: true, delaysMs: [3, 5, 7, 11] },
    );

    add(
      "with initial delay 0.075 seconds, retry twice at constant delay",
      { retries: 2, strategy: "constant", initialDelayMs: 75, defaults: [] },
      {},
      "duration-units",
      true,
      { supported: true, delaysMs: [75, 75] },
    );

    add(
      "cap at 500 milliseconds and retry five times, start with 3 seconds and double the delay",
      {
        retries: 5,
        strategy: "exponential",
        initialDelayMs: 3000,
        multiplier: 2,
        maxDelayMs: 500,
        defaults: [],
      },
      {},
      "cap",
      true,
      { supported: true, delaysMs: [500, 500, 500, 500, 500] },
    );

    add(
      "retry three times at constant delay of 50 ms and apply 0% jitter",
      {
        retries: 3,
        strategy: "constant",
        initialDelayMs: 50,
        jitter: { kind: "proportional", fraction: 0 },
        defaults: [],
      },
      {},
      "percentage-jitter",
      true,
      { supported: true, delaysMs: [50, 50, 50] },
    );

    add(
      "retry zero times with constant delay of 25 milliseconds",
      { retries: 0, strategy: "constant", initialDelayMs: 25, defaults: [] },
      {},
      "constant",
      true,
      { supported: true, delaysMs: [] },
    );

    add(
      "retry two times and multiply the delay by 4, start with one month",
      null,
      {},
      "unsupported",
      true,
    );

    add(
      "retry two times and multiply the delay by 4, start with 9007199254740991 ms",
      {
        retries: 2,
        strategy: "exponential",
        initialDelayMs: 9007199254740991,
        multiplier: 4,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: false, code: "overflow" },
    );

    add(
      "retry four times at fixed delay of 0.00001 seconds",
      null,
      {},
      "fractional-milliseconds",
      true,
    );

    add("retry once at fixed delay of 9007199254740993 ms", null, {}, "unsafe-duration", true);

    add(
      "could I retry four times at constant delay starting with 5 milliseconds?",
      null,
      {},
      "question",
      true,
    );

    add("retry twice at a fixed delay of 25 ms and double the delay", null, {}, "conflict", true);

    add(
      "make four attempts overall, including the initial one, at constant delay of 25 ms",
      null,
      {},
      "total-attempts",
      true,
    );
  }

  if (revision === 4) {
    const reference = "retry six times, double the delay and cap each delay at 12 seconds";

    add(reference, null, {}, "missing-initial", true);

    add(
      reference,
      {
        retries: 6,
        strategy: "exponential",
        initialDelayMs: 300,
        multiplier: 2,
        maxDelayMs: 12000,
        defaults: ["initialDelayMs"],
      },
      { defaults: { initialDelayMs: 300 } },
      "caller-default",
      true,
      { supported: true, delaysMs: [300, 600, 1200, 2400, 4800, 9600] },
    );

    add(
      "retry three times with a fixed delay of 40 milliseconds and triple the delay",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry four times at a constant delay of 80 ms without jitter",
      {
        retries: 4,
        strategy: "constant",
        initialDelayMs: 80,
        jitter: { kind: "none" },
        defaults: [],
      },
      { defaults: { jitter: { kind: "full" } } },
      "no-jitter",
      true,
      { supported: true, delaysMs: [80, 80, 80, 80] },
    );

    add(
      "retry three times with full jitter, no jitter, start with 60 ms at constant delay",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry three times, multiply the delay by 1.5 and start at 5 milliseconds",
      {
        retries: 3,
        strategy: "exponential",
        initialDelayMs: 5,
        multiplier: 1.5,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: true, delaysMs: [5, 8, 12] },
    );

    add(
      "retry twice with a fixed delay of 0.045 seconds",
      { retries: 2, strategy: "constant", initialDelayMs: 45, defaults: [] },
      {},
      "duration-units",
      true,
      { supported: true, delaysMs: [45, 45] },
    );

    add(
      "retry four times, initially wait 4 seconds, triple the delay and cap at 750 ms",
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 4000,
        multiplier: 3,
        maxDelayMs: 750,
        defaults: [],
      },
      {},
      "cap",
      true,
      { supported: true, delaysMs: [750, 750, 750, 750] },
    );

    add(
      "retry once at constant delay of 9007199254740994 milliseconds",
      null,
      {},
      "unsafe-duration",
      true,
    );

    add(
      "retry three times at fixed delay of 0.0002 milliseconds",
      null,
      {},
      "fractional-milliseconds",
      true,
    );

    add("can we retry twice with constant delay of 4 seconds?", null, {}, "question", true);

    add(
      "use five attempts overall, including the first, with a fixed delay of 70 ms",
      null,
      {},
      "total-attempts",
      true,
    );
  }

  if (revision === 5) {
    const reference = "retry four times, triple the delay with a cap of 15 seconds";

    add(reference, null, {}, "missing-initial", true);

    add(
      reference,
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 400,
        multiplier: 3,
        maxDelayMs: 15000,
        defaults: ["initialDelayMs"],
      },
      { defaults: { initialDelayMs: 400 } },
      "caller-default",
      true,
      { supported: true, delaysMs: [400, 1200, 3600, 10800] },
    );

    add(
      "retry four times at the same delay of 65 ms, then double the delay",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry twice at a constant delay, initially wait 90 milliseconds and disable jitter",
      {
        retries: 2,
        strategy: "constant",
        initialDelayMs: 90,
        jitter: { kind: "none" },
        defaults: [],
      },
      { defaults: { jitter: { kind: "equal" } } },
      "no-jitter",
      true,
      { supported: true, delaysMs: [90, 90] },
    );

    add(
      "retry twice with equal jitter and without jitter, start with 55 ms at fixed delay",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry four times, initially wait 7 ms and multiply each delay by 1.5",
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 7,
        multiplier: 1.5,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: true, delaysMs: [7, 11, 16, 24] },
    );

    add(
      "retry once with the same delay of 0.035 seconds",
      { retries: 1, strategy: "constant", initialDelayMs: 35, defaults: [] },
      {},
      "duration-units",
      true,
      { supported: true, delaysMs: [35] },
    );

    add(
      "retry twice, double the delay, start at 5 seconds and never exceed 600 milliseconds",
      {
        retries: 2,
        strategy: "exponential",
        initialDelayMs: 5000,
        multiplier: 2,
        maxDelayMs: 600,
        defaults: [],
      },
      {},
      "cap",
      true,
      { supported: true, delaysMs: [600, 600] },
    );

    add("retry twice with fixed delay of 9007199254740995 ms", null, {}, "unsafe-duration", true);

    add(
      "retry four times at a fixed delay of 0.0003 ms",
      null,
      {},
      "fractional-milliseconds",
      true,
    );

    add("may we retry three times with constant delay of 3 seconds?", null, {}, "question", true);
    add("use six attempts in total at a fixed delay of 45 ms", null, {}, "total-attempts", true);
  }

  if (revision >= 6) {
    const reference = "retry seven times, double the delay, with maximum delay of 10 seconds";

    add(reference, null, {}, "missing-initial", true);

    add(
      reference,
      {
        retries: 7,
        strategy: "exponential",
        initialDelayMs: 125,
        multiplier: 2,
        maxDelayMs: 10000,
        defaults: ["initialDelayMs"],
      },
      { defaults: { initialDelayMs: 125 } },
      "caller-default",
      true,
      { supported: true, delaysMs: [125, 250, 500, 1000, 2000, 4000, 8000] },
    );

    add(
      "retry twice with constant delay of 35 milliseconds, triple the delay",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry three times at a fixed delay of 110 ms and turn jitter off",
      {
        retries: 3,
        strategy: "constant",
        initialDelayMs: 110,
        jitter: { kind: "none" },
        defaults: [],
      },
      { defaults: { jitter: { kind: "full" } } },
      "no-jitter",
      true,
      { supported: true, delaysMs: [110, 110, 110] },
    );

    add(
      "retry twice at constant delay of 35 ms with full jitter and turn jitter off",
      null,
      {},
      "conflict",
      true,
    );

    add(
      "retry four times with exponential backoff, start at 4 milliseconds and multiply by 1.5",
      {
        retries: 4,
        strategy: "exponential",
        initialDelayMs: 4,
        multiplier: 1.5,
        defaults: [],
      },
      {},
      "exponential",
      true,
      { supported: true, delaysMs: [4, 6, 9, 14] },
    );

    add(
      "retry once at a fixed delay of 0.055 seconds",
      { retries: 1, strategy: "constant", initialDelayMs: 55, defaults: [] },
      {},
      "duration-units",
      true,
      { supported: true, delaysMs: [55] },
    );

    add(
      "retry three times, double the delay, start at 6 seconds and cap each delay at 800 ms",
      {
        retries: 3,
        strategy: "exponential",
        initialDelayMs: 6000,
        multiplier: 2,
        maxDelayMs: 800,
        defaults: [],
      },
      {},
      "cap",
      true,
      { supported: true, delaysMs: [800, 800, 800] },
    );

    add(
      "retry twice at the same delay of 9007199254740996 milliseconds",
      null,
      {},
      "unsafe-duration",
      true,
    );

    add(
      "retry four times with fixed delay of 0.0004 ms",
      null,
      {},
      "fractional-milliseconds",
      true,
    );

    add(
      "can the client retry three times at fixed delay of 6 seconds?",
      null,
      {},
      "question",
      true,
    );

    add(
      "use seven attempts overall, including the first, at a constant delay of 30 ms",
      null,
      {},
      "total-attempts",
      true,
    );
  }

  if (revision === 7) {
    for (const row of rows.filter((row) => row.critical)) {
      row.text = "For the service request, " + row.text;
    }
  }

  return rows;
}

if (import.meta.main) {
  const [scopePath, out, revisionText] = process.argv.slice(2);

  if (!scopePath || !out) throw new Error("usage: author <scope-freeze.json> <new-reserved.jsonl>");

  const scopeBytes = readFileSync(scopePath);
  const scope = JSON.parse(scopeBytes.toString());
  const rows = author(Number(revisionText ?? 1));

  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

  writeFileSync(out, body, { flag: "wx" });

  writeFileSync(
    out + ".meta.json",
    JSON.stringify(
      {
        sha256: hash(body),
        candidateSha256: scope.artifactFileSha256,
        scopeFreezeSha256: hash(scopeBytes),
        createdAt: new Date().toISOString(),
        trainingEligible: false,
        rows: rows.length,
        authorship:
          "AI-authored by the same implementation assistant " +
          "through a separate evaluation-only writer",
        independenceLimits:
          "No production/training imports. " +
          "Independent mathematical schedule targets and hand-checked critical outputs. " +
          "Bounded constructions and numeric variation are not independent human evidence " +
          "or population accuracy.",
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  console.log({ rows: rows.length, sha256: hash(body) });
}
