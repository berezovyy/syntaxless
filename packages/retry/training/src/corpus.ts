/** Seeded, bounded construction corpus with exact numeric policy supervision. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import type { Role } from "../../src/labels.js";
import { prepare } from "../../src/prepare.js";
import { resolveRoles } from "../../src/resolve.js";
import type { Context, RetryIntent } from "../../src/types.js";

export const FAMILIES = [
  "constant",
  "exponential",
  "cap",
  "duration-units",
  "caller-default",
  "full-jitter",
  "equal-jitter",
  "percentage-jitter",
  "no-jitter",
] as const;

export interface Row {
  text: string;
  context: Context;
  intent: RetryIntent | null;
  roles: readonly Role[];
  family: string;
  split: "train" | "validation" | "heldout";
  fingerprint: string;
  trainingEligible?: boolean;
}

export const hash = (bytes: string | Uint8Array) =>
  createHash("sha256").update(bytes).digest("hex");

export const readRows = (path: string): Row[] =>
  readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

const words: Partial<Record<Role, readonly string[]>> = {
  CONSTANT: ["constant", "fixed", "same"],
  EXPONENTIAL: ["exponential"],
  JITTER_FULL: ["full"],
  JITTER_EQUAL: ["equal"],
  JITTER_NONE: ["no", "without", "disable", "off"],
};

function* sample(count: number, seed: number, excluded = new Set<string>()): Generator<Row> {
  if (!Number.isSafeInteger(count) || count < 1 || count > 100000) {
    throw new Error("corpus rows must be in [1,100000]");
  }

  let state = seed;

  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

    return state / 2 ** 32;
  };

  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(random() * xs.length)]!;
  const seen = new Set<string>();
  let emitted = 0;

  for (let attempt = 0; emitted < count; attempt++) {
    if (attempt > count * 100) throw new Error("corpus deduplication exhausted");

    const context: Context = {};
    const family = pick(FAMILIES);
    const positive = random() < 0.67;

    let text = "";
    let roles: Role[] = [];

    const add = (piece: string, role: Role) => {
      if (!piece) return;
      text += (text ? " " : "") + piece;

      roles.push(
        ...prepare(piece, context).lexemes.map((l) => {
          if (l.unitMs !== undefined) return "UNIT";
          if (l.percent) return "PERCENT";
          if (l.multiplier !== undefined) return l.multiplier === 2 ? "DOUBLE" : "TRIPLE";
          if (["DOUBLE", "TRIPLE"].includes(role)) return "GLUE";

          return words[role] && !words[role]!.includes(l.text) ? "GLUE" : role;
        }),
      );
    };

    let prefix = pick([
      "",
      "please",
      "please configure the retry policy to",
      "please arrange the policy to",
      "for the client",
      "for the client request",
      "please configure retries so the application can",
      "configure retries so we can",
      "set up the operation to",
      "for the caller request",
      "for network calls",
      "for failed operations",
      "I would like the application to",
      "I need to",
      "I want to",
      "for the request",
      "for calls",
      "for the operation",
      "configure the operation to",
      "use the retry policy",
      "use a retry strategy",
      "use the retry settings to",
      "use retry settings to",
      "set the retry policy to",
    ]);

    if (random() < 0.35) {
      const preposition = pick(["for", "in", "within"]);

      const owner = pick([
        "the client",
        "the application",
        "the service",
        "our request",
        "the caller",
      ]);

      const noun = pick(["settings", "configuration", "policy", "options", "operation"]);

      prefix = `${preposition} ${owner} ${noun},`;
    }

    // Cover requests with or without a named beneficiary, independently of policy facts.
    if (random() < 0.25) {
      prefix = `${pick(["I", "we"])} ${pick(["want", "need", "would like"])} ${pick([
        "to",
        "the caller to",
        "the application to",
        "the service to",
      ])}`;
    }

    add(prefix, "O");

    const prefixRoles = [...roles];
    const chunks: { text: string; roles: Role[] }[] = [];

    let charMark = text.length;
    let roleMark = roles.length;

    function flush(): void {
      if (roles.length > roleMark) {
        let body = text.slice(charMark).trim();
        const bodyRoles = roles.slice(roleMark);

        while (body.startsWith(",")) {
          if (bodyRoles.shift() !== "GLUE") throw new Error("comma must have a neutral role");
          body = body.slice(1).trimStart();
        }

        chunks.push({ text: body, roles: bodyRoles });
      }

      charMark = text.length;
      roleMark = roles.length;
    }

    const retries = pick([0, 1, 2, 3, 4, 5, 6, 8, 10, 12]);

    const countWord = pick([
      String(retries),
      [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
      ][retries]!,
    ]);

    if (random() < 0.15 && [1, 2].includes(retries)) {
      add("retry", "GLUE");
      add(retries === 1 ? "once" : "twice", "RETRIES");
    } else if (random() < 0.7) {
      add(pick(["retry", "try again", "retry up to"]), "GLUE");
      add(countWord, "RETRIES");
      add(pick(["times", "retries", "more times"]), "GLUE");
    } else {
      add(countWord, "RETRIES");
      add(pick(["retries", "retries after the first attempt"]), "GLUE");
    }

    flush();

    const strategy =
      family === "constant" ? "constant" : pick(["constant", "exponential"] as const);

    const multiplier = pick([1, 1.25, 1.5, 2, 3, 4]);

    const joint =
      strategy === "constant" &&
      family !== "exponential" &&
      family !== "caller-default" &&
      random() < 0.4;

    if (family === "exponential") add("exponential", "EXPONENTIAL");

    if (strategy === "constant" && family !== "exponential" && !joint) {
      add(
        pick([
          "with constant delay",
          "at a fixed interval",
          "with the same delay",
          "at a fixed delay",
          "at constant delay",
        ]),
        "CONSTANT",
      );
    } else if (strategy !== "constant" && random() < 0.3) {
      add("with exponential backoff", "EXPONENTIAL");
    }

    flush();

    const unit = pick([
      { text: "ms", scale: 1 },
      { text: "milliseconds", scale: 1 },
      { text: "seconds", scale: 1000 },
      { text: "minutes", scale: 60000 },
      { text: "hours", scale: 3600000 },
    ]);

    const amount = pick(
      unit.scale === 1
        ? ["0", "1", "5", "10", "25", "100", "250", "500", "1000"]
        : ["0", "0.001", "0.1", "0.25", "1", "1.001", "2", "3", "5", "10"],
    );

    const initialDelayMs = Math.round(Number(amount) * unit.scale);
    const usesDefault = family === "caller-default";

    if (usesDefault) {
      context.defaults = { initialDelayMs };

      add(
        pick([
          "using the caller delay",
          "using the initial delay from the caller",
          "with caller initial delay",
          "",
        ]),
        "GLUE",
      );
    } else {
      if (joint) {
        add(
          pick([
            "with constant delay",
            "at a fixed delay",
            "with the same delay",
            "at constant delay",
          ]),
          "CONSTANT",
        );

        add("of", "GLUE");
      } else {
        add(
          pick([
            ", starting with",
            "with an initial delay of",
            "starting at",
            "initially wait",
            "start with",
            "with initial delay",
            "initial delay",
            "start at",
            "wait initially",
          ]),
          "GLUE",
        );
      }

      add(amount, "INITIAL");
      add(unit.text, "UNIT");
    }

    flush();

    const intent: RetryIntent = {
      retries,
      strategy: family === "exponential" ? "exponential" : strategy,
      initialDelayMs,
      defaults: usesDefault ? ["initialDelayMs"] : [],
    };

    if (intent.strategy === "exponential") {
      intent.multiplier = multiplier;

      if (multiplier === 2 && random() < 0.5) {
        add(", double", "DOUBLE");
        add("the delay", "GLUE");
      } else if (multiplier === 3 && random() < 0.5) {
        add(", triple", "TRIPLE");
        add("the delay", "GLUE");
      } else {
        add(
          pick([
            ", multiply the delay by",
            "with multiplier",
            "with a delay factor of",
            ", multiply by",
            "multiply each delay by",
            "multiply the delay by",
          ]),
          "GLUE",
        );

        add(String(multiplier), "FACTOR");
      }
    }

    flush();

    if (family === "cap" || random() < 0.3) {
      const cap = pick([0, 1, 2, 5, 10, 30, 60, 120]);

      add(
        pick([
          ", cap at",
          "with maximum delay",
          "never exceed",
          ", capped at",
          "cap the delay at",
          "cap it at",
          "cap each delay at",
          "with a cap of",
          "set the delay limit to",
          "maximum wait of",
          "limit the delay to",
          "maximum delay of",
        ]),
        "GLUE",
      );

      add(String(cap), "CAP");
      add("seconds", "UNIT");
      intent.maxDelayMs = cap * 1000;
    }

    flush();

    if (family === "full-jitter" || family === "equal-jitter") {
      intent.jitter = { kind: family === "full-jitter" ? "full" : "equal" };
      add(", with", "GLUE");

      add(
        family === "full-jitter" ? "full jitter" : "equal jitter",
        family === "full-jitter" ? "JITTER_FULL" : "JITTER_EQUAL",
      );
    } else if (family === "percentage-jitter") {
      const percent = pick([0, 10, 20, 25, 30, 50, 75, 100]);

      intent.jitter = { kind: "proportional", fraction: percent / 100 };

      add(
        pick([", with jitter of", ", with proportional jitter of", "apply jitter of", "apply"]),
        "GLUE",
      );

      add(String(percent), "JITTER_PERCENT");
      add(pick(["%", "percent"]), "PERCENT");
      add(pick(["", "jitter"]), "GLUE");
    }

    if (family === "no-jitter") {
      intent.jitter = { kind: "none" };

      add(
        pick(["without jitter", "with no jitter", "disable jitter", "turn jitter off"]),
        "JITTER_NONE",
      );

      if (random() < 0.5) context.defaults = { jitter: { kind: "full" } };
    }

    flush();

    for (let i = chunks.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));

      [chunks[i], chunks[j]] = [chunks[j]!, chunks[i]!];
    }

    text = prefix;
    roles = [...prefixRoles];

    for (const [i, chunk] of chunks.entries()) {
      if (i) add(pick(["", ",", "and", ", and", "then"]), "GLUE");
      text += (text ? " " : "") + chunk.text;
      roles.push(...chunk.roles);
    }

    // Assign the canonical base before negative wording or context changes.
    const fingerprint = hash(
      JSON.stringify([prepare(text, context).lexemes.map((l) => l.text), roles]),
    );

    const bucket = parseInt(fingerprint.slice(0, 8), 16) % 10;
    const split = bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout";

    let gold: RetryIntent | null = intent;
    let name: string = family;

    if (!positive) {
      const mode = pick([0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      name = [
        "question",
        "description",
        "uncertainty",
        "total-attempts",
        "unknown-unit",
        "duplicate-count",
        "fractional-count",
        "fractional-milliseconds",
        "missing-initial",
        "unsupported",
        "conflict",
      ][mode]!;

      if (mode === 0) {
        add("?", "UNSUPPORTED");
      } else if (mode === 1 || mode === 2) {
        if (mode === 2 && random() < 0.5) {
          add(pick(["maybe", "perhaps", "probably"]), "UNSUPPORTED");
        } else {
          const originalText = mode === 1 ? text.slice(prefix.length).trimStart() : text;
          const originalRoles = mode === 1 ? roles.slice(prefixRoles.length) : roles;

          text = "";
          roles = [];

          if (mode === 1) {
            add(
              pick([
                `the previous ${pick(["policy", "settings", "configuration", "request", "setup"])}`,
                `the old ${pick(["policy", "settings", "configuration", "request", "setup"])}`,
                `the saved ${pick(["policy", "settings", "configuration", "request", "setup"])}`,
                "the report",
                "the documentation",
              ]),
              "O",
            );

            add(
              pick(["used", "specified", "was", "had", "said", "says", "recorded", "described"]),
              "UNSUPPORTED",
            );

            if (random() < 0.7) add("to", "GLUE");
          } else {
            add(pick(["maybe", "perhaps", "probably"]), "UNSUPPORTED");
          }

          text += " " + originalText;
          roles.push(...originalRoles);
        }
      } else if (mode === 3) {
        add("including the initial attempt", "UNSUPPORTED");
      } else if (mode === 4) {
        add(", cap at", "GLUE");
        add("2", "CAP");
        add(pick(["frames", "weeks", "months"]), "UNIT");
      } else if (mode === 5) {
        add(", retry", "GLUE");
        add(String(retries + 1), "RETRIES");
        add("times", "GLUE");
      } else if (mode === 6) {
        add(", retry", "GLUE");
        add("2.5", "RETRIES");
        add("times", "GLUE");
      } else if (mode === 7) {
        add(", start with", "GLUE");
        add("0.0001", "INITIAL");
        add("milliseconds", "UNIT");
      } else if (mode === 10) {
        add(",", "GLUE");

        if (intent.strategy === "constant") {
          add(pick(["double", "triple"]), "DOUBLE");
          add("the delay", "GLUE");
        } else {
          add(pick(["at a fixed delay", "with constant delay"]), "CONSTANT");
        }
      } else if (mode === 8 && usesDefault) {
        delete context.defaults;
      } else {
        add(
          pick([
            "with unspecified jitter",
            "retry forever",
            "if an error occurs maybe",
            "increase the timeout instead",
            "using a custom formula",
          ]),
          "UNSUPPORTED",
        );
      }

      gold = null;
    }

    const prepared = prepare(text, context);
    let decoded: RetryIntent | null = null;

    try {
      decoded = resolveRoles(
        prepared,
        roles.map((role) => ({ role, score: 1 })),
        context,
      ).intent;
    } catch {}

    if (!isDeepStrictEqual(decoded, gold)) {
      throw new Error(
        `invalid supervision: ${text}: ${JSON.stringify(decoded)} vs ${JSON.stringify(gold)}`,
      );
    }

    if (excluded.has(hash(text.trim().toLowerCase())) || seen.has(text)) continue;
    seen.add(text);
    yield { text, context, intent: gold, roles, family: name, split, fingerprint };
    emitted++;
  }
}

export function generate(count = 8000, seed = 20261011): Row[] {
  return [...sample(count, seed)];
}

export function writeCorpus(path: string, count = 8000): Row[] {
  const rows: Row[] = [];

  const excluded = reservedHashes(
    resolve(import.meta.dir, "../../evaluation"),
    resolve(import.meta.dir, "../../evaluation/consumed"),
  );

  function* records(): Generator<Row> {
    for (const row of sample(count, 20261011, excluded)) {
      rows.push(row);
      yield row;
    }
  }

  const written = writeJsonl(path, records());

  writeFileSync(
    path + ".manifest.json",
    JSON.stringify(
      {
        rows: rows.length,
        sha256: written.sha256,
        generatorSha256: hash(readFileSync(import.meta.filename)),
        seed: 20261011,
        trainingEligible: true,
        excludedReservedTexts: excluded.size,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  return rows;
}
