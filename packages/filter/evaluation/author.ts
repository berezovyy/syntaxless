import { createHash } from "node:crypto";
/** Separate AI authoring: no parser, model or training renderer imports. */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import type { Context, FilterIntent, Scalar, Truth } from "../src/types.js";
import { WITNESSES } from "./fixtures.js";

const hash = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const revision = Number(process.argv[4] ?? 1);

if (![1, 2].includes(revision)) throw new Error("unsupported authoring revision");

export interface Acceptance {
  text: string;
  context: Context;
  intent: FilterIntent | null;
  family: string;
  roles: [];
  split: "heldout";
  fingerprint: string;
  trainingEligible: false;
  critical: boolean;
  compileTarget?: readonly Truth[];
}

function truth(node: FilterIntent, row: Record<string, unknown>, context: Context): Truth {
  if (node.kind === "not") {
    const t = truth(node.child, row, context);

    return t === null ? null : !t;
  }

  if (node.kind === "and" || node.kind === "or") {
    const children = node.children.map((child) => truth(child, row, context));

    if (node.kind === "and") {
      return children.includes(false) ? false : children.includes(null) ? null : true;
    }

    return children.includes(true) ? true : children.includes(null) ? null : false;
  }

  const field = context.fields.find((field) => field.id === node.field)!;
  const actual = row[field.column as string];

  if (node.kind === "null") return (actual === null || actual === undefined) === node.isNull;
  if (actual === null || actual === undefined) return null;

  const order = (a: unknown, b: unknown) =>
    typeof a === "string" && typeof b === "string"
      ? Buffer.compare(Buffer.from(a), Buffer.from(b))
      : (a as number) - (b as number);

  if (node.kind === "in") return node.values.includes(actual as Scalar);

  if (node.kind === "range") {
    return (
      (order(actual, node.lower.value) > 0 ||
        (node.lower.inclusive && order(actual, node.lower.value) === 0)) &&
      (order(actual, node.upper.value) < 0 ||
        (node.upper.inclusive && order(actual, node.upper.value) === 0))
    );
  }

  const c = order(actual, node.value);

  return node.op === "eq"
    ? actual === node.value
    : node.op === "ne"
      ? actual !== node.value
      : node.op === "lt"
        ? c < 0
        : node.op === "le"
          ? c <= 0
          : node.op === "gt"
            ? c > 0
            : c >= 0;
}

const positiveFamilies = [
  "eq",
  "ne",
  "lt",
  "le",
  "gt",
  "ge",
  "in",
  "range",
  "null",
  "and",
  "or",
  "exclusion",
  "not",
  "group",
];

const negativeFamilies = [
  "question",
  "description",
  "quotation",
  "negation",
  "historical",
  "ambiguity",
  "approximate",
  "unsupported-pattern",
  "unsupported-unit",
  "incomplete",
  "unknown-field",
  "inverted",
  "mixed-grouping",
  "unsafe-number",
];

function author(): Acceptance[] {
  const rows: Acceptance[] = [];
  const seen = new Set<string>();

  let next = 0;
  let state = 196732 + revision;

  const pick = <T>(items: readonly T[]): T => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

    return items[Math.floor((state / 2 ** 32) * items.length)]!;
  };

  function context(): Context {
    const id = next++;

    return {
      fields: [
        {
          id: `reservedAmount${id}`,
          aliases: [
            pick(["account value", "total credits", "paid amount", "balance figure"]) + " " + id,
          ],
          type: "number",
          column: "amount",
        },
        {
          id: `reservedLabel${id}`,
          aliases: [
            pick(["plan name", "state label", "region code", "account category"]) + " " + id,
          ],
          type: "string",
          column: "label",
        },
        {
          id: `reservedFlag${id}`,
          aliases: [
            pick(["trial member", "verified entry", "enabled flag", "approved account"]) + " " + id,
          ],
          type: "boolean",
          column: "flag",
        },
      ],
    };
  }

  function atom(family: string, ctx: Context): { text: string; intent: FilterIntent } {
    const numeric = !["eq", "ne", "in", "null"].includes(family);
    const field = numeric ? ctx.fields[0]! : pick(ctx.fields);
    const name = field.aliases![0]!;
    const n = pick([-20.25, -2.75, 0, 1, 18, 99.5, 100, 100.25, 101, 250, 1000]);

    const value: Scalar =
      field.type === "number"
        ? n
        : field.type === "boolean"
          ? pick([true, false])
          : pick(["active", "north", "O'Reilly", "Silver plan", "", `region-${next}`]);

    if (family === "null") {
      const isNull = pick([true, false]);

      return {
        text: `${name} ${isNull ? "is missing" : "is not null"}`,
        intent: { kind: "null", field: field.id, isNull },
      };
    }

    if (family === "in") {
      const values = [
        value,
        field.type === "number" ? n + 7 : field.type === "boolean" ? !value : "active",
      ] as Scalar[];

      return {
        text: `${name} is one of ${values.map((v) => JSON.stringify(v)).join(", ")}`,
        intent: { kind: "in", field: field.id, values },
      };
    }

    if (family === "range") {
      const upper = n + pick([1, 3.5, 25, 100]);

      return {
        text: `${name} from ${n} to ${upper}`,
        intent: {
          kind: "range",
          field: field.id,
          lower: { value: n, inclusive: true },
          upper: { value: upper, inclusive: true },
        },
      };
    }

    const op = family as "eq" | "ne" | "lt" | "le" | "gt" | "ge";

    const word = {
      eq: "is equal to",
      ne: "does not equal",
      lt: "below",
      le: "at most",
      gt: "above",
      ge: "at least",
    }[op];

    return {
      text: `${name} ${word} ${JSON.stringify(value)}`,
      intent: { kind: "compare", field: field.id, op, value },
    };
  }

  function positive(family: string, ctx: Context): { text: string; intent: FilterIntent } {
    if (!["and", "or", "exclusion", "not", "group"].includes(family)) return atom(family, ctx);

    const a = atom("gt", ctx);
    const b = atom("eq", ctx);
    const c = atom("null", ctx);

    if (family === "not") {
      return { text: `not ${a.text}`, intent: { kind: "not", child: a.intent } };
    }

    if (family === "group") {
      return {
        text: `( ${a.text} or ${b.text} ) and ${c.text}`,
        intent: {
          kind: "and",
          children: [{ kind: "or", children: [a.intent, b.intent] }, c.intent],
        },
      };
    }

    if (family === "exclusion") {
      return {
        text: `${a.text}, excluding ${b.text}`,
        intent: { kind: "and", children: [a.intent, { kind: "not", child: b.intent }] },
      };
    }

    const connective = family as "and" | "or";

    return {
      text: `${a.text} ${connective} ${b.text}`,
      intent: { kind: connective, children: [a.intent, b.intent] },
    };
  }

  function add(
    text: string,
    ctx: Context,
    intent: FilterIntent | null,
    family: string,
    critical = false,
  ): boolean {
    if (revision === 2 && critical) text = `For this selection: ${text}`;
    else if (revision === 2 && intent === null) text = `For this filter: ${text}`;
    const key = text.trim().toLowerCase();

    if (seen.has(key)) return false;
    seen.add(key);

    rows.push({
      text,
      context: ctx,
      intent,
      family,
      critical,
      roles: [],
      split: "heldout",
      fingerprint: hash(key),
      trainingEligible: false,
      ...(intent
        ? {
            compileTarget: WITNESSES.map((row) =>
              truth(intent, row as Record<string, unknown>, ctx),
            ),
          }
        : {}),
    });

    return true;
  }

  const carriers =
    revision === 2
      ? [
          (body: string) => `Show the entries where ${body}.`,
          (body: string) => `Select matching records with ${body}.`,
          (body: string) => `For the result set: ${body}.`,
          (body: string) => `Find all rows that satisfy ${body}.`,
          (body: string) => `Use this filter: ${body}.`,
          (body: string) => `Keep the records where ${body}.`,
        ]
      : [
          (body: string) => `Find entries where ${body}.`,
          (body: string) => `Show records satisfying ${body}.`,
          (body: string) => `For this table: ${body}.`,
          (body: string) => `Select entries with ${body}.`,
          (body: string) => `Filter these rows by ${body}.`,
          (body: string) => `Only show records where ${body}.`,
        ];

  for (const family of positiveFamilies) {
    let count = 0;

    for (let attempt = 0; count < 60; attempt++) {
      if (attempt > 10000) throw new Error("positive dedup exhausted");

      const ctx = context();
      const p = positive(family, ctx);

      count += Number(add(pick(carriers)(p.text), ctx, p.intent, family));
    }
  }

  for (const family of negativeFamilies) {
    let count = 0;

    for (let attempt = 0; count < 50; attempt++) {
      if (attempt > 10000) throw new Error("negative dedup exhausted");

      const ctx = context();
      const p = atom("gt", ctx);
      const name = ctx.fields[0]!.aliases![0]!;

      const text = {
        question: `Should our query use ${p.text}?`,
        description: `The specification mentions ${p.text}.`,
        quotation: `Someone wrote: "${p.text}".`,
        negation: `Never apply ${p.text}.`,
        historical: `The old query used ${p.text}.`,
        ambiguity: `Perhaps ${p.text}; we have not decided.`,
        approximate: `${name} above approximately ${next}.`,
        "unsupported-pattern": `${name} matching the pattern code-${next}.`,
        "unsupported-unit": `${name} above ${next} bytes.`,
        incomplete: `For these records: ${name} at least.`,
        "unknown-field": `Unlisted metric is above ${next}.`,
        inverted: `${name} between ${next + 3} and ${next}.`,
        "mixed-grouping": `${p.text} and ${atom("eq", ctx).text} or ${atom("null", ctx).text}.`,
        "unsafe-number": `${name} above 900719925474${10000 + next}.`,
      }[family]!;

      const fresh =
        revision === 2
          ? text
              .replace("Should our query use", "Should the filter apply")
              .replace("The specification mentions", "The help material discusses")
              .replace("Someone wrote:", "The note quotes:")
              .replace("Never apply", "Do not use")
              .replace("The old query used", "Our earlier rules specified")
              .replace("Perhaps", "Maybe")
              .replace("approximately", "around")
              .replace("matching the pattern", "using regex")
              .replace(" bytes", " percent")
              .replace("For these records:", "For this search:")
              .replace("Unlisted metric", "Undocumented score")
              .replace(" above 900719925474", " at least 900719925474")
          : text;

      count += Number(add(fresh, ctx, null, family));
    }
  }

  const ctx = context();
  const num = ctx.fields[0]!;
  const name = num.aliases![0]!;

  add(
    `For the record: ${name} above -2.75.`,
    ctx,
    { kind: "compare", field: num.id, op: "gt", value: -2.75 },
    "gt",
    true,
  );

  add(`${name} above 100?`, ctx, null, "question", true);
  add(`${name} above 9007199254740993.`, ctx, null, "unsafe-number", true);
  add(`${name} between 100.5 and 100.25.`, ctx, null, "inverted", true);
  add(`${name} above roughly 100.`, ctx, null, "approximate", true);
  add(`${name} above 100 seconds.`, ctx, null, "unsupported-unit", true);

  const reference: Context = {
    fields: [
      {
        id: "status",
        type: "string",
        column: "label",
        terms: [{ text: "active customers", value: "active" }],
      },
      { id: "spending", type: "number", column: "amount" },
      {
        id: "trial",
        type: "boolean",
        column: "flag",
        terms: [{ text: "trials", value: true }],
      },
    ],
  };

  add(
    "For these rows: active customers spending above 100, excluding trials.",
    reference,
    {
      kind: "and",
      children: [
        { kind: "compare", field: "status", op: "eq", value: "active" },
        { kind: "compare", field: "spending", op: "gt", value: 100 },
        {
          kind: "not",
          child: { kind: "compare", field: "trial", op: "eq", value: true },
        },
      ],
    },
    "exclusion",
    true,
  );

  return rows;
}

if (import.meta.main) {
  const [run, destination] = process.argv.slice(2);

  if (!run || !destination) throw new Error("usage: author <frozen-run> <new-reserved.jsonl>");

  const scope = readFileSync(join(run, "scope-freeze.json"));
  const rows = author();
  const body = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";

  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(destination, body, { flag: "wx" });

  writeFileSync(
    destination + ".meta.json",
    JSON.stringify(
      {
        sha256: hash(body),
        candidateSha256: JSON.parse(scope.toString()).artifactFileSha256,
        scopeFreezeSha256: hash(scope),
        createdAt: new Date().toISOString(),
        authorSourceSha256: hash(readFileSync(import.meta.filename)),
        fixtureSha256: hash(readFileSync(join(import.meta.dir, "fixtures.ts"))),
        authorship:
          "AI-authored by the implementation assistant using a separate writer " +
          "without training renderer or model inference",
        independenceLimits:
          "Same assistant implemented and authored evaluation; " +
          "limited generated variations are not independent human evidence or population accuracy",
        trainingEligible: false,
        authoringRevision: revision,
        rows: rows.length,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  console.log({ rows: rows.length, sha256: hash(body), evaluated: false });
}
