/** AI-authored acceptance. This writer imports no training renderer or parser.
 * Freeze a candidate and scope first. After evaluation these rows are regression
 * evidence, and a future candidate must receive newly authored reserved wording.
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import type { JsonSchema } from "../src/compiler/index.js";
import type { FieldIntent, FieldType } from "../src/types/intent.js";
import type { AcceptanceRow } from "../training/src/gate.js";

const hash = (data: string | Uint8Array) => createHash("sha256").update(data).digest("hex");
const types: FieldType[] = ["string", "email", "url", "uuid", "number", "integer", "boolean"];

const names: Record<FieldType, string[]> = {
  string: ["string", "text field", "text"],
  email: ["email", "email address", "e-mail"],
  url: ["url", "web address", "url field"],
  uuid: ["uuid", "uuid field"],
  number: ["number", "numeric field"],
  integer: ["integer", "whole number"],
  boolean: ["boolean", "boolean field"],
};

const presence = [
  { text: "required", value: true },
  { text: "mandatory", value: true },
  { text: "compulsory", value: true },
  { text: "optional", value: false },
  { text: "not required", value: false },
  { text: "not mandatory", value: false },
];

const revision = Number(process.argv[4] ?? 1);

if (![1, 2, 3, 4, 5].includes(revision)) throw new Error("unsupported authoring revision");

const carriers =
  revision === 5
    ? [
        (body: string) => `For the form input use ${body}.`,
        (body: string) => `Please validate this entry as ${body}.`,
        (body: string) => `The field requirement is ${body}.`,
        (body: string) => `For this parameter: ${body}.`,
        (body: string) => `Use ${body} for the form value.`,
        (body: string) => `This entry should be ${body}.`,
        (body: string) => `Make the form field ${body}.`,
        (body: string) => `Please set the input value to ${body}.`,
        (body: string) => `Validate this parameter as ${body}.`,
        (body: string) => `For this entry please use ${body}.`,
      ]
    : revision === 4
      ? [
          (body: string) => `For this entry use ${body}.`,
          (body: string) => `Please make the input ${body}.`,
          (body: string) => `For this form: ${body}.`,
          (body: string) => `The input value must be ${body}.`,
          (body: string) => `Use ${body} in this form.`,
          (body: string) => `Validate the form input as ${body}.`,
          (body: string) => `Please use ${body} for this value.`,
          (body: string) => `The requirement is: ${body}.`,
          (body: string) => `Set the value to ${body}.`,
          (body: string) => `For this input please validate as ${body}.`,
        ]
      : revision === 3
        ? [
            (body: string) => `Use ${body} for this entry.`,
            (body: string) => `Please set this field to ${body}.`,
            (body: string) => `For the input value: ${body}.`,
            (body: string) => `This form needs ${body}.`,
            (body: string) => `Make this value ${body}.`,
            (body: string) => `For the input use ${body}.`,
            (body: string) => `Validate as ${body} in this form.`,
            (body: string) => `The entry must be ${body}.`,
            (body: string) => `For this field please use ${body}.`,
            (body: string) => `Set the form input to ${body}.`,
          ]
        : revision === 2
          ? [
              (body: string) => `Make this input ${body}.`,
              (body: string) => `For the form, use ${body}.`,
              (body: string) => `Validate the value as ${body}.`,
              (body: string) => `This input must be ${body}.`,
              (body: string) => `For the entry: ${body}.`,
              (body: string) => `Set this input to ${body}.`,
              (body: string) => `Please use ${body} in the form.`,
              (body: string) => `For this parameter use ${body}.`,
              (body: string) => `The field should be ${body}.`,
              (body: string) => `Please validate this value as ${body}.`,
            ]
          : [
              (body: string) => `Use ${body}.`,
              (body: string) => `For this value, use ${body}.`,
              (body: string) => `Please validate as ${body}.`,
              (body: string) => `The input must be ${body}.`,
              (body: string) => `For this input: ${body}.`,
              (body: string) => `Set this field to ${body}.`,
              (body: string) => `Please use ${body} for the input.`,
              (body: string) => `For the value, please use ${body}.`,
              (body: string) => `This field must be ${body}.`,
              (body: string) => `Validate this field as ${body}.`,
            ];

let state = 733196 + revision;

function choose<T>(items: readonly T[]): T {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

  return items[Math.floor((state / 2 ** 32) * items.length)]!;
}

function number(): number {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

  return 1 + (state % 997);
}

function target(intent: FieldIntent): JsonSchema {
  // Independent paired gold calculation, without invoking production compile.
  const numeric = intent.type === "number" || intent.type === "integer";

  const schema: JsonSchema = {
    type:
      numeric || intent.type === "boolean"
        ? (intent.type as "number" | "integer" | "boolean")
        : "string",
  };

  if (intent.type === "email") schema.format = "email";
  if (intent.type === "url") schema.format = "uri";
  if (intent.type === "uuid") schema.format = "uuid";

  if (!numeric && intent.type !== "boolean") {
    if (intent.required || intent.minLength !== undefined) {
      schema.minLength = Math.max(intent.required ? 1 : 0, intent.minLength ?? 0);
    }

    if (intent.maxLength !== undefined) schema.maxLength = intent.maxLength;
  }

  for (const side of ["minimum", "maximum"] as const) {
    const bound = intent[side];

    if (!bound) continue;

    if (intent.type === "integer") {
      schema[side] =
        side === "minimum"
          ? Math.ceil(bound.value) + Number(!bound.inclusive && Number.isInteger(bound.value))
          : Math.floor(bound.value) - Number(!bound.inclusive && Number.isInteger(bound.value));
    } else if (bound.inclusive) {
      schema[side] = bound.value;
    } else if (side === "minimum") {
      schema.exclusiveMinimum = bound.value;
    } else {
      schema.exclusiveMaximum = bound.value;
    }
  }

  return schema;
}

function author(): AcceptanceRow[] {
  const rows: AcceptanceRow[] = [];
  const seen = new Set<string>();

  function add(
    text: string,
    intent: FieldIntent | null,
    family: string,
    critical = false,
  ): boolean {
    if (revision === 3 && critical) text = `For the input value: ${text}`;
    if (revision === 4 && critical) text = `For this entry: ${text}`;
    if (revision === 5 && critical) text = `For this parameter: ${text}`;
    const key = text.trim().toLowerCase();

    if (seen.has(key)) return false;
    seen.add(key);

    rows.push({
      id: `reserved-${rows.length}`,
      text,
      intent,
      family: intent ? "constraint" : "negative",
      template: intent?.type ?? family,
      spans: [],
      fingerprint: hash(key),
      split: "heldout",
      seed: 733196 + revision,
      trainingEligible: false,
      critical,
      ...(intent ? { compileTarget: target(intent) } : {}),
    });

    return true;
  }

  for (const type of types) {
    let emitted = 0;

    for (let attempt = 0; emitted < 100; attempt++) {
      if (attempt > 10000) throw new Error("reserved positive deduplication exhausted");

      const p = choose(presence);
      const noun = choose(names[type]);

      const intent: FieldIntent = { type, required: p.value };
      const numeric = type === "number" || type === "integer";
      const n = numeric ? number() - 499 + (choose([true, false]) ? 0.25 : 0) : number();
      const upper = n + number();
      const unit = numeric ? "" : " characters";
      const shape = type === "boolean" ? 0 : attempt % 8;
      let constraint = "";

      if (shape === 1 || shape === 3) {
        constraint = ` with ${shape === 1 ? "at least" : "more than"} ${n}${unit}`;
        if (numeric) intent.minimum = { value: n, inclusive: shape === 1 };
        else intent.minLength = n + Number(shape === 3);
      } else if (shape === 2 || shape === 4) {
        constraint = ` with ${shape === 2 ? "at most" : "less than"} ${upper}${unit}`;
        if (numeric) intent.maximum = { value: upper, inclusive: shape === 2 };
        else intent.maxLength = upper - Number(shape === 4);
      } else if (shape === 5 || shape === 7) {
        constraint =
          shape === 5
            ? ` with between ${n} and ${upper}${unit}`
            : ` with at least ${n} and at most ${upper}${unit}`;

        if (numeric) {
          intent.minimum = { value: n, inclusive: true };
          intent.maximum = { value: upper, inclusive: true };
        } else {
          intent.minLength = n;
          intent.maxLength = upper;
        }
      } else if (shape === 6) {
        const exact = numeric && type === "integer" ? Math.round(n) : n;

        constraint = ` with exactly ${exact}${unit}`;

        if (numeric) {
          intent.minimum = { value: exact, inclusive: true };
          intent.maximum = { value: exact, inclusive: true };
        } else {
          intent.minLength = n;
          intent.maxLength = n;
        }
      }

      const body = choose([`${p.text} ${noun}${constraint}`, `${noun} is ${p.text}${constraint}`]);

      emitted += Number(add(choose(carriers)(body), intent, type));
    }
  }

  const negativeFamilies = [
    "question",
    "description",
    "quotation",
    "negation",
    "historical",
    "ambiguity",
    "conflict",
    "inverted",
    "incomplete",
    "unsupported-unit",
    "nested",
    "unsupported-pattern",
    "unsafe-number",
    ...(revision >= 3 ? ["approximate"] : []),
  ];

  for (const family of negativeFamilies) {
    let emitted = 0;

    for (let attempt = 0; emitted < 50; attempt++) {
      if (attempt > 10000) throw new Error("reserved negative deduplication exhausted");

      const n = number();
      const p = choose(presence);
      const type = choose(types);
      const noun = choose(names[type]);

      const unitSuffix = type === "number" || type === "integer" ? "" : " characters";
      const base = `${p.text} ${noun} with at most ${n}${unitSuffix}`;

      const text = {
        question: revision === 2 ? `Should this input be ${base}?` : `Would you recommend ${base}?`,
        description:
          revision === 2 ? `The help page describes ${base}.` : `The manual describes ${base}.`,
        quotation: revision === 2 ? `The note reads: "${base}".` : `Someone wrote: "${base}".`,
        negation: revision === 2 ? `Do not apply ${base} to this input.` : `Do not use ${base}.`,
        historical:
          revision === 2 ? `Our previous form used ${base}.` : `Previously, we used ${base}.`,
        ambiguity:
          revision === 2
            ? `Perhaps ${base}; the choice is still undecided.`
            : `${base}, maybe; decide later.`,
        conflict: `${choose(carriers)(`required optional ${noun}`)} Max ${n} characters.`,
        inverted: choose(carriers)(`required number at least ${n + 3} and at most ${n}`),
        incomplete: choose(carriers)(
          `${p.text} ${noun} with ` +
            choose(["at least", "less than", "between", "more than", "exactly"]),
        ),
        "unsupported-unit": choose(carriers)(
          `required ${noun} at most ${n} ${choose(["bytes", "words", "seconds", "lines"])}`,
        ),
        nested: choose(carriers)(`required object containing ${noun} with max ${n} characters`),
        "unsupported-pattern": choose(carriers)(`required string matching the pattern code-${n}`),
        "unsafe-number": choose(carriers)(`required number at least 900719925474${10000 + n}`),
        approximate: choose(carriers)(
          `required ${noun} ` +
            `${choose(["around", "roughly", "about", "approximately", "~"])} ${n}${unitSuffix}`,
        ),
      }[family]!;

      const revised =
        revision >= 3
          ? text
              .replace(
                "Would you recommend",
                revision === 5
                  ? "Would this parameter require"
                  : revision === 4
                    ? "Should our field use"
                    : "Would this value need",
              )
              .replace(
                "The manual describes",
                revision === 5
                  ? "The reference page mentions"
                  : revision === 4
                    ? "The guide discusses"
                    : "The specification describes",
              )
              .replace(
                "Someone wrote:",
                revision === 5
                  ? "The example quotes:"
                  : revision === 4
                    ? "The note reads:"
                    : "Our note says:",
              )
              .replace(
                "Do not use",
                revision === 5
                  ? "Do not configure"
                  : revision === 4
                    ? "Avoid applying"
                    : "Never use",
              )
              .replace(
                "Previously, we used",
                revision === 5
                  ? "The previous input used"
                  : revision === 4
                    ? "This form used to have"
                    : "The old form used",
              )
              .replace(
                "maybe; decide later",
                revision === 5 ? "maybe; the choice is pending" : "perhaps; this remains undecided",
              )
          : text;

      emitted += Number(add(revised, null, family));
    }
  }

  // Predeclared safety controls stay visible even if aggregate rates pass.
  add(
    revision === 2
      ? "Validate this value as required number more than -2.75 and at most 6.5."
      : "Please use required number more than -2.75 and at most 6.5.",
    {
      type: "number",
      required: true,
      minimum: { value: -2.75, inclusive: false },
      maximum: { value: 6.5, inclusive: true },
    },
    "number",
    true,
  );

  add(
    revision === 2
      ? "For the entry: optional string exactly 0 characters."
      : "For this input: optional string exactly 0 characters.",
    { type: "string", required: false, minLength: 0, maxLength: 0 },
    "string",
    true,
  );

  add(
    revision === 2 ? "For this input, is an email required?" : "Required email?",
    null,
    "question",
    true,
  );

  add(
    revision === 2
      ? "Make this input required number at least 9007199254740995."
      : "Use required number at least 9007199254740993.",
    null,
    "unsafe-number",
    true,
  );

  add(
    revision === 2
      ? "Set this input to required integer greater than 3 and at most 3."
      : "Use required number greater than 10 and at most 10.",
    null,
    "inverted",
    true,
  );

  add(
    revision === 2
      ? "The field should be required text matching the pattern order-17."
      : "Use required string matching the pattern secret-123.",
    null,
    "unsupported-pattern",
    true,
  );

  if (revision >= 3) {
    for (const cue of ["around", "roughly", "about", "approximately", "~"]) {
      add(`Please use required string ${cue} 23 characters.`, null, "approximate", true);
    }
  }

  return rows;
}

if (import.meta.main) {
  const [run, destination] = process.argv.slice(2);

  if (!run || !destination) {
    throw new Error("usage: author-reserved <frozen-run> <new-reserved.jsonl>");
  }

  const scope = readFileSync(join(resolve(run), "scope-freeze.json"));
  const rows = author();
  const data = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";

  mkdirSync(dirname(resolve(destination)), { recursive: true });
  writeFileSync(destination, data, { flag: "wx" });

  const metadata = {
    sha256: hash(data),
    candidateSha256: JSON.parse(scope.toString()).artifactFileSha256,
    scopeFreezeSha256: hash(scope),
    rows: rows.length,
    authoringRevision: revision,
    createdAt: new Date().toISOString(),
    authorship:
      "AI-authored by the implementation assistant, using a separate evaluation writer " +
      "without training renderer, parser, or model inference",
    independenceLimits:
      "Same assistant designed implementation and evaluation; " +
      "generated variations of a limited number of authored constructions. " +
      "This is not independent human evidence, nor a population accuracy estimate.",
    authorSourceSha256: hash(readFileSync(import.meta.filename)),
    trainingEligible: false,
  };

  writeFileSync(`${destination}.meta.json`, JSON.stringify(metadata, null, 2) + "\n", {
    flag: "wx",
  });

  console.log({ rows: rows.length, sha256: metadata.sha256, evaluated: false });
}
