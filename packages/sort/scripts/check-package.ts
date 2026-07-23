import { join, resolve } from "node:path";

import { checkPackage } from "../../../scripts/check-package.js";

const CONSUMER = `import {
  parse,
  compile,
  sortRecords,
  explain,
  preload,
  dispose,
} from "@syntaxless/sort";
import type { Context } from "@syntaxless/sort";

const context: Context = {
  defaultNulls: "last",
  tieBreaker: "id",
  fields: [
    {
      id: "created",
      type: "number",
      column: "created",
      orderingAliases: [{ text: "newest", direction: "desc" }],
    },
    {
      id: "price",
      type: "number",
      column: "price",
      aliases: ["prices"],
      orderingAliases: [{ text: "cheapest", direction: "asc" }],
    },
    { id: "id", type: "number", column: "id" },
  ],
};

globalThis.fetch = async () => {
  throw new Error("offline parser attempted a network request");
};

await preload();

const result = await parse("newest first, then cheapest, missing prices last", context);

if (!result.ok || result.source !== "model") {
  throw new Error("packed reference parse failed: " + JSON.stringify(result));
}

const expected = {
  keys: [
    { field: "created", direction: "desc", nulls: "last", nullsSource: "caller-default" },
    { field: "price", direction: "asc", nulls: "last", nullsSource: "explicit" },
  ],
  tieBreaker: "id",
};

if (JSON.stringify(result.intent) !== JSON.stringify(expected)) {
  throw new Error("packed reference meaning mismatch");
}

if (!compile(result.intent, context).supported) {
  throw new Error("packed SQL compilation failed");
}

const records = [
  { id: 1, created: 2, price: 30 },
  { id: 2, created: 2, price: 10 },
  { id: 3, created: 2, price: 10 },
  { id: 4, created: null, price: 0 },
];

const sortedIds = sortRecords(records, result.intent, context).map((record) => record.id);

if (JSON.stringify(sortedIds) !== "[2,3,1,4]") {
  throw new Error("packed comparator priority/ties mismatch");
}

if (!explain(result.intent).includes("caller default")) {
  throw new Error("explanation dropped defaults");
}

const refusals = [
  "newest first?",
  "price ascending descending",
  "newest then cheapest then nulls first",
];

for (const text of refusals) {
  if ((await parse(text, context)).ok) {
    throw new Error("packed refusal failed: " + text);
  }
}

const withoutNullPolicy = { ...context };

delete withoutNullPolicy.defaultNulls;

if ((await parse("price descending", withoutNullPolicy)).ok) {
  throw new Error("missing null policy was guessed");
}

dispose();
await preload();
console.log("check:package: consumer runtime ok");
`;

const source = join(import.meta.dirname, "..");

checkPackage(source, CONSUMER, process.argv[2] ? resolve(process.argv[2]) : source);
