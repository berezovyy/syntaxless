import { join, resolve } from "node:path";

import { checkPackage } from "../../../scripts/check-package.js";

const CONSUMER = `import { compile, evaluate, explain, parse, preload } from "@syntaxless/filter";
import type { Context } from "@syntaxless/filter";

const context: Context = {
  fields: [
    {
      id: "status",
      type: "string",
      column: "status",
      terms: [{ text: "active customers", value: "active" }],
    },
    { id: "spending", type: "number", column: "spending" },
    { id: "trial", type: "boolean", column: "trial", terms: [{ text: "trials", value: true }] },
  ],
};

globalThis.fetch = async () => {
  throw new Error("offline inference attempted a network request");
};

await preload();

const result = await parse("active customers spending over 100, excluding trials", context);

if (!result.ok || result.source !== "model") {
  throw new Error("packed reference parse failed: " + JSON.stringify(result));
}

const compiled = compile(result.intent, context);
const expectedParameters = JSON.stringify(["active", 100, 1]);

if (!compiled.supported || JSON.stringify(compiled.parameters) !== expectedParameters) {
  throw new Error("packed SQL mismatch: " + JSON.stringify(compiled));
}

const aboveBoundary = { status: "active", spending: 100.25, trial: false };
const atBoundary = { status: "active", spending: 100, trial: false };
const excludedTrial = { status: "active", spending: 101, trial: true };

if (
  !evaluate(result.intent, aboveBoundary, context) ||
  evaluate(result.intent, atBoundary, context) ||
  evaluate(result.intent, excludedTrial, context)
) {
  throw new Error("packed evaluator boundary/exclusion mismatch");
}

if (!explain(result.intent).includes("NOT")) {
  throw new Error("explanation discarded exclusion");
}

const refusals = ["spending over 100?", "spending over 100 bytes", "maybe spending over 100"];

for (const text of refusals) {
  if ((await parse(text, context)).ok) {
    throw new Error("packed refusal failed: " + text);
  }
}

console.log("check:package: consumer runtime ok");
`;

const source = join(import.meta.dirname, "..");

checkPackage(source, CONSUMER, process.argv[2] ? resolve(process.argv[2]) : source);
