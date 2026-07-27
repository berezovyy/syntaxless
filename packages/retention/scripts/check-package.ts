import { join, resolve } from "node:path";

import { checkPackage } from "../../../scripts/check-package.js";

const CONSUMER = `import { parse, compile, explain, preload, dispose } from "@syntaxless/retention";
import type { RetentionIntent, CompileOptions } from "@syntaxless/retention";

globalThis.fetch = async () => {
  throw Error("offline inference attempted a network request");
};

await preload();

const parsed = await parse("keep daily backups for a week and monthly backups for a year");

const expected: RetentionIntent = {
  tiers: [
    { bucket: "day", window: { kind: "calendar", unit: "day", count: 7 } },
    { bucket: "month", window: { kind: "calendar", unit: "month", count: 12 } },
  ],
};

if (
  !parsed.ok ||
  parsed.source !== "model" ||
  JSON.stringify(parsed.intent) !== JSON.stringify(expected)
) {
  throw Error("packed reference failed: " + JSON.stringify(parsed));
}

const missing = compile(parsed.intent, {} as CompileOptions);

if (missing.supported || missing.code !== "missing-context") {
  throw Error("missing calendar context was guessed");
}

const options: CompileOptions = {
  timeZone: "UTC",
  referenceInstant: Date.parse("2024-03-15T12:00:00Z"),
  backups: [
    { id: "current", timestamp: Date.parse("2024-03-15T00:00:00Z") },
    { id: "old", timestamp: Date.parse("2022-01-01T00:00:00Z") },
  ],
};

const result = compile(parsed.intent, options);

const preview = {
  supported: true,
  keep: [{ id: "current", tierIndexes: [0, 1] }],
  remove: ["old"],
};

if (JSON.stringify(result) !== JSON.stringify(preview)) {
  throw Error("packed preview mismatch");
}

if (!explain(parsed.intent).includes("12 calendar month periods")) {
  throw Error("explanation omitted calendar scope");
}

const refusals = ["keep daily backups for a week?", "keep two backups per day for a week"];

for (const text of refusals) {
  if ((await parse(text)).ok) {
    throw Error("packed refusal failed: " + text);
  }
}

dispose();
await preload();
console.log("check:package: consumer runtime ok");
`;

const source = join(import.meta.dirname, "..");

checkPackage(source, CONSUMER, process.argv[2] ? resolve(process.argv[2]) : source);
