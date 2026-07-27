import { join, resolve } from "node:path";

import { checkPackage } from "../../../scripts/check-package.js";

const CONSUMER = `import { parse, compile, explain, preload, dispose } from "@syntaxless/format";
import type { FormatIntent, Context } from "@syntaxless/format";

globalThis.fetch = async () => {
  throw Error("offline inference attempted a network request");
};

await preload();

const parsed = await parse("euros, two decimals, negatives in parentheses");

if (
  !parsed.ok ||
  parsed.source !== "model" ||
  parsed.intent.kind !== "currency" ||
  parsed.intent.currency !== "EUR" ||
  parsed.intent.precision?.minimum !== 2 ||
  parsed.intent.precision?.maximum !== 2 ||
  parsed.intent.negative !== "parentheses"
) {
  throw Error("packed reference failed: " + JSON.stringify(parsed));
}

const missing = compile(parsed.intent);

if (missing.supported || missing.code !== "missing-context") {
  throw Error("caller locale was guessed");
}

const context: Context = { locale: "en-US" };
const compiled = compile(parsed.intent, context);

if (!compiled.supported) {
  throw Error(compiled.reason);
}

const shown = compiled.formatter.format(-1234.5);

if (!shown.supported || shown.text !== "(€1,234.50)") {
  throw Error("packed currency presentation mismatch");
}

const zero = compiled.formatter.format(-0);

if (!zero.supported || zero.text !== "(€0.00)") {
  throw Error("packed negative-zero mismatch");
}

if (!explain(parsed.intent).includes("caller must supply a supported locale")) {
  throw Error("explanation omitted caller context");
}

const date: FormatIntent = {
  kind: "date-time",
  dateStyle: "short",
  timeStyle: "short",
  hourCycle: "h23",
};

const clock = compile(date, { locale: "en-US", timeZone: "America/New_York" });

if (!clock.supported) {
  throw Error(clock.reason);
}

const time = clock.formatter.format(Date.parse("2024-03-10T07:30:00Z"));

if (!time.supported || time.text !== "3/10/24, 03:30") {
  throw Error("packed timezone mismatch");
}

const refusals = [
  "format number with negative two decimals",
  "use dollars with two decimals",
  "euros, two decimals?",
];

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
