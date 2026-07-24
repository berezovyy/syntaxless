import { join, resolve } from "node:path";

import { checkPackage } from "../../../scripts/check-package.js";

const CONSUMER = `import { parse, compile, explain, preload, dispose } from "@syntaxless/retry";
import type { RetryIntent } from "@syntaxless/retry";

globalThis.fetch = async () => {
  throw new Error("offline inference attempted a network request");
};

await preload();

const text = "retry three times, double the delay, cap at one minute";
const absent = await parse(text);

if (absent.ok || absent.code !== "missing-initial-delay") {
  throw new Error("missing initial delay was guessed: " + JSON.stringify(absent));
}

const parsed = await parse(text, { defaults: { initialDelayMs: 250 } });

if (!parsed.ok || parsed.source !== "model") {
  throw new Error("packed reference failed: " + JSON.stringify(parsed));
}

const expected: RetryIntent = {
  retries: 3,
  strategy: "exponential",
  initialDelayMs: 250,
  multiplier: 2,
  maxDelayMs: 60000,
  defaults: ["initialDelayMs"],
};

for (const key of Object.keys(expected) as (keyof RetryIntent)[]) {
  if (JSON.stringify(parsed.intent[key]) !== JSON.stringify(expected[key])) {
    throw new Error("packed meaning mismatch: " + key);
  }
}

const compiled = compile(parsed.intent);

if (!compiled.supported || JSON.stringify(compiled.delaysMs) !== "[250,500,1000]") {
  throw new Error("packed schedule mismatch");
}

if (!explain(parsed.intent).includes("250 ms (caller default)")) {
  throw new Error("explanation discarded caller default");
}

const compileSeededJitter = () => compile({ ...expected, jitter: { kind: "full" } }, { seed: 42 });
const jitter = compileSeededJitter();

if (!jitter.supported || JSON.stringify(jitter) !== JSON.stringify(compileSeededJitter())) {
  throw new Error("jitter seed mismatch");
}

const refusals = [
  "retry three times with constant delay of one second?",
  "retry three attempts including the first with constant delay of one second",
];

for (const text of refusals) {
  if ((await parse(text)).ok) {
    throw new Error("packed refusal failed: " + text);
  }
}

dispose();
await preload();
console.log("check:package: consumer runtime ok");
`;

const source = join(import.meta.dirname, "..");

checkPackage(source, CONSUMER, process.argv[2] ? resolve(process.argv[2]) : source);
