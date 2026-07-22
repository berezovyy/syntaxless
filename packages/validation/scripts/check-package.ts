import { join, resolve } from "node:path";

import { checkPackage } from "../../../scripts/check-package.js";

const CONSUMER = `import {
  compile,
  compileObject,
  explain,
  parse,
  preload,
  schema,
} from "@syntaxless/validation";

globalThis.fetch = async () => {
  throw new Error("offline inference attempted a network request");
};

await preload();

const modeled = await parse("required email with max 100 characters", { source: "model" });

if (
  !modeled.ok ||
  modeled.source !== "model" ||
  modeled.intent.type !== "email" ||
  modeled.intent.required !== true ||
  modeled.intent.maxLength !== 100
) {
  throw new Error("packed model intent mismatch: " + JSON.stringify(modeled));
}

const object = compileObject({ email: { type: "email", required: true } });

if (!object.supported || object.schema.required?.[0] !== "email") {
  throw new Error("object presence mismatch");
}

if (!explain({ type: "email", required: true }).includes("required-string default")) {
  throw new Error("explanation lost defaults");
}

const compiled = compile({ type: "email", required: true, maxLength: 100 });

if (typeof (compiled as { then?: unknown }).then === "function") {
  throw new Error("compile must be synchronous");
}

if (
  !compiled.supported ||
  compiled.schema.format !== "email" ||
  compiled.schema.maxLength !== 100
) {
  throw new Error("compile mismatch: " + JSON.stringify(compiled));
}

const parsed = await parse("required email with max 100 characters");

if (!parsed.ok || parsed.source !== "model") {
  throw new Error("packed default model parse failed: " + JSON.stringify(parsed));
}

if (parsed.intent.type !== "email" || parsed.intent.maxLength !== 100) {
  throw new Error("packed parse intent mismatch: " + JSON.stringify(parsed.intent));
}

const result = await schema("required email with max 100 characters");

if (!result.supported || result.schema.minLength !== 1) {
  throw new Error("packed schema mismatch: " + JSON.stringify(result));
}

const rejected = await schema("every weekday at 9");

if (rejected.supported || rejected.intent !== null) {
  throw new Error("packed schema should fail closed: " + JSON.stringify(rejected));
}

const approximate = await schema("required string around 10 characters");

if (approximate.supported || approximate.intent !== null) {
  throw new Error(
    "packed model discarded an unsupported approximation: " + JSON.stringify(approximate),
  );
}

console.log("check:package: consumer runtime ok");
`;

const source = join(import.meta.dirname, "..");

checkPackage(source, CONSUMER, process.argv[2] ? resolve(process.argv[2]) : source);
