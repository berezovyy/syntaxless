import { compile, cron, parse } from "../dist/index.js";

function check(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`smoke: ${message}`);
  }
}

const weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"] as const;
const joined = weekdays.join(",");

const compiled = compile({
  frequency: { unit: "week", interval: 1 },
  weekdays: [...weekdays],
  time: { hour: 9, minute: 0 },
});

check(compiled.supported, "compile should support a weekday schedule");

if (compiled.supported) {
  check(compiled.expression === "0 9 * * 1-5", `unexpected compile result: ${compiled.expression}`);
}

const parsed = await parse("every weekday at 9");

check(parsed.ok, "parse should succeed for 'every weekday at 9'");

if (parsed.ok) {
  check(parsed.intent.frequency.unit === "week", "parsed frequency should be weekly");
  check(parsed.intent.time?.hour === 9, "parsed time should be 09:00");
  check(parsed.intent.weekdays?.join(",") === joined, "parsed weekdays should be Mon-Fri");
}

const scheduled = await cron("every weekday at 9");

check(scheduled.supported, "cron should support 'every weekday at 9'");

check(
  typeof scheduled === "object" && "expression" in scheduled && "intent" in scheduled,
  "cron result must expose expression and intent",
);

if (scheduled.supported) {
  check(scheduled.expression === "0 9 * * 1-5", `unexpected expression: ${scheduled.expression}`);
  check(scheduled.intent.weekdays?.join(",") === joined, "cron intent weekdays mismatch");
}

const unsupported = await cron("every other Friday");

check(!unsupported.supported, "cron should not support 'every other Friday'");
check(unsupported.intent !== null, "unsupported cron must preserve the parsed intent");

console.log("smoke: ok");
