import type { PolicyField, RetryIntent } from "./types.js";
import { validate } from "./validate.js";

const capitalize = (text: string) => text[0]!.toUpperCase() + text.slice(1);

export function explain(intent: RetryIntent): string {
  validate(intent);

  const source = (field: PolicyField) =>
    intent.defaults.includes(field) ? " (caller default)" : "";

  const retryWord = intent.retries === 1 ? "retry" : "retries";
  const delays = `${capitalize(intent.strategy)} delays${source("strategy")}`;
  const initialDelay = `${intent.initialDelayMs} ms${source("initialDelayMs")}`;

  const parts = [
    `${intent.retries} ${retryWord} after the initial attempt${source("retries")}.`,
    `${delays}, starting at ${initialDelay}.`,
  ];

  if (intent.multiplier !== undefined) {
    const exponent = `to the retry index${source("multiplier")}`;

    parts.push(
      `Multiply the initial delay by ${intent.multiplier} ${exponent}, ` +
        "rounding up each exact result to whole milliseconds.",
    );
  }

  if (intent.maxDelayMs !== undefined) {
    parts.push(
      `Cap the base and final jitter range at ${intent.maxDelayMs} ms${source("maxDelayMs")}.`,
    );
  }

  if (intent.jitter && intent.jitter.kind !== "none") {
    const { jitter } = intent;
    const fraction = jitter.kind === "proportional" ? ` with fraction ${jitter.fraction}` : "";
    const description = `${capitalize(jitter.kind)} jitter${fraction}${source("jitter")}`;

    parts.push(`${description}, using an explicit caller seed.`);
  } else {
    parts.push(`No jitter is applied${intent.jitter ? source("jitter") : ""}.`);
  }

  parts.push("The package returns a delay schedule and does not execute requests.");

  return parts.join(" ");
}
