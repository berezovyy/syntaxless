import type { PolicyField, RetryIntent } from "./types.js";
import { validate } from "./validate.js";

export function explain(intent: RetryIntent): string {
  validate(intent);

  const source = (field: PolicyField) =>
    intent.defaults.includes(field) ? " (caller default)" : "";

  const parts = [
    `${intent.retries} retries after the initial attempt${source("retries")}.`,
    `${intent.strategy} delays${source("strategy")}, starting at ${intent.initialDelayMs} ms${source("initialDelayMs")}.`,
  ];

  if (intent.multiplier !== undefined) {
    parts.push(
      `Multiply the initial delay by ${intent.multiplier} to the retry index${source("multiplier")}, rounding up each exact result to whole milliseconds.`,
    );
  }

  if (intent.maxDelayMs !== undefined) {
    parts.push(
      `Cap the base and final jitter range at ${intent.maxDelayMs} ms${source("maxDelayMs")}.`,
    );
  }

  if (intent.jitter && intent.jitter.kind !== "none") {
    parts.push(
      `${intent.jitter.kind} jitter${intent.jitter.kind === "proportional" ? ` with fraction ${intent.jitter.fraction}` : ""}${source("jitter")}, using an explicit caller seed.`,
    );
  } else {
    parts.push(`No jitter is applied${intent.jitter ? source("jitter") : ""}.`);
  }

  parts.push("The package returns a delay schedule and does not execute requests.");

  return parts.join(" ");
}
