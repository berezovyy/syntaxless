/**
 * Canonicalizes free-form constraint text before parsing. Lowercase, NFKC,
 * whitespace-collapsed, trailing `.!` stripped (question marks are preserved). Idempotent.
 *
 * Allowed rewrites are unicode/whitespace only. Domain paraphrases belong in
 * lexicon.ts as alias rows, not here.
 */
export function normalize(text: string): string {
  if (typeof text !== "string") {
    throw new TypeError("normalize() expects a string");
  }

  let value = text.normalize("NFKC").toLowerCase();

  value = value.replace(/[\u2013\u2014]/g, "-");
  value = value.replace(/[\u2018\u2019]/g, "'");
  value = value.trim().replace(/[.!]+$/, "");
  value = value.replace(/\s+/g, " ").trim();

  return value;
}
