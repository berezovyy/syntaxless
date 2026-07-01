/**
 * Canonicalizes free-form schedule text before parsing. The result is lowercase and
 * whitespace-collapsed, so callers can compare or tokenize deterministically. The function
 * is idempotent: normalizing an already-normalized string returns the same string.
 */
export function normalize(text: string): string {
  if (typeof text !== "string") {
    throw new TypeError("normalize() expects a string");
  }

  let value = text.normalize("NFKC").toLowerCase();

  value = value.replace(/[\u2013\u2014]/g, "-");
  value = value.replace(/\ba\.?\s*m\.?/g, "am").replace(/\bp\.?\s*m\.?/g, "pm");
  value = value.replace(/[.!?]+$/, "");
  value = value.replace(/\s+/g, " ").trim();

  return value;
}
