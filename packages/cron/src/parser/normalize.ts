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
  // Fold punctuated meridiems without consuming an ordinary article followed
  // by an `m` word (`a monthly reminder` must not become `amonthly reminder`).
  value = value.replace(/\ba\.\s*m\.?/g, "am").replace(/\bp\.\s*m\.?/g, "pm");
  // A dotted clock is common in prose and some locales. Restrict the rewrite
  // to values carrying an explicit meridiem so decimal/version numbers remain
  // untouched.
  value = value.replace(/\b(\d{1,2})\.(\d{2})\s*(am|pm)\b/g, "$1:$2 $3");
  // Repair only boundaries whose interpretation is unambiguous. A generic
  // letter/digit split would corrupt ordinals (`15th`) and identifiers (`v2`).
  value = value.replace(/\b(every|each)\s*(\d+)/g, "$1 $2");
  value = value.replace(/\b(\d{1,2})\s*(am|pm)\b/g, "$1 $2");
  value = value.replace(/[.!?]+$/, "");
  value = value.replace(/\s+/g, " ").trim();

  return value;
}
