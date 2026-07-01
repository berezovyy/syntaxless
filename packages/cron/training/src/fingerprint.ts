/**
 * Structural fingerprints for rendered corpus examples.
 *
 * A fingerprint coarsens one rendered example to its frame: value slots become
 * `"slot"`, structural filler keeps its label and normalized literal, and
 * carrier prose (`O`) is hashed so wording still separates otherwise identical
 * frames. Paraphrases of one frame therefore collide, which is what lets the
 * held-out split exclude whole frames rather than renamed strings.
 *
 * The pattern is never derived from parsing: it is read from the same `Span`
 * labels the renderer chose at insertion time.
 */
import { createHash } from "node:crypto";

import { isStructural } from "../../src/model/labels";
import type { Span } from "./sentence";

/** The one token a value slot contributes to a pattern. */
export const SLOT = "slot";

/** The pattern entry for an example with no content label (all `O`). */
export const NEGATIVE = "negative";

/** SHA-256 as lowercase hex. */
export function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/**
 * Lowercase, collapse whitespace, and collapse every digit run to `#`. This is
 * the coarse form survives inside a literal or feeds the carrier hash.
 */
export function coarseText(text: string): string {
  return text.toLowerCase().replace(/\d+/g, "#").split(/\s+/).filter(Boolean).join(" ");
}

/**
 * Build the ordered structural pattern for a rendered example. Carrier spans
 * contribute `sha256(coarse).slice(0, 12)`; structural spans contribute
 * `"LABEL:coarse literal"`; content spans contribute {@link SLOT}. An example
 * with no non-`O` span is a single {@link NEGATIVE} entry.
 */
export function fingerprintPattern(text: string, spans: readonly Span[]): string[] {
  if (!spans.some((span) => span.label !== "O")) return [NEGATIVE];
  const pattern: string[] = [];

  for (const span of spans) {
    const literal = text.slice(span.start, span.end);

    if (span.label === "O") {
      pattern.push(sha256Hex(coarseText(literal)).slice(0, 12));
    } else if (isStructural(span.label)) {
      pattern.push(`${span.label}:${coarseText(literal)}`);
    } else {
      pattern.push(SLOT);
    }
  }

  return pattern;
}

/**
 * Structural pattern with carrier text erased: `O` collapses to a single
 * placeholder. Two examples that differ only by carrier wording share a
 * frame pattern; this is the view used to assert split disjointness.
 */
export function structuralPattern(text: string, spans: readonly Span[]): string[] {
  return fingerprintPattern(text, spans).map((entry) =>
    /^[0-9a-f]{12}$/.test(entry) ? "O" : entry,
  );
}

/** Fingerprint one rendered example as `sha256(JSON.stringify(pattern))`. */
export function fingerprintSpans(text: string, spans: readonly Span[]): string {
  return sha256Hex(JSON.stringify(fingerprintPattern(text, spans)));
}

/** Fingerprint a rendered example. */
export function fingerprintExample(example: { text: string; spans: readonly Span[] }): string {
  return fingerprintSpans(example.text, example.spans);
}
