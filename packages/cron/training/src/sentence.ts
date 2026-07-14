/**
 * Span-recording sentence builder for the deterministic renderer.
 *
 * Core rule: every span label is supplied by the `add()` call that inserted the
 * fragment. Nothing here or in the renderer reads `parse()` output, and no label
 * is ever inferred from rendered text.
 *
 * Alignment contract (enforced by Phase 13): each `add()` inserts one fragment
 * whose `[start, end)` covers a whole number of tokenizer tokens, so a fragment
 * never straddles a token boundary. JS strings are UTF-16, so offsets recorded
 * here line up with the tokenizer's `RawToken` offsets.
 */
import type { Label } from "../../src/model/labels";
import { isStructural } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import { normalize } from "../../src/parser/normalize";

export interface Span {
  start: number;
  end: number;
  label: Label;
  clauseStart: boolean;
  /** 1 iff this token is in the schedule to execute. Renderer-assigned. */
  actionable: boolean;
}

/** 1 iff the token belongs to the schedule the user is asking to run. */
export function defaultActionable(label: Label): boolean {
  return label !== "O";
}

export class Sentence {
  #text = "";
  #spans: Span[] = [];
  #pendingClause = false;

  /** The accumulated text. */
  get text(): string {
    return this.#text;
  }

  /** Spans in insertion order. */
  get spans(): Span[] {
    return this.#spans;
  }

  /**
   * Mark the next clause. The pending flag is consumed by the first content
   * label that follows (structural filler such as `RECUR`/`GLUE` does not
   * consume it), so `clauseStart` marks the clause's first value slot.
   */
  clause(): void {
    this.#pendingClause = true;
  }

  /**
   * Append `separator` (except at the start of the text), then `text`, and
   * record the resulting UTF-16 span.
   */
  add(text: string, label: Label = "O", separator = " ", actionable?: boolean): void {
    // The shipped model never sees raw user text: parseWithModel normalizes it
    // before tokenization. Normalize each renderer fragment before recording
    // offsets so training features and production features share exactly the
    // same casing, Unicode, meridiem, and terminal-punctuation policy while
    // preserving intentional empty separators such as `2-minute`.
    text = normalize(text);
    if (text.length === 0) return;
    if (this.#text.length > 0) this.#text += separator;
    const start = this.#text.length;

    this.#text += text;
    const end = this.#text.length;
    const clauseStart = this.#pendingClause && label !== "O" && !isStructural(label);

    if (clauseStart) this.#pendingClause = false;

    this.#spans.push({
      start,
      end,
      label,
      clauseStart,
      actionable: actionable ?? defaultActionable(label),
    });
  }
}

/**
 * Normalize the assembled sentence too: separate fragments can form `9:25am`,
 * even though each fragment was normalized. Production sees `9:25 am`.
 * Remap offsets by identical non-space tokens, never infer or rewrite roles.
 * Any normalization that changes non-space tokens must be fixed at the source;
 * guessing alignment would silently manufacture supervision.
 */
export function normalizeAnnotatedText(
  text: string,
  spans: readonly Span[],
): { text: string; spans: Span[] } {
  const normalized = normalize(text);

  if (normalized === text) return { text, spans: spans.map((span) => ({ ...span })) };
  const before = tokenize(text).filter((token) => token.kind !== 3);
  const after = tokenize(normalized).filter((token) => token.kind !== 3);

  if (before.length !== after.length || before.some((token, i) => token.text !== after[i]!.text)) {
    throw new Error("Normalization changes annotated tokens; repair the source construction");
  }

  const starts = new Map(before.map((token, i) => [token.start, after[i]!.start]));
  const ends = new Map(before.map((token, i) => [token.end, after[i]!.end]));

  return {
    text: normalized,
    spans: spans.map((span) => {
      const start = starts.get(span.start);
      const end = ends.get(span.end);

      if (start === undefined || end === undefined) {
        throw new Error("Cannot normalize a span that is not token-aligned");
      }

      return { ...span, start, end };
    }),
  };
}
