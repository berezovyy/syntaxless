/**
 * Discourse wrap renderer: wrap a labeled inner schedule, keep inner roles,
 * and record speech-act plus which tokens are actionable.
 *
 * Labels come from `add()`, never from `parse()`. Prefixes are ordinary English
 * and must not copy consumed human-evaluation wording.
 */
import type { SpeechAct } from "../../src/model/speech-acts";
import { Sentence, type Span } from "./sentence";

export interface DiscourseWrap {
  /** Closed template id, e.g. `shop-open`, `docs-quote`. */
  id: string;
  prefix?: string;
  suffix?: string;
  /** If true, wrap inner text in ASCII quotes as `O` spans. */
  quoted?: boolean;
}

function leadingSeparator(already: string, quoted: boolean, gap: string): string {
  if (already.length === 0) return gap;
  if (gap.length > 0) return gap;
  if (quoted && /["']$/.test(already)) return "";

  return " ";
}

/**
 * Wrap `inner` in a discourse frame. Inner role labels are copied unchanged.
 * `none` is not a wrap frame. The helper does not invent an intent.
 */
export function wrapDiscourse(
  inner: Sentence,
  frame: SpeechAct,
  template: DiscourseWrap,
): Sentence {
  if (frame === "none") throw new RangeError("wrapDiscourse: none is not a wrap frame");
  const wrapped = new Sentence();
  const keepInnerActionable = frame === "request" || frame === "correction";

  if (template.prefix !== undefined && template.prefix.length > 0) {
    wrapped.add(template.prefix, "O", " ", false);
  }

  if (template.quoted) {
    wrapped.add('"', "O", wrapped.text.length > 0 ? " " : "", false);
  }

  const text = inner.text;

  inner.spans.forEach((span: Span, index) => {
    const previousEnd = index === 0 ? 0 : inner.spans[index - 1]!.end;
    const gap = text.slice(previousEnd, span.start);
    const fragment = text.slice(span.start, span.end);

    const separator =
      index === 0 ? leadingSeparator(wrapped.text, template.quoted === true, gap) : gap;

    if (span.clauseStart) wrapped.clause();
    wrapped.add(fragment, span.label, separator, keepInnerActionable ? span.actionable : false);
  });

  if (template.quoted) wrapped.add('"', "O", "", false);

  if (template.suffix !== undefined && template.suffix.length > 0) {
    wrapped.add(template.suffix, "O", " ", false);
  }

  return wrapped;
}

/** Replay `from` onto `to` with labels preserved. `actionable` overrides every span. */
export function copySentence(from: Sentence, to: Sentence, actionable?: boolean): void {
  const text = from.text;

  from.spans.forEach((span, index) => {
    const previousEnd = index === 0 ? 0 : from.spans[index - 1]!.end;
    const gap = text.slice(previousEnd, span.start);
    const fragment = text.slice(span.start, span.end);
    const separator = index === 0 ? leadingSeparator(to.text, false, gap) : gap;

    if (span.clauseStart) to.clause();
    to.add(fragment, span.label, separator, actionable ?? span.actionable);
  });
}
