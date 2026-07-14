/**
 * Glue independent bag parts around a labeled schedule hole.
 *
 * Wrap tokens are `O`. Hole labels are copied unchanged. Quotes (when asked)
 * wrap the hole only, via `wrapDiscourse`. Returns null if the hole text would
 * move, split, or disappear.
 */
import type { SpeechAct } from "../../../src/model/speech-acts";
import { copySentence, wrapDiscourse } from "../discourse";
import { Sentence } from "../sentence";
import { holeActionableFor, type SlotDraw } from "./types";

export interface ComposeResult {
  sentence: Sentence;
  speechAct: SpeechAct;
  holeActionable: boolean;
  draw: SlotDraw;
}

function copyHole(hole: Sentence, actionable: boolean): Sentence {
  const copy = new Sentence();

  copySentence(hole, copy, actionable);

  return copy;
}

function addClause(sentence: Sentence, text: string): void {
  if (text.length === 0) return;
  if (sentence.text.length > 0) sentence.add(",", "O", "", false);
  sentence.add(text, "O", " ", false);
}

/** Scene bags are noun phrases; compose supplies a locative that can sit either side. */
export function locativeScene(scene: string, prep = "at"): string {
  if (scene.length === 0) return "";
  if (/^(at|in|on|for|by|with|from)\b/i.test(scene)) return scene;

  return `${prep} ${scene}`;
}

function copyCore(dest: Sentence, core: Sentence): void {
  if (dest.text.length > 0) dest.add(",", "O", "", false);
  copySentence(core, dest);
}

/**
 * Layout:
 * - mid:   hedge, at scene, speech.prefix HOLE speech.suffix, reason, tail
 * - left:  speech.prefix HOLE speech.suffix, reason, tail, hedge, at scene
 * - right: hedge, at scene, reason, speech.prefix HOLE speech.suffix, tail
 */
export function compose(hole: Sentence, draw: SlotDraw): ComposeResult | null {
  if (hole.text.length === 0) return null;
  const speechAct = draw.speech.act;
  const holeActionable = holeActionableFor(speechAct);
  // Position moves the complete speech frame, not its semantic operator.
  // Dropping "cancel" / "the manual says" turned negatives into bare requests
  // while retaining their refusal labels, poisoning both supervision and splits.
  const prefix = draw.speech.prefix;
  const suffix = draw.speech.suffix;
  const quoted = draw.speech.quoted === true;
  let core = copyHole(hole, holeActionable);

  if (
    quoted ||
    (prefix !== undefined && prefix.length > 0) ||
    (suffix !== undefined && suffix.length > 0)
  ) {
    const frame = speechAct === "none" ? "request" : speechAct;

    core = wrapDiscourse(core, frame, {
      id: draw.speech.id,
      ...(prefix !== undefined && prefix.length > 0 ? { prefix } : {}),
      ...(suffix !== undefined && suffix.length > 0 ? { suffix } : {}),
      ...(quoted ? { quoted: true } : {}),
    });
  }

  if (!core.text.includes(hole.text)) return null;

  const hedge = draw.hedge.text;
  const scene = locativeScene(draw.scene.text, draw.scene.prep ?? "at");
  const reason = draw.reason.text;
  const tail = draw.tail.text;
  const before: string[] = [];
  const after: string[] = [];

  switch (draw.position) {
    case "left":
      after.push(reason, tail, hedge, scene);
      break;
    case "right":
      before.push(hedge, scene, reason);
      after.push(tail);
      break;
    case "mid":
      before.push(hedge, scene);
      after.push(reason, tail);
      break;
  }

  const sentence = new Sentence();

  for (const part of before) addClause(sentence, part);
  copyCore(sentence, core);
  for (const part of after) addClause(sentence, part);
  if (!sentence.text.includes(hole.text)) return null;

  return { sentence, speechAct, holeActionable, draw };
}
