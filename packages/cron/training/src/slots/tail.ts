/**
 * Closers. First four are the former TRAILING_CARRIERS.
 */
import type { BagPart } from "./types";

const TEXTS: readonly string[] = [
  "if that is acceptable",
  "when you have a moment",
  "if that still holds",
  "once you confirm",
  "thanks",
  "should the board agree",
  "when you get a gap",
  "if the lock stays green",
  "no rush",
  "if the hatch is clear",
  "when the kettle cools",
  "if nobody needs the winch",
  "after you glance at it",
  "if the night desk agrees",
  "when you next sit down",
  "unless the winch is busy",
  "that is all",
  "ping me if not",
  "or leave it",
  "no need to reply",
  "whenever you like",
  "I will check later",
  "mark it done",
  "skip if jammed",
  "copy if you heard",
  "over and out",
  "that will do",
  "unless you see a snag",
  "or park it until dawn",
  "does that work",
  "thanks in advance",
  "whenever",
  "no reply needed",
  "I'll owe you",
  "that's the one",
  "ignore if already set",
  "make sense",
  "if not no worries",
  "yeah",
  "ok",
];

export const TAIL_PARTS: readonly BagPart[] = TEXTS.map((text, index) => ({
  id: `tail-${String(index + 1).padStart(2, "0")}`,
  text,
}));
