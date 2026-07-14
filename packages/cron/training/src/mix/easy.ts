/**
 * Easy-no records: all-O, speech-act none, intent null.
 */
import type { SpeechAct } from "../../../src/model/speech-acts";
import type { KindExample } from "../kinds";
import { Sentence } from "../sentence";

export function easyKindExample(text: string, index: number): KindExample {
  const sentence = new Sentence();

  sentence.add(text, "O", "", false);
  const template = `negative/easy/${index}`;

  return {
    kind: "negative",
    family: "negative",
    template,
    split: "train",
    intent: null,
    example: {
      text: sentence.text,
      spans: sentence.spans,
      template,
      style: "easy",
      speechAct: "none" satisfies SpeechAct,
    },
  };
}

export function assertEasyRecord(example: KindExample): void {
  if (example.intent !== null) throw new Error("easy-no must have intent null");

  if (example.example.speechAct !== "none") {
    throw new Error(`easy-no speechAct is ${example.example.speechAct}`);
  }

  if (example.example.spans.length === 0) throw new Error("easy-no has no spans");

  if (example.example.spans.some((span) => span.label !== "O")) {
    throw new Error("easy-no has a non-O span");
  }

  if (example.example.spans.some((span) => span.actionable)) {
    throw new Error("easy-no span is actionable");
  }
}
