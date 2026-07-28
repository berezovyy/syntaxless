import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import { decodeIntent, UnsafeScheduleContextError } from "../../src/parser/decode";
import type { ScheduleIntent } from "../../src/types/intent";
import { labeledTokensFromGold } from "../src/oracle";
import { Sentence } from "../src/sentence";
import { compose } from "../src/slots/compose";
import { EMPTY_PART, EMPTY_SPEECH, type SlotDraw, type SpeechFrame } from "../src/slots/types";

function weekdayNine(): Sentence {
  const hole = new Sentence();

  hole.clause();
  hole.add("every", "RECUR");
  hole.add("weekday", "DAYGROUP_WEEKDAY");
  hole.add("at", "GLUE");
  hole.add("9", "HOUR_9");

  return hole;
}

function draw(overrides: Partial<SlotDraw> = {}): SlotDraw {
  return {
    speech: EMPTY_SPEECH,
    scene: EMPTY_PART,
    hedge: EMPTY_PART,
    reason: EMPTY_PART,
    tail: EMPTY_PART,
    position: "mid",
    ...overrides,
  };
}

const intent: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  time: { hour: 9, minute: 0 },
};

const speechCases: Array<{ speech: SpeechFrame; actionable: boolean }> = [
  { speech: { id: "request", act: "request", prefix: "run this" }, actionable: true },
  {
    speech: { id: "quoted", act: "quoted", prefix: "the note says", quoted: true },
    actionable: false,
  },
];

test.each(speechCases)(
  "$speech.act wrapping preserves roles and applies the correct actionability",
  ({ speech, actionable }) => {
    const hole = weekdayNine();
    const original = structuredClone({ text: hole.text, spans: hole.spans });

    for (const position of ["left", "mid", "right"] as const) {
      const result = compose(
        hole,
        draw({ speech, position, hedge: { id: "h", text: "if possible" } }),
      );

      if (!result) throw new Error(`Missing composed ${speech.act}/${position}`);
      expect(result.speechAct).toBe(speech.act);
      expect(result.holeActionable).toBe(actionable);
      const { sentence } = result;

      expect(sentence.text).toContain(hole.text);

      expect(
        sentence.spans
          .filter((span) => span.label !== "O")
          .map((span) => ({
            text: sentence.text.slice(span.start, span.end),
            label: span.label,
            actionable: span.actionable,
            clauseStart: span.clauseStart,
          })),
      ).toEqual([
        { text: "every", label: "RECUR", actionable, clauseStart: false },
        { text: "weekday", label: "DAYGROUP_WEEKDAY", actionable, clauseStart: true },
        { text: "at", label: "GLUE", actionable, clauseStart: false },
        { text: "9", label: "HOUR_9", actionable, clauseStart: false },
      ]);

      const wrap = sentence.spans.filter((span) => span.label === "O");

      expect(wrap.length).toBeGreaterThan(0);
      expect(wrap.every((span) => span.actionable === false)).toBe(true);

      for (const token of tokenize(sentence.text).filter((token) => token.kind !== 3)) {
        expect(
          sentence.spans.filter((span) => token.start >= span.start && token.end <= span.end),
        ).toHaveLength(1);
      }

      const decode = () =>
        decodeIntent(
          labeledTokensFromGold(sentence.text, sentence.spans),
          undefined,
          sentence.text,
          speech.act,
        );

      if (actionable) expect(decode().intent).toEqual(intent);
      else expect(decode).toThrow(UnsafeScheduleContextError);
    }

    expect({ text: hole.text, spans: hole.spans }).toEqual(original);
  },
);

test("quotes enclose only the schedule and empty slots add no punctuation", () => {
  expect(compose(weekdayNine(), draw())?.sentence.text).toBe("every weekday at 9");
  expect(compose(new Sentence(), draw())).toBeNull();

  const result = compose(
    weekdayNine(),
    draw({
      speech: { id: "quoted", act: "quoted", prefix: "the note says", quoted: true },
      hedge: { id: "hedge", text: "if possible" },
      tail: { id: "tail", text: "thanks" },
    }),
  );

  expect(result?.sentence.text).toBe('if possible, the note says "every weekday at 9", thanks');
});
