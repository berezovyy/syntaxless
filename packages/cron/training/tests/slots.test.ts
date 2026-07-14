import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import { decodeIntent, UnsafeScheduleContextError } from "../../src/parser/decode";
import type { ScheduleIntent } from "../../src/types/intent";
import { labeledTokensFromGold, loadSlottedOracleRecords, oracleMatches } from "../src/oracle";
import { Sentence } from "../src/sentence";
import { compose } from "../src/slots/compose";
import { SPEECH_FRAMES } from "../src/slots/speech";
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
    speech: { id: "correction", act: "correction", prefix: "instead use" },
    actionable: true,
  },
  {
    speech: { id: "descriptive", act: "descriptive", prefix: "the timetable says" },
    actionable: false,
  },
  { speech: { id: "question", act: "question", suffix: "or not?" }, actionable: false },
  {
    speech: { id: "quoted", act: "quoted", prefix: "the note says", quoted: true },
    actionable: false,
  },
  {
    speech: { id: "one-off", act: "one-off", prefix: "just this once" },
    actionable: false,
  },
  {
    speech: { id: "cancel", act: "cancel", prefix: "stop running this" },
    actionable: false,
  },
  {
    speech: { id: "incomplete", act: "incomplete", prefix: "unless" },
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
          "off",
          speech.act,
        );

      if (actionable) expect(decode().intent).toEqual(intent);
      else expect(decode).toThrow(UnsafeScheduleContextError);
    }

    expect({ text: hole.text, spans: hole.spans }).toEqual(original);
  },
);

test.each([
  {
    position: "mid",
    expected:
      "if possible, in the office, run every weekday at 9 please, while the room is empty, thanks",
  },
  {
    position: "left",
    expected:
      "run every weekday at 9 please, while the room is empty, thanks, if possible, in the office",
  },
  {
    position: "right",
    expected:
      "if possible, in the office, while the room is empty, run every weekday at 9 please, thanks",
  },
] as const)(
  "$position placement orders all wrap slots without splitting the schedule",
  ({ position, expected }) => {
    const result = compose(
      weekdayNine(),
      draw({
        speech: { id: "request", act: "request", prefix: "run", suffix: "please" },
        scene: { id: "office", text: "the office", prep: "in" },
        hedge: { id: "hedge", text: "if possible" },
        reason: { id: "reason", text: "while the room is empty" },
        tail: { id: "tail", text: "thanks" },
        position,
      }),
    );

    expect(result?.sentence.text).toBe(expected);
  },
);

test.each([
  { act: "request", prefix: "run this", expected: "run this every weekday at 9" },
  { act: "cancel", prefix: "cancel", expected: "cancel every weekday at 9" },
  {
    act: "descriptive",
    prefix: "the manual says",
    expected: "the manual says every weekday at 9",
  },
] as const)(
  "left placement cannot erase the meaning of a $act frame",
  ({ act, prefix, expected }) => {
    const result = compose(
      weekdayNine(),
      draw({
        position: "left",
        speech: { id: act, act, prefix },
      }),
    );

    expect(result?.sentence.text).toBe(expected);
    expect(result?.speechAct).toBe(act);
  },
);

test("scenes with an existing preposition are not prefixed twice", () => {
  expect(
    compose(weekdayNine(), draw({ scene: { id: "office", text: "by the office" } }))?.sentence.text,
  ).toBe("by the office, every weekday at 9");
});

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

test("production slotted kind-seed wraps round-trip through the decoder", () => {
  const records = loadSlottedOracleRecords(200);

  expect(records).toHaveLength(200);

  expect(
    records
      .filter((record) => !oracleMatches(record))
      .map(({ id, text, speechAct }) => ({ id, text, speechAct })),
  ).toEqual([]);
});

test("the same generated wording never receives different speech acts across layouts", () => {
  const labels = new Map<string, SpeechFrame["act"]>();

  for (const speech of [EMPTY_SPEECH, ...SPEECH_FRAMES]) {
    for (const position of ["left", "mid", "right"] as const) {
      const row = compose(weekdayNine(), draw({ speech, position }));

      if (!row) throw new Error(`Cannot compose ${speech.id}/${position}`);
      const previous = labels.get(row.sentence.text);

      if (previous !== undefined) expect(row.speechAct).toBe(previous);
      labels.set(row.sentence.text, row.speechAct);
    }
  }
});

test.each([
  { prefix: "the shuttle departs at", act: "descriptive" },
  { prefix: "our workshop commences at", act: "descriptive" },
  { prefix: "the desk changed it to", act: "descriptive" },
  { prefix: "actually, change it to", act: "correction" },
] as const)("production speech supervision: $prefix is $act", ({ prefix, act }) => {
  const speech = SPEECH_FRAMES.find((frame) => frame.prefix === prefix);

  if (!speech) throw new Error(`Missing source phrase: ${prefix}`);
  const result = compose(weekdayNine(), draw({ speech }));

  expect(result?.speechAct).toBe(act);
  expect(result?.holeActionable).toBe(act === "correction");
});
