import { expect, test } from "bun:test";

import {
  SPEECH_ACT_COUNT,
  SPEECH_ACTS,
  speechActFromId,
  speechActId,
} from "../src/model/speech-acts.ts";

test("speech-act ids are contiguous from 0 and complete", () => {
  SPEECH_ACTS.forEach((act, index) => expect(speechActId[act]).toBe(index));
  expect(new Set(Object.values(speechActId)).size).toBe(SPEECH_ACT_COUNT);
  expect(SPEECH_ACT_COUNT).toBe(SPEECH_ACTS.length);
  expect(SPEECH_ACTS.length).toBe(9);
  expect(new Set(SPEECH_ACTS).size).toBe(SPEECH_ACTS.length);
});

test("frozen speech-act order is the design vocabulary", () => {
  expect(SPEECH_ACTS.join(",")).toBe(
    "none,request,descriptive,question,quoted,correction,one-off,cancel,incomplete",
  );

  expect(speechActFromId(0)).toBe("none");
  expect(speechActFromId(1)).toBe("request");
  expect(speechActId.request).toBe(1);
  expect(speechActFromId(SPEECH_ACT_COUNT - 1)).toBe("incomplete");
});

test("out-of-range predicted ids are a hard error, never none", () => {
  expect(() => speechActFromId(-1)).toThrow(RangeError);
  expect(() => speechActFromId(SPEECH_ACT_COUNT)).toThrow(RangeError);
  expect(() => speechActFromId(Number.NaN)).toThrow(RangeError);
  expect(speechActFromId(0)).toBe("none");
});
