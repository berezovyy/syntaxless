import { expect, test } from "bun:test";

import { createRng } from "../src/rng";
import {
  exceedsFingerprintCap,
  fingerprintReport,
  slotFingerprint,
} from "../src/slots/fingerprint";
import { sampleFingerprints, sampleSlotDraw } from "../src/slots/sample";
import { EMPTY_PART, EMPTY_SPEECH, type SlotDraw } from "../src/slots/types";

const bare: SlotDraw = {
  speech: EMPTY_SPEECH,
  scene: EMPTY_PART,
  hedge: EMPTY_PART,
  reason: EMPTY_PART,
  tail: EMPTY_PART,
  position: "right",
};

test("fingerprints distinguish meaning and slot placement, not synonymous speech-frame IDs", () => {
  const key = slotFingerprint(bare);

  expect(
    slotFingerprint({
      ...bare,
      speech: { id: "other-words", act: "request", prefix: "please" },
    }),
  ).toBe(key);

  expect(slotFingerprint({ ...bare, speech: { id: "question", act: "question" } })).not.toBe(key);
  expect(slotFingerprint({ ...bare, position: "left" })).not.toBe(key);

  for (const slot of ["scene", "hedge", "reason", "tail"] as const) {
    expect(
      slotFingerprint({ ...bare, [slot]: { id: "different", text: "different" } }),
      slot,
    ).not.toBe(key);
  }
});

test("fingerprint reports count concentration and empty wraps", () => {
  const empty = slotFingerprint(bare);

  expect(fingerprintReport([empty, "other", empty], 1)).toEqual({
    n: 3,
    maxShare: 2 / 3,
    emptyShare: 2 / 3,
    top: [{ fingerprint: empty, n: 2, share: 2 / 3 }],
  });

  expect(fingerprintReport([])).toEqual({ n: 0, maxShare: 0, emptyShare: 0, top: [] });
});

test("the concentration cap rejects at 2%, not only above it", () => {
  const unique = (n: number) => Array.from({ length: n }, (_, i) => `frame-${i}`);

  expect(exceedsFingerprintCap(unique(50))).toBe(true);
  expect(exceedsFingerprintCap(unique(51))).toBe(false);
});

test("production sampling stays below the 2% concentration cap", () => {
  const fingerprints = sampleFingerprints(10_000, 20260916);

  expect(fingerprints).toHaveLength(10_000);
  expect(fingerprintReport(fingerprints).maxShare).toBeLessThan(0.02);
});

test("sampled wraps fill one or two optional slots instead of stacking all four", () => {
  const rng = createRng(7);
  const counts = new Set<number>();

  for (let i = 0; i < 1000; i++) {
    const draw = sampleSlotDraw(rng);

    counts.add(
      [draw.scene, draw.hedge, draw.reason, draw.tail].filter((part) => part.id !== "empty").length,
    );
  }

  expect(counts).toEqual(new Set([1, 2]));
});

test("production sampling reaches every speech act and slot placement", () => {
  const rng = createRng(20260916);
  const acts = new Set<string>();
  const positions = new Set<string>();

  for (let index = 0; index < 10_000; index++) {
    const draw = sampleSlotDraw(rng);

    acts.add(draw.speech.act);
    positions.add(draw.position);
  }

  expect(acts).toEqual(
    new Set([
      "request",
      "descriptive",
      "question",
      "quoted",
      "correction",
      "one-off",
      "cancel",
      "incomplete",
    ]),
  );

  expect(positions).toEqual(new Set(["left", "mid", "right"]));
});
