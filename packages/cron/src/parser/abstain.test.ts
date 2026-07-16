import { expect, test } from "bun:test";

import { labelId } from "../model/labels";
import { sequenceEnergy } from "./abstain";

test("flat energy preserves token offsets and excludes structural and unknown roles", () => {
  const logits = Float32Array.from([
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 8, -2, -2, -2, -2, -2,
    -2, -2,
  ]);

  const roles = Uint8Array.from([
    labelId.O,
    labelId.RECUR,
    255,
    labelId.WEEKDAY_MON,
    labelId.HOUR_3,
  ]);

  const expected = (-(8 + Math.log(1 + 3 * Math.exp(-10))) + 2 - Math.log(4)) / 2;

  expect(sequenceEnergy(logits, roles, 4)).toBeCloseTo(expected, 12);
  expect(sequenceEnergy(logits, Uint8Array.from([labelId.O, labelId.RECUR, 255]), 4)).toBe(0);
});

test("energy stays finite for extreme positive and negative logits", () => {
  const content = Uint8Array.from([labelId.WEEKDAY_MON]);

  expect(sequenceEnergy(Float32Array.from([1000, 0, 0, 0]), content, 4)).toBe(-1000);

  expect(sequenceEnergy(Float32Array.from([-1000, -1000, -1000, -1000]), content, 4)).toBeCloseTo(
    1000 - Math.log(4),
    12,
  );
});
