/**
 * Constitution for the eval loop: a miss adds a kind, not a noun regex.
 *
 * Overlay catalogs are gone. This test pins that leftover regex does not
 * return, and forbids *new* eval-fossil patterns. It does not parse English
 * and does not call `parse()`.
 */
import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { resolveRepo } from "../src/evaluate";

const DECODE = readFileSync(resolveRepo("packages/cron/src/parser/decode.ts"), "utf8");
const MODEL = readFileSync(resolveRepo("packages/cron/src/parser/model.ts"), "utf8");
const EVAL_README = readFileSync(resolveRepo("packages/cron/evaluation/README.md"), "utf8");

test("overlay catalogs are deleted from the public parser", () => {
  expect(DECODE).not.toMatch(/function unsafeScheduleContext\(/);
  expect(DECODE).not.toMatch(/function hasDirectScheduleRequest\(/);
  expect(DECODE).not.toMatch(/function unrequestedDeclarativeContext\(/);
  expect(MODEL).not.toMatch(/function scopeActionableCorrection\(/);
  expect(DECODE).not.toMatch(/\/(?:flights\?\|planes\?)/);
  expect(DECODE).not.toMatch(/bakery/i);
  expect(DECODE).not.toMatch(/clock strikes/);
  expect(DECODE).not.toMatch(/daly\|daliy/);
  expect(DECODE).not.toMatch(/if \(overlayOn && explicitScheduleEvidence\) confidence = 1/);
  expect(MODEL).not.toMatch(/options\?\.overlay \?\? "on"/);
});

test("new eval-fossil noun regex is not added to decode or model", () => {
  for (const source of [DECODE, MODEL]) {
    expect(source).not.toMatch(/\bbakery\b/i);
    expect(source).not.toMatch(/\bflights?\b/i);
  }
});

test("evaluation README forbids noun regex after a consumed miss", () => {
  expect(EVAL_README).toMatch(/After a miss: kinds, not nouns/);
  expect(EVAL_README).toMatch(/regression suite\s+result/i);
  expect(EVAL_README).toMatch(/Forbidden: a decoder regex/);
});
