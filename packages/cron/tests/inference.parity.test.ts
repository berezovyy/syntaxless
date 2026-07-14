import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import type { WeightsArtifact } from "../src/model/decode.ts";
import { inferRows, inferTokens } from "../src/model/inference.ts";
import { PADDING_ROW, tokenize } from "../src/model/tokenizer.ts";
import { weights } from "../src/model/weights.gen.ts";

const FIXTURES = join(import.meta.dir, "fixtures");
const ARTIFACT = weights as WeightsArtifact;
const ROWS_PER_TOKEN = ARTIFACT.rowsPerToken;
const ROLE_CLASSES = ARTIFACT.roleClasses;
const SCHEDULE_KIND_CLASSES: number = ARTIFACT.scheduleKindClasses ?? 0;
const SPEECH_ACT_CLASSES: number = ARTIFACT.speechActClasses ?? 0;

const HAS_ACTIONABLE = ARTIFACT.segments.some(
  (segment) => segment.name === "actionable_head.weight",
);

const BOUNDARY_THRESHOLD = ARTIFACT.boundaryThreshold;
const TOLERANCE = 1e-3;

function arrayBuffer(path: string): ArrayBuffer {
  const buffer = readFileSync(path);

  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

const rows = new Uint16Array(arrayBuffer(join(FIXTURES, "parity.rows.bin")));
const offsets = new Uint32Array(arrayBuffer(join(FIXTURES, "parity.offsets.bin")));
const fixtureLogits = new Float32Array(arrayBuffer(join(FIXTURES, "parity.logits.bin")));
const fixtureBoundaries = new Float32Array(arrayBuffer(join(FIXTURES, "parity.boundaries.bin")));

const fixtureScheduleKinds =
  SCHEDULE_KIND_CLASSES === 0
    ? new Float32Array()
    : new Float32Array(arrayBuffer(join(FIXTURES, "parity.schedule-kinds.bin")));

const fixtureSpeechActs =
  SPEECH_ACT_CLASSES === 0 || !existsSync(join(FIXTURES, "parity.speech-acts.bin"))
    ? new Float32Array()
    : new Float32Array(arrayBuffer(join(FIXTURES, "parity.speech-acts.bin")));

const fixtureActionable =
  !HAS_ACTIONABLE || !existsSync(join(FIXTURES, "parity.actionable.bin"))
    ? new Float32Array()
    : new Float32Array(arrayBuffer(join(FIXTURES, "parity.actionable.bin")));

const sequences = offsets.length - 1;
const tokens = sequences >= 0 ? offsets[sequences]! : 0;

interface Predictions {
  labels: Uint8Array;
  boundaries: Uint8Array;
  logits: Float32Array;
  boundaryLogits: Float32Array;
  scheduleKindLogits: Float32Array;
  speechActLogits: Float32Array;
  actionableLogits: Float32Array;
}

function append(
  target: Predictions,
  source: Predictions,
  tokenBase: number,
  sequence: number,
): void {
  target.labels.set(source.labels, tokenBase);
  target.boundaries.set(source.boundaries, tokenBase);
  target.logits.set(source.logits, tokenBase * ROLE_CLASSES);
  target.boundaryLogits.set(source.boundaryLogits, tokenBase);
  target.scheduleKindLogits.set(source.scheduleKindLogits, sequence * SCHEDULE_KIND_CLASSES);
  target.speechActLogits.set(source.speechActLogits, sequence * SPEECH_ACT_CLASSES);
  target.actionableLogits.set(source.actionableLogits, tokenBase);
}

/** Run one fixture sequence, optionally padding its tail to `padTo` tokens. */
function predictSequence(sequence: number, padTo = 0): Predictions {
  const start = offsets[sequence]!;
  const end = offsets[sequence + 1]!;
  const length = end - start;
  const width = length + padTo;
  const padded = new Uint16Array(width * ROWS_PER_TOKEN).fill(PADDING_ROW);

  padded.set(rows.subarray(start * ROWS_PER_TOKEN, end * ROWS_PER_TOKEN));
  const result = inferRows(padded, length, true);

  return {
    labels: result.labels.subarray(0, length),
    boundaries: result.boundaries.subarray(0, length),
    logits: result.logits!.subarray(0, length * ROLE_CLASSES),
    boundaryLogits: result.boundaryLogits!.subarray(0, length),
    scheduleKindLogits: result.scheduleKindLogits ?? new Float32Array(),
    speechActLogits: result.speechActLogits ?? new Float32Array(),
    actionableLogits: result.actionableLogits?.subarray(0, length) ?? new Float32Array(),
  };
}

function predictAll(padTo = 0): Predictions {
  const total: Predictions = {
    labels: new Uint8Array(tokens),
    boundaries: new Uint8Array(tokens),
    logits: new Float32Array(tokens * ROLE_CLASSES),
    boundaryLogits: new Float32Array(tokens),
    scheduleKindLogits: new Float32Array(sequences * SCHEDULE_KIND_CLASSES),
    speechActLogits: new Float32Array(sequences * SPEECH_ACT_CLASSES),
    actionableLogits: new Float32Array(tokens),
  };

  for (let sequence = 0; sequence < sequences; sequence++) {
    append(total, predictSequence(sequence, padTo), offsets[sequence]!, sequence);
  }

  return total;
}

function maximumLogitError(computed: Predictions, reference: Float32Array): number {
  let maximum = 0;

  for (let index = 0; index < reference.length; index++) {
    const error = Math.abs(computed.logits[index]! - reference[index]!);

    if (error > maximum) maximum = error;
  }

  return maximum;
}

function maximumArrayError(computed: Float32Array, reference: Float32Array): number {
  let maximum = 0;

  for (let index = 0; index < reference.length; index++) {
    maximum = Math.max(maximum, Math.abs(computed[index]! - reference[index]!));
  }

  return maximum;
}

function argmaxMismatches(computed: Predictions, reference: Float32Array): number {
  let mismatches = 0;
  const total = computed.logits.length / ROLE_CLASSES;

  for (let token = 0; token < total; token++) {
    let expected = 0;
    let expectedBest = -Infinity;
    let actual = 0;
    let actualBest = -Infinity;

    for (let role = 0; role < ROLE_CLASSES; role++) {
      const fixture = reference[token * ROLE_CLASSES + role]!;

      if (fixture > expectedBest) {
        expectedBest = fixture;
        expected = role;
      }

      const value = computed.logits[token * ROLE_CLASSES + role]!;

      if (value > actualBest) {
        actualBest = value;
        actual = role;
      }
    }

    if (expected !== actual) mismatches += 1;
  }

  return mismatches;
}

function boundaryMismatches(computed: Predictions, reference: Float32Array): number {
  let mismatches = 0;

  for (let token = 0; token < reference.length; token++) {
    const expected = Number(reference[token]! >= BOUNDARY_THRESHOLD);

    if (computed.boundaries[token] !== expected) mismatches += 1;
  }

  return mismatches;
}

describe("inference parity fixture", () => {
  test("guards the frozen fixture dimensions", () => {
    expect(sequences).toBeGreaterThan(0);
    expect(rows.length).toBe(tokens * ROWS_PER_TOKEN);
    expect(fixtureLogits.length).toBe(tokens * ROLE_CLASSES);
    expect(fixtureBoundaries.length).toBe(tokens);
    expect(fixtureScheduleKinds.length).toBe(sequences * SCHEDULE_KIND_CLASSES);

    if (SPEECH_ACT_CLASSES > 0 && fixtureSpeechActs.length > 0) {
      expect(fixtureSpeechActs.length).toBe(sequences * SPEECH_ACT_CLASSES);
    }

    if (HAS_ACTIONABLE && fixtureActionable.length > 0) {
      expect(fixtureActionable.length).toBe(tokens);
    }
  });

  test("reproduces the frozen logits and boundaries", () => {
    const computed = predictAll();

    expect(argmaxMismatches(computed, fixtureLogits)).toBe(0);
    expect(boundaryMismatches(computed, fixtureBoundaries)).toBe(0);
    const error = maximumLogitError(computed, fixtureLogits);

    console.log(`full fixture max logit error: ${error}`);
    expect(error).toBeLessThan(TOLERANCE);

    expect(maximumArrayError(computed.scheduleKindLogits, fixtureScheduleKinds)).toBeLessThan(
      TOLERANCE,
    );

    if (SPEECH_ACT_CLASSES > 0 && fixtureSpeechActs.length > 0) {
      expect(maximumArrayError(computed.speechActLogits, fixtureSpeechActs)).toBeLessThan(
        TOLERANCE,
      );
    }

    if (HAS_ACTIONABLE && fixtureActionable.length > 0) {
      expect(maximumArrayError(computed.actionableLogits, fixtureActionable)).toBeLessThan(
        TOLERANCE,
      );
    }
  });

  test("short input keeps zero mismatches and a small logit error", () => {
    let shortest = 0;

    for (let sequence = 1; sequence < sequences; sequence++) {
      const length = offsets[sequence + 1]! - offsets[sequence]!;
      const best = offsets[shortest + 1]! - offsets[shortest]!;

      if (length < best) shortest = sequence;
    }

    const start = offsets[shortest]!;
    const length = offsets[shortest + 1]! - start;
    const computed = predictSequence(shortest);
    const reference = fixtureLogits.subarray(start * ROLE_CLASSES, (start + length) * ROLE_CLASSES);
    const referenceBoundaries = fixtureBoundaries.subarray(start, start + length);

    expect(argmaxMismatches(computed, reference)).toBe(0);
    expect(boundaryMismatches(computed, referenceBoundaries)).toBe(0);
    const error = maximumLogitError(computed, reference);

    console.log(`short input (${length} tokens) max logit error: ${error}`);
    expect(error).toBeLessThan(TOLERANCE);
  });

  test("padding the short input leaves valid predictions unchanged", () => {
    let shortest = 0;

    for (let sequence = 1; sequence < sequences; sequence++) {
      const length = offsets[sequence + 1]! - offsets[sequence]!;
      const best = offsets[shortest + 1]! - offsets[shortest]!;

      if (length < best) shortest = sequence;
    }

    const short = predictSequence(shortest);
    const padded = predictSequence(shortest, 9);

    expect(padded.labels).toEqual(short.labels);
    expect(padded.boundaries).toEqual(short.boundaries);
    let error = 0;

    for (let index = 0; index < short.logits.length; index++) {
      error = Math.max(error, Math.abs(padded.logits[index]! - short.logits[index]!));
    }

    for (let index = 0; index < short.boundaryLogits.length; index++) {
      error = Math.max(
        error,
        Math.abs(padded.boundaryLogits[index]! - short.boundaryLogits[index]!),
      );
    }

    for (let index = 0; index < short.scheduleKindLogits.length; index++) {
      error = Math.max(
        error,
        Math.abs(padded.scheduleKindLogits[index]! - short.scheduleKindLogits[index]!),
      );
    }

    for (let index = 0; index < short.speechActLogits.length; index++) {
      error = Math.max(
        error,
        Math.abs(padded.speechActLogits[index]! - short.speechActLogits[index]!),
      );
    }

    for (let index = 0; index < short.actionableLogits.length; index++) {
      error = Math.max(
        error,
        Math.abs(padded.actionableLogits[index]! - short.actionableLogits[index]!),
      );
    }

    console.log(`padding neutrality max delta: ${error}`);
    expect(error).toBeLessThan(TOLERANCE);
  });

  test("padding fidelity holds for every sequence", () => {
    const computed = predictAll(9);

    expect(argmaxMismatches(computed, fixtureLogits)).toBe(0);
    expect(boundaryMismatches(computed, fixtureBoundaries)).toBe(0);
    const error = maximumLogitError(computed, fixtureLogits);

    console.log(`padded fixture max logit error: ${error}`);
    expect(error).toBeLessThan(TOLERANCE);
  });

  test("legacy artifacts omit speech-act and actionable fields", () => {
    if (SPEECH_ACT_CLASSES > 0 || HAS_ACTIONABLE) return;
    const predictions = inferTokens(tokenize("every weekday at 9"));

    expect(predictions.speechAct).toBeUndefined();
    expect(predictions.actionable).toBeUndefined();
  });
});
