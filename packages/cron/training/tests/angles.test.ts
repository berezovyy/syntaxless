import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { ANGLE_DIMENSIONS, ANGLES } from "../src/llm/angles";
import type { Angle } from "../src/llm/angles";
import { FAMILY_WEIGHTS } from "../src/sampler";
import type { Family } from "../src/sampler";

const KNOWN_FAMILIES = new Set<Family>(Object.keys(FAMILY_WEIGHTS) as Family[]);

const PROMPTS_DIR = join(import.meta.dir, "..", "prompts");

function promptPath(angle: Angle): string {
  return join(PROMPTS_DIR, `${angle.promptId}.md`);
}

function readPrompt(angle: Angle): string {
  return readFileSync(promptPath(angle), "utf8");
}

function frontMatter(raw: string): Record<string, string> {
  const match = /^---\n([\s\S]*?)\n---/.exec(raw);

  expect(match).not.toBeNull();
  const fields: Record<string, string> = {};

  for (const line of match![1]!.split("\n")) {
    const separator = line.indexOf(":");

    if (separator === -1) continue;
    fields[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }

  return fields;
}

test("angle ids are unique and non-empty", () => {
  const ids = ANGLES.map((angle) => angle.id);

  expect(new Set(ids).size).toBe(ids.length);
  for (const id of ids) expect(id.length).toBeGreaterThan(0);
});

test("angle dimension tuples are unique", () => {
  const tuples = ANGLES.map((angle) =>
    [
      angle.output,
      angle.register,
      angle.domain,
      angle.numberStyle,
      angle.timeframe,
      angle.noise,
      angle.difficulty,
    ].join("|"),
  );

  expect(new Set(tuples).size).toBe(tuples.length);
});

test("every angle references a known family", () => {
  for (const angle of ANGLES) {
    expect(KNOWN_FAMILIES.has(angle.family)).toBe(true);
  }
});

test("output kind and family agree, and difficulty matches bounds/exception", () => {
  for (const angle of ANGLES) {
    if (angle.output === "negative") {
      expect(angle.family).toBe("negative");
    } else {
      expect(angle.family).not.toBe("negative");
    }

    if (angle.difficulty === "bounds") expect(angle.family).toBe("bounds");
    if (angle.difficulty === "exception") expect(angle.family).toBe("exception");
  }
});

test("every angle references an existing prompt file with matching id and version", () => {
  for (const angle of ANGLES) {
    const fields = frontMatter(readPrompt(angle));

    expect(fields.id).toBe(angle.promptId);
    expect(Number(fields.version)).toBe(angle.promptVersion);
    expect(fields.schemaVersion).toBe("1");
    expect(fields.angle).toBe(angle.output);
  }
});

test("every prompt body carries all five placeholders", () => {
  const seen = new Set<string>();

  for (const angle of ANGLES) {
    if (seen.has(angle.promptId)) continue;
    seen.add(angle.promptId);
    const raw = readPrompt(angle);

    for (const placeholder of [
      "{{lexicon}}",
      "{{family}}",
      "{{seedIntent}}",
      "{{count}}",
      "{{schema}}",
    ]) {
      expect(raw).toContain(placeholder);
    }
  }
});

test("every dimension value appears in at least one angle", () => {
  for (const [dimension, values] of Object.entries(ANGLE_DIMENSIONS)) {
    for (const value of values) {
      const used = ANGLES.some(
        (angle) => (angle as unknown as Record<string, string>)[dimension] === value,
      );

      expect(used).toBe(true);
    }
  }
});

test("every family is represented by at least one angle", () => {
  const families = new Set(ANGLES.map((angle) => angle.family));

  for (const family of KNOWN_FAMILIES) {
    expect(families.has(family)).toBe(true);
  }
});
