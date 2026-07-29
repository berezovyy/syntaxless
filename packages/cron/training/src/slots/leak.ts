/**
 * Refuse bag parts that copy eval wording, reserved carriers, or fossil nouns.
 */
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { normalizeText, RESERVED_PHRASES } from "../background";
import { HEDGE_PARTS } from "./hedge";
import { REASON_PARTS } from "./reason";
import { SCENE_PARTS } from "./scene";
import { SPEECH_FRAMES } from "./speech";
import { TAIL_PARTS } from "./tail";
import type { BagPart, SpeechFrame } from "./types";

const FOSSIL = [
  /\bbakery\b/i,
  /\bflights?\b/i,
  /clock strikes/i,
  /train leaves/i,
  /rehearsal begins/i,
];

export interface BagText {
  bag: string;
  id: string;
  text: string;
}

export function speechTexts(frame: SpeechFrame): string[] {
  return [frame.prefix, frame.suffix].filter(
    (part): part is string => part !== undefined && part.length > 0,
  );
}

export function allBagTexts(): BagText[] {
  const rows: BagText[] = [];

  for (const frame of SPEECH_FRAMES) {
    for (const text of speechTexts(frame)) rows.push({ bag: "speech", id: frame.id, text });
  }

  const listed: Array<readonly [string, readonly BagPart[]]> = [
    ["scene", SCENE_PARTS],
    ["hedge", HEDGE_PARTS],
    ["reason", REASON_PARTS],
    ["tail", TAIL_PARTS],
  ];

  for (const [bag, parts] of listed) {
    for (const part of parts) rows.push({ bag, id: part.id, text: part.text });
  }

  return rows;
}

export function fossilHits(text: string): string[] {
  return FOSSIL.filter((pattern) => pattern.test(text)).map((pattern) => String(pattern));
}

export function reservedHits(text: string): string[] {
  const normalized = normalizeText(text);
  const hits: string[] = [];

  for (const phrase of RESERVED_PHRASES) {
    if (normalized === phrase || normalized.includes(phrase)) hits.push(phrase);
  }

  return hits;
}

function evaluationDir(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../evaluation");
}

export function humanEvalTexts(): string[] {
  const dir = evaluationDir();
  const texts: string[] = [];

  for (const name of readdirSync(dir)) {
    if (!name.startsWith("human-v") || !name.endsWith(".jsonl")) continue;

    for (const line of readFileSync(join(dir, name), "utf8").split("\n")) {
      if (line.trim().length === 0) continue;
      const record = JSON.parse(line) as { text?: unknown };

      if (typeof record.text === "string") texts.push(normalizeText(record.text));
    }
  }

  return texts;
}

export function humanWordingHits(text: string, human: readonly string[]): string[] {
  const normalized = normalizeText(text);

  if (normalized.length === 0) return [];

  return human.filter((line) => {
    if (line === normalized) return true;

    if (normalized.length >= line.length && line.length >= 24 && normalized.includes(line)) {
      return true;
    }

    return false;
  });
}

export interface LeakHit {
  bag: string;
  id: string;
  text: string;
  kind: "fossil" | "reserved" | "human";
  detail: string;
}

export function leakHits(human = humanEvalTexts()): LeakHit[] {
  const hits: LeakHit[] = [];

  for (const row of allBagTexts()) {
    for (const fossil of fossilHits(row.text)) {
      hits.push({ ...row, kind: "fossil", detail: fossil });
    }

    for (const phrase of reservedHits(row.text)) {
      hits.push({ ...row, kind: "reserved", detail: phrase });
    }

    for (const line of humanWordingHits(row.text, human)) {
      hits.push({ ...row, kind: "human", detail: line });
    }
  }

  return hits;
}
