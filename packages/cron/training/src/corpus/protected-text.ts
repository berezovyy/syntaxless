import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { normalize } from "../../../src/parser/normalize";
import { RESERVED_PHRASES } from "../background";

export interface ProtectedSource {
  file: string;
  sha256: string;
  records: number;
}

/** Conservative protected-wording / ordinary-English grouping key, NOT schedule identity. */
export function textKey(text: string): string {
  return normalize(text)
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Runtime text identity must retain meaningful punctuation: mon/fri != mon-fri. */
export function corpusTextKey(text: string): string {
  return normalize(text);
}

/** A trie finds complete protected sentences inside a longer training message. */
export class ProtectedText {
  readonly #exact = new Set<string>();
  readonly #trie = new Map<string, unknown>();
  readonly sources: readonly ProtectedSource[];

  constructor(texts: Iterable<string>, sources: readonly ProtectedSource[] = []) {
    this.sources = sources;

    for (const text of texts) {
      const key = textKey(text);

      if (!key) continue;
      this.#exact.add(key);
      // Short schedule primitives can independently occur in legitimate data.
      // Full equality is still blocked; containment requires a sentence-sized key.
      if (key.length < 24) continue;
      let node = this.#trie;

      for (const word of key.split(" ")) {
        let child = node.get(word) as Map<string, unknown> | undefined;

        if (!child) {
          child = new Map();
          node.set(word, child);
        }

        node = child;
      }

      node.set("", true);
    }
  }

  matches(text: string): boolean {
    const key = textKey(text);

    if (this.#exact.has(key)) return true;
    const words = key.split(" ");

    for (let start = 0; start < words.length; start++) {
      let node: Map<string, unknown> | undefined = this.#trie;

      for (let index = start; index < words.length; index++) {
        node = node.get(words[index]!) as Map<string, unknown> | undefined;
        if (!node) break;
        if (node.has("")) return true;
      }
    }

    return false;
  }
}

/** Load all evaluation JSONL, including calibration, without exposing its wording. */
export function loadProtectedText(): ProtectedText {
  const training = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
  const evaluation = resolve(training, "../evaluation");

  const paths = readdirSync(evaluation)
    .filter((name) => name.endsWith(".jsonl"))
    .sort()
    .map((name) => join(evaluation, name));

  paths.push(join(training, "data/gold/reserved-carriers.jsonl"));
  const texts: string[] = [...RESERVED_PHRASES];
  const sources: ProtectedSource[] = [];

  for (const path of paths) {
    const bytes = readFileSync(path);
    let records = 0;

    for (const line of bytes.toString("utf8").split("\n")) {
      if (!line.trim()) continue;
      const row = JSON.parse(line) as { text?: unknown };

      if (typeof row.text !== "string") throw new Error(`Protected record has no text: ${path}`);
      texts.push(row.text);
      records++;
    }

    sources.push({
      file: path,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      records,
    });
  }

  return new ProtectedText(texts, sources);
}
