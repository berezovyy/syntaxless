/**
 * Load pinned Tatoeba easy-no sentences, or empty if not fetched.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { bootstrapEasySentences } from "./bootstrap";
import { acceptEasyNo, type EasyFilterOptions } from "./filter";

export const TATOEBA_DIR = join(dirname(fileURLToPath(import.meta.url)), "../../data/real/tatoeba");
export const TATOEBA_SENTENCES = join(TATOEBA_DIR, "sentences.txt");
export const TATOEBA_MANIFEST = join(TATOEBA_DIR, "manifest.json");

export type EasySourceKind = "tatoeba" | "bootstrap";

export interface EasySource {
  kind: EasySourceKind;
  sentences: readonly string[];
}

export function loadTatoebaSentences(path = TATOEBA_SENTENCES): string[] {
  if (!existsSync(path)) return [];

  return readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export function loadEasySource(
  options: EasyFilterOptions = {},
  path = TATOEBA_SENTENCES,
): EasySource {
  const pinned = loadTatoebaSentences(path).filter((text) => acceptEasyNo(text, options));

  if (pinned.length >= 2000) return { kind: "tatoeba", sentences: pinned };

  return {
    kind: "bootstrap",
    sentences: bootstrapEasySentences().filter((text) => acceptEasyNo(text, options)),
  };
}
