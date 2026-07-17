/** Load the local Tatoeba easy-negative source required by production. */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { acceptEasyNo, type EasyFilterOptions } from "./filter";

const directory = join(dirname(fileURLToPath(import.meta.url)), "../../data/real/tatoeba");

export const TATOEBA_SENTENCES = join(directory, "sentences.txt");
export const TATOEBA_MANIFEST = join(directory, "manifest.json");

export function loadTatoebaSentences(
  options: EasyFilterOptions = {},
  path = TATOEBA_SENTENCES,
): string[] {
  const sentences = readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && acceptEasyNo(line, options));

  if (sentences.length < 2000) {
    throw new Error(
      `Tatoeba source requires at least 2000 eligible sentences: ${path} (${sentences.length})`,
    );
  }

  return sentences;
}
