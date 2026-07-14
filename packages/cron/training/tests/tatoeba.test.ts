import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { normalizeText } from "../src/background";
import { fetchAndPinTatoeba, filterTatoebaLines } from "../src/mix/fetch-tatoeba";
import { tempDir } from "./helpers/temp-dir";

test("Tatoeba filtering reports distinct drop reasons and deduplicates kept rows", async () => {
  const kept = "Please close the window.";
  const blocked = "Please open the wooden door.";

  const sentences = [
    kept,
    kept,
    "lowercase text is not a sentence.",
    "Too short.",
    "The clock displays 10:30 clearly.",
    "Please close the window every day.",
    blocked,
  ];

  const result = await filterTatoebaLines(
    [
      ...sentences.map((text, index) => `${index}\teng\t${text}`),
      "8\tfra\tUne phrase en français.",
      "9\teng",
    ],
    new Set([normalizeText(blocked)]),
  );

  expect(result.kept).toEqual([kept]);

  expect(result.dropped).toEqual({
    language: 2,
    shape: 1,
    length: 1,
    normalization: 0,
    timeShaped: 1,
    timeWord: 1,
    reserved: 0,
    blocked: 1,
    duplicate: 1,
  });
});

test("an explicitly missing archive fails without downloading", async () => {
  const archive = join(tempDir("syntaxless-tatoeba-missing-"), "missing.bz2");

  await expect(fetchAndPinTatoeba(archive)).rejects.toThrow("archive not found");
});

test("a decompression failure is rejected before publishing corpus files", async () => {
  const archive = join(tempDir("syntaxless-tatoeba-corrupt-"), "corrupt.bz2");

  writeFileSync(archive, "not a bzip2 archive");
  await expect(fetchAndPinTatoeba(archive)).rejects.toThrow(/bunzip2.*failed/);
});
