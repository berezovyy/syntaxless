import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { corpusCodeHashes } from "../src/corpus/source-hashes";
import { tempDir } from "./helpers/temp-dir";

test("producer identity follows imports, reexports, data and cycles without executing code", () => {
  const directory = tempDir("syntaxless-code-identity-");
  const entry = join(directory, "entry.ts");

  writeFileSync(entry, 'import "./middle.js"; import("./lazy.ts"); export * from "./forward";');
  writeFileSync(join(directory, "middle.ts"), 'export * from "./leaf";');
  writeFileSync(join(directory, "leaf.ts"), 'throw new Error("must not execute");');
  writeFileSync(join(directory, "lazy.ts"), "export const version = 1;");
  writeFileSync(join(directory, "forward.ts"), 'import "./options.json"; export * from "./entry";');
  writeFileSync(join(directory, "options.json"), '{"limit": 2}');
  writeFileSync(join(directory, "unrelated.ts"), "export const mode = 1;");
  const url = pathToFileURL(entry).href;
  const initial = corpusCodeHashes(url);
  const cron = fileURLToPath(new URL("../../", import.meta.url));

  for (const name of [
    "entry.ts",
    "middle.ts",
    "leaf.ts",
    "lazy.ts",
    "forward.ts",
    "options.json",
  ]) {
    expect(Object.hasOwn(initial, relative(cron, join(directory, name)))).toBe(true);
  }

  expect(Object.hasOwn(initial, relative(cron, join(directory, "unrelated.ts")))).toBe(false);

  writeFileSync(join(directory, "unrelated.ts"), "export const mode = 2;");
  expect(corpusCodeHashes(url)).toEqual(initial);
  writeFileSync(join(directory, "leaf.ts"), 'throw new Error("changed dependency");');
  expect(corpusCodeHashes(url)).not.toEqual(initial);
  const changedCode = corpusCodeHashes(url);

  writeFileSync(join(directory, "options.json"), '{"limit": 3}');
  expect(corpusCodeHashes(url)).not.toEqual(changedCode);
});
