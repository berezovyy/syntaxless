import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { readLinesSync } from "../src/corpus/jsonl-io";
import { tempDir } from "./helpers/temp-dir";

test("readLinesSync splits lines and multi-byte characters across tiny buffers", () => {
  const path = join(tempDir("read-lines-"), "input.txt");
  const lines = ["first", "", "ünïcødé — ✓ 日本語", "x".repeat(50), "no newline at end"];

  writeFileSync(path, lines.join("\n"));
  // A 3-byte buffer cuts every multi-byte character and every line.
  expect([...readLinesSync(path, 3)]).toEqual(lines);
  writeFileSync(path, `${lines.join("\n")}\n`);
  expect([...readLinesSync(path, 3)]).toEqual(lines);
});
