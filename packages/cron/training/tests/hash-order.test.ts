import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readdirSync } from "node:fs";
import { join } from "node:path";

import { HashOrderedSpool } from "../src/corpus/production/hash-order";
import { tempDir } from "./helpers/temp-dir";

const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");

async function drain(spool: HashOrderedSpool): Promise<string[]> {
  const parts: Buffer[] = [];

  for await (const chunk of spool.chunks()) parts.push(chunk);
  const text = Buffer.concat(parts).toString("utf8");

  expect(text.endsWith("\n")).toBe(true);

  return text.slice(0, -1).split("\n");
}

function rows(count: number) {
  return Array.from({ length: count }, (_, index) => {
    const source = index < count / 2 ? "manual" : "public";
    const line = JSON.stringify({ source, index, text: `row ${index} — ünïcode` });

    return { hash: sha256(line), line };
  });
}

function hashOrder(input: ReturnType<typeof rows>): string[] {
  return [...input].sort((left, right) => (left.hash < right.hash ? -1 : 1)).map((row) => row.line);
}

test("external runs and multi-pass merges give the same order as one in-memory sort", async () => {
  const input = rows(20_000);
  const directory = join(tempDir("hash-order-external-"), "spool");
  // About 140 characters per entry: a 4 KiB budget spills hundreds of runs,
  // and fan-in 2 forces several intermediate merge passes.
  const spool = new HashOrderedSpool(directory, { runCharacters: 4096, mergeFanIn: 2 });

  for (const row of input) spool.add(row.hash, row.line);
  expect(spool.runCount).toBeGreaterThan(100);
  expect(await drain(spool)).toEqual(hashOrder(input));
  // Intermediate runs are deleted as soon as they are merged.
  expect(readdirSync(directory).length).toBeLessThanOrEqual(2);
  spool.remove();
});

test("memory held while adding stays below the run budget, independent of row count", () => {
  const budget = 8192;

  const spool = new HashOrderedSpool(join(tempDir("hash-order-bound-"), "spool"), {
    runCharacters: budget,
  });

  let largest = 0;

  for (const row of rows(10_000)) {
    spool.add(row.hash, row.line);
    largest = Math.max(largest, spool.queuedCharacters);
  }

  // A collecting implementation would hold all ~1.4M characters here.
  expect(largest).toBeLessThan(budget);
  expect(spool.runCount).toBeGreaterThan(100);
  spool.remove();
});
