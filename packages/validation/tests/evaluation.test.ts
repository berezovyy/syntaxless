/**
 * Regression replay of consumed evaluation wording against the bundled model.
 * Passing here is regression evidence only; it is not fresh acceptance.
 */
import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { parse } from "../src/index.ts";
import type { FieldIntent } from "../src/index.ts";

const evaluation = join(import.meta.dir, "../evaluation");

interface Row {
  id: string;
  text: string;
  intent: FieldIntent | null;
  critical?: boolean;
}

function readRows(path: string): Row[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as Row);
}

function consumedCriticalRows(): Row[] {
  const reserved = join(evaluation, "reserved");

  return readdirSync(reserved)
    .filter((name) => name.endsWith(".jsonl"))
    .filter((name) => {
      const digest = createHash("sha256")
        .update(readFileSync(join(reserved, name)))
        .digest("hex");

      return existsSync(join(evaluation, "consumed", `${digest}.json`));
    })
    .flatMap((name) => readRows(join(reserved, name)))
    .filter((row) => row.critical === true);
}

async function failures(rows: readonly Row[]) {
  const failed: { id: string; text: string }[] = [];

  for (const row of rows) {
    const result = await parse(row.text);

    const passed = row.intent
      ? result.ok && Bun.deepEquals(result.intent, row.intent, true)
      : !result.ok;

    if (!passed) failed.push({ id: row.id, text: row.text });
  }

  return failed;
}

test("the bundled model passes every critical row of consumed reserved sets", async () => {
  const rows = consumedCriticalRows();

  expect(rows.some((row) => row.intent !== null)).toBe(true);
  expect(rows.some((row) => row.intent === null)).toBe(true);
  expect(await failures(rows)).toEqual([]);
});

test("the bundled model refuses every development negative", async () => {
  const rows = readRows(join(evaluation, "dev-negatives.jsonl"));

  expect(rows.every((row) => row.intent === null)).toBe(true);
  expect(await failures(rows)).toEqual([]);
});
