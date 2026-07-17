import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { DiskHash256Set } from "../src/mix/disk-hash-set";
import { tempDir } from "./helpers/temp-dir";

const digest = (value: string | number): Buffer =>
  createHash("sha256").update(String(value)).digest();

test("the disk set is exact across transaction boundaries and deletes itself", () => {
  const path = join(tempDir("disk-hash-set-"), "set.sqlite");
  const set = new DiskHash256Set(path);
  // More inserts than one transaction holds, so membership must survive commits.
  const count = 70_000;

  for (let index = 0; index < count; index++) expect(set.add(digest(index))).toBe(true);
  expect(set.size).toBe(count);
  expect(set.add(digest(0))).toBe(false);
  expect(set.add(digest(count - 1))).toBe(false);
  expect(set.has(digest(12_345))).toBe(true);
  expect(set.has(digest(count))).toBe(false);
  expect(set.size).toBe(count);
  expect(existsSync(path)).toBe(true);
  set.close();
  set.close();
  expect(existsSync(path)).toBe(false);
  expect(() => set.has(digest(1))).toThrow("closed");
});
