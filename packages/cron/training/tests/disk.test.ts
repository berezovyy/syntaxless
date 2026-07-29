import { expect, test } from "bun:test";
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { diskPreflight, replaceJson } from "../src/corpus/disk";
import { tempDir } from "./helpers/temp-dir";

test(
  "a failed JSON commit preserves the destination, " + "removes its temporary and permits retry",
  () => {
    const directory = tempDir("cron-json-commit-");
    const destination = join(directory, "state.json");

    mkdirSync(destination);
    writeFileSync(join(destination, "prior"), "preserve me");
    expect(() => replaceJson(destination, { phase: "emit" })).toThrow();
    expect(readFileSync(join(destination, "prior"), "utf8")).toBe("preserve me");
    expect(readdirSync(directory)).toEqual(["state.json"]);
    rmSync(destination, { recursive: true });
    replaceJson(destination, { phase: "emit" });
    expect(readFileSync(destination, "utf8")).toBe('{\n  "phase": "emit"\n}\n');
    expect(readdirSync(directory)).toEqual(["state.json"]);
  },
);

test("disk preflight refuses a requirement larger than the free space", () => {
  expect(() => diskPreflight(tempDir("cron-disk-"), Number.MAX_SAFE_INTEGER)).toThrow(
    "Insufficient disk space",
  );

  expect(() => diskPreflight(tempDir("cron-disk-"), 0)).toThrow("positive safe-integer");
});
