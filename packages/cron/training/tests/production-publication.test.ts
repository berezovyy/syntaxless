import { expect, test } from "bun:test";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { diskPreflight, publishDirectory, replaceJson } from "../src/corpus/production/publication";
import { tempDir } from "./helpers/temp-dir";

test("a failed JSON commit preserves the destination, removes its temporary and permits retry", () => {
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
});

test("a killed publisher releases ownership and the next invocation can publish", async () => {
  const directory = tempDir("cron-publication-crash-");
  const output = join(directory, "v1");
  const ready = join(directory, "ready");
  const publisher = new URL("../src/corpus/production/publication.ts", import.meta.url).href;

  const child = Bun.spawn(
    [
      process.execPath,
      "--eval",
      `import { writeFileSync } from "node:fs";
     import { join } from "node:path";
     import { publishDirectory } from ${JSON.stringify(publisher)};
     await publishDirectory(${JSON.stringify(output)}, 1, async staging => {
       writeFileSync(join(staging, "partial"), "unfinished");
       writeFileSync(${JSON.stringify(ready)}, staging);
       await Bun.sleep(30_000);
       throw new Error("Fixture writer was not terminated");
     });`,
    ],
    { stdout: "ignore", stderr: "pipe" },
  );

  const stderr = new Response(child.stderr).text();

  try {
    const deadline = performance.now() + 2_000;

    while (!existsSync(ready)) {
      if (child.exitCode !== null || performance.now() >= deadline) {
        throw new Error("Fixture writer did not acquire publication ownership");
      }

      await Bun.sleep(5);
    }

    expect(child.exitCode).toBeNull();
    expect(existsSync(output)).toBe(false);
    const abandoned = readFileSync(ready, "utf8");

    await expect(
      publishDirectory(output, 1, async () => {
        throw new Error("A competing writer must not enter while the owner is alive");
      }),
    ).rejects.toThrow();

    child.kill("SIGKILL");
    expect(await child.exited).not.toBe(0);
    expect(await stderr).toBe("");

    await publishDirectory(output, 1, async (staging) => {
      expect(existsSync(output)).toBe(false);
      expect(staging).not.toBe(abandoned);
      writeFileSync(join(staging, "manifest.json"), "{}\n");
    });

    expect(readFileSync(join(output, "manifest.json"), "utf8")).toBe("{}\n");
    expect(readFileSync(join(abandoned, "partial"), "utf8")).toBe("unfinished");
  } finally {
    if (child.exitCode === null) child.kill("SIGKILL");
    await child.exited;
    await stderr;
  }
}, 5_000);

test("staged publication is invisible until success and an active version lock prevents another writer", async () => {
  const directory = tempDir("cron-publication-");
  const output = join(directory, "v1");

  const result = await publishDirectory(output, 1, async (staging) => {
    expect(existsSync(output)).toBe(false);

    await expect(
      publishDirectory(output, 1, async () => {
        throw new Error("must not run");
      }),
    ).rejects.toThrow();

    writeFileSync(join(staging, "manifest.json"), "{}\n");

    return "published";
  });

  expect(result).toBe("published");
  expect(readFileSync(join(output, "manifest.json"), "utf8")).toBe("{}\n");
  const failed = join(directory, "v2");

  await expect(
    publishDirectory(failed, 1, async (staging) => {
      writeFileSync(join(staging, "partial"), "unfinished");
      throw new Error("injected failure");
    }),
  ).rejects.toThrow("injected failure");

  expect(existsSync(failed)).toBe(false);
  expect(readdirSync(directory).sort()).toEqual([".v1.lock.sqlite", ".v2.lock.sqlite", "v1"]);

  expect(() => diskPreflight(directory, Number.MAX_SAFE_INTEGER)).toThrow(
    "Insufficient disk space",
  );
});
