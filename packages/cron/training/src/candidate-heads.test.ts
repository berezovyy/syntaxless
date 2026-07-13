import { describe, expect, test } from "bun:test";
/**
 * Real-path check: an exported artifact with speech-act heads loads in the
 * built engine, and public (heads-only) parse is not `missing-speech-act-head`.
 */
import { existsSync } from "node:fs";

import { loadEngine, resolveRepo } from "./evaluate";

const DIST = "packages/cron/dist";
const FIXTURE = "packages/cron/tests/fixtures/heads.weights.ts";

describe("exported heads artifact", () => {
  test("loads in the built engine and overlay-off is not missing-head", async () => {
    expect(existsSync(resolveRepo(DIST + "/index.js"))).toBe(true);
    expect(existsSync(resolveRepo(FIXTURE))).toBe(true);
    const module = await import(resolveRepo(FIXTURE));

    expect(module.weights.speechActClasses).toBeGreaterThan(0);

    const engine = await loadEngine(DIST, FIXTURE);

    try {
      const on = await engine.parse("every weekday at 9");

      expect(on.ok).toBe(true);

      if (on.ok) {
        expect(on.intent.frequency).toEqual({ unit: "week", interval: 1 });
        expect(on.intent.time).toEqual({ hour: 9, minute: 0 });
      }

      const off = await engine.parse("every weekday at 9", { overlay: "off" });
      const diagnosticCodes = off.ok ? [] : (off.diagnostics ?? []).map((note) => note.code);

      expect(diagnosticCodes).not.toContain("missing-speech-act-head");
      expect(off.ok ? "" : off.reason).not.toMatch(/missing-speech-act-head/);
      expect(off.ok).toBe(true);

      if (off.ok) {
        expect(off.intent.frequency).toEqual({ unit: "week", interval: 1 });
        expect(off.intent.time).toEqual({ hour: 9, minute: 0 });
      }
    } finally {
      await engine.dispose();
    }
  });
});
