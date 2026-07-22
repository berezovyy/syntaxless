import { expect, test } from "bun:test";

import { assignSplits, frameKey } from "../src/backward.ts";
import { expandKinds, kindIds } from "../src/kinds.ts";

test("every kind has frame-disjoint train, validation, and held-out support", () => {
  const examples = assignSplits(expandKinds());

  for (const kind of kindIds()) {
    const rows = examples.filter((example) => example.kind === kind);

    expect(rows.length, kind).toBeGreaterThan(0);

    expect(new Set(rows.map((example) => example.split)), kind).toEqual(
      new Set(["train", "validation", "heldout"]),
    );

    const frameSplits = new Map<string, Set<string>>();

    for (const row of rows) {
      const key = frameKey(row);
      const splits = frameSplits.get(key) ?? new Set<string>();

      splits.add(row.split);
      frameSplits.set(key, splits);
    }

    expect(
      [...frameSplits.values()].every((splits) => splits.size === 1),
      kind,
    ).toBe(true);
  }
});
