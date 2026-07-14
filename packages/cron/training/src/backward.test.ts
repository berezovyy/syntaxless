import { expect, test } from "bun:test";

import { assignSplits, frameKey } from "./backward";
import { expandKinds, type KindId } from "./kinds";

test("every selected kind has frame-disjoint train, validation, and held-out support", () => {
  const wanted = new Set<KindId>([
    "unit-interval",
    "day-rule",
    "exception-group",
    "frequency-idiom",
  ]);

  const examples = assignSplits(expandKinds(wanted));

  for (const kind of wanted) {
    const rows = examples.filter((example) => example.kind === kind);

    expect(new Set(rows.map((example) => example.split))).toEqual(
      new Set(["train", "validation", "heldout"]),
    );

    const frameSplits = new Map<string, Set<string>>();

    for (const row of rows) {
      const key = frameKey(row);
      const splits = frameSplits.get(key) ?? new Set<string>();

      splits.add(row.split);
      frameSplits.set(key, splits);
    }

    expect([...frameSplits.values()].every((splits) => splits.size === 1)).toBe(true);
  }
});
