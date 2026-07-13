import { describe, expect, test } from "bun:test";

import {
  exclusiveEndFromInclusiveShown,
  hour12From24,
  hour24From12,
  hourWithoutMeridiemIsInvertible,
  inclusiveShownFromExclusiveEnd,
  meridiemFrom24,
} from "./schedule-semantics.js";

describe("through inclusivity", () => {
  test("shown last day is inclusive; intent end is the next day", () => {
    expect(exclusiveEndFromInclusiveShown("2026-01-31")).toBe("2026-02-01");
    expect(inclusiveShownFromExclusiveEnd("2026-02-01")).toBe("2026-01-31");
  });
});

describe("12h clock invertibility", () => {
  test("1pm is 13:00 and is not invertible as a bare 1", () => {
    expect(hour12From24(13)).toBe(1);
    expect(meridiemFrom24(13)).toBe("pm");
    expect(hour24From12(1, "pm")).toBe(13);
    expect(hourWithoutMeridiemIsInvertible(1, 13)).toBe(false);
    expect(hourWithoutMeridiemIsInvertible(13, 13)).toBe(true);
  });

  test("midnight is 12am", () => {
    expect(hour24From12(12, "am")).toBe(0);
    expect(hour12From24(0)).toBe(12);
    expect(meridiemFrom24(0)).toBe("am");
  });
});
