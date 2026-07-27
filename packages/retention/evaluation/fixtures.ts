import type { Backup, Preview } from "../src/types.js";

/** Fixed backup instants and explicit calendar cases; no production preview import. */
export const REFERENCE = Date.parse("2024-03-15T12:00:00Z");

export const BACKUPS: readonly Backup[] = [
  { id: "current-early", timestamp: Date.parse("2024-03-15T01:00:00Z") },
  { id: "current-late", timestamp: Date.parse("2024-03-15T02:00:00Z") },
  { id: "march9", timestamp: Date.parse("2024-03-09T00:00:00Z") },
  { id: "march8", timestamp: Date.parse("2024-03-08T23:59:59Z") },
  { id: "february", timestamp: Date.parse("2024-02-20T00:00:00Z") },
  { id: "april2023", timestamp: Date.parse("2023-04-01T00:00:00Z") },
  { id: "march2023", timestamp: Date.parse("2023-03-31T23:59:59Z") },
];

export const REFERENCE_PREVIEW: Preview = {
  keep: [
    { id: "current-late", tierIndexes: [0, 1] },
    { id: "march9", tierIndexes: [0] },
    { id: "february", tierIndexes: [1] },
    { id: "april2023", tierIndexes: [1] },
  ],
  remove: ["current-early", "march8", "march2023"],
};
