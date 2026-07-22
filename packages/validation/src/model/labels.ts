/** Semantic roles are an artifact compatibility contract; never infer them from text. */
import type { FieldType } from "../types/intent.js";

export const LABELS = [
  "O",
  "TYPE_STRING",
  "TYPE_EMAIL",
  "TYPE_URL",
  "TYPE_UUID",
  "TYPE_NUMBER",
  "TYPE_INTEGER",
  "TYPE_BOOLEAN",
  "REQUIRED",
  "OPTIONAL",
  "GTE",
  "GT",
  "LTE",
  "LT",
  "BETWEEN",
  "EXACT",
  "NUMBER",
  "UNIT_CHARACTER",
  "GLUE",
  "NULLABLE",
  "UNSUPPORTED",
] as const;

export type Label = (typeof LABELS)[number];
export const LABEL_COUNT = LABELS.length;

export const labelId = Object.fromEntries(LABELS.map((name, id) => [name, id])) as Record<
  Label,
  number
>;

export type Role = number;

export const TYPE_LABELS = {
  string: "TYPE_STRING",
  email: "TYPE_EMAIL",
  url: "TYPE_URL",
  uuid: "TYPE_UUID",
  number: "TYPE_NUMBER",
  integer: "TYPE_INTEGER",
  boolean: "TYPE_BOOLEAN",
} as const satisfies Record<FieldType, Label>;

export const TYPE_BY_LABEL: Readonly<Partial<Record<Label, FieldType>>> = Object.fromEntries(
  Object.entries(TYPE_LABELS).map(([type, label]) => [label, type]),
);

export const STRUCTURAL_LABELS = LABELS.filter((label) => label !== "O" && label !== "NUMBER");
export type StructuralLabel = Exclude<Label, "O" | "NUMBER">;
export const CONTENT_LABELS = ["NUMBER"] as const;
export type ContentLabel = (typeof CONTENT_LABELS)[number];

export function isStructural(label: Label): boolean {
  return label !== "O" && label !== "NUMBER";
}

export function labelFromId(id: number): Label {
  const label = LABELS[id];

  if (label === undefined) throw new RangeError(`label id out of range: ${id}`);

  return label;
}
