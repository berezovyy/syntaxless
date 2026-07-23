/** Fixed SQL/JavaScript witness rows; acceptance author supplies independent truth targets. */
export const WITNESSES: readonly Record<string, unknown>[] = [
  { amount: -2.75, label: "active", flag: true },
  { amount: 0, label: "north", flag: false },
  { amount: 100, label: "active", flag: false },
  { amount: 100.25, label: "O'Reilly", flag: true },
  { amount: null, label: null, flag: null },
  { amount: 1000, label: "", flag: false },
  { amount: 100.5, label: "active", flag: false },
  { amount: 101, label: "active", flag: true },
];
