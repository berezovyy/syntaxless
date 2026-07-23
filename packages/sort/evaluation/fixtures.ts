/** Paired targets are authored separately from the production comparator/compiler. */
export const WITNESSES = [
  { id: 1, amount: 30, created: 5, label: "z", flag: true },
  { id: 2, amount: 10, created: 2, label: "a", flag: false },
  { id: 3, amount: null, created: 4, label: "q", flag: null },
  { id: 4, amount: 10, created: 2, label: "a", flag: false },
  { id: 5, amount: 20, created: null, label: "😀", flag: true },
  { id: 6, amount: 30, created: 5, label: "\uE000", flag: false },
  { id: 7, amount: 20, created: 1, label: null, flag: true },
  { id: 8, amount: null, created: 7, label: "a", flag: null },
] as const;
