import type { Demo } from "../ui.ts";
import { cronDemo } from "./cron.ts";
import { filterDemo } from "./filter.ts";
import { formatDemo } from "./format.ts";
import { retentionDemo } from "./retention.ts";
import { retryDemo } from "./retry.ts";
import { sortDemo } from "./sort.ts";
import { validationDemo } from "./validation.ts";

export const DEMOS: Demo[] = [
  cronDemo,
  validationDemo,
  filterDemo,
  sortDemo,
  retryDemo,
  retentionDemo,
  formatDemo,
];
