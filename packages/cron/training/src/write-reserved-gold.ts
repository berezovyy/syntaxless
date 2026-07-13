import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { resolveRepo } from "./evaluate";
import { generateReservedCarrierGold } from "./splits";

const out = resolveRepo("packages/cron/training/data/gold/reserved-carriers.jsonl");
const records = generateReservedCarrierGold();

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${records.map((record) => JSON.stringify(record)).join("\n")}\n`);
process.stdout.write(`wrote ${records.length} records -> ${out}\n`);
