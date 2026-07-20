/** The only training entry point: bun run train. */
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { prepareCompactCorpus } from "./compact";
import { replaceJson } from "./corpus/production/publication";
import { evaluateHumanCorpus } from "./human-evaluate";
import { runOracleSuite, summarizeOracleReport } from "./oracle";

const repo = fileURLToPath(new URL("../../../../", import.meta.url));

function run(
  command: readonly [string, ...string[]],
  environment: NodeJS.ProcessEnv = process.env,
): void {
  const [executable, ...args] = command;

  const result = spawnSync(executable, args, {
    cwd: repo,
    env: environment,
    stdio: "inherit",
  });

  if (result.error || result.status !== 0) {
    throw new Error(
      `${executable} failed: ${result.error?.message ?? result.signal ?? result.status}`,
    );
  }
}

async function train(): Promise<void> {
  const storage = join(repo, "training/cron/runs");

  mkdirSync(storage, { recursive: true });
  const directory = mkdtempSync(join(storage, "compact-"));
  const started = Date.now();

  const workflow = {
    command: "bun run train",
    corpusPrefix: null as string | null,
    runDirectory: directory,
    status: "failed",
    error: undefined as string | undefined,
    elapsedSeconds: 0,
    storage: [] as {
      phase: string;
      repositoryAllocatedBytes: number;
      runAllocatedBytes: number;
    }[],
  };

  function measure(phase: string): void {
    const allocated = (path: string) =>
      Number(execFileSync("du", ["-s", "-B1", path], { encoding: "utf8" }).split(/\s/)[0]);

    const snapshot = {
      phase,
      repositoryAllocatedBytes: allocated(repo),
      runAllocatedBytes: allocated(directory),
    };

    workflow.storage.push(snapshot);

    if (phase !== "finished" && snapshot.runAllocatedBytes > 128 * 1024 ** 2) {
      throw new Error("Compact cron run exceeds its 128 MiB storage budget");
    }
  }

  try {
    console.log(`run -> ${directory}`);
    measure("before preparation");
    // Exclude available evaluation wording and disclose historical gaps.
    // The historical production generator remains private and intact.
    const prepared = await prepareCompactCorpus(join(directory, "corpus"));
    const consumption = prepared.proof.consumption;

    const { verifiedPartitions, ledgerPartitions } = consumption;

    console.log(`consumed evidence: ${verifiedPartitions}/${ledgerPartitions} partitions verified`);

    if (consumption.limitation) console.warn(consumption.limitation);
    const prefix = prepared.prefix;

    workflow.corpusPrefix = prefix;
    measure("after preparation");
    run(["uv", "sync", "--project", "training/cron", "--locked"]);
    const oracle = await runOracleSuite();
    const summary = summarizeOracleReport(oracle);
    const oraclePath = join(directory, "oracle.json");

    writeFileSync(oraclePath, `${JSON.stringify(summary, null, 2)}\n`, { flag: "wx" });
    console.log(`oracle: ${oracle.correct}/${oracle.total} (failures ${oracle.failureCount})`);

    for (const [kind, stats] of Object.entries(summary.byKind).sort()) {
      if (stats.failures > 0) console.log(`  ${kind}: ${stats.failures}/${stats.total}`);
    }

    console.log(`report -> ${oraclePath}`);
    if (oracle.failureCount > 0) throw new Error(`Oracle gate failed; see ${oraclePath}`);
    // This process waits while Python trains; give its corpus-build garbage
    // back before the two run side by side.
    Bun.gc(true);
    const trainingStarted = Date.now();

    run(
      [
        "uv",
        "run",
        "--project",
        "training/cron",
        "--locked",
        "--no-sync",
        "python",
        "training/cron/train.py",
      ],
      {
        ...process.env,
        SYNTAXLESS_TRAIN_DATA_PREFIX: resolve(prefix),
        SYNTAXLESS_TRAIN_RUN_DIRECTORY: directory,
        SYNTAXLESS_BUN_BIN: process.execPath,
      },
    );

    const reportPath = join(directory, "report.json");
    const candidate = join(directory, "weights.gen.ts");

    const report = JSON.parse(readFileSync(reportPath, "utf8")) as {
      status?: string;
      candidateModule?: string;
    };

    if (report.status !== "exported" || report.candidateModule !== candidate) {
      throw new Error("Training did not finish checkpoint export");
    }

    const human = await evaluateHumanCorpus({
      dist: join(repo, "packages/cron/dist"),
      partition: "regression",
      weights: candidate,
    });

    const regressionPath = join(directory, "human-regression.json");

    writeFileSync(regressionPath, `${JSON.stringify(human, null, 2)}\n`, {
      flag: "wx",
    });

    replaceJson(reportPath, {
      ...report,
      status: "complete",
      humanRegression: {
        report: regressionPath,
        overlayOff: human.overlayOff,
        positiveSilentMeaningChanges: human.positiveSilentMeaningChanges,
        promotionEvidence: false,
      },
      totalRunSeconds: (Date.now() - trainingStarted) / 1000,
    });

    console.log(
      `regression: H-pos ${human.overlayOff.positiveCorrect}/${human.overlayOff.positiveTotal}, ` +
        `H-negOK ${human.overlayOff.negativeCorrect}/${human.overlayOff.negativeTotal}, ` +
        `H-silent ${human.positiveSilentMeaningChanges}, FA@cov ${human.overlayOff.faAtCov}`,
    );

    console.log(`report -> ${reportPath}`);
    workflow.status = "complete";
    measure("after training and export");
  } catch (error) {
    workflow.status = "failed";
    workflow.error = String(error);
    throw error;
  } finally {
    measure("finished");
    workflow.elapsedSeconds = (Date.now() - started) / 1000;

    writeFileSync(join(directory, "workflow.json"), `${JSON.stringify(workflow, null, 2)}\n`, {
      flag: "wx",
    });
  }
}

if (import.meta.main) {
  if (process.argv.length !== 2) {
    throw new Error("Training takes no flags. Run from the repository root: bun run train");
  }

  await train();
}
