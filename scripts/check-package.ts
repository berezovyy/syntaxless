/**
 * Packaged-artifact check. Packs the workspace, asserts the file list, then
 * installs the tarball into a throwaway consumer and exercises parse/compile/schema.
 */
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

export function checkPackage(sourceRoot: string, CONSUMER: string, packageRoot = sourceRoot): void {
  interface PackageManifest {
    name: string;
    files: string[];
  }

  const manifest = JSON.parse(
    readFileSync(join(packageRoot, "package.json"), "utf8"),
  ) as PackageManifest;

  function walk(dir: string): string[] {
    const files: string[] = [];

    for (const name of readdirSync(dir)) {
      const path = join(dir, name);

      if (statSync(path).isDirectory()) files.push(...walk(path));
      else files.push(path);
    }

    return files;
  }

  function expectedPackedFiles(): Set<string> {
    const expected = new Set<string>(["package/package.json"]);

    for (const entry of manifest.files) {
      const path = join(packageRoot, entry);

      if (!existsSync(path)) throw new Error(`check:package: files entry missing: ${entry}`);

      if (statSync(path).isDirectory()) {
        for (const file of walk(path)) expected.add(`package/${relative(packageRoot, file)}`);
      } else {
        expected.add(`package/${entry}`);
      }
    }

    return expected;
  }

  function assertExactPackedFiles(tarball: string): void {
    const actual = new Set(
      execFileSync("tar", ["-tzf", tarball], { encoding: "utf8" })
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0 && !line.endsWith("/")),
    );

    const expected = expectedPackedFiles();
    const missing = [...expected].filter((file) => !actual.has(file)).sort();
    const extra = [...actual].filter((file) => !expected.has(file)).sort();

    if (missing.length > 0 || extra.length > 0) {
      throw new Error(
        [
          "check:package: packed file list mismatch",
          ...missing.map((file) => `  missing: ${file}`),
          ...extra.map((file) => `  unexpected: ${file}`),
        ].join("\n"),
      );
    }

    console.log(`check:package: ${actual.size} packed files verified`);
  }

  function resolveTsc(): string {
    const candidates = [
      join(sourceRoot, "node_modules", "typescript", "bin", "tsc"),
      join(sourceRoot, "..", "..", "node_modules", "typescript", "bin", "tsc"),
    ];

    const found = candidates.find((path) => existsSync(path));

    if (found === undefined) throw new Error("check:package: tsc not found");

    return found;
  }

  const CONSUMER_TSCONFIG = {
    compilerOptions: {
      target: "ES2023",
      module: "ESNext",
      moduleResolution: "Bundler",
      lib: ["ES2023", "DOM", "DOM.Iterable"],
      strict: true,
      noEmit: true,
      skipLibCheck: true,
      types: [],
    },
    include: ["consumer.ts"],
  };

  function runConsumer(tarball: string, root: string): void {
    const consumerDir = join(root, "consumer");

    mkdirSync(consumerDir);

    writeFileSync(
      join(consumerDir, "package.json"),
      `${JSON.stringify({ name: "package-consumer", private: true, type: "module" }, null, 2)}\n`,
    );

    writeFileSync(join(consumerDir, "consumer.ts"), CONSUMER);

    writeFileSync(
      join(consumerDir, "tsconfig.json"),
      `${JSON.stringify(CONSUMER_TSCONFIG, null, 2)}\n`,
    );

    // This package has no runtime dependencies. Extract the real tarball into
    // the consumer's package resolution path without a registry/cache write.
    const installed = join(consumerDir, "node_modules", ...manifest.name.split("/"));

    mkdirSync(installed, { recursive: true });
    execFileSync("tar", ["-xzf", tarball, "-C", installed, "--strip-components=1"]);
    execFileSync("bun", ["run", "consumer.ts"], { cwd: consumerDir, stdio: "inherit" });

    execFileSync(
      process.execPath,
      [resolveTsc(), "-p", "tsconfig.json", "--noEmit", "false", "--outDir", "js"],
      {
        cwd: consumerDir,
        stdio: "inherit",
      },
    );

    // Exercise the same offline tarball consumer in Node, not only Bun.
    execFileSync("node", [join(consumerDir, "js/consumer.js")], {
      cwd: consumerDir,
      stdio: "inherit",
    });
  }

  function main(): void {
    const root = mkdtempSync(join(tmpdir(), "syntaxless-package-pack-"));

    try {
      const packed = execFileSync(
        "bun",
        ["pm", "pack", "--destination", root, "--ignore-scripts", "--quiet"],
        { cwd: packageRoot, encoding: "utf8" },
      );

      const tarball = packed
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .pop();

      if (tarball === undefined) throw new Error("check:package: bun pm pack produced no tarball");

      assertExactPackedFiles(tarball);
      runConsumer(tarball, root);
      console.log("check:package: ok");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }

  main();
}
