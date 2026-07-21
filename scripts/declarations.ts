/** Publish only reachable declarations, keeping shared private types inside each tarball. */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";

export function emitDeclarations(packageRoot: string, destination: string): void {
  const temporary = mkdtempSync(join(tmpdir(), "syntaxless-declarations-"));

  const packagesRoot = resolve(packageRoot, "..");
  const domain = basename(packageRoot);

  try {
    execFileSync(
      process.execPath,
      [
        resolve(packagesRoot, "../node_modules/typescript/bin/tsc"),
        "-p",
        "tsconfig.build.json",
        "--emitDeclarationOnly",
        "--declarationMap",
        "false",
        "--sourceMap",
        "false",
        "--rootDir",
        packagesRoot,
        "--outDir",
        temporary,
      ],
      { cwd: packageRoot, stdio: "inherit" },
    );

    const entry = join(temporary, domain, "src/index.d.ts");
    const reachable = new Map<string, string>();

    const specifiers = /((?:from\s*|import\(\s*)["'])([^"']+)(["'])/g;

    function dependency(file: string, specifier: string): string | undefined {
      return specifier.startsWith(".") && specifier.endsWith(".js")
        ? resolve(dirname(file), specifier.slice(0, -3) + ".d.ts")
        : undefined;
    }

    function output(file: string): string {
      const local = relative(join(temporary, domain, "src"), file);

      return local.startsWith("..")
        ? join(destination, "_shared", relative(join(temporary, "shared"), file))
        : join(destination, local);
    }

    const pending = [entry];

    while (pending.length) {
      const file = pending.pop()!;

      if (reachable.has(file)) continue;
      reachable.set(file, output(file));

      for (const match of readFileSync(file, "utf8").matchAll(specifiers)) {
        const child = dependency(file, match[2]!);

        if (child) pending.push(child);
      }
    }

    for (const [file, target] of reachable) {
      const source = readFileSync(file, "utf8").replace(
        specifiers,
        (whole, before, specifier, after) => {
          const child = dependency(file, specifier);

          if (!child) return whole;

          const path = relative(dirname(target), reachable.get(child)!)
            .replaceAll("\\", "/")
            .replace(/\.d\.ts$/, ".js");

          return before + (path.startsWith(".") ? path : "./" + path) + after;
        },
      );

      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, source);
    }

    console.log(`build: ${reachable.size} reachable declaration files`);
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
}
