/**
 * Build the playground into one self-contained HTML file.
 *
 * Every package is bundled from source with its model, and the lazily imported
 * weight chunks are inlined, so `demo/dist/index.html` opens straight from disk
 * (file://) and works offline.
 */
import { join } from "node:path";

const root = import.meta.dir;

const result = await Bun.build({
  entrypoints: [join(root, "src", "main.ts")],
  format: "iife",
  target: "browser",
  minify: true,
});

if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

const script = (await result.outputs[0]!.text()).replaceAll("</script", "<\\/script");
const page = await Bun.file(join(root, "src", "page.html")).text();
const html = page.replace("<!-- app -->", () => `<script>${script}</script>`);
const out = join(root, "dist", "index.html");

await Bun.write(out, html);
console.log(`${out} (${(html.length / 1024).toFixed(0)} KiB)`);
