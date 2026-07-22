import { resolve, sep } from "node:path";

/** Exercise the built package in Chromium, including lazy loading and offline use. */
import { chromium } from "playwright";

const directory = resolve(process.argv[2] ?? resolve(import.meta.dir, "../dist"));
const requests: string[] = [];

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/") {
      return new Response("<!doctype html><title>Validation browser check</title>", {
        headers: { "Content-Type": "text/html" },
      });
    }

    const path = resolve(directory, `.${url.pathname}`);

    if (!path.startsWith(directory + sep)) return new Response(null, { status: 403 });
    const file = Bun.file(path);

    return (await file.exists()) ? new Response(file) : new Response(null, { status: 404 });
  },
});

const origin = `http://127.0.0.1:${server.port}`;
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  await page.route("**/*", async (route) => {
    const request = route.request();

    if (
      new URL(request.url()).origin !== origin ||
      request.method() !== "GET" ||
      request.postData() !== null
    ) {
      throw new Error("unexpected browser request");
    }

    requests.push(new URL(request.url()).pathname);
    await route.continue();
  });

  await page.goto(origin);

  await page.evaluate(async (entry) => {
    await import(entry);
  }, "/index.js");

  if (requests.some((path) => path.endsWith("weights.gen.js"))) {
    throw new Error("entry import eagerly loaded weights");
  }

  const initial = await page.evaluate(async (entry) => {
    const api = await import(entry);

    const [parsed, schema] = await Promise.all([
      api.parse("required email with max 100 characters"),
      api.schema("optional number more than -2.75 and at most 6.5"),
      api.preload(),
    ]);

    return { parsed, schema };
  }, "/index.js");

  if (
    !initial.parsed.ok ||
    initial.parsed.source !== "model" ||
    JSON.stringify(initial.parsed.intent) !==
      JSON.stringify({ type: "email", required: true, maxLength: 100 }) ||
    !initial.schema.supported ||
    initial.schema.schema.exclusiveMinimum !== -2.75 ||
    initial.schema.schema.maximum !== 6.5
  ) {
    throw new Error(`browser meaning mismatch: ${JSON.stringify(initial)}`);
  }

  if (requests.filter((path) => path.endsWith("weights.gen.js")).length !== 1) {
    throw new Error("concurrent operations did not share one model load");
  }

  await page.context().setOffline(true);

  const offline = await page.evaluate(async (entry) => {
    const api = await import(entry);

    return {
      parsed: await api.parse("required email with max 100 characters"),
      rejected: await api.schema("required string around 10 characters"),
    };
  }, "/index.js");

  if (!offline.parsed.ok || offline.rejected.supported || offline.rejected.intent !== null) {
    throw new Error(`offline browser mismatch: ${JSON.stringify(offline)}`);
  }

  const coverage = "lazy weights, concurrent initialization, offline parsing";

  console.log(`browser: passed (${requests.length} local GET requests; ${coverage})`);
} finally {
  await browser?.close();
  server.stop(true);
}
