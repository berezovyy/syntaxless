import { resolve, sep } from "node:path";

/** Check the built package in real Chromium using only local GET requests. */
import { chromium } from "playwright";

const directory = resolve(process.argv[2] ?? resolve(import.meta.dir, "../dist"));
const requests: string[] = [];

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/") {
      return new Response("<!doctype html><title>Sort browser check</title>", {
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
    throw new Error("entry import eagerly loaded model");
  }

  const context = {
    defaultNulls: "last",
    tieBreaker: "id",
    fields: [
      {
        id: "created",
        type: "number",
        column: "created",
        orderingAliases: [{ text: "newest", direction: "desc" }],
      },
      {
        id: "price",
        type: "number",
        column: "price",
        aliases: ["prices"],
        orderingAliases: [{ text: "cheapest", direction: "asc" }],
      },
      { id: "id", type: "number", column: "id" },
    ],
  };

  const initial = await page.evaluate(
    async ({ entry, context }) => {
      const api = await import(entry);

      const [result] = await Promise.all([
        api.parse("newest first, then cheapest, missing prices last", context),
        api.preload(),
      ]);

      if (!result.ok) throw new Error("browser parse failed: " + JSON.stringify(result));
      if (!api.compile(result.intent, context).supported) throw new Error("browser compile failed");

      return {
        intent: result.intent,
        order: api
          .sortRecords(
            [
              { id: 1, created: 2, price: 20 },
              { id: 2, created: 2, price: 10 },
              { id: 3, created: 2, price: 10 },
            ],
            result.intent,
            context,
          )
          .map((r: { id: number }) => r.id),
      };
    },
    { entry: "/index.js", context },
  );

  if (
    JSON.stringify(initial.order) !== "[2,3,1]" ||
    initial.intent.keys[1]?.nullsSource !== "explicit"
  ) {
    throw new Error("browser priority/nulls mismatch");
  }

  if (requests.filter((path) => path.endsWith("weights.gen.js")).length !== 1) {
    throw new Error("concurrent model loads were not shared");
  }

  await page.context().setOffline(true);

  const offline = await page.evaluate(
    async ({ entry, context }) => {
      const api = await import(entry);

      return {
        parsed: await api.parse("price descending", context),
        rejected: await api.parse("newest first?", context),
      };
    },
    { entry: "/index.js", context },
  );

  if (!offline.parsed.ok || offline.rejected.ok) throw new Error("offline browser parsing failed");

  const coverage = "lazy weights, concurrent initialization, offline parsing";

  console.log(`browser: passed (${requests.length} local GET requests; ${coverage})`);
} finally {
  await browser?.close();
  server.stop(true);
}
