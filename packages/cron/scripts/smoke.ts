// Keep the bare-clock regression separate from the explicit morning reference.
const { cron }: typeof import("../src/index.js") = await import(
  new URL("../dist/index.js", import.meta.url).href
);

const result = await cron("every weekday at 9");

if (!result.supported || result.expression !== "0 9 * * 1-5") {
  throw new Error(`smoke: expected a weekday schedule at 09:00, got ${JSON.stringify(result)}`);
}

console.log("smoke: ok");
