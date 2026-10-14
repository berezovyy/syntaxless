// Prints a film's key timestamps (window.__marks) without rendering it.
// Usage: node tools/probe.cjs hero
const path = require("path");
const puppeteer = require(process.env.PUPPETEER || "puppeteer-core");
(async () => {
  const film = process.argv[2] || "hero";
  const browser = await puppeteer.launch({ executablePath: process.env.CHROME, args: ["--no-sandbox", "--allow-file-access-from-files"] });
  const page = await browser.newPage();
  page.on("console", (m) => console.log(m.text()));
  page.on("pageerror", (e) => console.log("pageerror", e.message));
  await page.evaluateOnNewDocument(() => (window.__timelines = {}));
  await page.goto("file://" + path.resolve(__dirname, "..", film, "index.html"));
  await new Promise((r) => setTimeout(r, 500));
  await browser.close();
})();
