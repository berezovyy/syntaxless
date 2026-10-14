# Playground

One HTML page with a tab for every package. Each tab runs the real bundled
model in the browser, so you can type instructions and see the intent, the
compiled output and the refusals.

```sh
bun run demo
```

This bundles every package from source, models included, into
`demo/dist/index.html`. Open that file directly; it needs no server and makes no
network requests.
