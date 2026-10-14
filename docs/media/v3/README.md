# README films, v3

The README films staged like product shots: one window on a pixel-art desert at dusk, sub-cards on
a grid inside it, colour only where it means something, and a hand on the mouse. Every film is a
[HyperFrames](https://github.com/heygen-com/hyperframes) composition: plain HTML and CSS driven by
one paused GSAP timeline, rendered frame by frame, so every render is identical.

| Film           | Shows                                                                                  |
| :------------- | :------------------------------------------------------------------------------------- |
| `hero`         | one browser app using all seven packages, offline after the first load, with each size |
| `how-it-works` | tokenize, tag, resolve, compile on real cron model output, and a refusal               |
| `cron`         | schedules to cron expressions and next runs, and what cron cannot express              |
| `validation`   | field rules to JSON Schema, and a refused input                                        |
| `filter`       | an English query to SQLite parameters and matching rows, edited and re-run             |
| `sort`         | an ordering in English re-sorting a live table                                         |
| `retry`        | a refusal for a missing value, then a bounded delay schedule                           |
| `retention`    | a retention policy previewed over 420 daily backups                                    |
| `format`       | one format intent compiled for two locales                                             |

## Render

Requires Node.js 22+, Python 3 with Pillow, and FFmpeg with `libwebp` and `libx264`.

```sh
./render.sh            # all films
./render.sh hero cron  # some films
```

Each film renders at 240fps (a streamed ProRes 4444 master, since 240fps PNG frames would need
tens of gigabytes of scratch space), and every four frames are averaged into one 60fps frame in
`renders/` (ignored by Git). That is motion blur, the way a camera shutter makes it; without it,
anything that travels strobes and the film looks choppy at 60fps. The 60fps frames then become a
looping 1920×1080 lossless animated WebP in `films/`, and an MP4 in `renders/` for sharing.
`tools/verify.py` then checks the frame count, the frame durations and that the last frame
matches the first. Keep the WebP lossless: lossy WebP leaves blocky trails on slow fades. The
WebP keeps an alpha channel: outside the stage's rounded corners the film is transparent, so it
sits on any page colour (the MP4 shows those corners as black).

Before rendering, look at stills: `npx hyperframes@0.8.150 snapshot hero --at 3,9.5`. To print a
film's key timestamps without rendering, run `node tools/probe.cjs hero` with `PUPPETEER` pointing
at a `puppeteer-core` install and `CHROME` at a Chrome binary.

## Conventions

- Every value on screen is the packages' real output, run with Bun, and every size is the built
  package at `gzip -9`, model included. Keep them in sync when the packages change.
- The stage is `shared/stage.png`, drawn by `shared/stage.py` (Pillow, seeded): a 320×180
  pixel-art desert scaled ×6 with hard edges. Edit the script and rerun it rather than editing
  the PNG. It sits outside `#scene`, so nothing a film does can fade it.
- One card, sub-cards on a grid: only the card has a title bar, window controls and a shadow.
  Inside it, flat sub-cards with 24px padding and 20px gutters, aligned, never overlapping. Text
  never sits on the art.
- Inside the card the palette is ayu Mirage, nearly monochrome: green for what matches or is
  kept, red for what fails or is removed, amber for the text cursor, and coloured window
  controls once the card has focus.
- Focus by contrast: a thin outline on the sub-card in play, the others' contents dimmed when it
  matters. No glows, bounces or scale pops.
- The pointer (`SX.pointer`, moved with `SX.hand`) does what a person would: clicks a field
  before typing in it, clicks tabs and the throttling menu, points at what the beat is about,
  rests out of the way, and is back at rest before the loop ends.
- Everything that changes per frame (typing, carets, counters) is computed from the current time
  in one driver tween (`SX.drive`); typing uses a seeded RNG, so its cadence looks human and
  renders the same every time.
- Motion is opacity and short moves: things open in place with a fade and a 14px rise, and new
  content appears after what it replaces has gone. Curves stay gentle (`sx.out`, `sx.in`,
  `sx.move`, `sx.swap` in `shared/sx.js`), and long travels get more time.
- Every film ends on its first frame so the WebP loops without a jump. Reset by dipping the
  sub-cards' contents, never whole windows: lossless WebP pays for every pixel that changes.

JetBrains Mono is bundled under the SIL Open Font License (`shared/fonts/OFL-JetBrainsMono.txt`).
