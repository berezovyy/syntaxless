#!/usr/bin/env python3
"""Draws the films' stage: a quiet pixel-art desert at dusk, 320x180 scaled x6 to 1920x1080.

A nod to the palette inside the card (ayu Mirage): mesas on a hazy horizon, a heat-shimmer band
that mirrors them, and dunes in the foreground. Seeded and Pillow-only, so it is reproducible:
    python3 shared/stage.py            # writes shared/stage.png
"""
import math
import random
import sys
from pathlib import Path

from PIL import Image

W, H, SCALE = 320, 180, 6
SEED = 11

# Muted dusk: slate blue overhead warming to a soft apricot horizon, plum mesas, sand dunes.
# The stage frames the card; it never competes with it.
P = {
    "sky0": (52, 58, 82),
    "sky1": (72, 72, 98),
    "sky2": (106, 86, 108),
    "sky3": (150, 104, 110),
    "sky4": (196, 132, 112),
    "sky5": (226, 168, 124),
    "sun": (246, 212, 160),
    "star": (150, 156, 186),
    "cloud": (176, 116, 116),
    "cloud_lit": (214, 150, 122),
    "mesa_far": (150, 102, 112),
    "mesa": (112, 78, 98),
    "mesa_lit": (138, 90, 104),
    "mesa_dark": (88, 62, 84),
    "haze": (208, 150, 122),
    "haze_dark": (176, 120, 114),
    "glint": (240, 196, 150),
    "sand": (178, 124, 92),
    "sand_lit": (204, 148, 104),
    "sand_dark": (140, 96, 82),
    "sand_deep": (110, 76, 74),
    "cactus": (84, 74, 70),
    "cactus_dark": (64, 56, 58),
    "pebble": (220, 170, 126),
}

BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]


def noise1d(rng, n, octaves):
    """Smooth 1D value noise in [0, 1], as a list of n samples."""
    out = [0.0] * n
    amp_total = 0.0
    for period, amp in octaves:
        knots = [rng.random() for _ in range(n // period + 3)]
        for x in range(n):
            i, f = divmod(x / period, 1)
            i = int(i)
            f = f * f * (3 - 2 * f)
            out[x] += amp * (knots[i] * (1 - f) + knots[i + 1] * f)
        amp_total += amp
    return [v / amp_total for v in out]


def main(out_path):
    rng = random.Random(SEED)
    img = Image.new("RGB", (W, H))
    px = img.load()

    def put(x, y, colour):
        if 0 <= x < W and 0 <= y < H:
            px[x, y] = colour

    # Sky: close bands from slate to apricot; each edge gets two rows of sparse dither.
    keys = ["sky0", "sky1", "sky2", "sky3", "sky4", "sky5"]
    bands = []
    for i in range(9):
        t = i / 8 * (len(keys) - 1)
        k = min(int(t), len(keys) - 2)
        a, b = P[keys[k]], P[keys[k + 1]]
        f = t - k
        bands.append(tuple(round(a[c] + (b[c] - a[c]) * f) for c in range(3)))
    horizon = 108
    step = 12
    for y in range(horizon + 1):
        band, offset = divmod(y, step)
        band = min(band, 8)
        for x in range(W):
            colour = bands[band]
            if band < 8 and offset == step - 2 and BAYER[y % 4][x % 4] < 4:
                colour = bands[band + 1]
            if band < 8 and offset == step - 1 and BAYER[y % 4][x % 4] < 12:
                colour = bands[band + 1]
            px[x, y] = colour

    # A few early stars in the top bands.
    for _ in range(18):
        x, y = rng.randrange(W), rng.randrange(2, 30)
        put(x, y, P["star"])

    # A large, low sun sitting on the haze, with a dithered halo.
    sx, sy, sr = 96, 96, 13
    for y in range(sy - sr - 4, horizon + 1):
        for x in range(sx - sr - 4, sx + sr + 5):
            d = math.hypot(x - sx, y - sy)
            if d <= sr:
                put(x, y, P["sun"])
            elif d <= sr + 3 and (x + y) % 2 == 0:
                put(x, y, P["sun"])

    # Long, thin dusk clouds: flat strokes with a lit underside.
    def streak(cx, cy, length, rows):
        for r in range(rows):
            half = length // 2 - r * 5
            for x in range(cx - half + r * 3, cx + half - r * 2):
                put(x, cy + r, P["cloud_lit"] if r == rows - 1 else P["cloud"])

    streak(60, 40, 64, 2)
    streak(210, 28, 90, 3)
    streak(282, 56, 52, 2)
    streak(150, 62, 38, 1)

    # Mesas: flat-topped buttes with sheer sides, a far hazy row and a nearer row.
    def mesa(x0, x1, top, colour, lit, shade, ledge=None):
        for x in range(x0, x1 + 1):
            # Sheer sides with a stepped talus at the foot.
            edge = min(x - x0, x1 - x)
            y_top = top if edge > 2 else top + (3 - edge) * 2
            for y in range(y_top, horizon + 1):
                c = colour
                if x - x0 < 4:
                    c = lit
                elif x1 - x < 6:
                    c = shade
                if ledge and y == ledge and (x + y) % 2 == 0:
                    c = shade
                put(x, y, c)
        # Talus: dithered skirts spreading out at the base.
        for y in range(horizon - 10, horizon + 1):
            spread = (y - horizon + 10) // 2
            for x in list(range(x0 - spread, x0)) + list(range(x1 + 1, x1 + spread + 1)):
                if BAYER[y % 4][x % 4] < 9:
                    put(x, y, shade)

    far = noise1d(rng, W, [(30, 1.0), (9, 0.3)])
    for x in range(W):
        for y in range(int(98 + (1 - far[x]) * 8), horizon + 1):
            px[x, y] = P["mesa_far"]
    mesa(170, 214, 80, P["mesa"], P["mesa_lit"], P["mesa_dark"])
    mesa(226, 238, 90, P["mesa"], P["mesa_lit"], P["mesa_dark"])
    mesa(18, 50, 86, P["mesa"], P["mesa_lit"], P["mesa_dark"])
    mesa(268, 300, 84, P["mesa"], P["mesa_lit"], P["mesa_dark"])

    # The mirage: a band of heat haze that mirrors the horizon, broken by shimmer lines.
    haze_top, haze_bottom = horizon + 1, 124
    for y in range(haze_top, 150):
        for x in range(W):
            src = 2 * horizon - y + 1
            c = P["haze"]
            if 0 <= src < H and px[x, src] in (P["mesa"], P["mesa_lit"], P["mesa_dark"], P["mesa_far"]):
                c = P["haze_dark"]
            px[x, y] = c
    for y in range(haze_top + 1, haze_bottom, 3):
        for x in range(W):
            if (x // 5 + y) % 3 == 0:
                px[x, y] = P["haze"]
    for y in range(haze_top + 1, haze_bottom - 1, 2):
        depth = (y - haze_top) / (haze_bottom - haze_top)
        half = int(sr * (1 - depth * 0.6))
        jitter = ((y * 5) % 5) - 2
        for x in range(sx - half + jitter, sx + half + jitter + 1):
            if (x + y // 2) % 6 != 0:
                put(x, y, P["glint"])

    # Dunes: overlapping crests, lit on the windward side, with dithered troughs.
    dunes = [
        (noise1d(rng, W, [(90, 1.0), (30, 0.25)]), 122, 12),
        (noise1d(rng, W, [(70, 1.0), (24, 0.3)]), 136, 14),
        (noise1d(rng, W, [(110, 1.0), (36, 0.2)]), 152, 16),
    ]
    for i, (n, base, amp) in enumerate(dunes):
        crest = [int(base + (1 - v) * amp) for v in n]
        for x in range(W):
            slope = crest[min(W - 1, x + 1)] - crest[max(0, x - 1)]
            for y in range(crest[x], H):
                depth = y - crest[x]
                c = P["sand_lit"] if slope > 0 and depth < 6 else P["sand"]
                if slope < 0 and depth < 5:
                    c = P["sand_dark"]
                if depth > 10 and BAYER[y % 4][x % 4] < min(15, (depth - 10)):
                    c = P["sand_dark"] if i < 2 else P["sand_deep"]
                px[x, y] = c
            # The crest line itself: one pixel of light.
            put(x, crest[x], P["sand_lit"])

    # Pebbles and ripples in the near sand.
    for _ in range(60):
        x = rng.randrange(W)
        y = rng.randrange(150, H)
        put(x, y, P["pebble"] if rng.random() < 0.5 else P["sand_dark"])
    for y in range(158, H, 4):
        for x in range(W):
            if (x + y * 3) % 23 < 4:
                put(x, y, P["sand_dark"])

    # Saguaros frame the left and right edges.
    def saguaro(bx, by, h, arms):
        for y in range(by - h, by + 1):
            for x in (bx - 1, bx, bx + 1):
                put(x, y, P["cactus_dark"] if x == bx + 1 else P["cactus"])
        put(bx - 1, by - h, P["cactus_dark"])
        put(bx + 1, by - h, P["cactus_dark"])
        for side, at, up in arms:
            ax = bx + side * 3
            for x in range(min(bx, ax), max(bx, ax) + 1):
                put(x, by - at, P["cactus"])
            for y in range(by - at - up, by - at + 1):
                put(ax, y, P["cactus"])
                put(ax + side, y, P["cactus_dark"])

    saguaro(10, 172, 40, [(-1, 18, 10), (1, 26, 9)])
    saguaro(30, 168, 24, [(1, 12, 6)])
    saguaro(304, 174, 46, [(1, 20, 12), (-1, 28, 8)])
    saguaro(284, 166, 20, [(-1, 9, 5)])

    img.resize((W * SCALE, H * SCALE), Image.NEAREST).save(out_path, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("stage.png"))
