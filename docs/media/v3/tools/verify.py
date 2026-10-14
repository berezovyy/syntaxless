#!/usr/bin/env python3
"""Checks a rendered film: frame count, frame durations and a seamless loop.

    python3 tools/verify.py films/hero.webp renders/hero-frames

The WebP must last as long as the 60fps sequence (the encoder merges identical frames into longer
ones, so it counts time, not frames), no frame may be shorter than 16 ms, and the last rendered
frame must equal the first so the loop closes without a jump. Frames are composited onto a solid
colour before comparing: transparent pixels can hold arbitrary RGB. Needs webpmux (libwebp).
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops

LOOP_TOLERANCE = 6  # max per-channel difference: blending noise, not a visible jump


def flat(img):
    base = Image.new("RGBA", img.size, (24, 26, 32, 255))
    return Image.alpha_composite(base, img.convert("RGBA")).convert("RGB")


def durations(webp):
    """Frame durations in ms, read with webpmux (Pillow does not report them per frame)."""
    info = subprocess.run(["webpmux", "-info", webp], capture_output=True, text=True, check=True).stdout
    rows = [line.split() for line in info.splitlines() if line.strip()[:1].isdigit() and ":" in line.split()[0]]
    return [int(row[6]) for row in rows]


def main(webp, frames_dir):
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    expected_ms = len(frames) * 1000 / 60
    ms = durations(webp)
    first = flat(Image.open(frames[0]))
    last = flat(Image.open(frames[-1]))
    problems = []
    if abs(sum(ms) - expected_ms) > 17:
        problems.append(f"WebP lasts {sum(ms)} ms, {len(frames)} frames at 60fps last {expected_ms:.0f} ms")
    if min(ms) < 16:
        problems.append(f"shortest frame lasts {min(ms)} ms")
    diff = ImageChops.difference(first, last)
    worst = max(hi for _, hi in diff.getextrema())
    box = diff.convert("L").point(lambda v: 255 if v > LOOP_TOLERANCE else 0).getbbox()
    if box:
        problems.append(f"loop seam: pixels differ by more than {LOOP_TOLERANCE} in {box} (max {worst})")
    print(f"{webp}: {len(ms)} stored frames, {sum(ms) / 1000:.2f} s, {min(ms)}-{max(ms)} ms per frame, seam max diff {worst}")
    if problems:
        print("  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
