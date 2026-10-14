#!/bin/sh
# Quick loop-seam check under the same GPU capture as a real render: renders a film at 4fps
# and compares its first frame with the last one. Usage: tools/seamtest.sh hero
set -eu
cd "$(dirname "$0")/.."
out="renders/seamtest-$1"
mkdir -p renders
npx --yes hyperframes@0.8.150 render "$1" --fps 4 --format png-sequence --workers 1 --quiet -o "$out" >/dev/null 2>&1
python3 - "$out" <<'PY'
import sys, glob
from PIL import Image, ImageChops
f = sorted(glob.glob(sys.argv[1] + "/frame_*.png"))
flat = lambda p: Image.alpha_composite(Image.new("RGBA", (1920, 1080), (24, 26, 32, 255)), Image.open(p).convert("RGBA")).convert("RGB")
d = ImageChops.difference(flat(f[0]), flat(f[-1]))
print(len(f), "frames; seam", d.convert("L").point(lambda v: 255 if v > 6 else 0).getbbox(), "max", max(h for _, h in d.getextrema()))
PY
