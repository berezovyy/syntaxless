#!/bin/sh
# Renders the README films, v3: a 240fps PNG sequence per film (HyperFrames), blended down to
# 60fps, then a looping lossless animated WebP at full 1920x1080 for inline playback on GitHub,
# plus an MP4 for sharing. Full size matters: on high-density screens GitHub draws the film about
# 1,800 pixels wide, and anything smaller gets stretched and soft.
# Usage: ./render.sh [film...]   (default: all films)
#
# Each 60fps frame averages the four 240fps frames it spans, the way a camera shutter does.
# Without that motion blur, anything that travels more than a few pixels per frame strobes, and
# the film reads as choppy even at 60fps.
#
# The 240fps master is a ProRes 4444 MOV with alpha: HyperFrames streams it to disk instead of
# storing 8,400 PNG frames (a 35s film needs about 70 GB of PNG scratch space). After that every
# step is lossless up to the WebP on purpose: lossy WebP stores mostly partial frames, and on slow
# fades of low-contrast UI those leave blocky trails.
set -eu

cd "$(dirname "$0")"
films=${*:-"hero how-it-works cron validation filter sort retry retention format"}
masters=${MASTERS:-renders}
out=films
mkdir -p "$masters" "$out"

for film in $films; do
    sub="$masters/$film-240.mov"
    frames="$masters/$film-frames"
    rm -rf "$sub" "$frames"
    # One worker streams frames straight into the encoder; more workers store them on disk first.
    npx --yes hyperframes@0.8.150 render "$film" --fps 240 --format mov --workers 1 --quiet -o "$sub"
    test -s "$sub"
    mkdir -p "$frames"
    ffmpeg -v error -y -i "$sub" \
        -vf "tmix=frames=4,select='eq(mod(n\,4)\,3)',setpts=N/(60*TB)" -r 60 \
        -pix_fmt rgba "$frames/frame_%06d.png"
    rm -f "$sub"
    ffmpeg -v error -y -framerate 60 -i "$frames/frame_%06d.png" \
        -vf "format=bgra" \
        -c:v libwebp_anim -lossless 1 -compression_level 4 -loop 0 \
        "$out/$film.webp"
    ffmpeg -v error -y -framerate 60 -i "$frames/frame_%06d.png" \
        -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p -movflags +faststart \
        "$masters/$film.mp4"
    if python3 tools/verify.py "$out/$film.webp" "$frames"; then
        rm -rf "$frames"
    else
        failed="${failed:-} $film" # keep its frames for diagnosis, render the rest
    fi
    echo "$film: $(du -h "$out/$film.webp" | cut -f1)"
done
if [ -n "${failed:-}" ]; then
    echo "failed checks:$failed"
    exit 1
fi
