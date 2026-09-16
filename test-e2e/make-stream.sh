#!/usr/bin/env bash
#
# Generates a small multi-bitrate HLS stream for the end-to-end browser test.
#
#   ./make-stream.sh [outdir] [h264|vp9]
#
# Six renditions whose BANDWIDTH values sit close to the training ladder in
# src/env.py ([300, 750, 1200, 1850, 2850, 4300] kbps), 24 s of video in
# 4-second segments (the research env's VIDEO_CHUNK_LEN).
#
# Codec choice:
#   h264 (default) — MPEG-TS segments, H.264 + AAC. What every shipping browser
#                    plays, and what you want if you are testing in your own
#                    Chrome/Firefox/Safari.
#   vp9            — fMP4 (CMAF) segments, VP9 + Opus. Chromium builds without
#                    proprietary codecs (Playwright's bundled Chromium, most
#                    Linux distro chromium packages) cannot decode H.264/AAC at
#                    all, so CI uses this variant.
#
set -euo pipefail

OUT="${1:-$(dirname "$0")/stream}"
CODEC="${2:-h264}"

rm -rf "$OUT"; mkdir -p "$OUT"

DUR=24
# height:video-bitrate(kbps)
LADDER=("180:300" "270:750" "360:1200" "480:1850" "540:2850" "720:4300")

VARIANTS=""
for i in "${!LADDER[@]}"; do
  IFS=: read -r H BR <<< "${LADDER[$i]}"
  W=$(( H * 16 / 9 )); W=$(( W - W % 2 ))

  if [ "$CODEC" = "vp9" ]; then
    ffmpeg -y -loglevel error \
      -f lavfi -i "testsrc2=size=${W}x${H}:rate=25:duration=${DUR}" \
      -f lavfi -i "sine=frequency=440:duration=${DUR}" \
      -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -row-mt 1 \
      -g 100 -keyint_min 100 \
      -b:v "${BR}k" -maxrate "${BR}k" -bufsize "$((BR*2))k" \
      -c:a libopus -b:a 64k -ac 1 \
      -f hls -hls_time 4 -hls_playlist_type vod -hls_list_size 0 \
      -hls_segment_type fmp4 \
      -hls_fmp4_init_filename "v${i}_init.mp4" \
      -hls_segment_filename "$OUT/v${i}_%03d.m4s" \
      "$OUT/v${i}.m3u8"
    CODECS="vp09.00.10.08,opus"
  else
    ffmpeg -y -loglevel error \
      -f lavfi -i "testsrc2=size=${W}x${H}:rate=25:duration=${DUR}" \
      -f lavfi -i "sine=frequency=440:duration=${DUR}" \
      -c:v libx264 -preset ultrafast -tune zerolatency \
      -g 100 -keyint_min 100 -sc_threshold 0 \
      -b:v "${BR}k" -maxrate "${BR}k" -bufsize "$((BR*2))k" \
      -c:a aac -b:a 64k -ac 1 \
      -f hls -hls_time 4 -hls_playlist_type vod -hls_list_size 0 \
      -hls_segment_filename "$OUT/v${i}_%03d.ts" \
      "$OUT/v${i}.m3u8"
    CODECS="avc1.42c01e,mp4a.40.2"
  fi

  BW=$(( (BR + 64) * 1000 ))
  VARIANTS+="#EXT-X-STREAM-INF:BANDWIDTH=${BW},RESOLUTION=${W}x${H},CODECS=\"${CODECS}\"\nv${i}.m3u8\n"
done

printf '#EXTM3U\n#EXT-X-VERSION:6\n%b' "$VARIANTS" > "$OUT/master.m3u8"
echo "✓ HLS master ($CODEC) → $OUT/master.m3u8"
du -sh "$OUT"
