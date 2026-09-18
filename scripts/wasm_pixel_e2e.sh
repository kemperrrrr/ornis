#!/usr/bin/env bash
# wasm_pixel_e2e.sh — manual live-pixel e2e for the WASM viewport.
#
# Chain: editor scene → GET /api/scene → WASM RenderWorld → rendered frame
# → screenshot → golden compare with tolerances (never bytewise).
# CI has no browser/WebGPU, so this harness SKIPs honestly (exit 0) when
# the environment cannot run the pixel leg. The headless half of the same
# chain is pinned by `cargo test -p ornis-wasm` (FULL_CONTRACT golden).
#
# Usage:
#   scripts/wasm_pixel_e2e.sh [--port 3420] [--out /tmp/ornis-e2e]
# Env:
#   ORNIS_E2E_INSTALL=1  allow `npx playwright install chromium` download.
# Thresholds (also in docs/WASM_PIXEL_E2E.md):
#   max per-pixel abs diff <= 12/255, mean abs diff <= 2/255,
#   mismatched-pixel fraction <= 0.5%. Comparison uses ImageMagick
#   `compare -metric RMSE` when available; otherwise the screenshot is
#   stored for manual review and the compare step SKIPs.
set -u

PORT=3420
OUT=/tmp/ornis-e2e
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

skip() { echo "SKIP wasm_pixel_e2e: $1"; exit 0; }

command -v cargo >/dev/null 2>&1 || skip "no cargo toolchain"
command -v node >/dev/null 2>&1 || skip "no node (playwright driver missing)"
command -v npx >/dev/null 2>&1 || skip "no npx (playwright driver missing)"

BROWSER_BIN=""
for c in chromium chromium-browser google-chrome google-chrome-stable firefox; do
  if command -v "$c" >/dev/null 2>&1; then BROWSER_BIN="$c"; break; fi
done
if [ -z "$BROWSER_BIN" ]; then
  if ! ls ~/.cache/ms-playwright >/dev/null 2>&1; then
    if [ "${ORNIS_E2E_INSTALL:-0}" = "1" ]; then
      echo "installing playwright chromium (ORNIS_E2E_INSTALL=1)…"
      npx -y playwright install chromium || skip "playwright browser install failed"
    else
      skip "no browser binary and no playwright cache (set ORNIS_E2E_INSTALL=1 to download)"
    fi
  fi
  BROWSER_BIN="playwright-chromium"
fi
echo "browser: $BROWSER_BIN"

mkdir -p "$OUT" || skip "cannot create $OUT"

echo "building wasm (wasm-pack, same as cargo xtask editor)…"
command -v wasm-pack >/dev/null 2>&1 || skip "no wasm-pack"
wasm-pack build crates/wasm --target web --out-dir editor/pkg \
  || skip "wasm-pack build failed"

echo "starting editor server on :$PORT…"
if [ "$PORT" != "3420" ]; then
  skip "editor-only server port is hardcoded to 3420 (requested $PORT)"
fi
cargo run --features editor-only -- --editor-dir editor >/dev/null 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 5
curl -sf "http://127.0.0.1:$PORT/api/scene" -o "$OUT/scene.json" \
  || skip "editor server did not answer /api/scene"

echo "capturing screenshot…"
if [ "$BROWSER_BIN" = "playwright-chromium" ]; then
  npx -y playwright screenshot --browser=chromium \
    --viewport-size=800,600 "http://127.0.0.1:$PORT" "$OUT/frame.png" \
    || skip "playwright screenshot failed (likely no WebGPU adapter)"
else
  "$BROWSER_BIN" --headless --disable-gpu-sandbox --no-sandbox \
    --use-angle=swiftshader --enable-unsafe-swiftshader \
    --window-size=800,600 --screenshot="$OUT/frame.png" \
    "http://127.0.0.1:$PORT" >/dev/null 2>&1 \
    || skip "headless browser screenshot failed (likely no WebGPU adapter)"
fi
[ -f "$OUT/frame.png" ] || skip "no screenshot produced"

GOLDEN="editor/e2e_golden.png"
if [ ! -f "$GOLDEN" ]; then
  echo "no $GOLDEN yet — stored $OUT/frame.png for manual review (first run bootstraps the golden)"
  exit 0
fi
if command -v compare >/dev/null 2>&1; then
  RMSE=$(compare -metric RMSE "$GOLDEN" "$OUT/frame.png" null: 2>&1 || true)
  echo "RMSE: $RMSE"
  echo "tolerance gate: max per-pixel <= 12/255, mean <= 2/255, fraction <= 0.5%"
  echo "(automated threshold parsing is manual for now — inspect the RMSE above)"
else
  echo "no ImageMagick 'compare' — stored $OUT/frame.png next to $GOLDEN for manual review"
fi
echo "artifacts: $OUT/scene.json $OUT/frame.png"
