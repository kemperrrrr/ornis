#!/usr/bin/env bash
# Ornis Engine — Browser Editor launcher (no wasm-pack)
# Fallback: uses cargo + wasm-bindgen-cli instead of wasm-pack
# Usage: ./scripts/run_editor_cargo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WASM_CRATE="${PROJECT_ROOT}/crates/wasm"
WASM_TARGET="wasm32-unknown-unknown"
WASM_OUT="${PROJECT_ROOT}/crates/ui/assets/editor/pkg"
EDITOR_URL="http://127.0.0.1:3420"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Ornis Engine — Browser Editor (cargo-only build)         ║"
echo "╠══════════════════════════════════════════════════════════════╣"

# ── Check wasm-bindgen-cli ────────────────────────────────────────
if ! command -v wasm-bindgen &>/dev/null; then
    echo "║  wasm-bindgen-cli not found. Installing...                   ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    cargo install wasm-bindgen-cli
fi

# ── Add wasm target if missing ────────────────────────────────────
if ! rustup target list --installed | grep -q "${WASM_TARGET}"; then
    echo "║  Adding ${WASM_TARGET} target...                              ║"
    rustup target add "${WASM_TARGET}"
fi

# ── Build WASM module ─────────────────────────────────────────────
echo "║  Step 1: Building WASM module via cargo...                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"

cd "${PROJECT_ROOT}"
cargo build --target "${WASM_TARGET}" -p ornis-wasm --release

# Locate the built .wasm
WASM_FILE="${PROJECT_ROOT}/target/${WASM_TARGET}/release/ornis_wasm.wasm"
if [ ! -f "${WASM_FILE}" ]; then
    WASM_FILE="${PROJECT_ROOT}/target/${WASM_TARGET}/release/ornis-wasm.wasm"
fi

mkdir -p "${WASM_OUT}"
wasm-bindgen "${WASM_FILE}" \
    --out-dir "${WASM_OUT}" \
    --target web \
    --no-typescript

echo "║  ✓ WASM built → crates/ui/assets/editor/pkg/                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"

# ── Run editor server ─────────────────────────────────────────────
echo "║  Step 2: Starting editor server...                           ║"
echo "║                                                              ║"
echo "║  Editor:  ${EDITOR_URL}                             ║"
echo "║  Status:  ${EDITOR_URL}/api/status                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Press Ctrl+C to stop                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cargo run --features editor-only
