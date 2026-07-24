#!/usr/bin/env bash
# Ornis Engine — Browser Editor launcher
# Usage: ./scripts/run_editor.sh
# Opens the editor in your browser at http://127.0.0.1:3420

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WASM_CRATE="${PROJECT_ROOT}/crates/wasm"
WASM_OUT="${PROJECT_ROOT}/crates/ui/assets/editor/pkg"
EDITOR_URL="http://127.0.0.1:3420"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Ornis Engine — Browser Editor Launcher             ║"
echo "╠══════════════════════════════════════════════════════════════╣"

# ── Check wasm-pack ───────────────────────────────────────────────
if ! command -v wasm-pack &>/dev/null; then
    echo "║  wasm-pack not found. Installing...                          ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    if command -v cargo &>/dev/null; then
        cargo install wasm-pack
    else
        echo "║  ERROR: cargo not found. Install Rust first.                 ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        exit 1
    fi
fi

# ── Build WASM module ─────────────────────────────────────────────
echo "║  Step 1: Building WASM module...                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"

cd "${WASM_CRATE}"
wasm-pack build \
    --target web \
    --out-dir "${WASM_OUT}" \
    --no-typescript \
    --no-pack

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

cd "${PROJECT_ROOT}"
cargo run --features editor-only
