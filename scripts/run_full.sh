#!/usr/bin/env bash
# Ornis Engine — Full Native Engine launcher
# Usage: ./scripts/run_full.sh
# Opens native winit window + remote editor on port 3420

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EDITOR_URL="http://127.0.0.1:3420"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Ornis Engine — Full Native Launcher                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Native window + HTTP editor server                          ║"
echo "║  Editor:  ${EDITOR_URL}                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Press Ctrl+C to stop                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cd "${PROJECT_ROOT}"
cargo run
