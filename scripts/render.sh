#!/bin/bash
# Render Ornis editor with the selected backend.
# Usage: ./render.sh [--blitz|--ornis] [width] [height] [output.png]

set -e
cd "$(dirname "$0")/.."

BACKEND="--ornis"
WIDTH=1920
HEIGHT=1080
OUTPUT="/tmp/render_comparison.png"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --blitz|--ornis) BACKEND="$1"; shift ;;
        *) 
            if [ -z "${WIDTH_SET+x}" ]; then
                WIDTH="$1"; WIDTH_SET=1
            elif [ -z "${HEIGHT_SET+x}" ]; then
                HEIGHT="$1"; HEIGHT_SET=1
            else
                OUTPUT="$1"
            fi
            shift ;;
    esac
done

if [ "$BACKEND" = "--blitz" ]; then
    echo "=== Rendering with Blitz (GPU) ==="
    # Build standalone HTML from Ornis template
    python3 -c "
import sys
sys.path.insert(0, 'crates/ui/examples')
# Generate the same CSS and HTML as the Ornis editor
css = open('bevy_editor_mockup/index.css').read()
html = open('bevy_editor_mockup/editor/index.html').read()
with open('/tmp/ornis_standalone.html', 'w') as f:
    f.write(f'<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>{css}</style></head><body>{html}</body></html>')
print(f'Written /tmp/ornis_standalone.html')
"
    cd forks/blitz
    cargo run -p ornis_screenshot --release -- \
        /tmp/ornis_standalone.html "$WIDTH" "$HEIGHT" 1.0 2>&1
    cp /tmp/blitz_ornis_editor_gpu.png "$OUTPUT" 2>/dev/null || \
    cp /tmp/blitz_ornis_editor.png "$OUTPUT"
    echo "=== Blitz render saved to $OUTPUT ==="
else
    echo "=== Rendering with Ornis (custom renderer) ==="
    cd "$(dirname "$0")/.."
    CONDA_NO_PLUGINS=true cargo run -p ornis-ui --features serialize \
        --example render_to_png -- "$WIDTH" "$HEIGHT" "$OUTPUT" 2>&1
    echo "=== Ornis render saved to $OUTPUT ==="
fi
