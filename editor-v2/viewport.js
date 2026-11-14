// Ornis editor-v2 — WebGPU WASM viewport.
//
// Build: `cargo xtask editor` (wasm-pack writes `editor-v2/pkg`).
// The module owns the backing store and applies the pointer/wheel orbit input;
// this file only boots it, keeps the 2D grid overlay in sync with
// `viewport_state_json()` and mirrors the applied viewport mode on the
// toolbar buttons. Same wire shape as `editor/viewport.js`.
import init, {
    start_renderer,
    set_viewport_mode,
    set_gizmo_mode,
    set_gizmo_space,
    set_snap,
    set_editor_mode,
    undo_gizmo,
    viewport_state_json,
} from './pkg/ornis_wasm.js';

const canvasEl = document.getElementById('canvas');
const viewportEl = document.getElementById('viewport');
const gridEl = document.getElementById('plane-grid');
const btn3d = document.getElementById('mode-3d');
const btn2d = document.getElementById('mode-2d');

// Pointer events (orbit camera, handled inside the WASM module) need
// touch-action:none so the browser does not hijack drags.
if (canvasEl) canvasEl.style.touchAction = 'none';

// Boot-time resize only prevents a wrong-size flash before the first frame;
// the WASM module resizes every frame from the viewport box at the clamped
// devicePixelRatio, so no resize listener is attached here.
const resizeCanvas = () => {
    if (!viewportEl || !canvasEl) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = viewportEl.getBoundingClientRect();
    canvasEl.width = Math.max(1, Math.trunc(rect.width * dpr));
    canvasEl.height = Math.max(1, Math.trunc(rect.height * dpr));
};
resizeCanvas();

const currentMode = () => {
    try {
        const state = JSON.parse(viewport_state_json());
        return state && state.mode === 'plane2d' ? 'plane2d' : 'perspective3d';
    } catch (e) {
        return 'perspective3d';
    }
};

let booted = false;
let gizmoMode = 'move';
let gizmoSpace = 'world';
let editorMode = 'edit';
let snap = { enabled: false, translate: 0.5, rotateDeg: 15, scale: 0.25 };

if (btn3d) btn3d.addEventListener('click', () => applyMode('perspective3d'));
if (btn2d) btn2d.addEventListener('click', () => applyMode('plane2d'));

// The WASM exports only exist after `init()`; queue nothing before then —
// the boot flash is short and a click on an uninitialized wrapper throws.
function applyMode(mode) {
    if (!booted) return Promise.resolve();
    return set_viewport_mode(mode === 'plane2d' ? 'plane2d' : 'perspective3d');
}

// Gizmo tool/space live in the WASM thread-locals; the toolbar asks for a
// change and keeps its own mirror for the button state.
function applyGizmoMode(mode) {
    const next = mode === 'rotate' || mode === 'scale' ? mode : mode === 'move' ? 'move' : 'off';
    gizmoMode = next;
    return booted ? set_gizmo_mode(next) : false;
}

function applyGizmoSpace(space) {
    gizmoSpace = space === 'local' ? 'local' : 'world';
    return booted ? set_gizmo_space(gizmoSpace) : false;
}

// The replica mirrors the authoritative play/edit mode so it can gate
// picking and gizmo drag off in Play (E4). JS relays every mode_changed
// here; the mirror is re-applied on boot if it changed before WASM ready.
function applyEditorMode(mode) {
    editorMode = mode === 'play' || mode === 'paused' ? mode : 'edit';
    return booted ? set_editor_mode(editorMode) : false;
}

// Snap steps live in the same WASM thread-locals as the gizmo tool/space;
// the controls keep a mirror and push the whole settings object on change.
// A step of 0 (or negative) is meaningful — the replica leaves that channel
// unquantized — so pass finite values through verbatim and only fall back
// when the entry is not a number.
function snapStep(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function applySnap(next) {
    snap = {
        enabled: !!next.enabled,
        translate: snapStep(next.translate, 0.5),
        rotateDeg: snapStep(next.rotateDeg, 15),
        scale: snapStep(next.scale, 0.25),
    };
    return booted ? set_snap(snap.enabled, snap.translate, snap.rotateDeg, snap.scale) : false;
}

function undoGizmoDrag() {
    return booted ? undo_gizmo() : false;
}

// Overlay: pointer-events:none div over the canvas, driven by
// `viewport_state_json()`. Minor/major grid lines via layered gradients;
// the mode buttons mirror the applied mode in the same pass.
const paintOverlay = () => {
    if (typeof viewport_state_json !== 'function') return;
    let state = null;
    try { state = JSON.parse(viewport_state_json()); } catch (e) { /* keep overlay hidden */ }
    const is2d = !!(state && state.mode === 'plane2d');
    if (btn3d) {
        btn3d.classList.toggle('active', !is2d);
        btn3d.setAttribute('aria-pressed', String(!is2d));
    }
    if (btn2d) {
        btn2d.classList.toggle('active', is2d);
        btn2d.setAttribute('aria-pressed', String(is2d));
    }
    const grid = is2d ? state.grid : null;
    if (!gridEl || !grid) {
        if (gridEl) gridEl.hidden = true;
        return;
    }
    gridEl.hidden = false;
    // Wire grid values are backing-store (device) pixels; the overlay paints
    // in CSS pixels using the measured display scale, exact even on
    // fractional layouts where the truncating backing store differs from the
    // nominal dpr. The two axes are measured separately.
    const rect = canvasEl ? canvasEl.getBoundingClientRect() : null;
    const fallback = (state && state.dpr) || window.devicePixelRatio || 1;
    const scaleX = rect && rect.width > 0 && canvasEl.width > 0
        ? canvasEl.width / rect.width
        : fallback;
    const scaleY = rect && rect.height > 0 && canvasEl.height > 0
        ? canvasEl.height / rect.height
        : fallback;
    const minorX = grid.minor_step_px / scaleX;
    const minorY = grid.minor_step_px / scaleY;
    const majorX = minorX * grid.major_every;
    const majorY = minorY * grid.major_every;
    const ox = grid.origin_px[0] / scaleX;
    const oy = grid.origin_px[1] / scaleY;
    gridEl.style.backgroundImage =
        'linear-gradient(rgba(255,255,255,.10) 1px, transparent 1px),' +
        'linear-gradient(90deg, rgba(255,255,255,.10) 1px, transparent 1px),' +
        'linear-gradient(rgba(255,255,255,.22) 1px, transparent 1px),' +
        'linear-gradient(90deg, rgba(255,255,255,.22) 1px, transparent 1px)';
    gridEl.style.backgroundSize =
        `${minorX}px ${minorY}px, ${minorX}px ${minorY}px, ${majorX}px ${majorY}px, ${majorX}px ${majorY}px`;
    gridEl.style.backgroundPosition =
        `${ox}px ${oy}px, ${ox}px ${oy}px, ${ox}px ${oy}px, ${ox}px ${oy}px`;
};

const overlayLoop = () => { paintOverlay(); requestAnimationFrame(overlayLoop); };

async function boot() {
    console.log('[ornis-editor] loading WASM…');
    await init();
    console.log('[ornis-editor] WASM loaded, starting renderer…');
    await start_renderer('canvas');
    booted = true;
    console.log('[ornis-editor] renderer started');
    // The toolbar mirror may have changed before the WASM module was ready
    // (the clicks only updated the JS state then). Apply it now, otherwise
    // the button shows Rotate while Move is actually active.
    set_gizmo_mode(gizmoMode);
    set_gizmo_space(gizmoSpace);
    set_snap(snap.enabled, snap.translate, snap.rotateDeg, snap.scale);
    set_editor_mode(editorMode);
    requestAnimationFrame(overlayLoop);
}

boot().catch((err) => {
    console.error('[ornis-editor] failed to start:', err);
    if (canvasEl) {
        canvasEl.classList.add('failed');
        canvasEl.title = 'WebGPU init failed: ' + err.message;
    }
});

// The welcome flow switches modes through the same module (no direct import
// from app.js, which is a classic script).
window.OrnisViewport = {
    setMode(mode) {
        return applyMode(mode);
    },
    mode() {
        return currentMode();
    },
    setGizmoMode(mode) {
        return applyGizmoMode(mode);
    },
    setGizmoSpace(space) {
        return applyGizmoSpace(space);
    },
    setEditorMode(mode) {
        return applyEditorMode(mode);
    },
    setSnap(next) {
        return applySnap(next || {});
    },
    undoGizmo() {
        return undoGizmoDrag();
    },
};
