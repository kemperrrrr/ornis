// Ornis editor-v2 — the viewport, in whichever way this host can render it.
//
// Three hosts, one file:
//
// 1. Chrome over `cargo xtask editor` — the WASM/WebGPU renderer below.
// 2. The native shell with a native viewport (`window.ornisShell.nativeViewport`)
//    — the shell draws the scene into a layer *under* this page, so the page
//    only reports where its viewport box sits and forwards input over it
//    (PLAN.md §j, вариант 1).
// 3. The native shell without one (WKWebView has no `navigator.gpu`) — the
//    WASM renderer cannot boot, so the viewport shows the shell's written
//    explanation instead of a dead red canvas.
//
// Build: `cargo xtask editor` (wasm-pack writes `editor-v2/pkg`). In native
// mode the package is never imported, so nothing is fetched or started — the
// `import()` is dynamic for exactly that reason. The module owns the backing
// store and applies the pointer/wheel orbit input; this file boots it, keeps
// the 2D grid overlay in sync with `viewport_state_json()` and mirrors the
// applied viewport mode on the toolbar buttons. Same wire shape as
// `editor/viewport.js`.
const canvasEl = document.getElementById('canvas');
const viewportEl = document.getElementById('viewport');
const gridEl = document.getElementById('plane-grid');
const btn3d = document.getElementById('mode-3d');
const btn2d = document.getElementById('mode-2d');
const noticeEl = document.getElementById('viewport-notice');
const noticeTitleEl = document.getElementById('viewport-notice-title');
const noticeBodyEl = document.getElementById('viewport-notice-body');
const noticeDetailEl = document.getElementById('viewport-notice-detail');

// Shell marker, installed before the page loads (`ornis_shell::init_script`).
// Absent in a plain browser, which is what makes the Chrome path unchanged.
const shell =
    window.ornisShell && typeof window.ornisShell === 'object' ? window.ornisShell : null;
const nativeViewport = !!(shell && shell.nativeViewport);

// The wasm-bindgen package, filled by `boot()`; `null` until then (and forever
// in native mode, where the shell renders instead).
let wasm = null;

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
    if (!wasm) return 'perspective3d';
    try {
        const state = JSON.parse(wasm.viewport_state_json());
        return state && state.mode === 'plane2d' ? 'plane2d' : 'perspective3d';
    } catch (e) {
        return 'perspective3d';
    }
};

let booted = false;
let readyCallbacks = [];
let gizmoMode = 'move';
let gizmoSpace = 'world';
let editorMode = 'edit';
let snap = { enabled: false, translate: 0.5, rotateDeg: 15, scale: 0.25 };

// ── written fallback ──────────────────────────────────────────────────────
// The written fallback: no WebGPU in this webview, so there is nothing to
// render here. Wording comes from the shell (it knows the environment); the
// page only decides that this is the state to paint.
function showNotice(reason) {
    if (!noticeEl) return;
    if (noticeTitleEl) noticeTitleEl.textContent = 'Вьюпорт недоступен';
    if (noticeBodyEl) {
        noticeBodyEl.textContent =
            (shell && shell.webgpuNotice) ||
            'WebGPU недоступен в этом окружении — откройте редактор в Chrome через `cargo xtask editor`';
    }
    if (noticeDetailEl) {
        noticeDetailEl.textContent = reason ? String(reason) : '';
        noticeDetailEl.hidden = !reason;
    }
    noticeEl.hidden = false;
    if (canvasEl) {
        // The red `#300` failure fill is gone: an explained viewport beats a
        // dead one, so the canvas steps aside instead.
        canvasEl.classList.remove('failed');
        canvasEl.hidden = true;
    }
}

function hideNotice() {
    if (noticeEl) noticeEl.hidden = true;
    if (canvasEl) canvasEl.hidden = false;
}

// `navigator.gpu` is the whole capability check: a present-but-useless adapter
// is a different failure (the renderer reports it with its own message), while
// a missing one means this webview cannot run the viewport at all. Checked
// before the WASM module is even fetched.
function webgpuMissing() {
    return !(typeof navigator !== 'undefined' && 'gpu' in navigator);
}

// ── native viewport client ────────────────────────────────────────────────
// The shell draws the scene; the page reports the box and forwards input. Wire
// format and validation live in `ornis_shell::viewport_ipc`.
function postShell(type, data) {
    if (!shell || typeof window.ipc === 'undefined' || !window.ipc) return;
    try {
        window.ipc.postMessage(JSON.stringify({ type: type, data: data }));
    } catch (e) {
        console.warn('[ornis-editor] viewport post failed:', e);
    }
}

function postBounds() {
    if (!viewportEl) return;
    const rect = viewportEl.getBoundingClientRect();
    postShell('viewport_bounds', {
        x: rect.left,
        y: rect.top,
        width: rect.width,
        height: rect.height,
        scale: window.devicePixelRatio || 1,
    });
}

function startNativeViewport() {
    document.body.classList.add('ornis-native-viewport');
    hideNotice();
    // The shell configures its layer from the first post onwards; layout
    // changes (docks, tabs, window resize) arrive as more posts.
    postBounds();
    if (typeof ResizeObserver === 'function' && viewportEl) {
        const observer = new ResizeObserver(postBounds);
        observer.observe(viewportEl);
    }
    window.addEventListener('resize', postBounds);
    if (!viewportEl) return;

    const postInput = (extra) => {
        const snapshot = {
            pressed_keys: [],
            pressed_mouse_buttons: [],
            pointer_position: [0, 0],
            pointer_delta: [0, 0],
            wheel_delta: 0,
        };
        if (extra) Object.assign(snapshot, extra);
        postShell('viewport_input', snapshot);
    };
    // DOM `buttons` bitmask -> engine mouse codes (0 left, 1 right, 2 middle).
    // The whole pressed set travels on every event, so a missed `pointerup`
    // cannot leave a button stuck.
    const pressedButtons = (mask) => {
        const held = [];
        if (mask & 1) held.push(0);
        if (mask & 2) held.push(1);
        if (mask & 4) held.push(2);
        return held;
    };
    let last = null;
    const pointer = (event) => {
        const rect = viewportEl.getBoundingClientRect();
        const position = [event.clientX - rect.left, event.clientY - rect.top];
        const delta = last ? [position[0] - last[0], position[1] - last[1]] : [0, 0];
        last = position;
        postInput({
            pressed_mouse_buttons: pressedButtons(event.buttons),
            pointer_position: position,
            pointer_delta: delta,
        });
    };
    viewportEl.addEventListener('pointerdown', (event) => {
        // Capture keeps a drag alive when the pointer leaves the box; the
        // shell's layer is under the page, so the page must own the drag.
        if (viewportEl.setPointerCapture) {
            try {
                viewportEl.setPointerCapture(event.pointerId);
            } catch (e) {
                /* capture is best-effort */
            }
        }
        pointer(event);
    });
    viewportEl.addEventListener('pointermove', pointer);
    const release = (event) => {
        pointer(event);
        last = null;
    };
    viewportEl.addEventListener('pointerup', release);
    viewportEl.addEventListener('pointercancel', release);
    viewportEl.addEventListener('lostpointercapture', () => {
        last = null;
        postInput({});
    });
    viewportEl.addEventListener(
        'wheel',
        (event) => {
            event.preventDefault();
            const rect = viewportEl.getBoundingClientRect();
            postInput({
                pointer_position: [event.clientX - rect.left, event.clientY - rect.top],
                wheel_delta: event.deltaY,
            });
        },
        { passive: false },
    );
    viewportEl.addEventListener('contextmenu', (event) => event.preventDefault());
    // A tab switch or window deactivation delivers no `pointerup`; the shell
    // would keep a button held and orbit forever.
    window.addEventListener('blur', () => {
        last = null;
        postInput({});
    });
    console.log('[ornis-editor] native viewport: shell renders, page lays out');
}

if (btn3d) btn3d.addEventListener('click', () => applyMode('perspective3d'));
if (btn2d) btn2d.addEventListener('click', () => applyMode('plane2d'));

// The WASM exports only exist after `init()`; queue nothing before then —
// the boot flash is short and a click on an uninitialized wrapper throws.
function applyMode(mode) {
    if (!booted) return Promise.resolve();
    return wasm.set_viewport_mode(mode === 'plane2d' ? 'plane2d' : 'perspective3d');
}

// Gizmo tool/space live in the WASM thread-locals; the toolbar asks for a
// change and keeps its own mirror for the button state.
function applyGizmoMode(mode) {
    const next = mode === 'rotate' || mode === 'scale' ? mode : mode === 'move' ? 'move' : 'off';
    gizmoMode = next;
    return booted && wasm ? wasm.set_gizmo_mode(next) : false;
}

function applyGizmoSpace(space) {
    gizmoSpace = space === 'local' ? 'local' : 'world';
    return booted && wasm ? wasm.set_gizmo_space(gizmoSpace) : false;
}

// The replica mirrors the authoritative play/edit mode so it can gate
// picking and gizmo drag off in Play (E4). JS relays every mode_changed
// here; the mirror is re-applied on boot if it changed before WASM ready.
function applyEditorMode(mode) {
    editorMode = mode === 'play' || mode === 'paused' ? mode : 'edit';
    return booted && wasm ? wasm.set_editor_mode(editorMode) : false;
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
    return booted && wasm ? wasm.set_snap(snap.enabled, snap.translate, snap.rotateDeg, snap.scale) : false;
}

function undoGizmoDrag() {
    return booted && wasm ? wasm.undo_gizmo() : false;
}

// Overlay: pointer-events:none div over the canvas, driven by
// `viewport_state_json()`. Minor/major grid lines via layered gradients;
// the mode buttons mirror the applied mode in the same pass.
const paintOverlay = () => {
    if (!wasm) return;
    let state = null;
    try { state = JSON.parse(wasm.viewport_state_json()); } catch (e) { /* keep overlay hidden */ }
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

// A control used before the wasm module is ready only updates the JS mirror,
// so a read-back taken then is stale (the effective rotate step reads `null`).
// Boot re-applies the mirrors first and only then runs the subscribers, which
// is what lets the toolbar repaint against the applied settings. Subscribing
// after boot runs immediately, so the order of the two modules cannot lose it.
// A subscriber is toolbar code, not the renderer: its failure must neither
// stop the remaining subscribers nor reach `boot().catch`, which would mark
// the canvas "WebGPU init failed" while the renderer is running. Both paths
// guard the call, so a subscriber behaves the same before and after boot.
function runReadyCallback(callback) {
    try {
        callback();
    } catch (err) {
        console.error('[ornis-editor] ready subscriber failed:', err);
    }
}

function onViewportReady(callback) {
    if (booted) {
        runReadyCallback(callback);
        return;
    }
    readyCallbacks.push(callback);
}

async function boot() {
    if (nativeViewport) {
        // The shell renders the scene into a layer under this page; nothing for
        // the page to boot (and no reason to fetch the wasm package).
        startNativeViewport();
        return;
    }
    if (webgpuMissing()) {
        // No point fetching the module: `start_renderer` can only fail.
        showNotice('navigator.gpu missing');
        console.warn('[ornis-editor] WebGPU unavailable in this webview, viewport disabled');
        return;
    }
    console.log('[ornis-editor] loading WASM…');
    wasm = await import('./pkg/ornis_wasm.js');
    await wasm.default();
    console.log('[ornis-editor] WASM loaded, starting renderer…');
    await wasm.start_renderer('canvas');
    booted = true;
    console.log('[ornis-editor] renderer started');
    // The toolbar mirror may have changed before the WASM module was ready
    // (the clicks only updated the JS state then). Apply it now, otherwise
    // the button shows Rotate while Move is actually active.
    wasm.set_gizmo_mode(gizmoMode);
    wasm.set_gizmo_space(gizmoSpace);
    wasm.set_snap(snap.enabled, snap.translate, snap.rotateDeg, snap.scale);
    wasm.set_editor_mode(editorMode);
    requestAnimationFrame(overlayLoop);
    const callbacks = readyCallbacks;
    readyCallbacks = [];
    callbacks.forEach(runReadyCallback);
}

boot().catch((err) => {
    console.error('[ornis-editor] failed to start:', err);
    const message = (err && err.message) || String(err);
    if (canvasEl) {
        canvasEl.classList.add('failed');
        canvasEl.title = 'WebGPU init failed: ' + message;
    }
    // A WebGPU-less webview is an environment fact, not an editor bug: the
    // written fallback is still better than a red canvas on top of it.
    showNotice('WebGPU init failed: ' + message);
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
    // The rotate grid coerces the requested step (divisor of 360°, half-turn
    // at the coarse end); the toolbar reads the effective value back rather
    // than duplicating the formula. `null` when the channel is unquantized —
    // wasm-bindgen hands back `undefined` for a Rust `None`, so normalize here
    // instead of leaving the documented contract to the caller's typeof check.
    effectiveRotateStep() {
        if (!booted) return null;
        const step = wasm.snap_effective_rotate_deg();
        return typeof step === 'number' ? step : null;
    },
    // Runs `callback` once boot has re-applied the toolbar mirrors (or
    // immediately if it already has), so a read-back is not left stale by a
    // control used before the wasm module was ready.
    onReady(callback) {
        onViewportReady(callback);
    },
    undoGizmo() {
        return undoGizmoDrag();
    },
};
