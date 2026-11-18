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

// ── copy ──────────────────────────────────────────────────────────────────
// Every user-visible string this module owns, in one place and in the same
// language as the rest of the editor UI. The shell owns the *environment*
// wording (`window.ornisShell.webgpuNotice`) — that one is Rust's, because only
// the shell knows which host the page is in.
const STRINGS = {
    noticeTitle: 'Viewport unavailable',
    noticeFallback:
        'WebGPU is unavailable in this environment — open the editor in Chrome via cargo xtask editor',
    // The wasm-pack bundle is a build artifact: without it the module import
    // itself fails. The fix is a command, so the notice names it.
    pkgMissingTitle: 'Viewport bundle is missing',
    pkgMissingBody:
        'editor-v2/pkg has not been built — run cargo xtask editor, then reload the editor',
    pkgMissingDetail: 'import ./pkg/ornis_wasm.js failed',
    // Shown over the viewport in native mode: the toolbar's 3D/2D, gizmo and
    // snap controls drive the WASM replica, which does not exist when the shell
    // renders. Saying so beats a button that silently does nothing.
    nativeToolbarNotice: 'Gizmo, snap and 3D/2D controls apply to the browser viewport (cargo xtask editor)',
};

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
// The written fallback: there is nothing for the page to render here, so it
// explains why instead of leaving a dead red canvas.
//
// `notice` carries the wording; `showNotice` takes the environment case (where
// the shell owns the body text because it knows the host) and
// `showBuildNotice` takes a build problem, whose fix is a command the page can
// name itself.
function showNotice(notice) {
    if (!noticeEl) return;
    if (noticeTitleEl) noticeTitleEl.textContent = notice.title;
    if (noticeBodyEl) noticeBodyEl.textContent = notice.body;
    if (noticeDetailEl) {
        noticeDetailEl.textContent = notice.detail ? String(notice.detail) : '';
        noticeDetailEl.hidden = !notice.detail;
    }
    noticeEl.hidden = false;
    if (canvasEl) {
        // The red `#300` failure fill is gone: an explained viewport beats a
        // dead one, so the canvas steps aside instead. `hidden` only wins over
        // `.viewport-canvas { display: block }` because styles.css also carries
        // `.viewport-canvas[hidden] { display: none }` — an attribute alone
        // does not beat a class rule.
        canvasEl.classList.remove('failed');
        canvasEl.hidden = true;
    }
}

function hideNotice() {
    if (noticeEl) noticeEl.hidden = true;
    if (canvasEl) canvasEl.hidden = false;
}

// The environment cannot do WebGPU (no `navigator.gpu`, or an adapter that
// resolved null). The shell knows the host, so its wording wins when present;
// the page's English fallback covers a plain browser tab.
function environmentNotice(detail) {
    return {
        title: STRINGS.noticeTitle,
        body: (shell && shell.webgpuNotice) || STRINGS.noticeFallback,
        detail: detail,
    };
}

// A build problem rather than a renderer defect: `editor-v2/pkg` has not been
// produced by `cargo xtask editor`. Kept separate so the message names the
// command instead of telling the user their GPU is at fault.
function buildNotice(detail) {
    return { title: STRINGS.pkgMissingTitle, body: STRINGS.pkgMissingBody, detail: detail };
}

// ── WebGPU capability ─────────────────────────────────────────────────────
// "This environment cannot do WebGPU" is decided by *fact*, never by matching
// an error message: a renderer error that merely mentions WebGPU or an adapter
// (a lost device, a validation error, "format … not supported") is a real
// defect and must stay a loud red canvas. Only two things are environment
// facts, and both are observable before rendering:
//
// 1. `navigator.gpu` does not exist — WKWebView today, and any browser without
//    WebGPU enabled;
// 2. `requestAdapter()` resolves `null` — Chrome on a box without a usable GPU
//    or the right flags.
//
// Both are checked in `boot()` before the wasm module is fetched, so the
// fallback costs no download and cannot be confused with a render failure.

// The one error string that still means "no adapter": `ornis-wasm` maps a
// failed `request_adapter` to exactly this text (`crates/wasm/src/lib.rs`,
// `map_err(|_| JsValue::from_str(ADAPTER_NOT_FOUND))`). Kept as an exact
// comparison rather than a regex/`includes` so new renderer messages cannot
// drift into this branch by sharing a word — `device: adapter not found in
// cache` is a real defect, not an environment limit.
//
// The two copies of this string cannot drift apart unnoticed: the Rust side has
// `pub const ADAPTER_NOT_FOUND` and a test
// (`the_page_recognises_exactly_this_adapter_message`) that compares this
// declaration against it. That test pins the *literal*, not the runtime path —
// it does not execute `map_err`, so changing the message the renderer actually
// throws would need its own test on the Rust side.
const ADAPTER_NOT_FOUND = 'adapter not found';

// Adapter options the pre-flight check and the renderer must agree on: the
// check is only meaningful if it asks for the same adapter the renderer will
// use. `ornis-wasm` requests `POWER_PREFERENCE`
// (`wgpu::PowerPreference::HighPerformance`, `crates/wasm/src/lib.rs`), so the
// probe asks for that too — a cheap extra adapter query is worth not checking
// one GPU and rendering on another. The Rust test
// `the_page_requests_the_high_performance_adapter` derives the expected string
// from that constant, so changing the preference breaks it.
const ADAPTER_OPTIONS = { powerPreference: 'high-performance' };

function webgpuMissing() {
    return !(typeof navigator !== 'undefined' && 'gpu' in navigator);
}

// Resolves whether this browser can hand out an adapter. A throw (some
// implementations reject instead of resolving `null`) is the same answer:
// no usable adapter here.
async function hasWebgpuAdapter() {
    if (webgpuMissing()) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter(ADAPTER_OPTIONS);
        return !!adapter;
    } catch (e) {
        return false;
    }
}

// Whether a *thrown* init error is the "no adapter" case. Deliberately exact:
// every other message is a renderer defect and stays a red canvas.
function isAdapterNotFound(message) {
    return String(message || '').trim() === ADAPTER_NOT_FOUND;
}

// ── native viewport client ────────────────────────────────────────────────
// The shell draws the scene; the page reports the box and forwards input. Wire
// format and validation live in `ornis_shell::viewport_ipc`.

// Interactive controls that live *inside* `#viewport` (mode switch, settings,
// gizmo toolbars, snap fields). A native-viewport handler must ignore events
// whose target is one of these, or a button click would also become a camera
// drag and `setPointerCapture` would steal the following events from the
// button. Kept as one selector so the filter cannot drift from the markup.
const VIEWPORT_CONTROL_SELECTOR =
    'button, input, select, textarea, a, [contenteditable], ' +
    '.viewport-mode, .viewport-gizmos, .viewport-settings, .snap-controls, .viewport-notice';

// Whether a pointer/wheel/keyboard event belongs to the engine rather than to a
// page control. `closest` walks up from the real target, so a click on an
// icon inside a button is still recognised as that button.
function isViewportSurfaceEvent(event) {
    const target = event && event.target;
    if (!target || typeof target.closest !== 'function') {
        // No element target (a synthetic event, or text selection): treat it as
        // the viewport itself, which is what a drag on the backing area is.
        return true;
    }
    if (target === viewportEl || target === canvasEl) return true;
    if (target.closest(VIEWPORT_CONTROL_SELECTOR)) return false;
    // Anything else inside the viewport box (the grid overlay, the canvas
    // container) is still backing surface.
    return viewportEl ? viewportEl.contains(target) : false;
}

function postShell(type, data) {
    if (!shell || typeof window.ipc === 'undefined' || !window.ipc) return;
    try {
        window.ipc.postMessage(JSON.stringify({ type: type, data: data }));
    } catch (e) {
        console.warn('[ornis-editor] viewport post failed:', e);
    }
}

function viewportRect() {
    return viewportEl
        ? viewportEl.getBoundingClientRect()
        : { left: 0, top: 0, width: 0, height: 0 };
}

function boundsPost(rect) {
    return {
        x: rect.left,
        y: rect.top,
        width: rect.width,
        height: rect.height,
        scale: window.devicePixelRatio || 1,
    };
}

function postBounds() {
    if (!viewportEl) return;
    postShell('viewport_bounds', boundsPost(viewportRect()));
}

// Tells the page that the toolbar cannot drive anything in this mode, so the
// controls read as unavailable instead of silently doing nothing.
function markToolbarUnavailable() {
    const controls = document.querySelectorAll(
        '.viewport-mode button, .viewport-gizmos .gizmo, .viewport-settings, .snap-controls',
    );
    for (const control of controls) {
        control.setAttribute('aria-disabled', 'true');
        control.classList.add('unavailable');
        if (!control.title) control.title = STRINGS.nativeToolbarNotice;
    }
    const banner = document.getElementById('viewport-native-hint');
    if (banner) {
        banner.textContent = STRINGS.nativeToolbarNotice;
        banner.hidden = false;
    }
}

function startNativeViewport() {
    document.body.classList.add('ornis-native-viewport');
    hideNotice();
    markToolbarUnavailable();
    // The shell configures its layer from the first post onwards; layout
    // changes (docks, tabs, window resize) arrive as more posts.
    postBounds();

    // A size observer alone misses a *move*: the viewport box can shift without
    // changing size (top bar grows, a banner appears, a panel scrolls) and the
    // native layer would then sit at the old place. So the rectangle is also
    // re-read once per frame and posted only when it actually changed — a
    // stable layout costs one `getBoundingClientRect` and one comparison.
    let lastBounds = boundsPost(viewportRect());
    const sameBounds = (a, b) =>
        a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height && a.scale === b.scale;
    const followBounds = () => {
        const next = boundsPost(viewportRect());
        if (!sameBounds(next, lastBounds)) {
            lastBounds = next;
            postShell('viewport_bounds', next);
        }
        requestAnimationFrame(followBounds);
    };
    if (typeof ResizeObserver === 'function' && viewportEl) {
        const observer = new ResizeObserver(postBounds);
        observer.observe(viewportEl);
    }
    window.addEventListener('resize', postBounds);
    requestAnimationFrame(followBounds);
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
    // Held keys, mirrored from DOM `code` -> raw engine codes. The shell's
    // `viewport_ipc` validates this list exactly like `pressed_buttons`.
    const heldKeys = new Set();
    const pointer = (event) => {
        const rect = viewportRect();
        const position = [event.clientX - rect.left, event.clientY - rect.top];
        const delta = last ? [position[0] - last[0], position[1] - last[1]] : [0, 0];
        last = position;
        postInput({
            pressed_keys: Array.from(heldKeys),
            pressed_mouse_buttons: pressedButtons(event.buttons),
            pointer_position: position,
            pointer_delta: delta,
        });
    };
    const releaseAll = () => {
        last = null;
        heldKeys.clear();
        postInput({});
    };
    viewportEl.addEventListener('pointerdown', (event) => {
        // A click on the 3D/2D switch, Settings, a gizmo tool or a snap field is
        // that control's, not a camera drag.
        if (!isViewportSurfaceEvent(event)) return;
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
    viewportEl.addEventListener('pointermove', (event) => {
        if (!isViewportSurfaceEvent(event)) return;
        pointer(event);
    });
    const release = (event) => {
        if (!isViewportSurfaceEvent(event)) return;
        pointer(event);
        last = null;
    };
    viewportEl.addEventListener('pointerup', release);
    viewportEl.addEventListener('pointercancel', release);
    viewportEl.addEventListener('lostpointercapture', releaseAll);
    viewportEl.addEventListener(
        'wheel',
        (event) => {
            // Wheel over a snap field must scroll/step that field, not zoom the
            // camera — and must not have its default suppressed.
            if (!isViewportSurfaceEvent(event)) return;
            event.preventDefault();
            const rect = viewportRect();
            postInput({
                pressed_keys: Array.from(heldKeys),
                pointer_position: [event.clientX - rect.left, event.clientY - rect.top],
                wheel_delta: event.deltaY,
            });
        },
        // Not a capture-phase listener: the filter above looks at
        // `event.target`, so a wheel over a snap field never reaches the shell
        // while the field's own handler and the default scroll still run.
        { passive: false },
    );
    viewportEl.addEventListener('contextmenu', (event) => {
        if (!isViewportSurfaceEvent(event)) return;
        event.preventDefault();
    });

    // Keyboard follows focus, like any editor viewport: a key *press* reaches
    // the engine only while focus is inside the viewport box, so typing in the
    // inspector or a shortcut field never moves the camera. The box itself is
    // made focusable so a click on the backing surface starts navigation.
    viewportEl.setAttribute('tabindex', '0');
    const focusInside = () => {
        const active = document.activeElement;
        return !!(viewportEl && active && viewportEl.contains(active));
    };
    const onKey = (event, pressed) => {
        const codes = gameKeyCodes(event.code);
        if (!codes || codes.length === 0) return;
        if (pressed) {
            // Only a press needs the gate; a release must always be delivered
            // (see below).
            if (!focusInside() || !isViewportSurfaceEvent(event)) return;
            event.preventDefault();
            for (const code of codes) heldKeys.add(code);
        } else {
            // A release is honoured whenever the engine thinks the key is down,
            // regardless of where focus is *now*. Otherwise: hold W in the
            // viewport, click into the inspector, release W — the key-up would
            // be dropped, `heldKeys` would keep W forever, and the camera would
            // keep flying until the window lost focus.
            let wasHeld = false;
            for (const code of codes) wasHeld = heldKeys.delete(code) || wasHeld;
            if (!wasHeld) return;
            event.preventDefault();
        }
        postInput({
            pressed_keys: Array.from(heldKeys),
            pointer_position: last || [0, 0],
        });
    };
    // The listener sits on the window so a key-up still arrives after focus
    // moved; `onKey` above decides what each direction means.
    window.addEventListener('keydown', (event) => onKey(event, true));
    window.addEventListener('keyup', (event) => onKey(event, false));
    // Focus moving out of the box ends keyboard navigation: release everything
    // rather than waiting for key-ups that may land in another control.
    viewportEl.addEventListener('focusout', (event) => {
        // `focusout` also fires when focus moves *within* the box (canvas ->
        // toolbar); only a move to somewhere outside ends navigation.
        const next = event && event.relatedTarget;
        if (next && viewportEl.contains(next)) return;
        releaseAll();
    });
    // Losing the window (tab switch) delivers no `keyup`/`pointerup`; without
    // this the shell keeps a key or button held forever.
    window.addEventListener('blur', releaseAll);
    console.log('[ornis-editor] native viewport: shell renders, page lays out');
}

// DOM `KeyboardEvent.code` -> the raw code aliases the engine reads, mirroring
// `ornis_input::KeyCode` (winit physical + ASCII for W/A/S/D, legacy DOM key
// codes for the arrows). Kept beside its only caller: the WASM path has the same
// table inside `ornis-wasm`, which cannot be reached from here.
function gameKeyCodes(code) {
    switch (code) {
        case 'KeyW':
            return [17, 87];
        case 'KeyA':
            return [30, 65];
        case 'KeyS':
            return [31, 83];
        case 'KeyD':
            return [32, 68];
        case 'ArrowUp':
            return [38];
        case 'ArrowDown':
            return [40];
        case 'ArrowLeft':
            return [37];
        case 'ArrowRight':
            return [39];
        default:
            return null;
    }
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
        showNotice(environmentNotice('navigator.gpu missing'));
        console.warn('[ornis-editor] WebGPU unavailable in this webview, viewport disabled');
        return;
    }
    // An absent adapter is the other environment fact, and it is cheap to ask
    // *before* downloading the module. Checking it here (rather than guessing
    // from a later error message) is what keeps real renderer failures loud.
    if (!(await hasWebgpuAdapter())) {
        showNotice(environmentNotice('no WebGPU adapter'));
        console.warn('[ornis-editor] no WebGPU adapter, viewport disabled');
        return;
    }
    console.log('[ornis-editor] loading WASM…');
    // A missing `editor-v2/pkg` (the wasm-pack output, built by
    // `cargo xtask editor`) fails the import itself, before any GPU work. That
    // is a *build* problem with a concrete fix, so it is reported as such
    // instead of as a renderer failure.
    try {
        wasm = await import('./pkg/ornis_wasm.js');
    } catch (err) {
        console.warn('[ornis-editor] wasm package missing:', err);
        showNotice(buildNotice(STRINGS.pkgMissingDetail));
        return;
    }
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
    // Only the exact "no adapter" string (what `ornis-wasm` emits when
    // `request_adapter` fails) is treated as an environment limit. Everything
    // else — a lost device, a validation error, an unsupported format — is a
    // renderer defect and keeps the loud red canvas, even if the message
    // mentions WebGPU or an adapter.
    if (!nativeViewport && (webgpuMissing() || isAdapterNotFound(message))) {
        showNotice(environmentNotice(message || 'no WebGPU adapter'));
        return;
    }
    if (canvasEl) {
        canvasEl.classList.add('failed');
        canvasEl.title = 'WebGPU init failed: ' + message;
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
