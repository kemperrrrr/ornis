// Page-side gate for the viewport host selection (PLAN.md §j).
//
// `editor-v2/viewport.js` decides, at boot, which of three hosts it is in:
// Chrome with WebGPU (the WASM renderer), the native shell with a native
// viewport (the page reports bounds and forwards input), or a webview without
// WebGPU (the written fallback notice instead of a red canvas).
//
// The real module is loaded through real ESM (a copy in a temp dir next to a
// stub `pkg/ornis_wasm.js`), so the import graph is exercised the way a
// browser would: a WebGPU-less or native-mode boot must not even import the
// WASM package. No browser, no GPU, no network.
//
// Run: `node --test editor-v2/tests/viewport_host.test.js` (CI: `cargo xtask
// quality`, stage `editor-viewport`). The explicit file matters: a bare
// `node --test editor-v2/tests/` is a *directory* argument, which Node 24
// rejects with "Cannot find module". A shell glob
// (`editor-v2/tests/*.test.js`) works too, but the named file does not depend
// on the caller's globbing. A missing `node` is a SKIP there, never a pass.
'use strict';

const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');

const SOURCE = fs.readFileSync(path.join(__dirname, '..', 'viewport.js'), 'utf8');

// Stub WASM package: every export the viewport may call records its name, and
// the default export is the wasm-bindgen `init()`.
const WASM_STUB = `
const record = (name) => (...args) => {
  (globalThis.__ornisCalls || (globalThis.__ornisCalls = [])).push([name, ...args]);
};
export default async () => record('init')();
export const start_renderer = async (id) => record('start_renderer')(id);
export const set_viewport_mode = record('set_viewport_mode');
export const set_gizmo_mode = record('set_gizmo_mode');
export const set_gizmo_space = record('set_gizmo_space');
export const set_snap = record('set_snap');
export const set_editor_mode = record('set_editor_mode');
export const undo_gizmo = record('undo_gizmo');
export const snap_effective_rotate_deg = () => 15;
export const viewport_state_json = () =>
  JSON.stringify({ mode: 'perspective3d', dpr: 1, grid: null });
`;

// `ornis-wasm` raises exactly this when `request_adapter` fails
// (crates/wasm/src/lib.rs: `map_err(|_| JsValue::from_str("adapter not found"))`).
// The page compares it verbatim, so the stub must use the real string.
const ADAPTER_NOT_FOUND = 'adapter not found';

// A host that *passes* the pre-flight adapter check but whose renderer then
// reports "adapter not found" anyway (a race: the adapter disappeared between
// the two calls). That exact string is still an environment limit.
const WASM_STUB_ADAPTER_LOST = `
export default async () => {
  throw new Error(${JSON.stringify('adapter not found')});
};
export const start_renderer = async () => {};
export const set_viewport_mode = () => true;
export const set_gizmo_mode = () => true;
export const set_gizmo_space = () => true;
export const set_snap = () => true;
export const set_editor_mode = () => true;
export const undo_gizmo = () => true;
export const snap_effective_rotate_deg = () => 15;
export const viewport_state_json = () => JSON.stringify({ mode: 'perspective3d', dpr: 1 });
`;

// Renderer defects whose text *looks* like an environment problem. All of them
// must keep the loud red canvas: matching on words is exactly the bug the
// pre-flight check exists to avoid.
const DEFECT_MESSAGES = [
  'WebGPU: device lost',
  'Texture format Rgba16Float not supported for storage',
  'WebGPU validation error: shader module invalid',
  'device lost (adapter limits exceeded)',
];

// Messages that *contain* `ADAPTER_NOT_FOUND` without being it. They are the
// reason the page compares with `===`: an `includes()` check would classify all
// three as an environment limit. Defined once and used from both tests below
// (the combined defects sweep and the dedicated equality test) so the two
// cannot drift apart.
const ADAPTER_LOOKALIKES = [
  'device: adapter not found in cache',
  'adapter not found: retrying after device loss',
  'adapter not found (device was destroyed)',
];

// A genuine defect after a working adapter: shader compilation failed. This is
// a bug, not an environment limit, so the loud red canvas must stay.
const WASM_STUB_BROKEN_SHADER = `
export default async () => {
  throw new Error('shader compilation failed: unresolved identifier');
};`;

let copyCounter = 0;

// Minimal element: the viewport only reads boxes, toggles classes/attributes
// and registers listeners. `dispatch` runs them, which is how input
// forwarding is observed.
//
// `parent` + `controls` model the real DOM shape the input filter walks: an
// event dispatched on a control bubbles to `#viewport`, so `target.closest()`
// must be able to find the control's selectors. Without that the "a click on the
// toolbar is not a camera drag" case could not be tested at all.
function element({
  width = 640,
  height = 480,
  left = 100,
  top = 60,
  hidden = false,
  controls = [],
  selectors = [],
} = {}) {
  const listeners = new Map();
  const classes = new Set();
  const attrs = new Map();
  const el = {
    hidden,
    title: '',
    textContent: '',
    dataset: {},
    style: {},
    classes,
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)),
      contains: (name) => classes.has(name),
    },
    setAttribute: (name, value) => attrs.set(name, String(value)),
    getAttribute: (name) => attrs.get(name),
    getBoundingClientRect: () => ({
      left,
      top,
      width,
      height,
      right: left + width,
      bottom: top + height,
    }),
    addEventListener: (type, handler) => {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    setPointerCapture: () => {},
    hasPointerCapture: () => true,
    // Containment is by explicit list: the stub has no tree to walk.
    contains: (node) => node === el || controls.includes(node),
    // Selector matching for `closest`: the test gives each element the list of
    // selectors it should answer for, and the walk goes element -> parent.
    closest: (selector) => {
      const wanted = selector.split(',').map((part) => part.trim());
      let node = el;
      while (node) {
        if (node.selectors && node.selectors.some((s) => wanted.includes(s))) return node;
        node = node.parent;
      }
      return null;
    },
    selectors,
    controls,
    parent: null,
    dispatch: (type, event) => {
      // Real events always answer `preventDefault`; the viewport calls it on
      // wheel to keep the page from scrolling.
      const withDefaults = Object.assign(
        { preventDefault: () => {}, target: el },
        event,
      );
      for (const handler of listeners.get(type) || []) handler(withDefaults);
    },
  };
  for (const control of controls) control.parent = el;
  return el;
}

// A toolbar control inside `#viewport` (a 3D/2D button, a gizmo tool, a snap
// field): it must keep its own pointer and wheel events.
function control(selectors) {
  return element({ width: 30, height: 30, selectors });
}

// Boots a fresh copy of `viewport.js` next to the stub WASM package with the
// DOM/window stubs installed as globals (the module reads them as globals,
// exactly like in a browser).
async function bootPage({ shell, gpu = true, adapter = true, stub = WASM_STUB }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ornis-viewport-'));
  fs.mkdirSync(path.join(dir, 'pkg'));
  fs.writeFileSync(
    path.join(dir, 'pkg', 'ornis_wasm.js'),
    stub,
  );
  copyCounter += 1;
  const entry = path.join(dir, `viewport-${copyCounter}.mjs`);
  fs.writeFileSync(entry, SOURCE);

  const dom = {
    viewport: element({
      controls: [
        control(['button']),
        control(['input']),
        control(['.viewport-gizmos']),
      ],
    }),
    canvas: element(),
    'plane-grid': element(),
    'mode-3d': element(),
    'mode-2d': element(),
    // `hidden` mirrors the attribute in index.html: the notice only becomes
    // visible when a boot path decides to show it.
    'viewport-notice': element({ hidden: true }),
    'viewport-notice-title': element(),
    'viewport-notice-body': element(),
    'viewport-notice-detail': element(),
    'viewport-native-hint': element({ hidden: true }),
  };
  const posted = [];
  const windowListeners = new Map();
  const bodyClasses = new Set();
  const consoleMessages = [];
  // `document.querySelectorAll` in the module asks for one selector (the
  // toolbar controls it disables in native mode); this answers it with the
  // stubbed controls.
  const TOOLBAR_SELECTOR =
    '.viewport-mode button, .viewport-gizmos .gizmo, .viewport-settings, .snap-controls';
  const queryStubs = [[[dom['mode-3d'], dom['mode-2d']], TOOLBAR_SELECTOR]];
  const saved = {
    window: globalThis.window,
    document: globalThis.document,
    navigator: globalThis.navigator,
    ResizeObserver: globalThis.ResizeObserver,
    requestAnimationFrame: globalThis.requestAnimationFrame,
    calls: globalThis.__ornisCalls,
    console: globalThis.console,
  };
  delete globalThis.__ornisCalls;

  const windowObject = {
    devicePixelRatio: 2,
    ornisShell: shell,
    ipc: { postMessage: (text) => posted.push(JSON.parse(text)) },
    addEventListener: (type, handler) => {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(handler);
    },
  };
  globalThis.window = windowObject;
  // A webview without WebGPU has `navigator` but no `gpu` member. Node's own
  // `navigator` is getter-only, so the stub is defined, not assigned — and
  // defined again on restore for the same reason.
  // `adapter` answers the page's pre-flight `requestAdapter()`: `true` gives an
  // adapter object, `false` gives `null` (the "no usable GPU" case), `'throw'`
  // rejects like an implementation that reports failure by throwing. The
  // options are recorded so a test can pin what the probe asked for.
  const adapterCalls = [];
  const requestAdapter = async (options) => {
    adapterCalls.push(options);
    if (adapter === 'throw') throw new Error('adapter request failed');
    return adapter ? { name: 'stub adapter' } : null;
  };
  const navigatorStub = gpu
    ? { gpu: { requestAdapter } }
    : { userAgent: 'Mozilla/5.0 AppleWebKit' };
  Object.defineProperty(globalThis, 'navigator', {
    value: navigatorStub,
    configurable: true,
    writable: true,
  });
  globalThis.document = {
    body: {
      classList: {
        add: (name) => bodyClasses.add(name),
        remove: (name) => bodyClasses.delete(name),
        contains: (name) => bodyClasses.has(name),
      },
    },
    // The toolbar controls `markToolbarUnavailable` walks: the DOM stub has no
    // tree, so the list is what `querySelectorAll` answers with. `mode-3d` and
    // `mode-2d` are in the map too, so they must be given a matching selector.
    querySelectorAll: (selector) => {
      for (const [list, matches] of queryStubs) {
        if (matches === selector) return list;
      }
      return [];
    },
    getElementById: (id) => dom[id] || null,
    // `document.activeElement` decides keyboard routing: only a focused element
    // inside the viewport box forwards keys to the engine.
    activeElement: null,
  };
  globalThis.ResizeObserver = function ResizeObserver() {
    this.observe = () => {};
    this.disconnect = () => {};
  };
  // The native client schedules a per-frame bounds check. Recording the
  // callbacks lets a test step the loop by hand instead of waiting for real
  // frames.
  const frames = [];
  globalThis.requestAnimationFrame = (callback) => {
    frames.push(callback);
    return frames.length;
  };
  globalThis.console = {
    log: (message) => consoleMessages.push(String(message)),
    warn: (message) => consoleMessages.push(String(message)),
    error: (message) => consoleMessages.push(String(message)),
  };

  await import(entry);
  // The page fetches the wasm module with a real dynamic `import()`, so a fixed
  // number of microtask ticks is not a wait for boot: on a cold tmpdir the chain
  // is still running when the ticks run out, and its tail then lands in the
  // *next* test's globals. Wait for a terminal boot state instead — the mirrors
  // applied (healthy), the written notice (environment limit), the red canvas
  // (defect) or the first bounds post (native shell) — and fail loudly if none
  // arrives, so a stuck boot cannot masquerade as an empty call log.
  const bootSettled = () =>
    (globalThis.__ornisCalls || []).some((call) => call[0] === 'set_editor_mode')
    || dom['viewport-notice'].hidden === false
    || dom.canvas.classes.has('failed')
    || posted.length > 0;
  for (let i = 0; i < 500 && !bootSettled(); i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.ok(bootSettled(), 'the page must reach a terminal boot state');
  // Let the tail after the marker land (ready subscribers run after the mirrors).
  let previous = -1;
  for (let i = 0; i < 50; i += 1) {
    const size = (globalThis.__ornisCalls || []).length;
    if (size === previous) break;
    previous = size;
    await new Promise((resolve) => setImmediate(resolve));
  }

  const calls = (globalThis.__ornisCalls || []).slice();
  const restore = () => {
    globalThis.window = saved.window;
    globalThis.document = saved.document;
    Object.defineProperty(globalThis, 'navigator', {
      value: saved.navigator,
      configurable: true,
      writable: true,
    });
    globalThis.ResizeObserver = saved.ResizeObserver;
    globalThis.requestAnimationFrame = saved.requestAnimationFrame;
    globalThis.console = saved.console;
    if (saved.calls === undefined) delete globalThis.__ornisCalls;
    else globalThis.__ornisCalls = saved.calls;
    fs.rmSync(dir, { recursive: true, force: true });
  };
  return {
    dom,
    posted,
    calls,
    frames,
    adapterCalls,
    bodyClasses,
    consoleMessages,
    dispatchWindow: (type, event) => {
      // Window-level listeners get the same `preventDefault` every real event
      // answers; the key handlers call it on a key the engine consumes.
      const withDefaults = Object.assign({ preventDefault: () => {} }, event);
      for (const handler of windowListeners.get(type) || []) handler(withDefaults);
    },
    restore,
  };
}

test('chrome path still boots the WASM renderer with no shell marker', async () => {
  const page = await bootPage({ shell: null });
  try {
    const names = page.calls.map(([name]) => name);
    assert.deepStrictEqual(
      names.slice(0, 2),
      ['init', 'start_renderer'],
      'Chrome must still init and start the WASM renderer',
    );
    assert.deepStrictEqual(page.calls[1], ['start_renderer', 'canvas']);
    for (const call of ['set_gizmo_mode', 'set_gizmo_space', 'set_snap', 'set_editor_mode']) {
      assert.ok(names.includes(call), `the toolbar mirror is applied once ready (${call})`);
    }
    assert.strictEqual(page.dom['viewport-notice'].hidden, true, 'no notice in Chrome');
    assert.ok(!page.bodyClasses.has('ornis-native-viewport'), 'Chrome keeps the opaque canvas');
    assert.strictEqual(page.posted.length, 0, 'Chrome talks HTTP, not the shell IPC');
  } finally {
    page.restore();
  }
});

test('shell without WebGPU shows the written notice, not a red canvas', async () => {
  const notice =
    'WebGPU недоступен в WKWebView — живой вьюпорт в Chrome через `cargo xtask editor`';
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: notice },
    gpu: false,
  });
  try {
    assert.strictEqual(page.dom['viewport-notice'].hidden, false, 'notice must be visible');
    assert.strictEqual(
      page.dom['viewport-notice-body'].textContent,
      notice,
      'the shell owns the wording; the page only paints it',
    );
    assert.strictEqual(
      page.dom['viewport-notice-title'].textContent,
      'Viewport unavailable',
      'the title is English like the rest of the editor UI',
    );
    assert.strictEqual(
      page.dom['viewport-notice-detail'].textContent,
      'navigator.gpu missing',
      'the detail line names the missing capability',
    );
    assert.ok(!page.dom.canvas.classes.has('failed'), 'the dead red fill must not come back');
    assert.strictEqual(page.dom.canvas.hidden, true, 'the canvas steps aside');
    assert.deepStrictEqual(
      page.calls,
      [],
      'a WebGPU-less webview must not even import the WASM package',
    );
  } finally {
    page.restore();
  }
});

test('shell with a native viewport reports bounds and forwards input', async () => {
  const page = await bootPage({
    shell: { nativeViewport: true, webgpuNotice: 'unused' },
    gpu: false,
  });
  try {
    assert.ok(
      page.bodyClasses.has('ornis-native-viewport'),
      'the page must make its viewport box transparent for the layer below',
    );
    assert.deepStrictEqual(page.calls, [], 'the shell renders, not the page');
    assert.strictEqual(page.posted[0].type, 'viewport_bounds');
    assert.deepStrictEqual(page.posted[0].data, {
      x: 100,
      y: 60,
      width: 640,
      height: 480,
      scale: 2,
    });

    // A drag over the box: position, delta and the pressed set travel as one
    // snapshot in the engine's wire shape.
    const viewport = page.dom.viewport;
    viewport.dispatch('pointerdown', { clientX: 110, clientY: 70, buttons: 1, pointerId: 1 });
    const down = page.posted.at(-1);
    assert.strictEqual(down.type, 'viewport_input');
    assert.deepStrictEqual(down.data.pointer_position, [10, 10]);
    assert.deepStrictEqual(down.data.pressed_mouse_buttons, [0]);

    viewport.dispatch('pointermove', { clientX: 130, clientY: 70, buttons: 1, pointerId: 1 });
    const moved = page.posted.at(-1);
    assert.deepStrictEqual(moved.data.pointer_position, [30, 10]);
    assert.deepStrictEqual(moved.data.pointer_delta, [20, 0]);
    assert.deepStrictEqual(moved.data.pressed_mouse_buttons, [0]);

    // Right-drag pans: engine code 1, distinct from left's 0.
    viewport.dispatch('pointermove', { clientX: 130, clientY: 70, buttons: 2, pointerId: 1 });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_mouse_buttons, [1]);

    // Wheel over the box.
    viewport.dispatch('wheel', { clientX: 120, clientY: 80, deltaY: -60 });
    const wheel = page.posted.at(-1);
    assert.strictEqual(wheel.data.wheel_delta, -60);
    assert.deepStrictEqual(wheel.data.pointer_position, [20, 20]);

    // Losing the window must release the held button, or the shell orbits
    // forever after a tab switch.
    page.dispatchWindow('blur', {});
    const blurred = page.posted.at(-1);
    assert.deepStrictEqual(blurred.data.pressed_mouse_buttons, []);
    assert.deepStrictEqual(blurred.data.pointer_delta, [0, 0]);
  } finally {
    page.restore();
  }
});

test('toolbar controls keep their own pointer and wheel events', async () => {
  // The blocker: `#viewport` contains the 3D/2D switch, Settings, the gizmo
  // toolbars and the snap step fields. A handler on the container that ignores
  // `event.target` turns every one of those into a camera drag, and
  // `setPointerCapture` then steals the events the control needs.
  const page = await bootPage({
    shell: { nativeViewport: true, webgpuNotice: 'unused' },
    gpu: false,
  });
  try {
    const viewport = page.dom.viewport;
    const [button, field, gizmos] = viewport.controls;
    const before = page.posted.length;

    // A press on a toolbar button must not post input nor capture the pointer.
    let captured = 0;
    viewport.setPointerCapture = () => {
      captured += 1;
    };
    viewport.dispatch('pointerdown', {
      clientX: 110,
      clientY: 70,
      buttons: 1,
      pointerId: 1,
      target: button,
    });
    assert.strictEqual(page.posted.length, before, 'a button press is not viewport input');
    assert.strictEqual(captured, 0, 'a button press must not steal pointer capture');

    // Nor a move or a release that started on the control.
    viewport.dispatch('pointermove', { clientX: 130, clientY: 70, buttons: 1, target: button });
    viewport.dispatch('pointerup', { clientX: 130, clientY: 70, buttons: 0, target: button });
    assert.strictEqual(page.posted.length, before, 'button events never reach the shell');

    // A wheel over a snap field must not be suppressed nor forwarded: the field
    // needs its default scrolling/stepping behaviour.
    let prevented = false;
    viewport.dispatch('wheel', {
      clientX: 120,
      clientY: 80,
      deltaY: -60,
      target: field,
      preventDefault: () => {
        prevented = true;
      },
    });
    assert.strictEqual(prevented, false, 'wheel over a field keeps its default');
    assert.strictEqual(page.posted.length, before, 'wheel over a field is not camera zoom');

    // The gizmo toolbar is a control container: a press on it is the toolbar's.
    viewport.dispatch('pointerdown', {
      clientX: 120,
      clientY: 80,
      buttons: 1,
      pointerId: 2,
      target: gizmos,
    });
    assert.strictEqual(page.posted.length, before, 'the gizmo toolbar owns its presses');

    // The backing surface still works: the filter must not swallow real drags.
    viewport.dispatch('pointerdown', { clientX: 110, clientY: 70, buttons: 1, pointerId: 3 });
    assert.strictEqual(page.posted.at(-1).type, 'viewport_input');
    assert.strictEqual(captured, 1, 'a drag on the surface still captures the pointer');
  } finally {
    page.restore();
  }
});

test('keys reach the engine only while the viewport has focus', async () => {
  const page = await bootPage({
    shell: { nativeViewport: true, webgpuNotice: 'unused' },
    gpu: false,
  });
  try {
    const before = page.posted.length;
    // Focus elsewhere (the inspector, a shortcut field): a WASD press is that
    // control's, not the camera's.
    globalThis.document.activeElement = page.dom['mode-3d'];
    page.dispatchWindow('keydown', { code: 'KeyW' });
    assert.strictEqual(page.posted.length, before, 'keys outside the viewport are ignored');

    // Focus inside the box: the key travels as the engine's raw aliases.
    const viewport = page.dom.viewport;
    globalThis.document.activeElement = viewport;
    page.dispatchWindow('keydown', { code: 'KeyW' });
    const down = page.posted.at(-1);
    assert.strictEqual(down.type, 'viewport_input');
    assert.deepStrictEqual(down.data.pressed_keys, [17, 87], 'W maps to its alias pair');

    // A second key adds to the held set rather than replacing it.
    page.dispatchWindow('keydown', { code: 'ArrowUp' });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, [17, 87, 38]);

    // Releasing one leaves the other held.
    page.dispatchWindow('keyup', { code: 'KeyW' });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, [38]);

    // A key the engine does not bind is not forwarded at all.
    const afterBound = page.posted.length;
    page.dispatchWindow('keydown', { code: 'KeyQ' });
    assert.strictEqual(page.posted.length, afterBound, 'unbound keys are not forwarded');

    // Losing the window releases held keys, or the camera would keep moving.
    page.dispatchWindow('blur', {});
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, []);
  } finally {
    page.restore();
  }
});

test('a key released after focus left the viewport is still released', async () => {
  // The blocker: hold W over the viewport, click into the inspector, release W.
  // The key-up arrives with focus elsewhere; dropping it left `heldKeys` holding
  // W forever, so the camera kept flying until the window lost focus.
  const page = await bootPage({
    shell: { nativeViewport: true, webgpuNotice: 'unused' },
    gpu: false,
  });
  try {
    const viewport = page.dom.viewport;
    globalThis.document.activeElement = viewport;
    page.dispatchWindow('keydown', { code: 'KeyW' });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, [17, 87]);

    // Focus moves to the inspector (a control outside the viewport box)...
    globalThis.document.activeElement = page.dom['mode-3d'].parent || page.dom['mode-3d'];
    // ...and W is released there. The release must still reach the engine.
    page.dispatchWindow('keyup', { code: 'KeyW' });
    const released = page.posted.at(-1);
    assert.strictEqual(released.type, 'viewport_input');
    assert.deepStrictEqual(
      released.data.pressed_keys,
      [],
      'a release is honoured regardless of where focus moved to',
    );

    // And nothing is left held: the next press is the only key down.
    globalThis.document.activeElement = viewport;
    page.dispatchWindow('keydown', { code: 'ArrowUp' });
    assert.deepStrictEqual(
      page.posted.at(-1).data.pressed_keys,
      [38],
      'W must not still count as held',
    );
  } finally {
    page.restore();
  }
});

test('focus leaving the viewport releases every held key', async () => {
  const page = await bootPage({
    shell: { nativeViewport: true, webgpuNotice: 'unused' },
    gpu: false,
  });
  try {
    const viewport = page.dom.viewport;
    globalThis.document.activeElement = viewport;
    page.dispatchWindow('keydown', { code: 'KeyW' });
    page.dispatchWindow('keydown', { code: 'ArrowLeft' });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, [17, 87, 37]);

    // A `focusout` that lands outside the box ends keyboard navigation: no
    // key-up is coming, so the page releases everything itself.
    viewport.dispatch('focusout', { relatedTarget: page.dom['mode-3d'] });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, []);

    // Moving focus *within* the box must not release anything.
    globalThis.document.activeElement = viewport;
    page.dispatchWindow('keydown', { code: 'KeyS' });
    assert.deepStrictEqual(page.posted.at(-1).data.pressed_keys, [31, 83]);
    viewport.dispatch('focusout', { relatedTarget: viewport });
    assert.deepStrictEqual(
      page.posted.at(-1).data.pressed_keys,
      [31, 83],
      'an internal focus move keeps the held keys',
    );
  } finally {
    page.restore();
  }
});

test('a moved viewport box re-posts its bounds without a resize', async () => {
  // `ResizeObserver` only fires on a size change. A viewport that merely shifts
  // (top bar grows, a banner appears) would leave the native layer behind, so
  // the client also re-reads the rectangle per frame and posts a change.
  const page = await bootPage({
    shell: { nativeViewport: true, webgpuNotice: 'unused' },
    gpu: false,
  });
  try {
    // The stub records scheduled frames; calling the newest one simulates the
    // next frame of the per-frame bounds check.
    const { frames } = page;
    assert.ok(frames.length > 0, 'the native client must schedule a layout follow-up');

    const postedBefore = page.posted.length;
    // Nothing moved: no post.
    frames.at(-1)();
    assert.strictEqual(page.posted.length, postedBefore, 'an unchanged box is not re-posted');

    // Shift the box by 12 px without changing its size.
    page.dom.viewport.getBoundingClientRect = () => ({
      left: 112,
      top: 60,
      width: 640,
      height: 480,
      right: 752,
      bottom: 540,
    });
    frames.at(-1)();
    const moved = page.posted.at(-1);
    assert.strictEqual(moved.type, 'viewport_bounds');
    assert.strictEqual(moved.data.x, 112, 'the shift is reported');
    assert.strictEqual(moved.data.width, 640, 'the size is unchanged');
  } finally {
    page.restore();
  }
});

test('a real init failure stays a loud red canvas, not the environment notice', async () => {
  // WebGPU is present, the adapter works, and the renderer still failed (a
  // shader bug): actionable, so the `failed` marker and the title must survive.
  // Only an environment limit (no `navigator.gpu`, or a null adapter) is the
  // case the written notice replaces.
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: 'environment notice' },
    gpu: true,
    stub: WASM_STUB_BROKEN_SHADER,
  });
  try {
    assert.ok(
      page.dom.canvas.classes.has('failed'),
      'the canvas must be marked failed for a real init error',
    );
    assert.ok(
      page.dom.canvas.title.startsWith('WebGPU init failed:'),
      `title reports the cause, got ${page.dom.canvas.title}`,
    );
    assert.strictEqual(
      page.dom['viewport-notice'].hidden,
      true,
      'the environment notice must not hide a real defect',
    );
  } finally {
    page.restore();
  }
});

test('a null adapter is an environment limit, not a red canvas', async () => {
  // `navigator.gpu` exists but `requestAdapter()` resolves null — Chrome on a
  // Linux box without the right flags. The page now discovers this *before*
  // fetching wasm, from the environment itself rather than from an error
  // message, so the written notice is the honest answer and the renderer is
  // never even blamed.
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: 'no GPU here' },
    gpu: true,
    adapter: false,
  });
  try {
    assert.strictEqual(
      page.dom['viewport-notice'].hidden,
      false,
      'the notice must be shown when the adapter cannot be created',
    );
    assert.strictEqual(page.dom['viewport-notice-body'].textContent, 'no GPU here');
    assert.ok(
      !page.dom.canvas.classes.has('failed'),
      'a missing adapter is not a renderer defect',
    );
    assert.deepStrictEqual(
      page.calls,
      [],
      'the module is not even downloaded when there is no adapter',
    );
  } finally {
    page.restore();
  }
});

test('an adapter query that throws is treated as no adapter', async () => {
  // Some implementations report failure by rejecting rather than resolving
  // null; the page must reach the same conclusion either way.
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: 'no GPU here' },
    gpu: true,
    adapter: 'throw',
  });
  try {
    assert.strictEqual(page.dom['viewport-notice'].hidden, false);
    assert.deepStrictEqual(page.calls, [], 'no download when the query throws');
  } finally {
    page.restore();
  }
});

test('only the exact "adapter not found" message is an environment limit', async () => {
  // The pre-flight check passed, yet the renderer still reports no adapter (it
  // disappeared between the two calls). `ornis-wasm` emits this one exact
  // string, so it is still environment — and it must be matched verbatim.
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: 'no GPU here' },
    gpu: true,
    stub: WASM_STUB_ADAPTER_LOST,
  });
  try {
    assert.strictEqual(
      page.dom['viewport-notice'].hidden,
      false,
      `${ADAPTER_NOT_FOUND} is an environment limit`,
    );
    assert.ok(!page.dom.canvas.classes.has('failed'));
  } finally {
    page.restore();
  }
});

test('a message merely containing "adapter not found" is a defect', async () => {
  // Pins the comparison as `===` rather than `includes`: if the check ever
  // loosens, these pass the substring test and must still be red. This is the
  // mutation a reviewer applied by hand; here it fails a named test.
  for (const message of ADAPTER_LOOKALIKES) {
    assert.notStrictEqual(
      message,
      ADAPTER_NOT_FOUND,
      'the lookalikes must differ from the exact string',
    );
    assert.ok(
      message.includes(ADAPTER_NOT_FOUND),
      `"${message}" must contain the exact string, or it does not test \`includes\``,
    );
    const page = await bootPage({
      shell: { nativeViewport: false, webgpuNotice: 'environment notice' },
      gpu: true,
      stub: `
export default async () => {
  throw new Error(${JSON.stringify(message)});
};`,
    });
    try {
      assert.ok(
        page.dom.canvas.classes.has('failed'),
        `an \`includes\` match must not soften this defect: ${message}`,
      );
      assert.strictEqual(
        page.dom['viewport-notice'].hidden,
        true,
        `the notice must not appear for a lookalike: ${message}`,
      );
    } finally {
      page.restore();
    }
  }
});

test('the adapter pre-flight asks for the same adapter the renderer uses', async () => {
  // A check that queries a different adapter than the renderer proves nothing.
  // `ornis-wasm` requests `PowerPreference::HighPerformance`, so the probe must
  // pass the matching options — otherwise it could validate the integrated GPU
  // and render on the discrete one (or vice versa).
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: 'unused' },
    gpu: true,
  });
  try {
    assert.ok(page.adapterCalls.length >= 1, 'the pre-flight must query an adapter');
    assert.deepStrictEqual(
      page.adapterCalls[0],
      { powerPreference: 'high-performance' },
      'the probe must request the high-performance adapter, like ornis-wasm',
    );
  } finally {
    page.restore();
  }
});

test('renderer defects that mention WebGPU or adapters stay a red canvas', async () => {
  // The bug this pins: a broad message regex turned real defects into a help
  // blurb. Every one of these mentions WebGPU, an adapter, or "not supported"
  // and must still be a loud failure.
  for (const message of DEFECT_MESSAGES) {
    const page = await bootPage({
      shell: { nativeViewport: false, webgpuNotice: 'environment notice' },
      gpu: true,
      stub: `
export default async () => {
  throw new Error(${JSON.stringify(message)});
};`,
    });
    try {
      assert.ok(
        page.dom.canvas.classes.has('failed'),
        `must stay a red canvas: ${message}`,
      );
      assert.ok(
        page.dom.canvas.title.startsWith('WebGPU init failed:'),
        `title must report the defect: ${message}`,
      );
      assert.strictEqual(
        page.dom['viewport-notice'].hidden,
        true,
        `the environment notice must not hide a defect: ${message}`,
      );
    } finally {
      page.restore();
    }
  }
});

test('a healthy WebGPU host never shows the notice', async () => {
  // The other half of the contract: the fallback must not appear when the
  // renderer boots, otherwise a working Chrome session loses its viewport.
  const page = await bootPage({
    shell: { nativeViewport: false, webgpuNotice: 'no WebGPU here' },
    gpu: true,
  });
  try {
    assert.strictEqual(page.dom['viewport-notice'].hidden, true);
    assert.ok(!page.dom.canvas.classes.has('failed'));
    assert.strictEqual(page.dom.canvas.hidden, false);
  } finally {
    page.restore();
  }
});
