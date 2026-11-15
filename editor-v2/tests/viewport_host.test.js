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
// Run: `node --test editor-v2/tests/` (CI: `cargo xtask quality`, stage
// `editor-viewport`). A missing `node` is a SKIP there, never a pass.
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

let copyCounter = 0;

// Minimal element: the viewport only reads boxes, toggles classes/attributes
// and registers listeners. `dispatch` runs them, which is how input
// forwarding is observed.
function element({ width = 640, height = 480, left = 100, top = 60, hidden = false } = {}) {
  const listeners = new Map();
  const classes = new Set();
  const attrs = new Map();
  return {
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
    dispatch: (type, event) => {
      // Real events always answer `preventDefault`; the viewport calls it on
      // wheel to keep the page from scrolling.
      const withDefaults = Object.assign({ preventDefault: () => {} }, event);
      for (const handler of listeners.get(type) || []) handler(withDefaults);
    },
  };
}

// Boots a fresh copy of `viewport.js` next to the stub WASM package with the
// DOM/window stubs installed as globals (the module reads them as globals,
// exactly like in a browser).
async function bootPage({ shell, gpu = true }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ornis-viewport-'));
  fs.mkdirSync(path.join(dir, 'pkg'));
  fs.writeFileSync(path.join(dir, 'pkg', 'ornis_wasm.js'), WASM_STUB);
  copyCounter += 1;
  const entry = path.join(dir, `viewport-${copyCounter}.mjs`);
  fs.writeFileSync(entry, SOURCE);

  const dom = {
    viewport: element(),
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
  };
  const posted = [];
  const windowListeners = new Map();
  const bodyClasses = new Set();
  const consoleMessages = [];
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
  const navigatorStub = gpu ? { gpu: {} } : { userAgent: 'Mozilla/5.0 AppleWebKit' };
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
    getElementById: (id) => dom[id] || null,
  };
  globalThis.ResizeObserver = function ResizeObserver() {
    this.observe = () => {};
    this.disconnect = () => {};
  };
  globalThis.requestAnimationFrame = () => 0;
  globalThis.console = {
    log: (message) => consoleMessages.push(String(message)),
    warn: (message) => consoleMessages.push(String(message)),
    error: (message) => consoleMessages.push(String(message)),
  };

  await import(entry);
  // Let the boot promise chain (the stub import included) settle.
  for (let i = 0; i < 50; i += 1) await Promise.resolve();
  await new Promise((resolve) => setImmediate(resolve));

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
    bodyClasses,
    consoleMessages,
    dispatchWindow: (type, event) => {
      for (const handler of windowListeners.get(type) || []) handler(event);
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
    assert.strictEqual(page.dom['viewport-notice-title'].textContent, 'Вьюпорт недоступен');
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
