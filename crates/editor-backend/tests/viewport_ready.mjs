// Ornis editor-v2 — executable check of the viewport readiness queue.
//
// Runs the real `editor-v2/viewport.js` in Node against a stub wasm package
// and a minimal DOM, so the onReady queue, the effective-step repaint and the
// subscriber error isolation are exercised instead of grepped for. The label
// renderer is extracted from the real `live.js` and executed, not copied.
// Node builtins only — no npm dependencies. Driven by
// `editor_v2_viewport_ready_runs_in_node` (crates/editor-backend/src/remote.rs).
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const editorDir = fileURLToPath(new URL('../../../editor-v2/', import.meta.url));
const liveSource = readFileSync(join(editorDir, 'live.js'), 'utf8');

// The stub mirrors crates/editor/src/snap.rs: the step is coerced to a divisor
// of 360° and the coarse end is clamped to a half-turn. `undefined` for an
// unquantized channel is deliberate — that is what wasm-bindgen hands back for
// a Rust `None`, so the null contract of effectiveRotateStep is really tested.
// The formula is copied by hand and this harness covers the JS wiring, not the
// grid: a change to the Rust side would not fail here. It is owned by
// `effective_rotate_deg_reports_the_grid_in_use`,
// `rotate_step_is_coerced_to_a_divisor_of_a_full_turn` and
// `a_rotate_step_wider_than_a_half_turn_still_turns` (crates/editor/src/snap.rs)
// plus `effective_rotate_step_is_readable_from_the_browser`
// (crates/wasm/src/snap.rs) — keep this stub in step with them.
// The failure that must stay a loud red canvas is a *renderer defect* after a
// working adapter (here: shader compilation), not an environment limit: the page
// routes "no navigator.gpu" and "no adapter" to the written notice instead, so a
// message naming an adapter would test the wrong branch.
function stubSource(failInit) {
    const initBody = failInit
        ? "throw new Error('shader compilation failed: unresolved identifier');"
        : 'await gate;';
    return `
export const calls = [];
export let snapState = { enabled: false, translate: 0.5, rotateDeg: 15, scale: 0.25 };
let release;
export const gate = new Promise((resolve) => { release = resolve; });
export function openGate() { release(); }
export default async function init() { calls.push('init'); ${initBody} }
export async function start_renderer() { calls.push('start_renderer'); }
export function set_viewport_mode(m) { calls.push('set_viewport_mode:' + m); return true; }
export function set_gizmo_mode(m) { calls.push('set_gizmo_mode:' + m); return true; }
export function set_gizmo_space(s) { calls.push('set_gizmo_space:' + s); return true; }
export function set_editor_mode(m) { calls.push('set_editor_mode:' + m); return true; }
export function undo_gizmo() { calls.push('undo_gizmo'); return true; }
export function viewport_state_json() { return JSON.stringify({ mode: 'perspective3d', dpr: 1 }); }
export function set_snap(enabled, translate, rotateDeg, scale) {
    calls.push('set_snap');
    snapState = { enabled, translate, rotateDeg, scale };
    return true;
}
export function snap_effective_rotate_deg() {
    calls.push('read_step');
    const deg = snapState.rotateDeg;
    if (!snapState.enabled || !Number.isFinite(deg) || deg <= 0) return undefined;
    const divisions = Math.round(360 / deg);
    if (!Number.isFinite(divisions)) return undefined;
    return 360 / Math.max(divisions, 2);
}
`;
}

function makeEl() {
    const el = {
        style: {},
        title: '',
        width: 0,
        height: 0,
        hidden: false,
        added: [],
        // A real `classList` supports removal too; the viewport drops the
        // `failed` mark when it shows the written fallback instead.
        classList: {
            add: (c) => el.added.push(c),
            remove: (c) => {
                const at = el.added.indexOf(c);
                if (at >= 0) el.added.splice(at, 1);
            },
            toggle: () => {},
        },
        getBoundingClientRect: () => ({ width: 800, height: 600 }),
        addEventListener: () => {},
        setAttribute: () => {},
    };
    return el;
}

// Scratch directories are registered as they are created and removed on exit,
// so a failing assertion does not leave ornis-viewport-* behind in os.tmpdir().
const dirs = [];
process.on('exit', () => {
    for (const dir of dirs) rmSync(dir, { recursive: true, force: true });
});

// Each sandbox gets its own directory so the module is imported fresh: boot()
// runs once per module instance, and the failure path needs a second one.
async function sandbox(failInit) {
    const dir = mkdtempSync(join(tmpdir(), 'ornis-viewport-'));
    dirs.push(dir);
    mkdirSync(join(dir, 'pkg'));
    writeFileSync(join(dir, 'pkg', 'ornis_wasm.js'), stubSource(failInit));
    copyFileSync(join(editorDir, 'viewport.js'), join(dir, 'viewport.js'));
    const els = new Map();
    globalThis.document = {
        getElementById: (id) => {
            if (!els.has(id)) els.set(id, makeEl());
            return els.get(id);
        },
    };
    globalThis.window = { devicePixelRatio: 1 };
    // The harness models a WebGPU-capable browser: the viewport only boots the
    // wasm module when `navigator.gpu` exists (a WKWebView without it gets the
    // written fallback, covered by editor-v2/tests/viewport_host.test.js).
    // Node's own `navigator` is getter-only, hence `defineProperty`.
    Object.defineProperty(globalThis, 'navigator', {
        value: { gpu: {} },
        configurable: true,
        writable: true,
    });
    globalThis.requestAnimationFrame = () => 0;
    const wasm = await import(pathToFileURL(join(dir, 'pkg', 'ornis_wasm.js')).href);
    await import(pathToFileURL(join(dir, 'viewport.js')).href);
    return { dir, wasm, api: globalThis.window.OrnisViewport, canvas: els.get('canvas') };
}

// Pulls a whole function out of live.js by brace depth, so the label text is
// produced by the shipped code rather than by a paraphrase of it.
function extractFunction(source, name) {
    const start = source.indexOf(`function ${name}(`);
    assert.ok(start >= 0, `live.js must define ${name}`);
    let depth = 0;
    for (let i = source.indexOf('{', start); i < source.length; i += 1) {
        if (source[i] === '{') depth += 1;
        else if (source[i] === '}') {
            depth -= 1;
            if (depth === 0) return source.slice(start, i + 1);
        }
    }
    throw new Error(`unbalanced braces in ${name}`);
}

// The extracted function is loaded as a module of its own: its two free
// variables become module-level bindings, while `window` stays a global lookup
// so it resolves to the sandbox the call is made from.
async function loadLabelRenderer(dir) {
    const file = join(dir, 'label.mjs');
    writeFileSync(
        file,
        `export const label = { textContent: '' };
export const rotateInput = { value: '25' };
const snapRotateEffective = label;
const snapRotate = rotateInput;
${extractFunction(liveSource, 'renderEffectiveRotateStep')}
export { renderEffectiveRotateStep };
`,
    );
    return import(pathToFileURL(file).href);
}

async function settle(predicate) {
    for (let i = 0; i < 200 && !predicate(); i += 1) {
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
}

const realError = console.error;
function captureErrors() {
    const seen = [];
    console.error = (...args) => seen.push(args.map(String).join(' '));
    return { seen, restore: () => { console.error = realError; } };
}

// ── healthy boot: queue order, repaint, subscriber isolation ───────────────
const s = await sandbox(false);
const { api, wasm } = s;
assert.equal(typeof api.onReady, 'function', 'onReady must be exposed');
assert.equal(api.effectiveRotateStep(), null, 'before boot the read-back is null');

// The user enables snap and types 25° while wasm is still loading.
api.setSnap({ enabled: true, translate: 0.5, rotateDeg: 25, scale: 0.25 });
assert.equal(api.effectiveRotateStep(), null, 'the mirror alone cannot answer');
assert.ok(!wasm.calls.includes('set_snap'), 'the replica is not written before boot');

const { label, rotateInput, renderEffectiveRotateStep: render } = await loadLabelRenderer(s.dir);
render();
assert.equal(label.textContent, '', 'before boot the label stays empty');

// A throwing subscriber must not stop the next one, and must not reach
// boot().catch — the renderer is up, so the canvas is not a WebGPU failure.
const ran = [];
api.onReady(() => { throw new Error('cb boom'); });
api.onReady(() => { ran.push('second'); render(); });
assert.deepEqual(ran, [], 'subscribers wait for boot');

const bootErrors = captureErrors();
wasm.openGate();
await settle(() => ran.length > 0);
bootErrors.restore();

assert.deepEqual(ran, ['second'], 'the subscriber after the throwing one still runs');
assert.ok(wasm.calls.includes('set_snap'), 'boot re-applies the snap mirror');
assert.ok(
    bootErrors.seen.some((line) => line.includes('ready subscriber failed')),
    'the subscriber failure is logged',
);
assert.ok(!s.canvas.added.includes('failed'), 'the canvas must not be marked failed');
assert.equal(s.canvas.title, '', 'no "WebGPU init failed" title for a subscriber throw');
assert.ok(Math.abs(api.effectiveRotateStep() - 360 / 14) < 1e-3, 'the grid coerces 25°');
assert.equal(label.textContent, '≈25.714°', 'the repaint shows the coerced step');

// ── label text across the grid cases ───────────────────────────────────────
for (const [requested, expected] of [['15', '15°'], ['300', '≈180°'], ['7', '≈7.059°']]) {
    rotateInput.value = requested;
    api.setSnap({ enabled: true, translate: 0.5, rotateDeg: Number(requested), scale: 0.25 });
    render();
    assert.equal(label.textContent, expected, `requested ${requested}°`);
}

// Snap off: the stub returns undefined, the API must normalize it to null.
api.setSnap({ enabled: false, translate: 0.5, rotateDeg: 15, scale: 0.25 });
assert.equal(api.effectiveRotateStep(), null, 'undefined must be normalized to null');
render();
assert.equal(label.textContent, '', 'an unquantized channel shows no step');

// ── subscribing after boot ─────────────────────────────────────────────────
let late = 0;
api.onReady(() => { late += 1; });
assert.equal(late, 1, 'a late subscriber runs immediately, there is no second boot');

const lateErrors = captureErrors();
api.onReady(() => { throw new Error('late boom'); });
lateErrors.restore();
assert.ok(
    lateErrors.seen.some((line) => line.includes('ready subscriber failed')),
    'a late subscriber throw is logged, not propagated to the caller',
);

// ── a real init failure still marks the canvas ─────────────────────────────
const bad = await sandbox(true);
await settle(() => bad.canvas.added.includes('failed'));
assert.ok(bad.canvas.added.includes('failed'), 'a genuine init failure marks the canvas');
assert.ok(
    bad.canvas.title.startsWith('WebGPU init failed:'),
    `the title reports the init error, got ${bad.canvas.title}`,
);
assert.equal(bad.api.effectiveRotateStep(), null, 'without a replica there is no grid');

console.log('OK: onReady queue, effective-step repaint and error isolation verified in Node');
