# WASM pixel e2e — status and manual runbook

Live pixel e2e chain: editor scene → `GET /api/scene` → WASM `RenderWorld`
→ rendered frame → screenshot → golden compare.

## Status

There is no live pixel run in CI: CI has no browser and no WebGPU adapter.
This is recorded honestly, not papered over:

- `crates/wasm/src/lib.rs`, test `browser_pixel_e2e_gated_on_browser_and_webgpu`:
  passes with a `SKIP` message unless `ORNIS_E2E_BROWSER=1` is set **and**
  a browser binary is on `PATH`. The enabled path pins the contract input
  the harness renders, so contract drift still fails the test.
- `cargo test -p ornis-wasm` pins the same chain headlessly up to
  extraction: `FULL_CONTRACT` golden bytes
  (`full_contract_extract_golden_bytes_pinned`), snapshot versioning
  (`snapshot_versioning_replaces_without_stale_instances`), malformed-JSON
  robustness (`malformed_scene_json_is_rejected_without_panic`), plus the
  pre-existing `scene_snapshot_replace_scene_extracts_without_browser`
  parity (`frame_upload` vs `extract_render_data`).
- `crates/editor-backend/tests/scene_contract.rs` pins the server half of
  the boundary (`version`/`sequence` as `u64`, `entities`/`lights` arrays,
  `camera: null` placeholder as the documented WASM fallback signal).

## Manual run

Prerequisites: `cargo`, `node`/`npx`, `wasm-pack`, a browser
(`chromium`/`google-chrome`/`firefox`) or a playwright cache, and a WebGPU
adapter (real GPU or SwiftShader with `--enable-unsafe-swiftshader`).

```sh
cargo xtask e2e [--port 3420] [--out /tmp/ornis-e2e]
ORNIS_E2E_INSTALL=1 cargo xtask e2e   # allow browser download
```

The task builds the viewport exactly like `cargo xtask editor`
(`wasm-pack build crates/wasm --target web --out-dir editor/pkg`), starts
`cargo run --features editor-only` (fixed port 3420), polls `/api/scene`,
captures `frame.png` of the `#bevy` canvas (`editor/index.html`), and —
when ImageMagick `compare` exists and `editor/e2e_golden.png` is present —
reports the RMSE. Any missing capability is an honest `SKIP` (exit 0).

## Golden tolerances (never bytewise)

- max per-pixel absolute diff ≤ 12/255,
- mean absolute diff ≤ 2/255,
- mismatched-pixel fraction ≤ 0.5%.

First run bootstraps the golden: with no `editor/e2e_golden.png` the
harness stores the screenshot under `--out` for manual review; promote it
to `editor/e2e_golden.png` only after visual sign-off. Re-render with the
default orbit camera (`viewport.js` → `start_renderer('bevy')`) and a
fixed 800×600 viewport so runs are comparable.

## Known gap 2026-09-20: headless canvas renders solid red

Two harness runs captured a solid dark-red `#bevy` canvas even with a
warm published scene (5 entities, v5) after `#bevy` + 3 s settle, so the
golden is deliberately NOT bootstrapped from those shots. Probable cause:
headless playwright-chromium gets no WebGPU adapter (the CLI cannot
forward `--enable-unsafe-swiftshader`; only the system-chromium path
carries SwiftShader flags). Next step: software-WebGPU provisioning for
the playwright path (or a node-driven launch with explicit args), then
re-run → visual sign-off → promote. The harness already waits for a
published non-empty scene before shooting, so the empty-scene failure
mode is closed regardless.
