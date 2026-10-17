# Anti-aliasing resolve (#123, groundwork)

`ornis_render::aa` provides an opt-in GPU resolve. It does **not** enable TAA
in the editor or replace the default `RenderFrame3D` pipeline.

## Modes and cost model

| Mode | Geometry samples requested | Fullscreen draws | Additional work |
| --- | ---: | ---: | --- |
| Off | 1 | 1 (copy) | no filtering |
| MSAA | 4 | 1 (copy) | negotiate geometry count against adapter |
| FXAA | 1 | 1 | contrast early-out, bounded directional filtering |
| SMAA-like | 1 | 3 | edge detection, bounded 16-pixel searches, neighbourhood blend |
| TAA | 1 | 1 | reprojection, point-depth rejection, 3×3 clamp |

The morphological implementation is lookup-free with analytic approximate
area weights, **not reference SMAA**: diagonal and corner refinement and the
reference search/area LUTs are not implemented.

The current owner eagerly allocates four RGBA16Float targets: two history
buffers, edge and weight buffers, **32 bytes/pixel** (63.3 MiB at 1920×1080).
No readbacks or texture copies are recorded by the resolve. Uniform buffers
are frame-local, so multiple frames recorded before submission cannot all
observe the final frame's parameters. These are structural cost estimates,
not measured GPU timings; performance acceptance remains outstanding.

## Contract

1. Create one `AaResolve` per camera, for the output format and viewport.
2. Before rendering geometry, call `jitter()` and `jitter_projection()`.
   Shadows and UI use their unjittered projections.
3. Supply a single-sampled linear color texture before tone mapping/UI and
   a same-size RGBA16Float motion texture:
   - `xy`: previous UV minus current UV (UV y points down), including jitter;
   - `z`: current positive linear depth;
   - `w`: expected previous positive linear depth, or negative for rejection.
4. Call `render`; tone map its output and draw UI afterwards.
5. Call `resize` when resizing, and `reset_history` for camera cuts,
   teleports, projection discontinuities, scene replacements or dropped frames.
   `set_mode` invalidates history automatically. Do not share an instance
   between cameras or render targets. Submit recorded frames in order.

`motion_vector` converts two clip positions to the UV convention above.
For moving/skinned meshes, previous clip coordinates must come from the
**previous object transform and previous deformed vertex**, not the current
vertex and previous camera alone. Depth is stored in half precision and must
fit its finite range. Disocclusion uses expected previous depth; interpolated
history depth must not be used across silhouettes.

`AaPass` (re-exported by `frame_passes`, native only) declares reads of `Hdr`
and imported `AaMotion`, and writes a separate transient `AaColor`. A custom
plan must register these resources, provide the imported motion view, run the
node after lighting, and feed `AaColor` to its composite. History is owned by
the node, never pooled or aliased. Its resize/reset calls must accompany plan
resize/camera cuts. On wasm call `AaResolve` directly: the scheduler requires
`Send` passes, while wasm wgpu handles are not `Send`.

## Validation and remaining integration

Tests cover Halton periodicity, clip-space jitter, motion sign, a CPU temporal
convergence reference and bright-trail clamp/rejection. A headless GPU test
records all five modes, multiple frames per encoder and checks a flat-field
readback. Shader parsing/validation is also tested with Naga.

Still required before closing #123:

- real G-buffer velocity output and previous transforms/deformed vertices,
  including skinning, extraction lifetime/reset rules and first-frame handling;
- production frame-plan/composite wiring and editor/wasm mode selection;
- reference SMAA search/area tables and diagonal/corner handling;
- static-scene GPU goldens and moving/skinned-scene ghosting regressions;
- measured GPU pass timings and a green full quality gate.
