# Material mipmaps and KTX2

`ornis-texture` is a GPU-independent seam shared by `ornis-gltf` and
`ornis-render`. `LoadedImage` is now `TextureImage::{Rgba8, Ktx2}` rather than
an RGBA-only struct. Material image fields are grouped under
`LoadedMaterial::textures` (`LoadedTextures`); `material.texture(role)` remains
the role-based accessor. PNG/JPEG callers can inspect `image.rgba8()`; callers
uploading imported images should use `TextureCache::upload_image_cached`.
The existing `CpuImage` / `upload_texture` / `upload_cached` entry points remain.
They do not take an anisotropy level (see below).

## Filtering and storage

- RGBA8 gets `floor(log2(max(width,height))) + 1` levels. Area filtering
  includes odd-sized edges and handles rectangular / single-axis images.
- Material samplers request anisotropic filtering: `AnisotropyLevel::DEFAULT`
  is 16, `wgpu`'s clamp ceiling and the `maxAnisotropy` every target driver
  reports. The level that a frame actually samples with is
  `RenderBackendConfig::material_anisotropy` (default 16), stored on
  `Renderer3D` and bound as the one shared `material_sampler` of the
  textured-forward pass. `AnisotropyLevel::new` saturates a raw request into
  `1..=16` before that value is stored on the config or passed to
  `set_material_anisotropy`, which rebuilds the sampler when the pass
  already exists. Upload entry points do not take a level:
  `GpuTexture::sampler` is fixed at `AnisotropyLevel::DEFAULT`, and the
  direct footprint gate builds the `OFF` and `DEFAULT` samplers itself.
  The cache key is the image and its role. All filters stay linear (what `wgpu`
  requires above 1), and an adapter without
  `DownlevelFlags::ANISOTROPIC_FILTERING` silently receives level 1 — the
  documented `wgpu` degradation, not an error — so requesting 16 is safe on
  every backend. Shadow and post-process samplers are
  `SamplerKind::LinearClamp` and stay isotropic: their fullscreen quads never
  magnify a grazing surface.
- Base color and emission average RGB in linear light and re-encode sRGB;
  alpha and metallic/roughness remain linear. Normal RGB is decoded to
  [-1,1], averaged and normalized; a cancelling vector becomes +Z.
- KTX2 preserves **every authored level**, including partial chains, without
  regenerating or renormalizing it. Filtering responsibility belongs to the
  offline author for these images.
- ETC1S/BasisLZ and UASTC LDR (uncompressed or Zstd) transcode to BC7, ASTC
  4×4, ETC2 RGBA8, or RGBA8, in that preference order. Selection uses the
  **enabled device** features, not just adapter advertisement.
  Basis' bundled Zstd feature is disabled: inflation uses `ruzstd >= 0.9.1`
  (RUSTSEC-2026-0344 fixed), bounded output and a per-mip frame-window limit.
- Native RGBA8, BC1 RGBA, BC3, BC7, ASTC 4×4 LDR and ETC2 RGBA8 pass through
  when supported. Unsupported native compression decodes all levels to
  RGBA8. Compressed base dimensions not divisible by four use RGBA8 to
  comply with WebGPU texture-creation constraints.
- Uploads use physical block-rounded mip extents, including 2×2/1×1 tails.
  `Queue::write_texture` accepts tight rows; only buffer copies need
  256-byte row alignment. sRGB vs linear GPU formats follow material role.
- `texture_compression_features(adapter.features())` supplies the optional
  features at device creation (enabled in the desktop runner). Clients
  creating their own device can omit them and receive RGBA8 fallback.

KTX2 material support is deliberately 2D, non-array, single-face, LDR, at
most 8192 pixels per edge. `level_count == 0` is rejected rather than
interpreted as a request to generate levels: authored KTX2 levels are preserved.
Unsupported shapes/codecs and native-block
supercompression return typed errors. Basis decompression lengths are
bounded before transcoding. HDR/IBL textures retain their separate path.

## glTF integration

`KHR_texture_basisu` works with required or optional extensions, with or
without a core fallback `source`. A valid extension source takes precedence.
For optional Basis, an unsupported/corrupt container or a missing/unavailable
image source selects the core PNG/JPEG when one exists. Required Basis or
optional Basis without a core source still returns the error. Unsafe paths
never trigger fallback. Only selected sources are decoded: a missing or broken
unused fallback does not reject an otherwise valid material.
It is validated and promoted before **normal gltf-rs document validation**;
unknown required extensions and invalid core indices still fail. KTX2 image
bytes may come from a buffer view, data URI or a confined external path.
KTX2 stays encoded through import and deduplicates before transcoding. Source
selection validates containers; corrupted codec bitstreams discovered only by
deferred GPU transcoding still return an upload error.

`KHR_texture_basisu` must be listed in `extensionsUsed`; explicit MIME types
must agree with the image signature, and KTX2 is rejected in a core-only
texture. PNG/JPEG imports retain dimensions above 8192 (CPU asset users may
need them); the render upload API, not the importer, enforces GPU size limits.
The four imported material slots use the same texture-indexed source resolver.

The normal image is retained and can be uploaded under `TextureRole::Normal`.
This work does not add tangent-space normal shading to the renderer, change
UV-set/sampler import, or change the editor's existing scalar-only
`scene_from_model` conversion. Callers using the material-texture API receive
the new mip/compression behavior; the editor scene conversion still discards
texture bindings and mesh UVs as before.

## Tests and memory accounting

```sh
cargo test -p ornis-texture
cargo test -p ornis-gltf
cargo test -p ornis-render --test material_mips -- --test-threads=1
cargo test -p ornis-render textures::tests
cargo clippy --workspace --all-targets -- -D warnings
```

`material_mips` requires an adapter in CI (lavapipe). It compares a
perspective tilted checker with an independent analytic golden and checks
that the base-only control aliases. It samples all six authored levels of
real ETC1S/UASTC fixtures on a device with no compression features and, if
available, on a compression-enabled device. Additional GPU gates force native
BC7/ASTC/ETC2 to RGBA8 without device features, and verify Basis/native 30×30
images on a compression-enabled device. A CPU BC1-alpha regression pins the
RGBA8 transcode fallback; KTX2 proptests cover arbitrary bytes and structured
mutations. Run the same test on a Mac for
Metal/ASTC/BC verification; a software-adapter result is not a Mac result.

The tilted-plane golden is isotropic by construction (`AnisotropyLevel::OFF`):
the checker averages to the same linear-light 188 at every level the plane
samples, so the golden pins the mip chain and not the anisotropy. A second
footprint gate reuses that plane with 32-texel bands along `v`: the isotropic
pass averages them to the same 188, while `AnisotropyLevel::DEFAULT` keeps
black/white stripes in the lower, stretched half, where the footprint's short
axis sizes the level. Both renders are counted on one interior window (the
plane's silhouette is excluded — derivatives are one-sided there, the same
boundary the golden gate budgets), and the gate requires the anisotropic pass
to produce many more band-coloured pixels than the isotropic one, so a level
of 1 and a missing mip chain both fail. On an adapter without
`ANISOTROPIC_FILTERING` it instead asserts the default-level image is
byte-identical to the isotropic one (`wgpu`'s degradation contract, exercised
as behavior rather than as an error). That gate uploads the image once and
builds both samplers with `sampler_descriptor_for_role`, so it does not
read `GpuTexture::sampler`.
`forward_tilted_bands_follow_backend_anisotropy` repeats it through
`create_render_backend`: the striped image is uploaded once, and
`RenderBackendConfig::material_anisotropy` of `OFF` and `DEFAULT` must
produce the same contrast. The stored sampler is `DEFAULT`, so a forward
pass that ignored the config and sampled it fails the `OFF` frame. After
the pass exists the setter is raised to 16 and then lowered back to `OFF`;
each frame has to match the config level.

`GpuTexture::payload_bytes` sums the exact uploaded mip payloads. CPU tests
verify these counts from actual transcoding, including block-rounded tails:

| Image, full chain | RGBA8 | BC7 / ASTC 4×4 / ETC2 RGBA8 | Reduction |
|---|---:|---:|---:|
| 40×40 fixture, 6 mips | 8,520 B | 2,240 B | 3.80× |
| 2048×2048, 12 mips | 22,369,620 B | 5,592,432 B | 4.00× |

Native BC1 uses 2,796,216 B for the 2048² chain (8.00× reduction).
For comparison, the former RGBA8 **base-only** 2048² upload was 16,777,216 B;
adding the full chain costs about 33%, whereas a BC7 full chain is 3× smaller
than that old base-only upload. These are exact texture payload budgets,
**not measured driver VRAM residency**: page alignment and driver metadata
are not exposed by this counter. The GPU test prints the chosen format and
payload after upload. Physical Mac VRAM profiling remains a manual check.

## Offline bake (`ornis-assets::bake`, issue #129)

The KTX2 containers above used to be something an external toolchain had to
produce. The `crates/assets/src/bake/` module (`bake.rs` for the policy and the
entry points, `encoder.rs` for the tool driver, `identity.rs` for the
tool version mixed into the cache key, `cache.rs` for the keys, `model.rs`
for the slot pass) now bakes them at import time, so a PNG/JPEG
model gets the same runtime treatment as a `KHR_texture_basisu` one: authored
levels, no runtime mip generation.

- **Codec policy.** `BasisMode::for_role` picks UASTC for
  `TextureRole::Normal` (quality where block artifacts hurt) and ETC1S for
  albedo, ORM and emissive (size). `BakeSettings::overrides` replaces any single
  slot, e.g. `{ BaseColor: Uastc }` for a hero texture.
- **Cache.** The key is FNV-1a 64 over bake schema `1`, the decoded pixels,
  their dimensions, the role, the codec and the producer, stored as
  `<key>.ktx2` under `BakeSettings::cache_dir`. It hashes *decoded content*,
  not the source file bytes, so a re-encoded PNG with identical texels hits
  the same entry, and two primitives sharing an image share one bake. The
  producer is `rgba8-writer-v1` for the in-process RGBA8 writer, or the tool's
  `-version` stdout plus `bin:` and an FNV-1a digest of the binary (the digest
  alone when `-version` fails) and the pinned flags, excluding file paths.
  Version text alone would collide: two builds that both print `basisu 2.10.0`
  (different commits, or SSE on versus off) must not share an entry. The digest
  is a cache key, not a security hash. An RGBA8 fallback and a `basisu` run
  therefore never share an entry, and upgrading the tool or changing a flag
  misses the old container instead of reusing it. The entry is written to a
  temporary file in the same directory and published with `rename`, so a reader
  never observes a partial container. `EncoderSource::Path`
  resolves only absolute `PATH` entries and canonicalizes the file, so an
  empty or relative entry cannot change the key with the current directory.
  An entry that stops parsing as KTX2 counts as a miss.
- **Encoder.** Basis ships no pure-Rust encoder, so compressed payloads come
  from the reference `basisu` command line — the project the runtime transcoder
  port (`crates/texture/src/basis.rs`) comes from. The shape follows the two
  established offline pipelines rather than inventing one: Bevy's
  asset-processing `basisu` step (per-slot `uastc`/`etc1s` policy) and
  KTX-Software's `toktx` (`--uastc`, `--bcmp`, generated mipmaps in the same
  KTX2 container). The pixel handoff is an uncompressed 32-bit TGA written
  beside the cache entry. Colour space is pinned per role, and the mip kernel
  is named so a tool default cannot move it. The reference tool's Kaiser
  kernel is not the in-process area filter; the two chains are different
  producers and do not share a cache entry.

  | Role | Colour-space flags |
  | --- | --- |
  | BaseColor, Emissive | `-srgb -mip_srgb` |
  | MetallicRoughness | `-linear -mip_linear` |
  | Normal | `-normal_map -linear -mip_linear -mip_renorm` |

  Codec flags are `-uastc -uastc_level 2` or `-etc1s -q 128`. Every command
  also passes `-mip_clamp -mip_filter kaiser`. BaseColor ETC1S:

  `basisu -ktx2 -mipmap -output_file <key>.tool.ktx2 -etc1s -q 128 -srgb -mip_srgb -mip_clamp -mip_filter kaiser <key>.tga`

  Normal UASTC (the role default; an ETC1S override keeps the normal flags):

  `basisu -ktx2 -mipmap -output_file <key>.tool.ktx2 -uastc -uastc_level 2 -normal_map -linear -mip_linear -mip_renorm -mip_clamp -mip_filter kaiser <key>.tga`

  `EncoderSource::Path` (the default) looks for `basisu` on absolute `PATH`
  entries, `EncoderSource::Binary` pins an exact tool, `EncoderSource::InProcess`
  never spawns a process. Each external process (`-version` and the encode) is
  killed after `BakeSettings::encoder_timeout` (default two minutes,
  `BakeError::TimedOut`). On Unix the child is its own process group and the
  deadline sends `SIGKILL` to that group, so a wrapper that forks instead of
  `exec` does not leave a grandchild holding the pipes. After the child
  exits (any status), pipe readers are joined through EOF until that same
  deadline, so a slow `-version` is not truncated into a different cache key.
  A grandchild that still holds a pipe at the deadline is killed with the
  group and the call is `TimedOut`; the partial output is not cached. After a timeout the readers are joined only for a
  short bound, then detached. Off Unix only the direct child is killed; the
  join bound still returns. Nothing from that run is cached. Tool output is validated through `Ktx2Image::from_bytes`
  (reader, DFD, level-length and dimension rules) *before* anything replaces
  the slot, and only the validated bytes are cached.
- **No encoder is a visible downgrade, not the size goal.** Without a `basisu`
  the bake still writes a container from `encode_rgba8_ktx2` — the same
  authored chain, RGBA8 blocks instead of the compressed ones — so the runtime
  mip cost disappears either way. That RGBA8 container does **not** meet the
  #129 compressed-size goal (the 4.00× / 8.00× figures above). An explicit
  `EncoderSource::InProcess` is `BakeOrigin::Authored`. A `PATH` lookup that
  finds nothing is `BakeOrigin::Downgraded`: `BakeStats::downgraded` counts
  those slots and `BakeStats::last_failure` records the reason. A later import
  that hits the cache stays `Downgraded` (it is not reported as `Cached`), so
  `bake_stats` still shows the downgrade and the reason. An explicit
  `InProcess` hit stays `Cached`. The two producers do not share a cache entry
  with a later `basisu` run.
- **The source stays.** A failed bake (missing pinned tool, non-zero exit,
  unparsable output) returns a typed `BakeError`; `bake_model` keeps the decoded
  PNG/JPEG pixels of that slot and counts it in `BakeStats::fallback`, with the
  reason in `BakeStats::last_failure`.

```rust
let mut server = AssetServer::new()
    .with_texture_baking(BakeSettings::new("target/ornis-texture-cache"));
let id = server.load_gltf_file(path)?;
let stats = server.bake_stats(); // encoded / cached / authored / downgraded / fallback
```

`bake_stats()` sums every model import and reload on that server. A later
model does not clear an earlier downgrade or fallback. `last_failure` keeps
the first reason unless a later pass has a failed slot, which replaces it.

The bake is import-time and lives on `AssetServer` (feature `gltf`): every
model written through `store`/`insert_at`, including `reload`, bakes its decoded
slots when a policy is set and is untouched when it is not. A slot already
holding KTX2 (`KHR_texture_basisu`) keeps the container the file authored.

### Tests for the bake

- `crates/texture/src/encode.rs` — round trip through the reader for all four
  roles, role/transfer-function pairing, determinism, invalid sources,
  rectangular chains, unknown formats rejected instead of guessed.
- `crates/texture/tests/ktx2.rs` — `native_container` is again an independent
  writer (a real Basis header plus a per-format DFD), so the native-format
  expectations are not fed by `ktx2_container`. A separate test checks that
  writer agrees with the oracle on level bytes. The bake fixtures assert sRGB
  for albedo UASTC and linear for the normal-map UASTC.
- `crates/assets/src/bake.rs` — role defaults and overrides, cache hits per
  (content, role, producer), the schema byte in the key, byte-level
  determinism, pinned CLI flags for every role and codec plus TGA bytes via a
  stub encoder, committed ETC1S/UASTC/normal-map round trips, the `ORNIS_BASISU`
  live-encoder check (a visible skip when unset), non-zero exit reporting, a
  killed hung encoder, atomic cache publish, the model slot pass, fallback,
  and the counted RGBA8 downgrade on a cache hit as well as the first write.
- `crates/texture/tests/data/bake-gradient.png` plus `bake-etc1s.ktx2`,
  `bake-uastc.ktx2` and `bake-normal-uastc.ktx2` — a 16×16 ramp encoded by
  basisu 2.10.0 (see that directory's README). The normal-map file uses
  `-normal_map -linear`.
- `crates/assets/src/server.rs` — a glTF with a PNG albedo keeps its decoded
  texels without a policy, and becomes a validated KTX2 chain with one.

## Next steps

- Ship a `basisu` build with the toolchain (or vendor a UASTC encoder) so the
  default path produces compressed containers without a manual install.
- Cache baked containers next to the source asset in the editor so a fresh
  checkout skips the first re-encode.
