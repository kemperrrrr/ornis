# Material texture API migration (PR #128)

This unreleased change affects the Rust import/render API:

- `ornis_gltf::LoadedImage` is now the shared
  `TextureImage::{Rgba8(CpuImage), Ktx2(Ktx2Image)}` enum. Use `dimensions()`
  for either variant, `rgba8()` for decoded pixels, and
  `TextureCache::upload_image_cached` to upload without expanding KTX2 early.
- The former `LoadedMaterial::{base_color_texture, metallic_roughness_texture,
  emissive_texture}` fields move to `LoadedMaterial::textures`
  (`LoadedTextures`). Untextured literals use `textures: Default::default()`.
  `material.texture(role)` remains the role-based accessor.
- `TextureRole` gains `Normal`; exhaustive matches must include it.
  `LoadedTextures::normal_texture` retains the image, and CPU mip generation
  renormalizes it. Tangent-space normal shading is not added here.
- The RGBA8 `CpuImage`, `upload_texture` and `TextureCache::upload_cached`
  entry points remain; uploads now generate full mip chains.
- Material samplers request anisotropic filtering. `upload_texture`,
  `upload_image`, `TextureCache::upload_cached` and
  `TextureCache::upload_image_cached` take an `AnisotropyLevel`
  (`OFF` = 1, `DEFAULT` = 16, `new(u16)` saturates into `1..=16`);
  `SamplerKind::LinearRepeat` carries the level, so exhaustive matches on it
  need the new payload. The level a frame samples with is
  `RenderBackendConfig::material_anisotropy` (default 16), forwarded to the
  textured-forward `material_sampler`. The texture cache key does **not**
  include the level: the same image at another level reuses one GPU texture.
  `SamplerKind::LinearClamp` (composite, HDR layers, shadow maps) stays
  isotropic and unchanged.
- Optional `KHR_texture_basisu` now falls back to the core PNG/JPEG on
  unsupported/corrupt containers or unavailable sources. Required Basis
  returns an error. Unused image/fallback payloads are no longer decoded.
  Invalid indices are still rejected even when their payload is unused.
- Basis must be declared in `extensionsUsed`, MIME declarations must match
  signatures, and core-only KTX2 images are rejected.
- CPU PNG/JPEG import retains the old ability to decode edges over 8192;
  GPU upload remains bounded by the portable/device limits.

KTX2 dependencies remain enabled by default and mandatory in this iteration.
Making them optional needs a coordinated feature design in `ornis-texture`,
`ornis-gltf` and `ornis-render` (the public image type is shared); this is not
silently promised by a glTF-only switch. Compression **device features** are
requested at runtime from `wgpu::Adapter::features`, not through Cargo flags.
The transitive `paste` advisory is unmaintained, not ignored or newly allowed;
replacement belongs with a maintained native-block decoder update.

# Offline texture bake (issue #129)

Additive: nothing existing changed meaning, and every new behaviour is opt-in.

- `ornis-texture` exports the KTX2 writer behind the runtime loader:
  `encode_rgba8_ktx2(&CpuImage, TextureRole) -> Vec<u8>` writes the role-filtered
  mip chain (transfer function follows the role: sRGB for base color/emissive,
  linear otherwise), and `ktx2_container(format, width, height, &levels)` writes
  one from caller-authored level payloads. Both are pure functions of their
  inputs, so a container is byte-reproducible.
- New `ornis_assets::bake`: `BakeSettings` (per-role `BasisMode` overrides, cache
  directory, `EncoderSource`), `bake_texture`, `bake_model`, `Baked`,
  `BakeOrigin`, `BakeStats` and the typed `BakeError`. Only `bake_model`
  requires the `gltf` feature; the rest of the crate's default build has it on.
- `ornis-assets` now depends on `ornis-texture` unconditionally (the bake types
  are in the public API); `crates/assets/Cargo.toml` and `Cargo.lock` are the
  only manifest changes.
- `AssetServer` gains `with_texture_baking(BakeSettings)` and `bake_stats()`
  (feature `gltf`). Without `with_texture_baking` the import path is unchanged:
  decoded PNG/JPEG pixels, mips generated at upload. With it, decoded material
  slots become validated KTX2 containers at import and reload time, and a slot
  whose bake fails keeps its decoded pixels instead of failing the import.
- Compressed payloads need the reference `basisu` binary. `EncoderSource::InProcess`
  writes the in-process RGBA8 chain on purpose (`BakeOrigin::Authored`).
  `EncoderSource::Path` with no `basisu` on an absolute `PATH` entry does the
  same write but records `BakeOrigin::Downgraded` and `BakeStats::downgraded`:
  RGBA8 does not meet the compressed-size goal. The cache key includes schema,
  tool identity and the role's pinned flags, so that RGBA8 entry is not reused
  once `basisu` is available. Colour space is pinned per role (`-srgb` /
  `-linear` / `-normal_map -mip_renorm`).
