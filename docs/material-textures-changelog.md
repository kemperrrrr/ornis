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
