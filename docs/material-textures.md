# Material mipmaps and KTX2

`ornis-texture` is a GPU-independent seam shared by `ornis-gltf` and
`ornis-render`. `LoadedImage` is now `TextureImage::{Rgba8, Ktx2}` rather than
an RGBA-only struct. PNG/JPEG callers can inspect `image.rgba8()`; callers
uploading imported images should use `TextureCache::upload_image_cached`.
The existing `CpuImage` / `upload_texture` / `upload_cached` API remains.

## Filtering and storage

- RGBA8 gets `floor(log2(max(width,height))) + 1` levels. Area filtering
  includes odd-sized edges and handles rectangular / single-axis images.
- Base color and emission average RGB in linear light and re-encode sRGB;
  alpha and metallic/roughness remain linear. Normal RGB is decoded to
  [-1,1], averaged and normalized; a cancelling vector becomes +Z.
- KTX2 preserves **every authored level**, including partial chains, without
  regenerating or renormalizing it. Filtering responsibility belongs to the
  offline author for these images.
- ETC1S/BasisLZ and UASTC LDR (uncompressed or Zstd) transcode to BC7, ASTC
  4×4, ETC2 RGBA8, or RGBA8, in that preference order. Selection uses the
  **enabled device** features, not just adapter advertisement.
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
most 8192 pixels per edge. Unsupported shapes/codecs and native-block
supercompression return typed errors. Basis decompression lengths are
bounded before transcoding. HDR/IBL textures retain their separate path.

## glTF integration

`KHR_texture_basisu` works with required or optional extensions, with or
without a core fallback `source`. The extension source takes precedence.
It is validated and promoted before **normal gltf-rs document validation**;
unknown required extensions and invalid core indices still fail. KTX2 image
bytes may come from a buffer view, data URI or a confined external path.
Images stay encoded through import and deduplicate before transcoding.

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
available, on a compression-enabled device. Run the same test on a Mac for
Metal/ASTC/BC verification; a software-adapter result is not a Mac result.

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
