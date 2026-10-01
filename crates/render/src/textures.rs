//! GPU upload of glTF texture images (`LoadedImage` → `wgpu` texture).
//!
//! CPU side mirrors `ornis-gltf` field-for-field without depending on it:
//! [`CpuImage`] is `LoadedImage` (`width`/`height`/RGBA8 `pixels`), and
//! [`TextureRole`] is the three-slot role enum — the wiring stays mechanical
//! (match on the role, upload, bind). Sampler parameters and `texCoord` sets
//! are intentionally not carried: the import ignores them and the upload uses
//! its own defaults (see [`sampler_descriptor_for_role`]).
//!
//! # Storage and lifetime
//!
//! [`GpuTexture`] owns one uploaded image (texture + view + sampler).
//! [`TextureCache`] owns the set: the caller keeps it next to `Renderer3D`
//! (e.g. inside `GpuFrameState`) and hands out `u32` handles. Identical
//! uploads (same role, size and pixels) deduplicate to one entry. There is no
//! snapshot system: like `RenderSubmit`, callers read the component lanes
//! directly (`extract_render_data` canon) and feed images here per frame.
//!
//! # Sampler defaults and the sRGB decision
//!
//! Filtering is linear (`mag`/`min`, `mipmap` reserved for a future mip
//! chain — a single mip is uploaded today), wrapping is `Repeat` on every
//! axis (the glTF default; the import carries no wrap mode), LOD clamps are
//! the full `0.0..=32.0` range, anisotropy is `1`, and there is no depth
//! comparison. Color roles ([`TextureRole::BaseColor`],
//! [`TextureRole::Emissive`]) upload as `Rgba8UnormSrgb`, so the hardware
//! decodes sRGB to linear on sample and the render stays in linear light
//! (the same convention as the composite pass color inputs). The data role
//! ([`TextureRole::MetallicRoughness`], green holds roughness, blue holds
//! metallic) uploads as `Rgba8Unorm`: linear data that must never be
//! decoded. CPU mirrors of both conversions live here
//! ([`sample_albedo_linear`], [`sample_metallic_roughness`]) so tests pin
//! the convention without an adapter.
//!
//! # What this module does not do
//!
//! The scalar `is_metallic` switch is untouched: a bound
//! metallic-roughness texture does not move it, the future shader path
//! samples the texture at runtime instead (UVs already vary through the
//! g-buffer as `input.uv`, so sampling needs only new bindings — no vertex
//! change). Skeleton work needs nothing from textures: texture binds are
//! per-material data, independent of skinning.
//!
//! `KHR_texture_transform` and WebP stay out of scope (the import rejects
//! WebP before decode).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Maximum texture edge in pixels accepted by [`CpuImage::validate`].
///
/// Matches the guaranteed WebGPU limit (`maxTextureDimension2D` floor), so an
/// image passing validation is uploadable on every backend.
pub const MAX_TEXTURE_EDGE: u32 = 8192;
/// Bytes per RGBA8 texel.
const RGBA8_BYTES: usize = 4;
/// Scale from an 8-bit channel to unit float.
const U8_TO_UNIT: f32 = 255.0;
/// sRGB IEC 61966-2-1 linear/power junction (normalized channel).
const SRGB_LINEAR_THRESHOLD: f32 = 0.04045;
/// Reciprocal of the sRGB linear-segment slope.
const SRGB_LINEAR_SLOPE: f32 = 12.92;
/// sRGB power-segment offset (numerator).
const SRGB_POWER_OFFSET: f32 = 0.055;
/// sRGB power-segment scale (denominator).
const SRGB_POWER_SCALE: f32 = 1.055;
/// sRGB power-segment exponent.
const SRGB_POWER_GAMMA: f32 = 2.4;

/// Which material slot an uploaded image feeds.
///
/// Mirrors `ornis_gltf::TextureRole` without depending on that crate; the
/// upload matches on this to pick the GPU format and the material binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureRole {
    /// `baseColorTexture`: albedo multiplier (sRGB color data).
    BaseColor,
    /// `metallicRoughnessTexture`: green holds roughness, blue holds
    /// metallic (linear data, never sRGB-decoded).
    MetallicRoughness,
    /// `emissiveTexture`: emission multiplier (sRGB color data).
    Emissive,
}

/// CPU-side texture image: always RGBA8, row-major, top row first.
///
/// Field-for-field mirror of `ornis_gltf::LoadedImage` (PNG and JPEG sources
/// both land here — the decoder normalizes channels, so the upload needs no
/// format switch). `pixels` holds exactly `width * height * 4` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 bytes, row-major from the top row.
    pub pixels: Vec<u8>,
}

impl CpuImage {
    /// Builds an image, validating dimensions and pixel length.
    ///
    /// # Errors
    ///
    /// Returns [`TextureUploadError`] when either dimension is zero, either
    /// edge exceeds [`MAX_TEXTURE_EDGE`], or `pixels` is not exactly
    /// `width * height * 4` bytes.
    pub fn from_rgba8(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<Self, TextureUploadError> {
        let image = Self {
            width,
            height,
            pixels,
        };
        image.validate()?;
        Ok(image)
    }

    /// Checks dimensions and pixel length (see [`Self::from_rgba8`]).
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_rgba8`].
    pub fn validate(&self) -> Result<(), TextureUploadError> {
        if self.width == 0 || self.height == 0 {
            return Err(TextureUploadError::EmptyDimensions {
                width: self.width,
                height: self.height,
            });
        }
        if self.width > MAX_TEXTURE_EDGE || self.height > MAX_TEXTURE_EDGE {
            return Err(TextureUploadError::ImageTooLarge {
                width: self.width,
                height: self.height,
            });
        }
        let expected = self.width as u64 * self.height as u64 * RGBA8_BYTES as u64;
        if self.pixels.len() as u64 != expected {
            return Err(TextureUploadError::PixelLengthMismatch {
                expected,
                actual: self.pixels.len() as u64,
            });
        }
        Ok(())
    }

    /// RGBA bytes of the texel at `(x, y)` (origin: top-left).
    ///
    /// Returns `None` when the coordinates are out of bounds. Pure helper
    /// for the CPU sampling mirrors below.
    pub fn texel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = (y as usize * self.width as usize + x as usize) * RGBA8_BYTES;
        let slice = self.pixels.get(offset..offset + RGBA8_BYTES)?;
        Some([slice[0], slice[1], slice[2], slice[3]])
    }
}

/// Rejection of a [`CpuImage`] before any GPU work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TextureUploadError {
    /// Either dimension is zero — there is no honest GPU texture for it.
    #[error("texture has empty dimensions ({width}x{height})")]
    EmptyDimensions {
        /// Offending width.
        width: u32,
        /// Offending height.
        height: u32,
    },
    /// An edge exceeds [`MAX_TEXTURE_EDGE`] (not uploadable everywhere).
    #[error("texture too large ({width}x{height})")]
    ImageTooLarge {
        /// Offending width.
        width: u32,
        /// Offending height.
        height: u32,
    },
    /// `pixels` is not exactly `width * height * 4` bytes.
    #[error("texture pixel length mismatch: expected {expected}, got {actual}")]
    PixelLengthMismatch {
        /// Required byte count.
        expected: u64,
        /// Actual byte count.
        actual: u64,
    },
}

/// GPU format for `role`: sRGB for color roles, linear for the data role.
///
/// [`TextureRole::BaseColor`] and [`TextureRole::Emissive`] are sRGB-encoded
/// color, so they upload as `Rgba8UnormSrgb` and the hardware decodes to
/// linear on sample. [`TextureRole::MetallicRoughness`] is linear
/// roughness/metallic data, so it uploads as `Rgba8Unorm` and is never
/// decoded. Pure (no GPU access).
pub fn texture_format_for_role(role: TextureRole) -> wgpu::TextureFormat {
    match role {
        TextureRole::BaseColor | TextureRole::Emissive => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureRole::MetallicRoughness => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Upload sampler defaults for `role` (all roles share them today).
///
/// Linear `mag`/`min` filtering, linear `mipmap` (reserved: a single mip is
/// uploaded, the chain is future work), `Repeat` wrapping on every axis (the
/// glTF default — the import carries no wrap mode), full LOD range,
/// anisotropy `1`, no depth comparison. Pure (no GPU access); the label is
/// `None` so the descriptor is `'static`.
pub fn sampler_descriptor_for_role(role: TextureRole) -> wgpu::SamplerDescriptor<'static> {
    let _ = role;
    crate::flags::SamplerKind::LinearRepeat.descriptor()
}

/// Decodes one sRGB-encoded channel byte to linear light.
///
/// This is what the hardware does on sample for the sRGB formats above
/// (exact transfer function, not the `pow(c, 2.2)` approximation): below the
/// `0.04045` junction the curve is linear (`c / 12.92`), above it is the
/// `((c + 0.055) / 1.055) ^ 2.4` power. Pure CPU mirror for tests.
pub fn srgb_channel_to_linear(byte: u8) -> f32 {
    let c = f32::from(byte) / U8_TO_UNIT;
    if c <= SRGB_LINEAR_THRESHOLD {
        c / SRGB_LINEAR_SLOPE
    } else {
        ((c + SRGB_POWER_OFFSET) / SRGB_POWER_SCALE).powf(SRGB_POWER_GAMMA)
    }
}

/// Samples one albedo/emissive texel as linear RGBA.
///
/// RGB channels decode via [`srgb_channel_to_linear`] (what the `Srgb`
/// format does in hardware); alpha is linear (`a / 255`). Pure CPU mirror
/// of texel-center `Nearest` sampling.
pub fn sample_albedo_linear(texel: [u8; 4]) -> [f32; 4] {
    [
        srgb_channel_to_linear(texel[0]),
        srgb_channel_to_linear(texel[1]),
        srgb_channel_to_linear(texel[2]),
        f32::from(texel[3]) / U8_TO_UNIT,
    ]
}

/// Samples one metallic-roughness texel as `(roughness, metallic)`.
///
/// Green holds roughness, blue holds metallic; both are linear data, so the
/// channels scale by `1/255` with no sRGB decode (red/alpha are ignored).
/// Pure CPU mirror of texel-center `Nearest` sampling.
pub fn sample_metallic_roughness(texel: [u8; 4]) -> (f32, f32) {
    (
        f32::from(texel[1]) / U8_TO_UNIT,
        f32::from(texel[2]) / U8_TO_UNIT,
    )
}

/// Deterministic content key of one upload (role + size + pixels).
///
/// Dedup key for [`TextureCache::upload_cached`]: identical images under the
/// same role share one GPU entry (a color image and a data image with equal
/// bytes must not alias — their formats differ). Same hashing discipline as
/// `mesh_upload::soup_hash`.
pub fn texture_cache_key(image: &CpuImage, role: TextureRole) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    role.hash(&mut hasher);
    image.width.hash(&mut hasher);
    image.height.hash(&mut hasher);
    image.pixels.hash(&mut hasher);
    hasher.finish()
}

/// One uploaded image: texture plus its view and sampler.
///
/// The view and sampler are created from the texture at upload; all three
/// live and die together with this owner (see the module docs for the
/// cache-level lifetime).
pub struct GpuTexture {
    /// Device texture holding the image bytes.
    pub texture: wgpu::Texture,
    /// Default view of [`GpuTexture::texture`].
    pub view: wgpu::TextureView,
    /// Sampler with [`sampler_descriptor_for_role`] defaults.
    pub sampler: wgpu::Sampler,
    /// Role the image was uploaded under (selects format and binding).
    pub role: TextureRole,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Uploads `image` under `role`: texture + view + sampler.
///
/// The texture uses [`texture_format_for_role`] with `TEXTURE_BINDING |
/// `COPY_DST` usage and one mip; rows are padded to the 256-byte copy
/// alignment, so any validated size uploads. The sampler uses
/// [`sampler_descriptor_for_role`].
///
/// # Errors
///
/// Returns [`TextureUploadError`] when `image` fails validation — no GPU
/// object is created.
pub fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &CpuImage,
    role: TextureRole,
) -> Result<GpuTexture, TextureUploadError> {
    image.validate()?;
    let format = texture_format_for_role(role);
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf texture upload"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // `write_texture` rows must honor the 256-byte copy alignment; pad any
    // narrow image instead of restricting uploads to wide ones.
    let unpadded_bytes_per_row = image.width as usize * RGBA8_BYTES;
    let padded_bytes_per_row =
        unpadded_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
    let mut staging = vec![0u8; padded_bytes_per_row * image.height as usize];
    for (dst_row, src_row) in staging
        .chunks_exact_mut(padded_bytes_per_row)
        .zip(image.pixels.chunks_exact(unpadded_bytes_per_row))
    {
        dst_row[..unpadded_bytes_per_row].copy_from_slice(src_row);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &staging,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded_bytes_per_row as u32),
            rows_per_image: Some(image.height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&sampler_descriptor_for_role(role));
    Ok(GpuTexture {
        texture,
        view,
        sampler,
        role,
        width: image.width,
        height: image.height,
    })
}

/// Handle of one uploaded image in [`TextureCache`]: entry index.
///
/// Newtype over `u32` so texture handles never mix with material indices
/// or entity ids at the type level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureHandle(u32);

impl TextureHandle {
    /// Wraps a raw `u32` cache index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` cache index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Cache index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for TextureHandle {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for TextureHandle {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<TextureHandle> for u32 {
    fn from(h: TextureHandle) -> Self {
        h.0
    }
}

impl From<TextureHandle> for usize {
    fn from(h: TextureHandle) -> Self {
        h.0 as usize
    }
}

/// Owner of the frame's uploaded images, with content dedup.
///
/// The caller (e.g. next to `Renderer3D` in `GpuFrameState`) owns one cache;
/// entries live as long as it does. Handles are entry indices. Uploads go
/// through [`TextureCache::upload_cached`] so identical images share one
/// GPU texture; [`TextureCache::get`] resolves a handle for bind-group
/// assembly. No snapshot system is involved — callers feed lane-read images
/// (the `RenderSubmit` direct-read canon) straight in.
#[derive(Default)]
pub struct TextureCache {
    /// Uploaded entries, in upload order (the handle is the index).
    entries: Vec<GpuTexture>,
    /// Content key ([`texture_cache_key`]) to entry index.
    index: HashMap<u64, TextureHandle>,
}

impl TextureCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Number of uploaded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no entry has been uploaded yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolves a handle from [`TextureCache::upload_cached`], if live.
    pub fn get(&self, handle: TextureHandle) -> Option<&GpuTexture> {
        self.entries.get(handle.index())
    }

    /// Uploads `image` under `role`, or reuses the identical entry.
    ///
    /// On a content hit ([`texture_cache_key`]) no GPU work happens and the
    /// existing handle returns; on a miss the image uploads via
    /// [`upload_texture`] and the new handle returns.
    ///
    /// # Errors
    ///
    /// Returns [`TextureUploadError`] when `image` fails validation —
    /// failures never populate the cache, so a retry re-attempts the upload.
    pub fn upload_cached(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &CpuImage,
        role: TextureRole,
    ) -> Result<TextureHandle, TextureUploadError> {
        image.validate()?;
        let key = texture_cache_key(image, role);
        if let Some(handle) = self.index.get(&key) {
            return Ok(*handle);
        }
        let entry = upload_texture(device, queue, image, role)?;
        let handle = TextureHandle::from(self.entries.len());
        self.entries.push(entry);
        self.index.insert(key, handle);
        Ok(handle)
    }
}

/// Per-material texture bind: optional cache handle per [`TextureRole`].
///
/// Pairs with one `OpenPBRMaterial` table entry: `None` means the slot is
/// unbound and the scalar factor stands alone (legacy path, pixel-identical
/// — textured materials are new paths only). The metallic-roughness handle,
/// when set, is sampled by the shader at runtime; it never moves the scalar
/// `is_metallic` switch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterialTextureSet {
    /// Handle of the albedo image ([`TextureRole::BaseColor`]), if bound.
    pub base_color: Option<TextureHandle>,
    /// Handle of the roughness/metallic image
    /// ([`TextureRole::MetallicRoughness`]), if bound.
    pub metallic_roughness: Option<TextureHandle>,
    /// Handle of the emission image ([`TextureRole::Emissive`]), if bound.
    pub emissive: Option<TextureHandle>,
}

impl MaterialTextureSet {
    /// Binds `handle` to the `role` slot, replacing any previous bind.
    pub fn set(&mut self, role: TextureRole, handle: TextureHandle) {
        match role {
            TextureRole::BaseColor => self.base_color = Some(handle),
            TextureRole::MetallicRoughness => self.metallic_roughness = Some(handle),
            TextureRole::Emissive => self.emissive = Some(handle),
        }
    }

    /// Returns the handle bound to the `role` slot, if any.
    pub fn binding(&self, role: TextureRole) -> Option<TextureHandle> {
        match role {
            TextureRole::BaseColor => self.base_color,
            TextureRole::MetallicRoughness => self.metallic_roughness,
            TextureRole::Emissive => self.emissive,
        }
    }

    /// True when no slot is bound (legacy scalar-only material).
    pub fn is_empty(&self) -> bool {
        self.base_color.is_none() && self.metallic_roughness.is_none() && self.emissive.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(width: u32, height: u32, texel: [u8; 4]) -> CpuImage {
        CpuImage {
            width,
            height,
            pixels: texel
                .iter()
                .cycle()
                .take(width as usize * height as usize * 4)
                .copied()
                .collect(),
        }
    }

    #[test]
    fn rejects_empty_oversize_and_mismatched_images() {
        assert!(matches!(
            CpuImage::from_rgba8(0, 4, Vec::new()),
            Err(TextureUploadError::EmptyDimensions { .. })
        ));
        assert!(matches!(
            CpuImage::from_rgba8(4, 0, Vec::new()),
            Err(TextureUploadError::EmptyDimensions { .. })
        ));
        assert!(matches!(
            CpuImage::from_rgba8(MAX_TEXTURE_EDGE + 1, 1, vec![0; 4]),
            Err(TextureUploadError::ImageTooLarge { .. })
        ));
        assert!(matches!(
            CpuImage::from_rgba8(2, 2, vec![0; 15]),
            Err(TextureUploadError::PixelLengthMismatch { .. })
        ));
        assert!(matches!(
            CpuImage::from_rgba8(2, 2, vec![0; 17]),
            Err(TextureUploadError::PixelLengthMismatch { .. })
        ));
        // Exact fit validates.
        assert!(CpuImage::from_rgba8(2, 2, vec![9; 16]).is_ok());
        // Display strings carry the mismatched sizes.
        let error = TextureUploadError::PixelLengthMismatch {
            expected: 16,
            actual: 15,
        };
        assert_eq!(
            error.to_string(),
            "texture pixel length mismatch: expected 16, got 15"
        );
    }

    #[test]
    fn format_decision_color_is_srgb_data_is_linear() {
        // Albedo and emission are sRGB color: hardware decodes to linear.
        assert_eq!(
            texture_format_for_role(TextureRole::BaseColor),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            texture_format_for_role(TextureRole::Emissive),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        // Roughness/metallic is linear data: uploading it as sRGB would
        // corrupt both channels on sample, so it stays `Unorm`.
        assert_eq!(
            texture_format_for_role(TextureRole::MetallicRoughness),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn sampler_defaults_are_linear_repeat() {
        // All roles share the upload defaults today (glTF samplers and
        // `texCoord` sets are ignored on import by design).
        for role in [
            TextureRole::BaseColor,
            TextureRole::MetallicRoughness,
            TextureRole::Emissive,
        ] {
            let sampler = sampler_descriptor_for_role(role);
            assert_eq!(sampler.mag_filter, wgpu::FilterMode::Linear);
            assert_eq!(sampler.min_filter, wgpu::FilterMode::Linear);
            assert_eq!(sampler.mipmap_filter, wgpu::MipmapFilterMode::Linear);
            assert_eq!(sampler.address_mode_u, wgpu::AddressMode::Repeat);
            assert_eq!(sampler.address_mode_v, wgpu::AddressMode::Repeat);
            assert_eq!(sampler.address_mode_w, wgpu::AddressMode::Repeat);
            assert_eq!(sampler.lod_min_clamp, 0.0);
            assert_eq!(sampler.lod_max_clamp, 32.0);
            assert_eq!(sampler.compare, None);
            assert_eq!(sampler.anisotropy_clamp, 1);
        }
    }

    #[test]
    fn srgb_to_linear_spot_values() {
        // Junction points of the exact transfer function.
        assert_eq!(srgb_channel_to_linear(0), 0.0);
        assert_eq!(srgb_channel_to_linear(255), 1.0);
        // Linear segment just below the junction (`10/255 ≈ 0.039 < 0.04045`).
        let low = srgb_channel_to_linear(10);
        assert!((low - (10.0 / 255.0) / 12.92).abs() < 1e-6, "{low}");
        // Power segment: mid-gray lands near 0.214, never near the
        // `pow(c, 2.2)` approximation drift for dark values (black stays 0).
        let mid = srgb_channel_to_linear(128);
        assert!((mid - 0.2159).abs() < 1e-3, "{mid}");
        assert!(srgb_channel_to_linear(1) > 0.0, "dark values stay nonzero");
    }

    #[test]
    fn metallic_roughness_samples_green_blue_linear() {
        // (G, B) = (roughness, metallic); red/alpha ignored, no sRGB decode.
        assert_eq!(
            sample_metallic_roughness([9, 255, 0, 200]),
            (1.0, 0.0),
            "full roughness, dielectric"
        );
        assert_eq!(
            sample_metallic_roughness([9, 0, 255, 200]),
            (0.0, 1.0),
            "smooth metal"
        );
        let (roughness, metallic) = sample_metallic_roughness([0, 128, 64, 255]);
        assert!((roughness - 128.0 / 255.0).abs() < 1e-6);
        assert!((metallic - 64.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn albedo_samples_decode_srgb_to_linear() {
        // Pure red: R decodes to 1, G/B stay 0, alpha stays linear.
        assert_eq!(sample_albedo_linear([255, 0, 0, 255]), [1.0, 0.0, 0.0, 1.0]);
        // Half alpha is linear (never sRGB-decoded).
        let sampled = sample_albedo_linear([255, 255, 255, 128]);
        assert_eq!(sampled[..3], [1.0, 1.0, 1.0]);
        assert!((sampled[3] - 128.0 / 255.0).abs() < 1e-6);
        // Mid-gray decodes through the power segment, not `c/255`.
        let gray = sample_albedo_linear([128, 128, 128, 255]);
        assert!((gray[0] - 0.2159).abs() < 1e-3, "{gray:?}");
    }

    #[test]
    fn upload_path_validates_then_samples_without_adapter() {
        // Adapter-free upload→sample round trip: a 2×2 RGBA8 image (the
        // `LoadedImage` contract) validates, then every texel samples
        // through the same mirrors the GPU path uses — albedo through the
        // sRGB decode, metallic-roughness through the linear (G, B) pick.
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255, //
            18, 128, 64, 128,
        ];
        let image = CpuImage::from_rgba8(2, 2, rgba).expect("valid 2x2");
        assert_eq!(image.texel(0, 0), Some([255, 0, 0, 255]));
        assert_eq!(image.texel(1, 1), Some([18, 128, 64, 128]));
        assert_eq!(image.texel(2, 0), None, "out of bounds");
        // Albedo role: red texel decodes to linear red.
        let red = sample_albedo_linear(image.texel(0, 0).expect("red"));
        assert_eq!(red, [1.0, 0.0, 0.0, 1.0]);
        // Data role: (G, B) = (128/255 roughness, 64/255 metallic), linear.
        let (roughness, metallic) = sample_metallic_roughness(image.texel(1, 1).expect("data"));
        assert!((roughness - 128.0 / 255.0).abs() < 1e-6);
        assert!((metallic - 64.0 / 255.0).abs() < 1e-6);
        // The GPU formats/staging behind this path stay pinned above, so a
        // future adapter run samples the same values the CPU mirrors give.
        assert_eq!(
            texture_format_for_role(TextureRole::BaseColor),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            texture_format_for_role(TextureRole::MetallicRoughness),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn cache_key_is_stable_and_role_sensitive() {
        let image = solid_image(2, 2, [10, 20, 30, 40]);
        assert_eq!(
            texture_cache_key(&image, TextureRole::BaseColor),
            texture_cache_key(&image, TextureRole::BaseColor),
            "stable across calls"
        );
        assert_ne!(
            texture_cache_key(&image, TextureRole::BaseColor),
            texture_cache_key(&image, TextureRole::MetallicRoughness),
            "same bytes under another role must not alias (formats differ)"
        );
        assert_ne!(
            texture_cache_key(&image, TextureRole::BaseColor),
            texture_cache_key(&solid_image(2, 2, [11, 20, 30, 40]), TextureRole::BaseColor),
            "one byte flips the key"
        );
    }

    #[test]
    fn material_set_binds_per_role_and_starts_empty() {
        let mut set = MaterialTextureSet::default();
        assert!(set.is_empty(), "untextured materials bind nothing");
        assert_eq!(set.binding(TextureRole::BaseColor), None);
        set.set(TextureRole::BaseColor, TextureHandle::from_raw(0));
        set.set(TextureRole::MetallicRoughness, TextureHandle::from_raw(2));
        set.set(TextureRole::Emissive, TextureHandle::from_raw(5));
        assert!(!set.is_empty());
        assert_eq!(
            set.binding(TextureRole::BaseColor),
            Some(TextureHandle::from_raw(0))
        );
        assert_eq!(
            set.binding(TextureRole::MetallicRoughness),
            Some(TextureHandle::from_raw(2))
        );
        assert_eq!(
            set.binding(TextureRole::Emissive),
            Some(TextureHandle::from_raw(5))
        );
        // Rebinding replaces (one slot per role, never a list).
        set.set(TextureRole::BaseColor, TextureHandle::from_raw(7));
        assert_eq!(
            set.binding(TextureRole::BaseColor),
            Some(TextureHandle::from_raw(7))
        );
    }

    #[test]
    fn empty_cache_resolves_nothing() {
        let cache = TextureCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.get(TextureHandle::from_raw(0)).is_none());
    }

    #[test]
    fn gpu_upload_roundtrip_skips_without_adapter() {
        // Real `upload_texture` + `upload_cached` dedup when an adapter
        // exists; a loud skip otherwise (same contract as the pixel-parity
        // harness — CI provides lavapipe, adapter-less machines stay green).
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
            return;
        };
        // Narrow (2 px) rows: exercises the 256-byte staging padding.
        let image = solid_image(2, 3, [200, 128, 64, 255]);
        let uploaded =
            upload_texture(&device, &queue, &image, TextureRole::BaseColor).expect("2x3 uploads");
        assert_eq!(uploaded.width, 2);
        assert_eq!(uploaded.height, 3);
        assert_eq!(uploaded.role, TextureRole::BaseColor);
        assert_eq!(
            uploaded.texture.format(),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(uploaded.texture.size().width, 2);
        assert_eq!(uploaded.texture.size().height, 3);

        let mut cache = TextureCache::new();
        let first = cache
            .upload_cached(&device, &queue, &image, TextureRole::BaseColor)
            .expect("first upload");
        let second = cache
            .upload_cached(&device, &queue, &image, TextureRole::BaseColor)
            .expect("identical reuses");
        assert_eq!(
            (first, second),
            (TextureHandle::from_raw(0), TextureHandle::from_raw(0)),
            "identical images deduplicate"
        );
        assert_eq!(cache.len(), 1);
        assert!(cache.get(TextureHandle::from_raw(0)).is_some());
        assert!(cache.get(TextureHandle::from_raw(1)).is_none());
        // Same bytes under the data role must not alias the color entry.
        let data_handle = cache
            .upload_cached(&device, &queue, &image, TextureRole::MetallicRoughness)
            .expect("data role uploads separately");
        assert_eq!(data_handle, TextureHandle::from_raw(1));
        assert_eq!(cache.len(), 2);
        assert_eq!(
            cache
                .get(TextureHandle::from_raw(1))
                .expect("data entry")
                .texture
                .format(),
            wgpu::TextureFormat::Rgba8Unorm
        );
        // Validation failures never populate the cache.
        let bad = CpuImage {
            width: 2,
            height: 2,
            pixels: vec![0; 15],
        };
        assert!(
            cache
                .upload_cached(&device, &queue, &bad, TextureRole::BaseColor)
                .is_err()
        );
        assert_eq!(cache.len(), 2, "failures are not cached");
    }

    async fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    }
}
