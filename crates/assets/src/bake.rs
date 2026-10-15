//! Offline texture baking: decoded sources → KTX2 with the authored mip chain.
//!
//! The runtime never generates material mips for a baked asset: `ornis-texture`
//! uploads the chain the container carries. Baking is where that chain is
//! decided, once, offline, and cached by content:
//!
//! - [`BasisMode`] is the codec policy. [`BasisMode::for_role`] holds the
//!   defaults — UASTC where a lossy block encoder hurts most (tangent-space
//!   normals), ETC1S for the color and ORM slots — and
//!   [`BakeSettings::overrides`] replaces any single slot.
//! - The cache key is the decoded source pixels plus the policy inputs (role,
//!   codec), so a re-imported PNG/JPEG that produced the same texels reuses the
//!   stored container. Nothing is inferred from file names, sizes or
//!   timestamps.
//! - Basis has no pure-Rust encoder, so compressed payloads come from the
//!   reference `basisu` command line (the project the runtime transcoder port
//!   comes from). [`EncoderSource`] picks the tool: a configured binary, one
//!   discovered on `PATH`, or none at all. When no tool runs, the bake still
//!   removes the runtime mip cost through [`encode_rgba8_ktx2`]: the same
//!   authored levels, RGBA8 blocks instead of the ~4× smaller compressed ones.
//! - A bake never replaces the source: [`bake_texture`] hands a container back
//!   only on success, and [`bake_model`] keeps the decoded PNG/JPEG pixels of
//!   any slot whose bake failed. A host that never calls
//!   [`AssetServer::with_texture_baking`](crate::AssetServer::with_texture_baking)
//!   keeps the previous import behaviour unchanged.
//!
//! ```no_run
//! use ornis_assets::bake::{BakeSettings, EncoderSource, bake_texture};
//! use ornis_texture::{CpuImage, TextureRole};
//!
//! let settings = BakeSettings {
//!     encoder: EncoderSource::InProcess,
//!     ..BakeSettings::new("target/ornis-texture-cache")
//! };
//! let source = CpuImage::from_rgba8(2, 2, vec![128; 16])?;
//! let baked = bake_texture(&settings, &source, TextureRole::BaseColor)?;
//! assert_eq!(baked.image.dimensions(), (2, 2));
//! # Ok::<(), ornis_assets::bake::BakeError>(())
//! ```

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use ornis_texture::{CpuImage, Ktx2Image, TextureRole, TextureUploadError, encode_rgba8_ktx2};

#[cfg(feature = "gltf")]
use ornis_gltf::LoadedTextures;
#[cfg(feature = "gltf")]
use ornis_texture::TextureImage;

/// Reference encoder program, discovered on `PATH` when none is configured.
const ENCODER_PROGRAM: &str = "basisu";
/// Environment variable holding the executable search path.
const PATH_VARIABLE: &str = "PATH";
/// Cache file suffix (validated KTX2 containers).
const OUTPUT_SUFFIX: &str = ".ktx2";
/// Suffix of the pixels handed to the encoder.
const SOURCE_SUFFIX: &str = ".tga";
/// Suffix of the container the encoder writes, before validation.
const DUMP_SUFFIX: &str = ".tool.ktx2";
/// Flags of every encoder run: KTX2 output with a generated mip chain.
const FIXED_FLAGS: [&str; 3] = ["-ktx2", "-mipmap", "-output_file"];
/// UASTC effort level, spelled out instead of relying on a tool default.
const UASTC_LEVEL: &str = "2";
/// ETC1S quality level, likewise spelled out (the tool's own default).
const ETC1S_QUALITY: &str = "128";
/// Encoder stderr bytes kept in [`BakeError::Exit`].
const STDERR_TAIL_BYTES: usize = 512;
/// FNV-1a 64 offset basis, the cache key's starting state.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Channels in one RGBA texel.
const RGBA_CHANNELS: usize = 4;
/// Bytes in an uncompressed true-color TGA header (no ID, no color map).
const TGA_HEADER_BYTES: usize = 18;
/// TGA image type 2: uncompressed true-color.
const TGA_UNCOMPRESSED_TRUE_COLOR: u8 = 2;
/// Color-map specification bytes of an uncolored TGA header.
const TGA_COLOR_MAP_BYTES: usize = 5;
/// X/Y origin fields of a TGA header (both zero: bottom-up rows).
const TGA_ORIGIN_BYTES: usize = 4;
/// Bits per texel handed to the encoder.
const TGA_RGBA_BITS: u8 = 32;
/// Alpha depth 8 with the origin bit clear (rows bottom-up, the TGA default).
const TGA_DESCRIPTOR: u8 = 8;
/// Red channel offset of one RGBA texel.
const RED_CHANNEL: usize = 0;
/// Blue channel offset of one RGBA texel.
const BLUE_CHANNEL: usize = 2;

/// Codec the offline encoder is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisMode {
    /// UASTC LDR 4×4 (`-uastc`): high quality, 8 bits per texel in memory.
    Uastc,
    /// ETC1S (`-etc1s`): compressed endpoints and selectors, low bitrate.
    Etc1s,
}

impl BasisMode {
    /// Default codec of a material slot: UASTC where block artifacts hurt most
    /// (tangent-space normals), ETC1S for albedo, ORM and emission.
    pub const fn for_role(role: TextureRole) -> Self {
        match role {
            TextureRole::Normal => Self::Uastc,
            TextureRole::BaseColor | TextureRole::MetallicRoughness | TextureRole::Emissive => {
                Self::Etc1s
            }
        }
    }
}

/// Where compressed payloads come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderSource {
    /// Exactly this binary; its failure is reported, never swallowed.
    Binary(PathBuf),
    /// `basisu` discovered on the executable search path, the in-process
    /// authored chain when it is not installed.
    Path,
    /// Never spawn a process: always the in-process authored chain.
    InProcess,
}

/// Bake policy: codec overrides, cache directory and encoder source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakeSettings {
    /// Per-slot codec overrides; a slot left out uses [`BasisMode::for_role`].
    pub overrides: HashMap<TextureRole, BasisMode>,
    /// Directory holding the `<key>.ktx2` entries (created on demand).
    pub cache_dir: PathBuf,
    /// Where compressed payloads come from.
    pub encoder: EncoderSource,
}

impl BakeSettings {
    /// Role defaults, `basisu` discovered on `PATH`, entries under `cache_dir`.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            overrides: HashMap::new(),
            cache_dir: cache_dir.into(),
            encoder: EncoderSource::Path,
        }
    }

    /// Codec of `role`: the override when there is one, the default otherwise.
    pub fn mode_of(&self, role: TextureRole) -> BasisMode {
        self.overrides
            .get(&role)
            .copied()
            .unwrap_or_else(|| BasisMode::for_role(role))
    }
}

/// Where a baked container came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BakeOrigin {
    /// Read back from the content-addressed cache.
    Cached,
    /// Written by the reference encoder.
    Encoded,
    /// Written in-process by the RGBA8 writer: authored mips, uncompressed.
    Authored,
}

/// One baked container and how it was produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Baked {
    /// Validated KTX2 container, ready for a model's material slots.
    pub image: Ktx2Image,
    /// Provenance of `image`.
    pub origin: BakeOrigin,
}

/// Counters of one [`bake_model`] pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BakeStats {
    /// Containers written by the reference encoder.
    pub encoded: usize,
    /// Containers served from the cache.
    pub cached: usize,
    /// Containers written in-process (no encoder ran).
    pub authored: usize,
    /// Slots left as decoded PNG/JPEG after a failed bake.
    pub fallback: usize,
    /// Formatted reason of the last failure (`None` when nothing failed).
    pub last_failure: Option<String>,
}

/// Why a bake could not produce a container.
#[derive(Debug, thiserror::Error)]
pub enum BakeError {
    /// A container was requested for bytes that are not a material texture.
    #[error("invalid texture: {0}")]
    Image(#[from] TextureUploadError),
    /// Cache or scratch-file access failed.
    #[error("bake cache IO at {path}: {source}")]
    Io {
        /// Failing path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// The encoder binary could not be started.
    #[error("cannot run encoder `{program}`: {source}")]
    Spawn {
        /// Program that failed to start.
        program: String,
        /// Underlying spawn error.
        source: std::io::Error,
    },
    /// The encoder ran and exited non-zero.
    #[error("encoder `{program}` exited with code {code}: {stderr}")]
    Exit {
        /// Program that failed.
        program: String,
        /// Exit code (`-1` when a signal terminated the process).
        code: i32,
        /// Tail of the tool's stderr.
        stderr: String,
    },
}

/// Bakes `image` for `role`: cache hit, else encoder, else in-process chain.
///
/// The cache key covers the decoded pixels, their dimensions and both policy
/// inputs (role, codec), so two sources that decode to the same image share one
/// entry. An entry that no longer parses as KTX2 counts as a miss and is
/// rewritten; only containers that passed [`Ktx2Image::from_bytes`] are ever
/// stored. The key does not cover the encoder build: after upgrading the tool,
/// delete `cache_dir` to re-encode.
///
/// # Errors
///
/// [`BakeError`] when `image` is not a valid texture, the cache is unwritable,
/// or a configured encoder fails or returns a container this workspace rejects.
/// A missing `basisu` is not an error — the RGBA8 chain still removes the
/// runtime mip cost, so a host without the tool keeps working.
pub fn bake_texture(
    settings: &BakeSettings,
    image: &CpuImage,
    role: TextureRole,
) -> Result<Baked, BakeError> {
    image.validate()?;
    let key = cache_key(image, role, settings.mode_of(role));
    if let Some(cached) = load_cached(settings, &key) {
        return Ok(cached);
    }
    encode_fresh(settings, image, role, &key)
}

/// Runs the configured encoder when there is one, the in-process writer
/// otherwise.
fn encode_fresh(
    settings: &BakeSettings,
    image: &CpuImage,
    role: TextureRole,
    key: &str,
) -> Result<Baked, BakeError> {
    match encoder(settings) {
        Some(program) => encode_with_tool(settings, &program, image, role, key),
        None => author(settings, image, role, key),
    }
}

/// Encoder binary to run, unless the policy keeps the bake in-process.
fn encoder(settings: &BakeSettings) -> Option<PathBuf> {
    match &settings.encoder {
        EncoderSource::InProcess => None,
        EncoderSource::Path => find_on_path(ENCODER_PROGRAM),
        EncoderSource::Binary(program) => Some(program.clone()),
    }
}

/// First `program` entry of the executable search path, if any.
fn find_on_path(program: &str) -> Option<PathBuf> {
    std::env::var_os(PATH_VARIABLE).and_then(|path| first_entry(&path, program))
}

/// First existing file named `program` in `path`.
fn first_entry(path: &OsStr, program: &str) -> Option<PathBuf> {
    std::env::split_paths(path)
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file())
}

/// In-process fallback: the authored mip chain, stored as RGBA8 blocks.
fn author(
    settings: &BakeSettings,
    image: &CpuImage,
    role: TextureRole,
    key: &str,
) -> Result<Baked, BakeError> {
    let bytes = encode_rgba8_ktx2(image, role)?;
    let container = Ktx2Image::from_bytes(bytes)?;
    store_cached(settings, key, container.bytes())?;
    Ok(Baked {
        image: container,
        origin: BakeOrigin::Authored,
    })
}

/// One encoder run: pixel handoff, CLI call, validation, cache store.
fn encode_with_tool(
    settings: &BakeSettings,
    program: &Path,
    image: &CpuImage,
    role: TextureRole,
    key: &str,
) -> Result<Baked, BakeError> {
    ensure_dir(settings)?;
    let source = cache_file(settings, key, SOURCE_SUFFIX);
    let dump = cache_file(settings, key, DUMP_SUFFIX);
    let args = basis_args(settings.mode_of(role), &source, &dump);
    std::fs::write(&source, tga_bytes(image)).map_err(|error| io_error(&source, error))?;
    run_encoder(program, args)?;
    let container = read_container(settings, &dump, key)?;
    cleanup(&source);
    cleanup(&dump);
    Ok(Baked {
        image: container,
        origin: BakeOrigin::Encoded,
    })
}

/// Runs the encoder once; a non-zero exit carries the tool's stderr tail.
fn run_encoder(program: &Path, args: Vec<OsString>) -> Result<(), BakeError> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|error| BakeError::Spawn {
            program: program.display().to_string(),
            source: error,
        })?;
    failure_of(program, &output)
}

/// Non-zero exit status as a typed failure.
fn failure_of(program: &Path, output: &std::process::Output) -> Result<(), BakeError> {
    if output.status.success() {
        return Ok(());
    }
    Err(BakeError::Exit {
        program: program.display().to_string(),
        code: exit_code(&output.status),
        stderr: tail(&output.stderr),
    })
}

/// Exit code of a finished process (`-1` for a signal-terminated one).
fn exit_code(status: &std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

/// Trailing, lossily decoded slice of an encoder's stderr.
fn tail(bytes: &[u8]) -> String {
    let start = bytes.len().saturating_sub(STDERR_TAIL_BYTES);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

/// Reads back what the encoder wrote: validated first, cached second.
fn read_container(settings: &BakeSettings, dump: &Path, key: &str) -> Result<Ktx2Image, BakeError> {
    let bytes = std::fs::read(dump).map_err(|error| io_error(dump, error))?;
    let container = Ktx2Image::from_bytes(bytes)?;
    store_cached(settings, key, container.bytes())?;
    Ok(container)
}

/// Pinned argument vector for one encoder run.
///
/// The flags are the reference tool's documented names: KTX2 output with
/// generated mips, the codec of `mode`, and an explicit effort or quality level
/// so a bake cannot drift when a tool default changes. Only the codec is
/// per-bake; everything else is fixed here.
fn basis_args(mode: BasisMode, source: &Path, dump: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = FIXED_FLAGS.map(OsString::from).to_vec();
    args.push(dump.into());
    args.extend(mode_flags(mode));
    args.push(source.into());
    args
}

/// Codec flags of one bake mode, with the level pinned explicitly.
fn mode_flags(mode: BasisMode) -> [OsString; 3] {
    match mode {
        BasisMode::Uastc => ["-uastc", "-uastc_level", UASTC_LEVEL].map(OsString::from),
        BasisMode::Etc1s => ["-etc1s", "-q", ETC1S_QUALITY].map(OsString::from),
    }
}

/// Uncompressed 32-bit TGA of `image`, rows bottom-up.
///
/// The reference encoder reads PNG/JPEG/TGA/DDS/QOI; a TGA header plus raw
/// texels is the cheapest lossless handoff once the imported model only holds
/// decoded pixels. Edges above `u16` are unreachable: [`bake_texture`] rejects
/// anything past `MAX_TEXTURE_EDGE` first.
fn tga_bytes(image: &CpuImage) -> Vec<u8> {
    let mut bytes = tga_header(image);
    bytes.extend_from_slice(&tga_pixels(image));
    bytes
}

/// 18-byte uncompressed true-color TGA header with a bottom-left origin.
fn tga_header(image: &CpuImage) -> Vec<u8> {
    let mut header = Vec::with_capacity(TGA_HEADER_BYTES);
    header.extend_from_slice(&[0, 0, TGA_UNCOMPRESSED_TRUE_COLOR]);
    header.extend_from_slice(&[0; TGA_COLOR_MAP_BYTES]);
    header.extend_from_slice(&[0; TGA_ORIGIN_BYTES]);
    header.extend_from_slice(&(image.width as u16).to_le_bytes());
    header.extend_from_slice(&(image.height as u16).to_le_bytes());
    header.extend_from_slice(&[TGA_RGBA_BITS, TGA_DESCRIPTOR]);
    header
}

/// Texels of `image` bottom-up and in TGA channel order (B, G, R, A).
fn tga_pixels(image: &CpuImage) -> Vec<u8> {
    let stride = image.width as usize * RGBA_CHANNELS;
    let mut pixels: Vec<u8> = image
        .pixels
        .chunks(stride)
        .rev()
        .flatten()
        .copied()
        .collect();
    swap_red_and_blue(&mut pixels);
    pixels
}

/// Reorders every four-byte texel in place into TGA channel order.
fn swap_red_and_blue(pixels: &mut [u8]) {
    pixels
        .as_chunks_mut::<RGBA_CHANNELS>()
        .0
        .iter_mut()
        .for_each(|texel| texel.swap(RED_CHANNEL, BLUE_CHANNEL));
}

/// Content-addressed key of one bake input.
///
/// FNV-1a over the exact bytes: a cache key, not a security hash. A collision
/// costs one stale container that still has to parse as KTX2 before it can
/// reach the renderer.
fn cache_key(image: &CpuImage, role: TextureRole, mode: BasisMode) -> String {
    let mut hash = ContentHash::new();
    hash.write(role_tag(role));
    hash.write(mode_tag(mode));
    hash.write(&image.width.to_le_bytes());
    hash.write(&image.height.to_le_bytes());
    hash.write(&image.pixels);
    hash.hex()
}

/// Distinguishes material slots in the cache key (stable, never parsed).
fn role_tag(role: TextureRole) -> &'static [u8] {
    match role {
        TextureRole::BaseColor => b"base-color",
        TextureRole::MetallicRoughness => b"metallic-roughness",
        TextureRole::Emissive => b"emissive",
        TextureRole::Normal => b"normal",
    }
}

/// Distinguishes codecs in the cache key (stable, never parsed).
fn mode_tag(mode: BasisMode) -> &'static [u8] {
    match mode {
        BasisMode::Uastc => b"uastc",
        BasisMode::Etc1s => b"etc1s",
    }
}

/// Streaming FNV-1a 64 state for cache keys.
#[derive(Debug)]
struct ContentHash(u64);

impl ContentHash {
    /// Fresh state at the FNV offset basis.
    fn new() -> Self {
        Self(FNV_OFFSET_BASIS)
    }

    /// Mixes `bytes` into the state.
    fn write(&mut self, bytes: &[u8]) {
        self.0 = bytes.iter().fold(self.0, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        });
    }

    /// Lowercase hex digest of everything written so far.
    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// One file of the cache directory: `<key><suffix>`.
fn cache_file(settings: &BakeSettings, key: &str, suffix: &str) -> PathBuf {
    settings.cache_dir.join(format!("{key}{suffix}"))
}

/// Reads the stored container of `key`; a missing or invalid entry is a miss.
fn load_cached(settings: &BakeSettings, key: &str) -> Option<Baked> {
    let path = cache_file(settings, key, OUTPUT_SUFFIX);
    let image = Ktx2Image::from_bytes(std::fs::read(path).ok()?).ok()?;
    Some(Baked {
        image,
        origin: BakeOrigin::Cached,
    })
}

/// Creates the cache directory (idempotent).
fn ensure_dir(settings: &BakeSettings) -> Result<(), BakeError> {
    std::fs::create_dir_all(&settings.cache_dir)
        .map_err(|error| io_error(&settings.cache_dir, error))
}

/// Writes one validated container into the cache under `key`.
fn store_cached(settings: &BakeSettings, key: &str, bytes: &[u8]) -> Result<(), BakeError> {
    ensure_dir(settings)?;
    let path = cache_file(settings, key, OUTPUT_SUFFIX);
    std::fs::write(&path, bytes).map_err(|error| io_error(&path, error))
}

/// Removes one encoder scratch file (best effort: the cache entry is valid).
fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// IO failure tagged with the path it happened on.
fn io_error(path: &Path, source: std::io::Error) -> BakeError {
    BakeError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Material slots [`bake_model`] bakes, in upload order.
#[cfg(feature = "gltf")]
const BAKED_ROLES: [TextureRole; 4] = [
    TextureRole::BaseColor,
    TextureRole::MetallicRoughness,
    TextureRole::Emissive,
    TextureRole::Normal,
];

/// Bakes every decoded material image of an imported model in place.
///
/// Slots already holding KTX2 keep the chain the file authored, and a slot
/// whose bake fails keeps its decoded PNG/JPEG pixels, so the import result
/// stays uploadable either way; [`BakeStats`] says what happened. Content
/// reached by two primitives in the same role is encoded once: the second slot
/// is served from the cache.
#[cfg(feature = "gltf")]
pub fn bake_model(settings: &BakeSettings, model: &mut crate::Model) -> BakeStats {
    let mut stats = BakeStats::default();
    for primitive in &mut model.primitives {
        let textures = &mut primitive.material.textures;
        BAKED_ROLES
            .iter()
            .for_each(|role| bake_slot(settings, slot_mut(textures, *role), *role, &mut stats));
    }
    stats
}

/// Bakes one material slot when it still holds decoded pixels.
#[cfg(feature = "gltf")]
fn bake_slot(
    settings: &BakeSettings,
    slot: &mut Option<TextureImage>,
    role: TextureRole,
    stats: &mut BakeStats,
) {
    let Some(source) = slot.as_ref().and_then(decoded) else {
        return;
    };
    apply_baked(slot, bake_texture(settings, source, role), stats);
}

/// Mutable slot of one material role.
#[cfg(feature = "gltf")]
fn slot_mut(textures: &mut LoadedTextures, role: TextureRole) -> &mut Option<TextureImage> {
    match role {
        TextureRole::BaseColor => &mut textures.base_color_texture,
        TextureRole::MetallicRoughness => &mut textures.metallic_roughness_texture,
        TextureRole::Emissive => &mut textures.emissive_texture,
        TextureRole::Normal => &mut textures.normal_texture,
    }
}

/// Decoded pixels of a slot, if it still needs baking.
#[cfg(feature = "gltf")]
fn decoded(image: &TextureImage) -> Option<&CpuImage> {
    match image {
        TextureImage::Rgba8(pixels) => Some(pixels),
        TextureImage::Ktx2(_) => None,
    }
}

/// Stores a produced container, or counts the pixels left as fallback.
#[cfg(feature = "gltf")]
fn apply_baked(
    slot: &mut Option<TextureImage>,
    result: Result<Baked, BakeError>,
    stats: &mut BakeStats,
) {
    match result {
        Ok(baked) => {
            count_origin(stats, baked.origin);
            *slot = Some(TextureImage::Ktx2(baked.image));
        }
        Err(error) => {
            stats.fallback += 1;
            stats.last_failure = Some(error.to_string());
        }
    }
}

/// Adds one produced container to the counters.
#[cfg(feature = "gltf")]
fn count_origin(stats: &mut BakeStats, origin: BakeOrigin) {
    match origin {
        BakeOrigin::Cached => stats.cached += 1,
        BakeOrigin::Encoded => stats.encoded += 1,
        BakeOrigin::Authored => stats.authored += 1,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Root directory of one test: the cache lives inside it, next to any fake
    /// tool the test installs.
    fn test_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ornis-bake-{}-{name}", std::process::id()))
    }

    /// Cache directory of one test, wiped before it is used.
    fn cache_dir(name: &str) -> PathBuf {
        let dir = test_root(name).join("cache");
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Policy that never spawns a process, so a test sees the same bytes
    /// whether or not `basisu` happens to be installed.
    fn in_process(name: &str) -> BakeSettings {
        BakeSettings {
            encoder: EncoderSource::InProcess,
            ..BakeSettings::new(cache_dir(name))
        }
    }

    /// 8×8 RGBA image: top half opaque red, bottom half half-transparent blue.
    fn checker() -> CpuImage {
        let mut pixels = Vec::with_capacity(8 * 8 * RGBA_CHANNELS);
        for row in 0..8 {
            let texel = if row < 4 {
                [200, 40, 10, 255]
            } else {
                [20, 60, 220, 128]
            };
            pixels.extend(texel.repeat(8));
        }
        CpuImage::from_rgba8(8, 8, pixels).expect("valid 8x8 image")
    }

    /// Key of one bake input, as the bake itself computes it.
    fn key_of(settings: &BakeSettings, image: &CpuImage, role: TextureRole) -> String {
        cache_key(image, role, settings.mode_of(role))
    }

    /// File permissions of the fake encoder (`rwxr-xr-x`).
    #[cfg(unix)]
    const EXECUTABLE_BITS: u32 = 0o755;

    /// Fake encoder: records its argv, keeps a copy of the pixels it was
    /// handed, and publishes `payload` as if it had encoded them.
    #[cfg(unix)]
    fn fake_encoder(name: &str, payload: &[u8]) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;
        // `$4` is the `-output_file` argument and `$8` the source bitmap, per
        // the layout `basis_args` pins (asserted in the tests below).
        const BODY: &str = r#"#!/bin/sh
printf '%s\n' "$@" >> "$0.args"
cp "$8" "$(dirname "$0")/read.tga"
cp "$(dirname "$0")/payload.ktx2" "$4"
"#;
        let dir = test_root(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("script dir");
        std::fs::write(dir.join("payload.ktx2"), payload).expect("payload");
        let program = dir.join("fake-basisu.sh");
        std::fs::write(&program, BODY).expect("script body");
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(EXECUTABLE_BITS))
            .expect("executable");
        program
    }

    /// Every argument the fake encoder has been called with so far.
    #[cfg(unix)]
    fn invocations(program: &Path) -> Vec<String> {
        let log = PathBuf::from(format!("{}.args", program.display()));
        std::fs::read_to_string(log)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// Pixels the fake encoder was handed (TGA header stripped).
    #[cfg(unix)]
    fn encoder_read(program: &Path) -> Vec<u8> {
        let tga = std::fs::read(program.with_file_name("read.tga")).expect("pixels handed over");
        tga[TGA_HEADER_BYTES..].to_vec()
    }

    /// One primitive whose four material slots hold the same decoded image.
    #[cfg(feature = "gltf")]
    fn textured_primitive() -> crate::ModelPrimitive {
        use ornis_gltf::{LoadedMaterial, LoadedMesh};
        let slot = Some(TextureImage::Rgba8(checker()));
        crate::ModelPrimitive {
            node: ornis_gltf::NodeIdx(0),
            mesh: LoadedMesh {
                positions: vec![[0.0, 0.0, 0.0]],
                indices: vec![0],
                normals: None,
                uvs: None,
                joints: None,
                weights: None,
            },
            material: LoadedMaterial {
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 1.0,
                emission: [0.0, 0.0, 0.0],
                textures: LoadedTextures {
                    base_color_texture: slot.clone(),
                    metallic_roughness_texture: slot.clone(),
                    emissive_texture: slot.clone(),
                    normal_texture: slot,
                },
            },
        }
    }

    /// One-primitive model holding [`textured_primitive`].
    #[cfg(feature = "gltf")]
    fn textured_model() -> crate::Model {
        crate::Model {
            name: "baked".into(),
            nodes: Vec::new(),
            roots: Vec::new(),
            primitives: vec![textured_primitive()],
            skins: Vec::new(),
            skel_clips: Vec::new(),
            anim_clips: Vec::new(),
            stats: Default::default(),
        }
    }

    #[test]
    fn role_defaults_cover_quality_and_size_slots() {
        assert_eq!(BasisMode::for_role(TextureRole::Normal), BasisMode::Uastc);
        assert_eq!(
            BasisMode::for_role(TextureRole::BaseColor),
            BasisMode::Etc1s
        );
        assert_eq!(
            BasisMode::for_role(TextureRole::MetallicRoughness),
            BasisMode::Etc1s
        );
        assert_eq!(BasisMode::for_role(TextureRole::Emissive), BasisMode::Etc1s);
        let mut settings = in_process("modes");
        settings
            .overrides
            .insert(TextureRole::BaseColor, BasisMode::Uastc);
        assert_eq!(settings.mode_of(TextureRole::BaseColor), BasisMode::Uastc);
        assert_eq!(settings.mode_of(TextureRole::Normal), BasisMode::Uastc);
        assert_eq!(settings.mode_of(TextureRole::Emissive), BasisMode::Etc1s);
        assert_eq!(
            BakeSettings::new("cache").mode_of(TextureRole::Emissive),
            BasisMode::Etc1s
        );
    }

    #[test]
    fn the_cache_key_covers_content_role_and_codec() {
        let source = checker();
        let other =
            CpuImage::from_rgba8(8, 8, vec![7; 8 * 8 * RGBA_CHANNELS]).expect("valid image");
        let keys = [
            cache_key(&source, TextureRole::BaseColor, BasisMode::Etc1s),
            cache_key(&source, TextureRole::BaseColor, BasisMode::Uastc),
            cache_key(&source, TextureRole::Normal, BasisMode::Etc1s),
            cache_key(&source, TextureRole::Emissive, BasisMode::Etc1s),
            cache_key(&other, TextureRole::BaseColor, BasisMode::Etc1s),
        ];
        let unique: HashSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "colliding keys: {keys:?}");
        assert!(
            keys.iter().all(|key| key.len() == 16),
            "key width: {keys:?}"
        );
        assert!(
            keys.iter()
                .all(|key| key.chars().all(|c| c.is_ascii_hexdigit()))
        );
    }

    #[test]
    fn in_process_bake_authors_the_role_chain_once_and_caches_it() {
        let settings = in_process("cache");
        let source = checker();
        let first = bake_texture(&settings, &source, TextureRole::BaseColor).expect("bakes");
        assert_eq!(first.origin, BakeOrigin::Authored);
        assert_eq!(first.image.dimensions(), (8, 8));
        let again = bake_texture(&settings, &source, TextureRole::BaseColor).expect("bakes");
        assert_eq!(again.origin, BakeOrigin::Cached, "the second call is a hit");
        assert_eq!(again.image.bytes(), first.image.bytes());
        // A different slot re-filters the chain, so it is a separate entry.
        let normal = key_of(&settings, &source, TextureRole::Normal);
        assert_ne!(normal, key_of(&settings, &source, TextureRole::BaseColor));
        assert!(bake_texture(&settings, &source, TextureRole::Normal).is_ok());
        let entry = settings.cache_dir.join(format!("{normal}{OUTPUT_SUFFIX}"));
        assert!(entry.is_file(), "role entry: {entry:?}");
        // An entry that stops parsing is a miss, not a failed bake.
        std::fs::write(&entry, b"junk").expect("corrupt entry");
        let repaired = bake_texture(&settings, &source, TextureRole::Normal).expect("rebakes");
        assert_eq!(repaired.origin, BakeOrigin::Authored);
    }

    #[test]
    fn baked_bytes_are_a_pure_function_of_the_input() {
        let source = checker();
        let one = in_process("determinism-a");
        let two = in_process("determinism-b");
        let first = bake_texture(&one, &source, TextureRole::MetallicRoughness).expect("bakes");
        let second = bake_texture(&two, &source, TextureRole::MetallicRoughness).expect("bakes");
        assert_eq!(first.image.bytes(), second.image.bytes());
        let direct = encode_rgba8_ktx2(&source, TextureRole::MetallicRoughness).expect("encodes");
        assert_eq!(first.image.bytes(), direct.as_slice());
    }

    #[test]
    fn the_authored_container_loads_with_the_full_mip_chain() {
        let settings = in_process("chain");
        let source = checker();
        let baked = bake_texture(&settings, &source, TextureRole::BaseColor).expect("bakes");
        let levels = baked
            .image
            .transcode(ornis_texture::TextureEncoding::Rgba8)
            .expect("loads");
        assert_eq!(levels.dimensions(), (8, 8));
        assert_eq!(levels.levels().len(), 4, "8x8 keeps four levels");
        assert_eq!(levels.levels()[0], source.pixels);
        let expected: Vec<Vec<u8>> =
            ornis_texture::generate_mipmaps(&source, TextureRole::BaseColor)
                .expect("chain")
                .into_iter()
                .map(|level| level.pixels)
                .collect();
        assert_eq!(levels.levels(), expected.as_slice());
    }

    #[test]
    fn a_missing_encoder_binary_is_a_typed_failure_and_leaves_no_entry() {
        let settings = BakeSettings {
            encoder: EncoderSource::Binary(PathBuf::from("/nonexistent/basisu")),
            ..in_process("missing")
        };
        let source = checker();
        let failure = bake_texture(&settings, &source, TextureRole::BaseColor);
        assert!(
            matches!(failure, Err(BakeError::Spawn { .. })),
            "unexpected {failure:?}"
        );
        let entry = settings.cache_dir.join(format!(
            "{}{OUTPUT_SUFFIX}",
            key_of(&settings, &source, TextureRole::BaseColor)
        ));
        assert!(!entry.exists(), "a failed bake must not be cached");
    }

    #[test]
    #[cfg(unix)]
    fn an_encoder_is_handed_the_pixels_and_the_pinned_flags() {
        let payload = encode_rgba8_ktx2(&checker(), TextureRole::BaseColor).expect("encodes");
        let program = fake_encoder("run", &payload);
        let settings = BakeSettings {
            encoder: EncoderSource::Binary(program.clone()),
            ..BakeSettings::new(cache_dir("run"))
        };
        let source = checker();
        let baked = bake_texture(&settings, &source, TextureRole::BaseColor).expect("bakes");
        assert_eq!(baked.origin, BakeOrigin::Encoded);
        assert_eq!(baked.image.bytes(), payload);
        let key = key_of(&settings, &source, TextureRole::BaseColor);
        let dump = settings
            .cache_dir
            .join(format!("{key}{DUMP_SUFFIX}"))
            .display()
            .to_string();
        let bitmap = settings
            .cache_dir
            .join(format!("{key}{SOURCE_SUFFIX}"))
            .display()
            .to_string();
        assert_eq!(
            invocations(&program),
            [
                "-ktx2",
                "-mipmap",
                "-output_file",
                &dump,
                "-etc1s",
                "-q",
                "128",
                &bitmap,
            ]
        );
        assert_eq!(
            encoder_read(&program),
            tga_bytes(&source)[TGA_HEADER_BYTES..]
        );
        // The second call is a cache hit: the tool does not run again.
        let again = bake_texture(&settings, &source, TextureRole::BaseColor).expect("bakes");
        assert_eq!(again.origin, BakeOrigin::Cached);
        assert_eq!(invocations(&program).len(), FIXED_FLAGS.len() + 5);
        assert!(
            !settings
                .cache_dir
                .join(format!("{key}{DUMP_SUFFIX}"))
                .exists()
        );
        assert!(
            !settings
                .cache_dir
                .join(format!("{key}{SOURCE_SUFFIX}"))
                .exists()
        );
    }

    #[test]
    #[cfg(unix)]
    fn an_override_replaces_the_role_default_in_the_encoder_flags() {
        let payload = encode_rgba8_ktx2(&checker(), TextureRole::Normal).expect("encodes");
        let program = fake_encoder("override", &payload);
        let settings = BakeSettings {
            encoder: EncoderSource::Binary(program.clone()),
            overrides: HashMap::from([(TextureRole::Normal, BasisMode::Etc1s)]),
            cache_dir: cache_dir("override"),
        };
        assert_eq!(settings.mode_of(TextureRole::Normal), BasisMode::Etc1s);
        let baked = bake_texture(&settings, &checker(), TextureRole::Normal).expect("bakes");
        assert_eq!(baked.origin, BakeOrigin::Encoded);
        let argv = invocations(&program);
        assert_eq!(argv[4..7], ["-etc1s", "-q", "128"], "override in {argv:?}");
    }

    #[test]
    #[cfg(unix)]
    fn an_encoder_failure_reports_its_exit_code_and_stderr() {
        let program = fake_encoder("exit", b"");
        std::fs::write(
            &program,
            "#!/bin/sh\ncat >&2 <<'END'\nbad pixels\nEND\nexit 3\n",
        )
        .expect("failing tool");
        let settings = BakeSettings {
            encoder: EncoderSource::Binary(program),
            ..BakeSettings::new(cache_dir("exit"))
        };
        let source = checker();
        let failure = bake_texture(&settings, &source, TextureRole::BaseColor);
        let Err(BakeError::Exit { code, stderr, .. }) = failure else {
            panic!("expected an exit failure, got {failure:?}");
        };
        assert_eq!(code, 3);
        assert!(stderr.contains("bad pixels"), "stderr was {stderr:?}");
    }

    #[test]
    #[cfg(unix)]
    fn unparsable_tool_output_is_reported_and_uncached() {
        let program = fake_encoder("garbage", b"not a container");
        let settings = BakeSettings {
            encoder: EncoderSource::Binary(program),
            ..BakeSettings::new(cache_dir("garbage"))
        };
        let source = checker();
        let failure = bake_texture(&settings, &source, TextureRole::BaseColor);
        assert!(
            matches!(failure, Err(BakeError::Image(_))),
            "unexpected {failure:?}"
        );
        let key = key_of(&settings, &source, TextureRole::BaseColor);
        assert!(
            !settings
                .cache_dir
                .join(format!("{key}{OUTPUT_SUFFIX}"))
                .exists()
        );
    }

    #[test]
    #[cfg(feature = "gltf")]
    fn model_bake_writes_every_slot_and_keeps_pixels_on_failure() {
        let settings = in_process("model");
        let mut model = textured_model();
        model.primitives.push(textured_primitive());
        let stats = bake_model(&settings, &mut model);
        assert_eq!(stats.authored, 4, "four slots bake once each");
        assert_eq!(stats.cached, 4, "the second primitive reuses them");
        assert_eq!(stats.fallback, 0);
        for role in BAKED_ROLES {
            let slot = model.primitives[0].material.texture(role);
            let Some(TextureImage::Ktx2(image)) = slot else {
                panic!("slot {role:?} is not baked: {slot:?}");
            };
            assert_eq!(image.dimensions(), (8, 8));
        }
        // A second pass has nothing to do: KTX2 slots are skipped, not counted
        // and not re-encoded (a reload re-imports the pixels and hits the cache).
        let again = bake_model(&settings, &mut model);
        assert_eq!(again, BakeStats::default());
        // A broken tool keeps the decoded pixels of every slot.
        let broken = BakeSettings {
            encoder: EncoderSource::Binary(PathBuf::from("/nonexistent/basisu")),
            ..in_process("model-broken")
        };
        let mut failing = textured_model();
        let stats = bake_model(&broken, &mut failing);
        assert_eq!(stats.fallback, 4);
        assert_eq!(stats.authored, 0);
        assert!(stats.last_failure.is_some());
        let kept = failing.primitives[0]
            .material
            .texture(TextureRole::BaseColor);
        assert!(matches!(kept, Some(TextureImage::Rgba8(pixels)) if *pixels == checker()));
    }
}
