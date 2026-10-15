//! Content-addressed bake cache: keys, entries and the file plumbing.

use std::path::{Path, PathBuf};

use ornis_texture::{CpuImage, Ktx2Image, TextureRole};

use super::{BakeError, BakeOrigin, BakeSettings, Baked, BasisMode};

/// Cache file suffix (validated KTX2 containers).
pub(super) const OUTPUT_SUFFIX: &str = ".ktx2";

/// FNV-1a 64 offset basis, the cache key's starting state.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Content-addressed key of one bake input.
///
/// FNV-1a over the exact bytes: a cache key, not a security hash. A collision
/// costs one stale container that still has to parse as KTX2 before it can
/// reach the renderer.
pub(super) fn cache_key(image: &CpuImage, role: TextureRole, mode: BasisMode) -> String {
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
pub(super) fn cache_file(settings: &BakeSettings, key: &str, suffix: &str) -> PathBuf {
    settings.cache_dir.join(format!("{key}{suffix}"))
}

/// Reads the stored container of `key`; a missing or invalid entry is a miss.
pub(super) fn load_cached(settings: &BakeSettings, key: &str) -> Option<Baked> {
    let path = cache_file(settings, key, OUTPUT_SUFFIX);
    let image = Ktx2Image::from_bytes(std::fs::read(path).ok()?).ok()?;
    Some(Baked {
        image,
        origin: BakeOrigin::Cached,
    })
}

/// Creates the cache directory (idempotent).
pub(super) fn ensure_dir(settings: &BakeSettings) -> Result<(), BakeError> {
    std::fs::create_dir_all(&settings.cache_dir)
        .map_err(|error| io_error(&settings.cache_dir, error))
}

/// Writes one validated container into the cache under `key`.
pub(super) fn store_cached(
    settings: &BakeSettings,
    key: &str,
    bytes: &[u8],
) -> Result<(), BakeError> {
    ensure_dir(settings)?;
    let path = cache_file(settings, key, OUTPUT_SUFFIX);
    std::fs::write(&path, bytes).map_err(|error| io_error(&path, error))
}

/// Removes one encoder scratch file (best effort: the cache entry is valid).
pub(super) fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// IO failure tagged with the path it happened on.
pub(super) fn io_error(path: &Path, source: std::io::Error) -> BakeError {
    BakeError::Io {
        path: path.to_path_buf(),
        source,
    }
}
