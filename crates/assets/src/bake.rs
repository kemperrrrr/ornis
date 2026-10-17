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
//!   codec), the bake schema and the producer. The in-process writer is tagged
//!   `rgba8-writer-v1`; a tool contributes its `-version` text (or a content
//!   hash of the binary) and the pinned flags, without file paths. An RGBA8
//!   fallback and a `basisu` run therefore never share an entry, and replacing
//!   the tool or a flag misses the old container. Nothing is inferred from
//!   file names, sizes or timestamps. The key still hashes decoded texels, not
//!   the source file bytes, so a re-encoded PNG with the same pixels hits.
//! - Basis has no pure-Rust encoder, so compressed payloads come from the
//!   reference `basisu` command line (the project the runtime transcoder port
//!   comes from). [`EncoderSource`] picks the tool: a configured binary, one
//!   discovered on an absolute `PATH` entry, or none at all. When no tool runs,
//!   the bake still removes the runtime mip cost through
//!   [`encode_rgba8_ktx2`](ornis_texture::encode_rgba8_ktx2): the same
//!   authored levels, RGBA8 blocks instead of the compressed ones. That RGBA8
//!   container does not meet the compressed-size goal. An explicit
//!   [`EncoderSource::InProcess`] is [`BakeOrigin::Authored`]; a `PATH` lookup
//!   that finds nothing is [`BakeOrigin::Downgraded`] and is counted apart.
//! - A bake never replaces the source: [`bake_texture`] hands a container back
//!   only on success, and [`bake_model`] keeps the decoded PNG/JPEG pixels of
//!   any slot whose bake failed. A host that never calls
//!   [`AssetServer::with_texture_baking`](crate::AssetServer::with_texture_baking)
//!   keeps the previous import behaviour unchanged.
//! - This file holds the policy types and the two entry points; the CLI driver
//!   and TGA handoff, the content-addressed cache and the model pass live in
//!   the private submodules beside it.
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
use std::path::PathBuf;

use ornis_texture::{CpuImage, Ktx2Image, TextureRole, TextureUploadError};

mod cache;
mod encoder;
#[cfg(feature = "gltf")]
mod model;

use self::cache::{cache_key, load_cached};
use self::encoder::{author, encode_with_tool, plan};
#[cfg(feature = "gltf")]
pub use self::model::bake_model;

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
    /// `basisu` discovered on an absolute `PATH` entry (canonicalized), the
    /// in-process authored chain when it is not installed. Empty and relative
    /// entries are ignored, so the result does not depend on the current
    /// directory. That fallback is [`BakeOrigin::Downgraded`], not a silent
    /// cache hit under the compressed codec.
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
    /// Written in-process by the RGBA8 writer, because the caller asked for it.
    Authored,
    /// Written in-process because [`EncoderSource::Path`] found no `basisu`.
    ///
    /// The container is RGBA8. It removes the runtime mip cost and does not
    /// meet the compressed-size goal. [`BakeStats::downgraded`] counts these
    /// slots, and [`BakeStats::last_failure`] records why when nothing harder
    /// failed.
    Downgraded,
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
    /// Containers written in-process because the caller chose that
    /// ([`EncoderSource::InProcess`]).
    pub authored: usize,
    /// Containers written in-process because [`EncoderSource::Path`] found no
    /// `basisu`. These are RGBA8 and do not meet the compressed-size goal.
    pub downgraded: usize,
    /// Slots left as decoded PNG/JPEG after a failed bake.
    pub fallback: usize,
    /// Formatted reason of the last failure, or the RGBA8 downgrade reason when
    /// a `PATH` miss wrote RGBA8 and nothing harder failed.
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
/// The cache key covers the decoded pixels, their dimensions, the role, the
/// codec, the bake schema and the producer (in-process tag, or tool identity
/// plus pinned flags). Two sources that decode to the same image share one
/// entry only when the same producer would write it. An entry that no longer
/// parses as KTX2 counts as a miss and is rewritten; only containers that
/// passed [`Ktx2Image::from_bytes`] are ever stored. Upgrading `basisu` or
/// changing a pinned flag changes the key, so the old container is left unused.
///
/// # Errors
///
/// [`BakeError`] when `image` is not a valid texture, the cache is unwritable,
/// or a configured encoder fails or returns a container this workspace rejects.
/// A missing `basisu` on `PATH` is not an error — the RGBA8 chain still removes
/// the runtime mip cost — but the result is [`BakeOrigin::Downgraded`], which
/// does not meet the compressed-size goal.
pub fn bake_texture(
    settings: &BakeSettings,
    image: &CpuImage,
    role: TextureRole,
) -> Result<Baked, BakeError> {
    image.validate()?;
    let plan = plan(settings, role)?;
    let key = cache_key(image, role, settings.mode_of(role), plan.producer());
    if let Some(cached) = load_cached(settings, &key) {
        return Ok(cached);
    }
    encode_fresh(settings, image, role, &key, &plan)
}

/// Runs the planned encoder, or the in-process writer when the plan has none.
fn encode_fresh(
    settings: &BakeSettings,
    image: &CpuImage,
    role: TextureRole,
    key: &str,
    plan: &encoder::EncodePlan,
) -> Result<Baked, BakeError> {
    match plan.program() {
        Some(program) => encode_with_tool(settings, program, image, role, key),
        None => author(settings, image, role, key, plan.origin()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    #[cfg(feature = "gltf")]
    use ornis_gltf::LoadedTextures;
    #[cfg(feature = "gltf")]
    use ornis_texture::TextureImage;
    use ornis_texture::encode_rgba8_ktx2;

    use super::cache::{BAKE_SCHEMA, OUTPUT_SUFFIX, Producer, RGBA8_WRITER_TAG, cache_key};
    use super::encoder::{
        DUMP_SUFFIX, RGBA_CHANNELS, SOURCE_SUFFIX, TGA_HEADER_BYTES, first_absolute, pinned_flags,
        tga_bytes,
    };
    #[cfg(feature = "gltf")]
    use super::model::BAKED_ROLES;
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
        let plan = super::encoder::plan(settings, role).expect("encoder resolves");
        cache_key(image, role, settings.mode_of(role), plan.producer())
    }

    /// File permissions of the fake encoder (`rwxr-xr-x`).
    #[cfg(unix)]
    const EXECUTABLE_BITS: u32 = 0o755;

    /// Per-channel error of the committed ETC1S fixture's base level.
    ///
    /// Measured on the 16×16 ramp encoded by basisu 2.10.0 (`-q 128`): the
    /// worst channel differs by 22.
    const ETC1S_LEVEL0_TOLERANCE: u8 = 22;

    /// Per-channel error of the committed UASTC fixture's base level.
    ///
    /// Measured on the same ramp at `-uastc_level 2`: the worst channel
    /// differs by 5.
    #[cfg(unix)]
    const UASTC_LEVEL0_TOLERANCE: u8 = 5;

    /// Body of an encoder that refuses `-version` and still copies a payload.
    #[cfg(unix)]
    const SILENT_BODY: &str = r#"#!/bin/sh
# @@MARKER@@
if [ "$1" = "-version" ]; then
    exit 1
fi
printf '%s\n' "$@" >> "$0.args"
out=""
src=""
skip=""
for arg in "$@"; do
    if [ -n "$skip" ]; then
        if [ "$skip" = "out" ]; then
            out=$arg
        fi
        skip=""
        continue
    fi
    case "$arg" in
        -output_file) skip=out ;;
        -q|-uastc_level|-mip_filter) skip=val ;;
        -*) ;;
        *) src=$arg ;;
    esac
done
cp "$src" "$(dirname "$0")/read.tga"
cp "$(dirname "$0")/payload.ktx2" "$out"
"#;

    /// 16×16 ramp committed next to the Basis fixtures.
    fn gradient_png() -> CpuImage {
        let image =
            image::load_from_memory(include_bytes!("../../texture/tests/data/bake-gradient.png"))
                .expect("png")
                .to_rgba8();
        CpuImage::from_rgba8(image.width(), image.height(), image.into_raw()).expect("gradient")
    }

    /// Every channel of `level` is within `tolerance` of `source`.
    fn assert_level0_close(level: &[u8], source: &[u8], tolerance: u8, label: &str) {
        assert_eq!(level.len(), source.len(), "{label}");
        for (index, (got, expected)) in level.iter().zip(source).enumerate() {
            let delta = got.abs_diff(*expected);
            assert!(
                delta <= tolerance,
                "{label} byte {index} differs by {delta} (limit {tolerance})"
            );
        }
    }

    /// Test-harness filter of `name` (crate prefix stripped).
    fn harness_filter(name: &str) -> String {
        let module = module_path!();
        let rest = module.split_once("::").map_or(module, |(_, rest)| rest);
        format!("{rest}::{name}")
    }

    /// Fake encoder: prints `version` for `-version`, records its argv, keeps
    /// a copy of the pixels it was handed, and publishes `payload`.
    #[cfg(unix)]
    fn fake_encoder(name: &str, payload: &[u8]) -> PathBuf {
        fake_encoder_version(name, payload, "fake-basisu 1")
    }

    /// [`fake_encoder`] whose `-version` stdout is `version`.
    #[cfg(unix)]
    fn fake_encoder_version(name: &str, payload: &[u8], version: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;
        // Flags that take a value are skipped so the source path is the last
        // positional argument, whatever the role/codec flags are.
        const BODY: &str = r#"#!/bin/sh
if [ "$1" = "-version" ]; then
    printf '%s\n' '@@VERSION@@'
    exit 0
fi
printf '%s\n' "$@" >> "$0.args"
out=""
src=""
skip=""
for arg in "$@"; do
    if [ -n "$skip" ]; then
        if [ "$skip" = "out" ]; then
            out=$arg
        fi
        skip=""
        continue
    fi
    case "$arg" in
        -output_file) skip=out ;;
        -q|-uastc_level|-mip_filter) skip=val ;;
        -*) ;;
        *) src=$arg ;;
    esac
done
cp "$src" "$(dirname "$0")/read.tga"
cp "$(dirname "$0")/payload.ktx2" "$out"
"#;
        let dir = test_root(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("script dir");
        std::fs::write(dir.join("payload.ktx2"), payload).expect("payload");
        let program = dir.join("fake-basisu.sh");
        std::fs::write(&program, BODY.replace("@@VERSION@@", version)).expect("script body");
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

    /// Full argv the encoder must see for `role` and `mode`.
    ///
    /// Hard-coded, so a change to the production flag list fails the tests
    /// instead of following them.
    #[cfg(unix)]
    fn expected_argv(role: TextureRole, mode: BasisMode, dump: &str, source: &str) -> Vec<String> {
        let mut args: Vec<String> = expected_flags(role, mode)
            .into_iter()
            .map(str::to_string)
            .collect();
        args.insert(2, "-output_file".to_string());
        args.insert(3, dump.to_string());
        args.push(source.to_string());
        args
    }

    /// Pinned flags without paths. Colour space is per role; the mip kernel is
    /// always named.
    #[cfg(unix)]
    fn expected_flags(role: TextureRole, mode: BasisMode) -> Vec<&'static str> {
        let mut flags = vec!["-ktx2", "-mipmap"];
        match mode {
            BasisMode::Uastc => flags.extend(["-uastc", "-uastc_level", "2"]),
            BasisMode::Etc1s => flags.extend(["-etc1s", "-q", "128"]),
        }
        match role {
            TextureRole::BaseColor | TextureRole::Emissive => flags.extend(["-srgb", "-mip_srgb"]),
            TextureRole::MetallicRoughness => flags.extend(["-linear", "-mip_linear"]),
            TextureRole::Normal => {
                flags.extend(["-normal_map", "-linear", "-mip_linear", "-mip_renorm"]);
            }
        }
        flags.extend(["-mip_clamp", "-mip_filter", "kaiser"]);
        flags
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
        let writer = Producer::InProcess;
        let keys = [
            cache_key(&source, TextureRole::BaseColor, BasisMode::Etc1s, &writer),
            cache_key(&source, TextureRole::BaseColor, BasisMode::Uastc, &writer),
            cache_key(&source, TextureRole::Normal, BasisMode::Etc1s, &writer),
            cache_key(&source, TextureRole::Emissive, BasisMode::Etc1s, &writer),
            cache_key(&other, TextureRole::BaseColor, BasisMode::Etc1s, &writer),
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
        assert_eq!(BAKE_SCHEMA, 1);
        assert_eq!(RGBA8_WRITER_TAG, "rgba8-writer-v1");
    }

    #[test]
    fn the_cache_key_separates_the_writer_from_the_tool_and_its_flags() {
        let source = checker();
        let role = TextureRole::BaseColor;
        let mode = BasisMode::Etc1s;
        let writer = cache_key(&source, role, mode, &Producer::InProcess);
        let tool = |identity: &str, flags: &str| {
            cache_key(
                &source,
                role,
                mode,
                &Producer::Tool {
                    identity: identity.to_string(),
                    flags: flags.to_string(),
                },
            )
        };
        let flags = pinned_flags(role, mode).join(" ");
        let encoded = tool("fake-basisu 1", &flags);
        let other_version = tool("fake-basisu 2", &flags);
        let other_flags = tool(
            "fake-basisu 1",
            &pinned_flags(TextureRole::Normal, BasisMode::Etc1s).join(" "),
        );
        assert_ne!(writer, encoded, "rgba8 and basisu must not share a key");
        assert_ne!(encoded, other_version, "tool version is part of the key");
        assert_ne!(encoded, other_flags, "a pinned flag is part of the key");
        let unique: HashSet<&String> = [&writer, &encoded, &other_version, &other_flags]
            .into_iter()
            .collect();
        assert_eq!(unique.len(), 4);
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
        let cached = std::fs::read_dir(&settings.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .any(|entry| entry.path().extension().is_some_and(|ext| ext == "ktx2"))
            })
            .unwrap_or(false);
        assert!(!cached, "a failed bake must not be cached");
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
            expected_argv(TextureRole::BaseColor, BasisMode::Etc1s, &dump, &bitmap)
        );
        assert_eq!(
            encoder_read(&program),
            tga_bytes(&source)[TGA_HEADER_BYTES..]
        );
        // The second call is a cache hit: the tool does not run again.
        let first_run = invocations(&program);
        let again = bake_texture(&settings, &source, TextureRole::BaseColor).expect("bakes");
        assert_eq!(again.origin, BakeOrigin::Cached);
        assert_eq!(invocations(&program), first_run, "the tool runs once");
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
        let bitmap = settings
            .cache_dir
            .join(format!(
                "{}{SOURCE_SUFFIX}",
                key_of(&settings, &checker(), TextureRole::Normal)
            ))
            .display()
            .to_string();
        let dump = settings
            .cache_dir
            .join(format!(
                "{}{DUMP_SUFFIX}",
                key_of(&settings, &checker(), TextureRole::Normal)
            ))
            .display()
            .to_string();
        assert_eq!(
            invocations(&program),
            expected_argv(TextureRole::Normal, BasisMode::Etc1s, &dump, &bitmap)
        );
    }

    #[test]
    #[cfg(unix)]
    fn every_role_and_mode_pins_colour_space_in_the_encoder_flags() {
        let payload = encode_rgba8_ktx2(&checker(), TextureRole::BaseColor).expect("encodes");
        let program = fake_encoder("roles", &payload);
        let roles = [
            TextureRole::BaseColor,
            TextureRole::MetallicRoughness,
            TextureRole::Emissive,
            TextureRole::Normal,
        ];
        let modes = [BasisMode::Etc1s, BasisMode::Uastc];
        for role in roles {
            for mode in modes {
                let settings = BakeSettings {
                    encoder: EncoderSource::Binary(program.clone()),
                    overrides: HashMap::from([(role, mode)]),
                    cache_dir: cache_dir(&format!("roles-{role:?}-{mode:?}")),
                };
                let source = checker();
                let baked = bake_texture(&settings, &source, role).expect("bakes");
                assert_eq!(baked.origin, BakeOrigin::Encoded);
                let key = key_of(&settings, &source, role);
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
                    expected_argv(role, mode, &dump, &bitmap),
                    "argv for {role:?} {mode:?}"
                );
                assert_eq!(
                    key,
                    cache_key(
                        &source,
                        role,
                        mode,
                        &Producer::Tool {
                            identity: "fake-basisu 1".into(),
                            flags: expected_flags(role, mode).join(" "),
                        },
                    ),
                    "the cache key must carry these flags"
                );
                let _ = std::fs::write(format!("{}.args", program.display()), "");
            }
        }
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
    #[cfg(unix)]
    fn in_process_and_tool_bakes_do_not_share_a_cache_entry() {
        let payload = encode_rgba8_ktx2(&checker(), TextureRole::BaseColor).expect("encodes");
        let program = fake_encoder("split", &payload);
        let source = checker();
        let dir = cache_dir("split");
        let authored = bake_texture(&in_process_at(&dir), &source, TextureRole::BaseColor)
            .expect("in-process");
        assert_eq!(authored.origin, BakeOrigin::Authored);
        let encoded = bake_texture(
            &binary_at(&dir, program.clone()),
            &source,
            TextureRole::BaseColor,
        )
        .expect("tool");
        assert_eq!(
            encoded.origin,
            BakeOrigin::Encoded,
            "a tool must not reuse the rgba8 entry"
        );
        let reverse = cache_dir("split-reverse");
        let first = bake_texture(
            &binary_at(&reverse, program),
            &source,
            TextureRole::BaseColor,
        )
        .expect("tool first");
        assert_eq!(first.origin, BakeOrigin::Encoded);
        let second = bake_texture(&in_process_at(&reverse), &source, TextureRole::BaseColor)
            .expect("writer second");
        assert_eq!(
            second.origin,
            BakeOrigin::Authored,
            "the writer must not reuse the tool entry"
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_tool_version_change_misses_the_previous_entry() {
        let payload = encode_rgba8_ktx2(&checker(), TextureRole::BaseColor).expect("encodes");
        let program = fake_encoder_version("version", &payload, "fake-basisu 1");
        let source = checker();
        let dir = cache_dir("version");
        let first = bake_texture(
            &binary_at(&dir, program.clone()),
            &source,
            TextureRole::BaseColor,
        )
        .expect("first version");
        assert_eq!(first.origin, BakeOrigin::Encoded);
        let replacement = fake_encoder_version("version-2", &payload, "fake-basisu 2");
        let second = bake_texture(
            &binary_at(&dir, replacement),
            &source,
            TextureRole::BaseColor,
        )
        .expect("second version");
        assert_eq!(
            second.origin,
            BakeOrigin::Encoded,
            "version is part of the key"
        );
        assert_ne!(
            key_of(&binary_at(&dir, program), &source, TextureRole::BaseColor),
            key_of(
                &binary_at(
                    &dir,
                    fake_encoder_version("version-3", &payload, "fake-basisu 2")
                ),
                &source,
                TextureRole::BaseColor,
            )
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_tool_without_version_output_is_keyed_by_its_bytes() {
        let payload = encode_rgba8_ktx2(&checker(), TextureRole::BaseColor).expect("encodes");
        let source = checker();
        let alpha = silent_encoder("silent-a", &payload, "alpha");
        let beta = silent_encoder("silent-b", &payload, "beta");
        let key_alpha = key_of(
            &binary_at(&cache_dir("silent-a"), alpha.clone()),
            &source,
            TextureRole::BaseColor,
        );
        let key_beta = key_of(
            &binary_at(&cache_dir("silent-b"), beta.clone()),
            &source,
            TextureRole::BaseColor,
        );
        assert_ne!(key_alpha, key_beta, "content hash separates the binaries");
        let baked = bake_texture(
            &binary_at(&cache_dir("silent-run"), alpha),
            &source,
            TextureRole::BaseColor,
        )
        .expect("hash fallback still encodes");
        assert_eq!(baked.origin, BakeOrigin::Encoded);
    }

    #[test]
    fn relative_and_empty_path_entries_do_not_resolve_an_encoder() {
        assert!(first_absolute(std::ffi::OsStr::new(""), "basisu").is_none());
        assert!(first_absolute(std::ffi::OsStr::new("relative-bin"), "basisu").is_none());
        assert!(first_absolute(std::ffi::OsStr::new(":"), "basisu").is_none());
        let listed = std::env::join_paths([Path::new("relative"), Path::new("")]).expect("join");
        assert!(first_absolute(&listed, "basisu").is_none());
    }

    #[test]
    #[cfg(unix)]
    fn an_absolute_path_entry_is_canonicalized() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = test_root("path-hit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let program = dir.join("basisu");
        std::fs::write(&program, b"#!/bin/sh\nexit 0\n").expect("tool");
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(EXECUTABLE_BITS))
            .expect("executable");
        let listed = std::env::join_paths([Path::new("."), &dir]).expect("join");
        let found = first_absolute(&listed, "basisu").expect("absolute entry");
        assert_eq!(found, program.canonicalize().expect("canonical"));
        assert!(found.is_absolute());
    }

    #[test]
    fn an_empty_path_downgrades_instead_of_hiding_the_rgba8_container() {
        if std::env::var_os("ORNIS_BAKE_EMPTY_PATH").is_some() {
            assert_empty_path_downgrade();
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                &harness_filter("an_empty_path_downgrades_instead_of_hiding_the_rgba8_container"),
                "--test-threads",
                "1",
            ])
            .env("PATH", "")
            .env("ORNIS_BAKE_EMPTY_PATH", "1")
            .output()
            .expect("reinvoke");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "empty PATH child failed\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// Bake under an empty `PATH`: the container is RGBA8 and the model pass
    /// counts it as a downgrade.
    fn assert_empty_path_downgrade() {
        let settings = BakeSettings::new(cache_dir("empty-path"));
        assert!(matches!(settings.encoder, EncoderSource::Path));
        let baked =
            bake_texture(&settings, &checker(), TextureRole::BaseColor).expect("downgrades");
        assert_eq!(baked.origin, BakeOrigin::Downgraded);
        assert_eq!(
            baked.image.encoding(),
            Some(ornis_texture::TextureEncoding::Rgba8)
        );
        let again = bake_texture(&settings, &checker(), TextureRole::BaseColor).expect("cached");
        assert_eq!(again.origin, BakeOrigin::Cached);
        #[cfg(feature = "gltf")]
        {
            let settings = BakeSettings::new(cache_dir("empty-path-model"));
            let mut model = textured_model();
            let stats = bake_model(&settings, &mut model);
            assert_eq!(stats.downgraded, 4);
            assert_eq!(stats.authored, 0);
            assert_eq!(stats.encoded, 0);
            assert_eq!(stats.fallback, 0);
            assert_eq!(
                stats.last_failure.as_deref(),
                Some(super::model::DOWNGRADE_REASON)
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn a_real_basis_container_round_trips_through_the_bake() {
        let source = gradient_png();
        for (name, bytes, mode, tolerance) in [
            (
                "etc1s",
                include_bytes!("../../texture/tests/data/bake-etc1s.ktx2").as_slice(),
                BasisMode::Etc1s,
                ETC1S_LEVEL0_TOLERANCE,
            ),
            (
                "uastc",
                include_bytes!("../../texture/tests/data/bake-uastc.ktx2").as_slice(),
                BasisMode::Uastc,
                UASTC_LEVEL0_TOLERANCE,
            ),
        ] {
            let program = fake_encoder(&format!("basis-{name}"), bytes);
            let settings = BakeSettings {
                encoder: EncoderSource::Binary(program),
                overrides: HashMap::from([(TextureRole::BaseColor, mode)]),
                cache_dir: cache_dir(&format!("basis-{name}")),
            };
            let baked = bake_texture(&settings, &source, TextureRole::BaseColor).expect(name);
            assert_eq!(baked.origin, BakeOrigin::Encoded, "{name}");
            assert_eq!(baked.image.encoding(), None, "{name} is Basis, not RGBA8");
            let levels = baked
                .image
                .transcode(ornis_texture::TextureEncoding::Rgba8)
                .expect(name);
            let edge = source.width.max(source.height);
            assert_eq!(levels.levels().len(), edge.ilog2() as usize + 1, "{name}");
            assert_level0_close(&levels.levels()[0], &source.pixels, tolerance, name);
        }
    }

    #[test]
    fn a_configured_basisu_is_deterministic_when_present() {
        let Ok(program) = std::env::var("ORNIS_BASISU") else {
            eprintln!("ORNIS_BASISU is unset; the real encoder round trip was not run");
            return;
        };
        let program = PathBuf::from(program);
        assert!(
            program.is_file(),
            "ORNIS_BASISU does not point at a file: {}",
            program.display()
        );
        let source = gradient_png();
        let first = bake_texture(
            &binary_at(&cache_dir("real-a"), program.clone()),
            &source,
            TextureRole::BaseColor,
        )
        .expect("first real bake");
        let second = bake_texture(
            &binary_at(&cache_dir("real-b"), program),
            &source,
            TextureRole::BaseColor,
        )
        .expect("second real bake");
        assert_eq!(first.origin, BakeOrigin::Encoded);
        assert_eq!(second.origin, BakeOrigin::Encoded);
        assert_eq!(first.image.bytes(), second.image.bytes());
        assert_eq!(first.image.encoding(), None);
        let levels = first
            .image
            .transcode(ornis_texture::TextureEncoding::Rgba8)
            .expect("transcode");
        assert_eq!(
            levels.levels().len(),
            source.width.max(source.height).ilog2() as usize + 1
        );
        assert_level0_close(
            &levels.levels()[0],
            &source.pixels,
            ETC1S_LEVEL0_TOLERANCE,
            "real basisu",
        );
    }

    /// Settings that write with the in-process RGBA8 writer into `dir`.
    fn in_process_at(dir: &Path) -> BakeSettings {
        BakeSettings {
            encoder: EncoderSource::InProcess,
            ..BakeSettings::new(dir)
        }
    }

    /// Settings that run `program` and store containers in `dir`.
    fn binary_at(dir: &Path, program: PathBuf) -> BakeSettings {
        BakeSettings {
            encoder: EncoderSource::Binary(program),
            ..BakeSettings::new(dir)
        }
    }

    /// Encoder whose `-version` fails, so the key falls back to a content hash.
    ///
    /// `marker` is a comment, so two otherwise identical tools hash apart.
    #[cfg(unix)]
    fn silent_encoder(name: &str, payload: &[u8], marker: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;
        let program = fake_encoder(name, payload);
        let body = SILENT_BODY.replace("@@MARKER@@", marker);
        std::fs::write(&program, body).expect("silent tool");
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(EXECUTABLE_BITS))
            .expect("executable");
        program
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
        assert_eq!(stats.downgraded, 0);
        assert_eq!(stats.fallback, 0);
        assert_eq!(stats.last_failure, None);
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
        assert_eq!(stats.downgraded, 0);
        assert!(stats.last_failure.is_some());
        let kept = failing.primitives[0]
            .material
            .texture(TextureRole::BaseColor);
        assert!(matches!(kept, Some(TextureImage::Rgba8(pixels)) if *pixels == checker()));
    }
}
