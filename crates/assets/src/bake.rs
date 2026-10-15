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
//!   removes the runtime mip cost through
//!   [`encode_rgba8_ktx2`](ornis_texture::encode_rgba8_ktx2): the same
//!   authored levels, RGBA8 blocks instead of the ~4× smaller compressed ones.
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
use self::encoder::{author, encode_with_tool, encoder};
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    #[cfg(feature = "gltf")]
    use ornis_gltf::LoadedTextures;
    #[cfg(feature = "gltf")]
    use ornis_texture::TextureImage;
    use ornis_texture::encode_rgba8_ktx2;

    use super::cache::{OUTPUT_SUFFIX, cache_key};
    use super::encoder::{DUMP_SUFFIX, RGBA_CHANNELS, SOURCE_SUFFIX, TGA_HEADER_BYTES, tga_bytes};
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
