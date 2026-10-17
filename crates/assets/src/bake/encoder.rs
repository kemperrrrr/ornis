//! Reference `basisu` invocation, pixel handoff and the in-process fallback.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::SystemTime;

use ornis_texture::{CpuImage, Ktx2Image, TextureRole, encode_rgba8_ktx2};

use super::cache::{Producer, cache_file, cleanup, ensure_dir, fnv_hex, io_error, store_cached};
use super::{BakeError, BakeOrigin, BakeSettings, Baked, BasisMode, EncoderSource};

/// Reference encoder program, discovered on `PATH` when none is configured.
const ENCODER_PROGRAM: &str = "basisu";

/// Environment variable holding the executable search path.
const PATH_VARIABLE: &str = "PATH";

/// Suffix of the pixels handed to the encoder.
pub(super) const SOURCE_SUFFIX: &str = ".tga";

/// Suffix of the container the encoder writes, before validation.
pub(super) const DUMP_SUFFIX: &str = ".tool.ktx2";

/// Container flags of every encoder run, before the output path.
const CONTAINER_FLAGS: [&str; 2] = ["-ktx2", "-mipmap"];

/// Flag that names the container the tool must write.
const OUTPUT_FLAG: &str = "-output_file";

/// UASTC effort level, spelled out instead of relying on a tool default.
const UASTC_LEVEL: &str = "2";

/// ETC1S quality level, likewise spelled out (the tool's own default).
const ETC1S_QUALITY: &str = "128";

/// UASTC codec flags, with the effort pinned explicitly.
const UASTC_FLAGS: [&str; 3] = ["-uastc", "-uastc_level", UASTC_LEVEL];

/// ETC1S codec flags, with the quality pinned explicitly.
const ETC1S_FLAGS: [&str; 3] = ["-etc1s", "-q", ETC1S_QUALITY];

/// Mip flags pinned so a tool default cannot move the chain.
///
/// `kaiser` is the reference tool's own default kernel. It is not the
/// in-process area filter: the two writers must not be treated as the same
/// chain, which is why the cache key records these flags.
const MIP_FLAGS: [&str; 3] = ["-mip_clamp", "-mip_filter", "kaiser"];

/// sRGB colour and mip filtering (albedo and emission).
const SRGB_FLAGS: [&str; 2] = ["-srgb", "-mip_srgb"];

/// Linear colour and mip filtering (ORM data).
const LINEAR_FLAGS: [&str; 2] = ["-linear", "-mip_linear"];

/// Linear metrics, linear mips and renormalized normal mips.
const NORMAL_FLAGS: [&str; 4] = ["-normal_map", "-linear", "-mip_linear", "-mip_renorm"];

/// Encoder stderr bytes kept in [`BakeError::Exit`].
const STDERR_TAIL_BYTES: usize = 512;

/// Channels in one RGBA texel.
pub(super) const RGBA_CHANNELS: usize = 4;

/// Bytes in an uncompressed true-color TGA header (no ID, no color map).
pub(super) const TGA_HEADER_BYTES: usize = 18;

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

/// Argument the tool prints its version for.
const VERSION_FLAG: &str = "-version";

/// Identity of one encoder binary, remembered for the process.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IdentityStamp {
    /// Canonical program path.
    path: PathBuf,
    /// `mtime` when the platform reports one.
    modified: Option<SystemTime>,
    /// File length, so a same-timestamp rewrite still misses.
    len: u64,
}

/// Versions already read, keyed by [`IdentityStamp`].
static IDENTITIES: LazyLock<Mutex<std::collections::HashMap<IdentityStamp, String>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// How a fresh bake will be produced, and the producer mixed into its key.
pub(super) struct EncodePlan {
    /// Tool to spawn, when the plan is not the in-process writer.
    program: Option<PathBuf>,
    /// Origin recorded when this plan writes a new container.
    origin: BakeOrigin,
    /// Producer half of the cache key.
    producer: Producer,
}

impl EncodePlan {
    /// Producer mixed into the cache key of this plan.
    pub(super) fn producer(&self) -> &Producer {
        &self.producer
    }

    /// Tool path, when this plan spawns one.
    pub(super) fn program(&self) -> Option<&Path> {
        self.program.as_deref()
    }

    /// Origin of a container this plan writes itself.
    pub(super) fn origin(&self) -> BakeOrigin {
        self.origin
    }
}

/// Resolves `settings` into a writer or a tool, including the cache producer.
///
/// # Errors
///
/// [`BakeError::Spawn`] when a configured binary can neither print `-version`
/// nor be read for the content-hash fallback. A missing `basisu` on `PATH` is
/// not an error: the plan is the in-process writer with
/// [`BakeOrigin::Downgraded`].
pub(super) fn plan(settings: &BakeSettings, role: TextureRole) -> Result<EncodePlan, BakeError> {
    match &settings.encoder {
        EncoderSource::InProcess => Ok(writer_plan(BakeOrigin::Authored)),
        EncoderSource::Path => match find_on_path(ENCODER_PROGRAM) {
            Some(program) => tool_plan(program, role, settings.mode_of(role)),
            None => Ok(writer_plan(BakeOrigin::Downgraded)),
        },
        EncoderSource::Binary(program) => {
            tool_plan(canonical_or_keep(program), role, settings.mode_of(role))
        }
    }
}

/// In-process writer plan. The producer tag does not depend on `origin`.
fn writer_plan(origin: BakeOrigin) -> EncodePlan {
    EncodePlan {
        program: None,
        origin,
        producer: Producer::InProcess,
    }
}

/// Tool plan: identity from `-version`, or a content hash when that fails.
fn tool_plan(
    program: PathBuf,
    role: TextureRole,
    mode: BasisMode,
) -> Result<EncodePlan, BakeError> {
    let identity = tool_identity(&program)?;
    Ok(EncodePlan {
        program: Some(program),
        origin: BakeOrigin::Encoded,
        producer: Producer::Tool {
            identity,
            flags: pinned_flag_text(role, mode),
        },
    })
}

/// First existing `basisu` on an absolute `PATH` entry, canonicalized.
fn find_on_path(program: &str) -> Option<PathBuf> {
    std::env::var_os(PATH_VARIABLE).and_then(|path| first_absolute(&path, program))
}

/// First `program` file in an absolute directory of `path`.
///
/// Empty and relative entries are absent, so the result does not depend on
/// the process's current directory.
pub(super) fn first_absolute(path: &OsStr, program: &str) -> Option<PathBuf> {
    std::env::split_paths(path).find_map(|directory| absolute_file(&directory, program))
}

/// Canonical `directory/program` when `directory` is absolute and the file exists.
fn absolute_file(directory: &Path, program: &str) -> Option<PathBuf> {
    if !directory.is_absolute() {
        return None;
    }
    let candidate = directory.join(program);
    if candidate.is_file() {
        candidate.canonicalize().ok()
    } else {
        None
    }
}

/// Canonical path when the file exists, the original path otherwise.
fn canonical_or_keep(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Tool identity: `-version` stdout, else a content hash of the binary.
fn tool_identity(program: &Path) -> Result<String, BakeError> {
    if let Some(hit) = remembered(program) {
        return Ok(hit);
    }
    let identity = version_stdout(program)
        .or_else(|| file_hash(program))
        .ok_or_else(|| identity_failure(program))?;
    remember(program, &identity);
    Ok(identity)
}

/// Trimmed `-version` stdout, when the tool exits successfully with some.
fn version_stdout(program: &Path) -> Option<String> {
    let output = std::process::Command::new(program)
        .arg(VERSION_FLAG)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `bin:` plus the FNV-1a digest of the tool's bytes.
fn file_hash(program: &Path) -> Option<String> {
    let bytes = std::fs::read(program).ok()?;
    Some(format!("bin:{}", fnv_hex(&bytes)))
}

/// Failure reported when neither `-version` nor the content hash works.
fn identity_failure(program: &Path) -> BakeError {
    match std::process::Command::new(program)
        .arg(VERSION_FLAG)
        .output()
    {
        Err(source) => BakeError::Spawn {
            program: program.display().to_string(),
            source,
        },
        Ok(_) => match std::fs::read(program) {
            Err(source) => io_error(program, source),
            Ok(_) => BakeError::Exit {
                program: program.display().to_string(),
                code: 1,
                stderr: "encoder -version produced no identity".to_string(),
            },
        },
    }
}

/// Cached identity for `program`, when its stamp still matches.
fn remembered(program: &Path) -> Option<String> {
    let stamp = identity_stamp(program)?;
    let guard = IDENTITIES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.get(&stamp).cloned()
}

/// Stores `identity` under the current stamp of `program`.
fn remember(program: &Path, identity: &str) {
    let Some(stamp) = identity_stamp(program) else {
        return;
    };
    let mut guard = IDENTITIES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.insert(stamp, identity.to_string());
}

/// Stamp of `program`, or `None` when the file cannot be stated.
fn identity_stamp(program: &Path) -> Option<IdentityStamp> {
    let meta = std::fs::metadata(program).ok()?;
    Some(IdentityStamp {
        path: program.to_path_buf(),
        modified: meta.modified().ok(),
        len: meta.len(),
    })
}

/// In-process fallback: the authored mip chain, stored as RGBA8 blocks.
pub(super) fn author(
    settings: &BakeSettings,
    image: &CpuImage,
    role: TextureRole,
    key: &str,
    origin: BakeOrigin,
) -> Result<Baked, BakeError> {
    let bytes = encode_rgba8_ktx2(image, role)?;
    let container = Ktx2Image::from_bytes(bytes)?;
    store_cached(settings, key, container.bytes())?;
    Ok(Baked {
        image: container,
        origin,
    })
}

/// One encoder run: pixel handoff, CLI call, validation, cache store.
pub(super) fn encode_with_tool(
    settings: &BakeSettings,
    program: &Path,
    image: &CpuImage,
    role: TextureRole,
    key: &str,
) -> Result<Baked, BakeError> {
    ensure_dir(settings)?;
    let source = cache_file(settings, key, SOURCE_SUFFIX);
    let dump = cache_file(settings, key, DUMP_SUFFIX);
    let args = basis_args(role, settings.mode_of(role), &source, &dump);
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

/// Pinned flags of one run, without the output path or the source path.
///
/// Colour space is per role: sRGB for albedo and emission, linear for ORM,
/// and linear plus renormalized mips for normals. The codec and the mip
/// kernel are pinned beside that, so a tool default cannot move the bake.
pub(super) fn pinned_flags(role: TextureRole, mode: BasisMode) -> Vec<&'static str> {
    let mut flags = Vec::new();
    flags.extend(CONTAINER_FLAGS);
    flags.extend(mode_flag_strs(mode));
    flags.extend(role_flags(role));
    flags.extend(MIP_FLAGS);
    flags
}

/// Space-joined [`pinned_flags`], the flag half of a tool cache key.
pub(super) fn pinned_flag_text(role: TextureRole, mode: BasisMode) -> String {
    pinned_flags(role, mode).join(" ")
}

/// Pinned argument vector for one encoder run.
fn basis_args(role: TextureRole, mode: BasisMode, source: &Path, dump: &Path) -> Vec<OsString> {
    let mut args = Vec::new();
    for (index, flag) in pinned_flags(role, mode).into_iter().enumerate() {
        args.push(OsString::from(flag));
        if index + 1 == CONTAINER_FLAGS.len() {
            args.push(OsString::from(OUTPUT_FLAG));
            args.push(dump.into());
        }
    }
    args.push(source.into());
    args
}

/// Codec flags of one bake mode.
fn mode_flag_strs(mode: BasisMode) -> &'static [&'static str] {
    match mode {
        BasisMode::Uastc => &UASTC_FLAGS,
        BasisMode::Etc1s => &ETC1S_FLAGS,
    }
}

/// Colour-space flags of one material role.
fn role_flags(role: TextureRole) -> &'static [&'static str] {
    match role {
        TextureRole::BaseColor | TextureRole::Emissive => &SRGB_FLAGS,
        TextureRole::MetallicRoughness => &LINEAR_FLAGS,
        TextureRole::Normal => &NORMAL_FLAGS,
    }
}

/// Uncompressed 32-bit TGA of `image`, rows bottom-up.
///
/// The reference encoder reads PNG/JPEG/TGA/DDS/QOI; a TGA header plus raw
/// texels is the cheapest lossless handoff once the imported model only holds
/// decoded pixels. Edges above `u16` are unreachable: [`bake_texture`] rejects
/// anything past `MAX_TEXTURE_EDGE` first.
pub(super) fn tga_bytes(image: &CpuImage) -> Vec<u8> {
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
