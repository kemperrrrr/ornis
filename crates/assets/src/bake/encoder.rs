//! Reference `basisu` invocation, pixel handoff and the in-process fallback.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use ornis_texture::{CpuImage, Ktx2Image, TextureRole, encode_rgba8_ktx2};

use super::cache::{cache_file, cleanup, ensure_dir, io_error, store_cached};
use super::{BakeError, BakeOrigin, BakeSettings, Baked, BasisMode, EncoderSource};

/// Reference encoder program, discovered on `PATH` when none is configured.
const ENCODER_PROGRAM: &str = "basisu";

/// Environment variable holding the executable search path.
const PATH_VARIABLE: &str = "PATH";

/// Suffix of the pixels handed to the encoder.
pub(super) const SOURCE_SUFFIX: &str = ".tga";

/// Suffix of the container the encoder writes, before validation.
pub(super) const DUMP_SUFFIX: &str = ".tool.ktx2";

/// Flags of every encoder run: KTX2 output with a generated mip chain.
const FIXED_FLAGS: [&str; 3] = ["-ktx2", "-mipmap", "-output_file"];

/// UASTC effort level, spelled out instead of relying on a tool default.
const UASTC_LEVEL: &str = "2";

/// ETC1S quality level, likewise spelled out (the tool's own default).
const ETC1S_QUALITY: &str = "128";

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

/// Encoder binary to run, unless the policy keeps the bake in-process.
pub(super) fn encoder(settings: &BakeSettings) -> Option<PathBuf> {
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
pub(super) fn author(
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
