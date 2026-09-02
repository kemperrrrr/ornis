//! Symphonia-based decoding of audio files and in-memory bytes into clips.
//!
//! [`decode_file`] / [`decode_bytes`] probe the container (the path's
//! extension feeds the format hint), pick a codec decoder and produce
//! interleaved f32 samples normalized to [-1, 1]. Undecodable packets are
//! skipped rather than failing the whole decode; unknown sample rate or
//! channel layout falls back to 44.1 kHz stereo.
//!
use std::path::Path;
use std::sync::Arc;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::source::AudioClip;

/// Failure modes of [`decode_file`] / [`decode_bytes`].
#[derive(Debug)]
pub enum DecodeError {
    /// The file could not be opened or read.
    Io(std::io::Error),
    /// Symphonia rejected the container/metadata (e.g. not audio at all).
    Format(&'static str),
    /// No decodable track, or no Symphonia decoder registered for the codec.
    Unsupported,
}

impl From<std::io::Error> for DecodeError {
    fn from(e: std::io::Error) -> Self {
        DecodeError::Io(e)
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Io(e) => write!(f, "IO error: {}", e),
            DecodeError::Format(msg) => write!(f, "Format error: {}", msg),
            DecodeError::Unsupported => write!(f, "Unsupported audio format"),
        }
    }
}

impl std::error::Error for DecodeError {}

fn decode_inner(
    mss: MediaSourceStream,
    hint: Hint,
    format_opts: FormatOptions,
    metadata_opts: MetadataOptions,
    decoder_opts: AudioDecoderOptions,
) -> Result<AudioClip, DecodeError> {
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, format_opts, metadata_opts)
        .map_err(|_| DecodeError::Format("failed to probe format"))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or(DecodeError::Unsupported)?;

    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .ok_or(DecodeError::Unsupported)?;
    let audio_params = codec_params.audio().ok_or(DecodeError::Unsupported)?;
    let sample_rate = audio_params.sample_rate.unwrap_or(44100);
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count() as u16)
        .unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &decoder_opts)
        .map_err(|_| DecodeError::Unsupported)?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut tmp: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(pkt)) => pkt,
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(_) => break,
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let interleaved = decoded.samples_interleaved();
        if tmp.len() < interleaved {
            tmp.resize(interleaved, 0.0);
        }
        decoded.copy_to_slice_interleaved(&mut tmp[..interleaved]);
        all_samples.extend_from_slice(&tmp[..interleaved]);
    }

    Ok(AudioClip {
        sample_rate,
        channels,
        samples: Arc::new(all_samples),
    })
}

/// Decode any Symphonia-supported audio file into an interleaved f32 clip.
///
/// The path's extension feeds the format probe's hint. Samples come out
/// interleaved and normalized to [-1, 1]; unknown sample rate/channel count
/// falls back to 44100 Hz / stereo. Undecodable packets are skipped rather
/// than failing the whole decode.
///
/// # Errors
/// [`DecodeError::Io`] when the path can't be read, [`DecodeError::Format`]
/// when probing fails, [`DecodeError::Unsupported`] without a usable track.
pub fn decode_file<P: AsRef<Path>>(path: P) -> Result<AudioClip, DecodeError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    decode_inner(
        mss,
        hint,
        FormatOptions::default(),
        MetadataOptions::default(),
        AudioDecoderOptions::default(),
    )
}

/// In-memory counterpart of [`decode_file`] for embedded/fetched audio.
///
/// `extension` (e.g. `"wav"`) drives the format hint; same error contract
/// and sample normalization as [`decode_file`].
pub fn decode_bytes(data: &[u8], extension: &str) -> Result<AudioClip, DecodeError> {
    let owned = data.to_vec();
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(owned)), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension);

    decode_inner(
        mss,
        hint,
        FormatOptions::default(),
        MetadataOptions::default(),
        AudioDecoderOptions::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample-3s.wav")
    }

    #[test]
    fn decode_file_reads_real_wav() {
        let clip = decode_file(fixture_path()).expect("decode sample-3s.wav");
        // Header: PCM 16-bit, 2 channels, 44100 Hz.
        assert_eq!(clip.sample_rate, 44100);
        assert_eq!(clip.channels, 2);
        // 563712 bytes of 16-bit stereo PCM = 281856 interleaved f32 samples
        // (563712 / (2 channels * 2 bytes-per-sample) * 2 channels).
        assert_eq!(clip.samples.len(), 281856);
        // Duration ~3.19 s (281856 / 2 channels / 44100).
        let dur = clip.duration_secs();
        assert!(dur > 3.0 && dur < 3.5, "duration was {dur}");
        // Samples are normalized f32 in [-1, 1].
        let max_abs = clip.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0, "sample out of range: {max_abs}");
    }

    #[test]
    fn decode_bytes_matches_file() {
        let bytes = std::fs::read(fixture_path()).expect("read fixture");
        let clip = decode_bytes(&bytes, "wav").expect("decode bytes");
        assert_eq!(clip.sample_rate, 44100);
        assert_eq!(clip.channels, 2);
        assert_eq!(clip.samples.len(), 281856);
    }

    #[test]
    fn decode_bytes_rejects_garbage() {
        // Not a valid audio container at all.
        let junk = b"this is definitely not audio data, just some text";
        let err = decode_bytes(junk, "wav");
        assert!(err.is_err(), "garbage should fail to decode");
        match err {
            Err(DecodeError::Format(_)) | Err(DecodeError::Unsupported) => {}
            other => panic!("expected Format or Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn decode_file_missing_path_errors() {
        let missing = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/does-not-exist.wav");
        let err = decode_file(missing);
        assert!(matches!(err, Err(DecodeError::Io(_))));
    }
}
