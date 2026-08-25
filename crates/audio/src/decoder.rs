use std::path::Path;
use std::sync::Arc;

use symphonia::core::audio::SampleBuffer as SymphSampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::source::AudioClip;

#[derive(Debug)]
pub enum DecodeError {
    Io(std::io::Error),
    Format(&'static str),
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

pub fn decode_file<P: AsRef<Path>>(path: P) -> Result<AudioClip, DecodeError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|_| DecodeError::Format("failed to probe format"))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or(DecodeError::Unsupported)?;

    let codec_params = &track.codec_params;
    let track_id = track.id;
    let sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(codec_params, &decoder_opts)
        .map_err(|_| DecodeError::Unsupported)?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(pkt) => pkt,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let frames_count = decoded.frames();
        let spec = *decoded.spec();
        let mut buf = SymphSampleBuffer::<f32>::new(frames_count as u64, spec);
        buf.copy_interleaved_ref(decoded);
        let frames = buf.samples().to_vec();
        let num_samples = frames_count * spec.channels.count();
        all_samples.extend_from_slice(&frames[..num_samples.min(frames.len())]);
    }

    Ok(AudioClip {
        sample_rate,
        channels,
        samples: Arc::new(all_samples),
    })
}

pub fn decode_bytes(data: &[u8], extension: &str) -> Result<AudioClip, DecodeError> {
    let owned = data.to_vec();
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(owned)), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension);

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|_| DecodeError::Format("failed to probe format"))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or(DecodeError::Unsupported)?;

    let codec_params = &track.codec_params;
    let track_id = track.id;
    let sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(codec_params, &decoder_opts)
        .map_err(|_| DecodeError::Unsupported)?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(pkt) => pkt,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let frames_count = decoded.frames();
        let spec = *decoded.spec();
        let mut buf = SymphSampleBuffer::<f32>::new(frames_count as u64, spec);
        buf.copy_interleaved_ref(decoded);
        let frames = buf.samples().to_vec();
        let num_samples = frames_count * spec.channels.count();
        all_samples.extend_from_slice(&frames[..num_samples.min(frames.len())]);
    }

    Ok(AudioClip {
        sample_rate,
        channels: channels as u16,
        samples: Arc::new(all_samples),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sample-3s.wav")
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
