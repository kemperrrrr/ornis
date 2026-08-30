//! Audio data model: clips, per-entity sources and mix inputs.
//!
//! [`SampleBuffer`] shares interleaved f32 samples cheaply via `Arc`.
//! [`AudioSource`] is the per-entity component whose [`AudioState`]
//! transitions are observed by [`crate::engine::AudioEngine::step`];
//! [`MixInput`] and [`SpatialParams`] are the per-play snapshots handed to
//! the backend for mixing.

use std::sync::Arc;

/// Interleaved f32 samples normalized to [-1, 1], cheaply shared on clone.
pub type SampleBuffer = Arc<Vec<f32>>;

/// Playback state of a source, driven by game logic each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioState {
    /// Actively producing output; a source entering this state triggers playback.
    Playing,
    /// Silenced but resumable to `Playing` without restarting the clip.
    Paused,
    /// Not playing. Default state of a fresh source.
    #[default]
    Stopped,
}

/// Per-entity playback component referencing a clip registered in the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioSource {
    /// Index of the clip in the engine's registry; `None` plays nothing.
    pub clip_id: Option<usize>,
    /// Linear gain multiplier, clamped to [0, 1] at mix time.
    pub volume: f32,
    /// Playback rate multiplier (1.0 = normal). Reserved for future resampling;
    /// currently not applied by the mixers.
    pub pitch: f32,
    /// Restart from the beginning when the clip finishes.
    pub looping: bool,
    /// Compute distance/azimuth attenuation relative to the listener
    /// (requires a position component on the same entity).
    pub spatial: bool,
    /// Current playback state; transitions are observed by [`crate::engine::AudioEngine::step`].
    pub state: AudioState,
}

impl AudioSource {
    /// Stopped, silent source with unit volume/pitch and no clip attached.
    pub fn new() -> Self {
        Self {
            clip_id: None,
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: false,
            state: AudioState::Stopped,
        }
    }
}

impl Default for AudioSource {
    fn default() -> Self {
        Self::new()
    }
}

/// Global listener resource: master gain ducking every source equally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioListener {
    /// Master multiplier applied on top of each source's volume; negative
    /// values passed to the engine are clamped to 0.
    pub gain: f32,
}

impl AudioListener {
    /// Listener at full (1.0) master gain.
    pub fn new() -> Self {
        Self { gain: 1.0 }
    }
}

impl Default for AudioListener {
    fn default() -> Self {
        Self::new()
    }
}

/// Fully decoded, ready-to-mix sound: interleaved f32 samples plus format info.
#[derive(Debug, Clone)]
pub struct AudioClip {
    /// Sample rate in Hz (e.g. 44100).
    pub sample_rate: u32,
    /// Interleaved channel count (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Interleaved samples normalized to [-1, 1]; length is a multiple of `channels`.
    pub samples: SampleBuffer,
}

impl AudioClip {
    /// Length of the clip in seconds (`samples / (rate * channels)`); 0 for empty clips.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }
}

/// Positional-audio parameters for one playing sound, shared by native and
/// wasm backends so both hear the same falloff and panning.
///
/// Geometry convention: the listener faces +z; `azimuth` is a true geometric
/// angle in radians ([-pi/2, pi/2], -pi/2 = hard left), `elevation` rotates up.
#[derive(Debug, Clone, Copy)]
pub struct SpatialParams {
    /// Distance from listener to source in world units.
    pub distance: f32,
    /// Horizontal angle in radians; 0 = straight ahead, ±π/2 = hard side.
    pub azimuth: f32,
    /// Vertical angle in radians; 0 = level with the listener.
    pub elevation: f32,
    /// How fast gain decays beyond `reference_distance` (1.0 = linear falloff).
    pub rolloff_factor: f32,
    /// Distance within which no attenuation is applied (full gain).
    pub reference_distance: f32,
}

impl Default for SpatialParams {
    fn default() -> Self {
        Self {
            distance: 0.0,
            azimuth: 0.0,
            elevation: 0.0,
            rolloff_factor: 1.0,
            reference_distance: 1.0,
        }
    }
}

/// One sound handed to a backend for playback: clip samples flattened with
/// per-source settings already resolved by the engine.
#[derive(Debug, Clone)]
pub struct MixInput {
    /// Shared interleaved clip samples.
    pub samples: SampleBuffer,
    /// Final linear gain (source volume × listener master gain).
    pub volume: f32,
    /// Whether the backend should restart the clip at its end.
    pub looping: bool,
    /// `Some` when the source requested spatialization.
    pub spatial: Option<SpatialParams>,
    /// Backend-owned read cursor into `samples` (interleaved index).
    pub cursor: usize,
}

impl MixInput {
    /// Bind a clip to its source component: copies volume/looping, shares the
    /// sample buffer, starts playback from the beginning.
    pub fn new(clip: &AudioClip, source: &AudioSource, spatial: Option<SpatialParams>) -> Self {
        Self {
            samples: clip.samples.clone(),
            volume: source.volume,
            looping: source.looping,
            spatial,
            cursor: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_source_new_defaults() {
        let s = AudioSource::new();
        assert_eq!(s.clip_id, None);
        assert_eq!(s.volume, 1.0);
        assert_eq!(s.pitch, 1.0);
        assert!(!s.looping);
        assert!(!s.spatial);
        assert_eq!(s.state, AudioState::Stopped);
        // Default impl matches new().
        assert_eq!(AudioSource::default(), s);
    }

    #[test]
    fn audio_listener_defaults() {
        let l = AudioListener::new();
        assert_eq!(l.gain, 1.0);
        assert_eq!(AudioListener::default(), l);
    }

    #[test]
    fn spatial_params_defaults() {
        let sp = SpatialParams::default();
        assert_eq!(sp.distance, 0.0);
        assert_eq!(sp.azimuth, 0.0);
        assert_eq!(sp.elevation, 0.0);
        assert_eq!(sp.rolloff_factor, 1.0);
        assert_eq!(sp.reference_distance, 1.0);
    }

    #[test]
    fn audio_clip_duration_secs() {
        // 44100 Hz, 2 channels, 88200 interleaved samples = 1.0 s.
        let clip = AudioClip {
            sample_rate: 44100,
            channels: 2,
            samples: std::sync::Arc::new(vec![0.0f32; 88200]),
        };
        assert_eq!(clip.duration_secs(), 1.0);

        // Mono, 48000 Hz, 48000 samples = 1.0 s.
        let mono = AudioClip {
            sample_rate: 48000,
            channels: 1,
            samples: std::sync::Arc::new(vec![0.0f32; 48000]),
        };
        assert_eq!(mono.duration_secs(), 1.0);

        // Zero samples -> 0 duration, must not divide by zero.
        let empty = AudioClip {
            sample_rate: 44100,
            channels: 2,
            samples: std::sync::Arc::new(vec![]),
        };
        assert_eq!(empty.duration_secs(), 0.0);
    }

    #[test]
    fn mix_input_binds_clip_and_source() {
        let clip = AudioClip {
            sample_rate: 44100,
            channels: 1,
            samples: std::sync::Arc::new(vec![0.5f32; 100]),
        };
        let mut src = AudioSource::new();
        src.volume = 0.25;
        src.looping = true;

        let input = MixInput::new(&clip, &src, None);
        assert_eq!(input.samples.len(), 100);
        assert_eq!(input.volume, 0.25);
        assert!(input.looping);
        assert!(input.spatial.is_none());
        assert_eq!(input.cursor, 0);

        let sp = SpatialParams::default();
        let input2 = MixInput::new(&clip, &src, Some(sp));
        assert!(input2.spatial.is_some());
    }
}
