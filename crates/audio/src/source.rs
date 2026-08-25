use std::sync::Arc;

pub type SampleBuffer = Arc<Vec<f32>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioState {
    Playing,
    Paused,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioSource {
    pub clip_id: Option<usize>,
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub state: AudioState,
}

impl AudioSource {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioListener {
    pub gain: f32,
}

impl AudioListener {
    pub fn new() -> Self {
        Self { gain: 1.0 }
    }
}

impl Default for AudioListener {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct AudioClip {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: SampleBuffer,
}

impl AudioClip {
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialParams {
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub rolloff_factor: f32,
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

#[derive(Debug, Clone)]
pub struct MixInput {
    pub samples: SampleBuffer,
    pub volume: f32,
    pub looping: bool,
    pub spatial: Option<SpatialParams>,
    pub cursor: usize,
}

impl MixInput {
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
