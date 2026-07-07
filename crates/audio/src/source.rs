use std::sync::Arc;

pub type SampleBuffer = Arc<Vec<f32>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioState {
    Playing,
    Paused,
    Stopped,
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
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
