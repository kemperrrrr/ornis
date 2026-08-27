use std::collections::HashMap;

use glam::Vec3;
use ornis_core::{Entity, SmartStore};

use crate::backend::{AudioBackend, AudioBackendTrait};
use crate::source::{AudioClip, AudioSource, AudioState, MixInput, SpatialParams};

/// Bridges the ECS world and the audio backend.
///
/// Each [`step`](AudioEngine::step) scans every entity carrying an
/// [`AudioSource`]: entities newly marked [`AudioState::Playing`] get their
/// mix input (with spatial parameters derived relative to the listener)
/// submitted to the backend once, then tracked as active; paused/stopped or
/// despawned entities are untracked. Clip storage is owned here and
/// referenced by registration index.
pub struct AudioEngine {
    backend: Box<dyn AudioBackendTrait>,
    clips: Vec<AudioClip>,
    active: HashMap<Entity, usize>,
    _listener_pos: Vec3,
    _listener_gain: f32,
}

impl AudioEngine {
    /// Create an engine on the platform's real audio output.
    ///
    /// # Errors
    /// Propagates backend construction failure (no mixer thread).
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            backend: Box::new(AudioBackend::new()?),
            clips: Vec::new(),
            active: HashMap::new(),
            _listener_pos: Vec3::ZERO,
            _listener_gain: 1.0,
        })
    }

    /// Build an engine around an arbitrary backend — used by headless tests.
    pub fn with_backend(backend: Box<dyn AudioBackendTrait>) -> Self {
        Self {
            backend,
            clips: Vec::new(),
            active: HashMap::new(),
            _listener_pos: Vec3::ZERO,
            _listener_gain: 1.0,
        }
    }

    /// Store a clip and return its stable registration id for
    /// [`AudioSource::clip_id`]. Ids are never reused.
    pub fn register_clip(&mut self, clip: AudioClip) -> usize {
        let id = self.clips.len();
        self.clips.push(clip);
        id
    }

    /// Move/listen the virtual listener. Spatial attenuation and panning are
    /// computed relative to this position; `gain` is a master multiplier
    /// applied to every source's volume (e.g. a global mute/duck).
    pub fn set_listener(&mut self, pos: Vec3, gain: f32) {
        self._listener_pos = pos;
        self._listener_gain = gain.max(0.0);
    }

    /// Access the underlying backend (typically downcast via
    /// [`AudioBackendTrait::as_any`] in tests).
    pub fn backend(&self) -> &dyn AudioBackendTrait {
        &*self.backend
    }

    /// One frame of orchestration: submit newly-playing sources, drop
    /// stopped/paused/despawned ones. Cheap no-op when no [`AudioSource`]
    /// lane exists in `store`.
    pub fn step(&mut self, store: &SmartStore) {
        self.update_sources(store);
    }

    fn update_sources(&mut self, store: &SmartStore) {
        let sources = match store.read_lane::<AudioSource>() {
            Some(lane) => lane,
            None => return,
        };

        let positions = store.read_lane::<Vec3>();
        let mut seen: std::collections::HashSet<Entity> = std::collections::HashSet::new();

        for i in 0..sources.data.len() {
            let entity = sources.entities[i];
            let source = &sources.data[i];
            seen.insert(entity);

            match source.state {
                AudioState::Playing => {
                    let clip_id = match source.clip_id {
                        Some(id) if id < self.clips.len() => id,
                        _ => continue,
                    };

                    if self.active.contains_key(&entity) {
                        continue;
                    }

                    let spatial = if source.spatial {
                        positions.as_ref().and_then(|pos_lane| {
                            pos_lane.get(entity).map(|pos| {
                                let diff = *pos - self._listener_pos;
                                let distance = diff.length();
                                let azimuth = if distance > 0.001 {
                                    (diff.x / distance).asin()
                                } else {
                                    0.0
                                };
                                SpatialParams {
                                    distance,
                                    azimuth,
                                    elevation: 0.0,
                                    rolloff_factor: 1.0,
                                    reference_distance: 1.0,
                                }
                            })
                        })
                    } else {
                        None
                    };

                    let mut input = MixInput::new(&self.clips[clip_id], source, spatial);
                    // Master listener gain ducks every source equally.
                    input.volume *= self._listener_gain;
                    self.backend.play(input);
                    self.active.insert(entity, clip_id);
                }
                AudioState::Stopped | AudioState::Paused => {
                    self.active.remove(&entity);
                }
            }
        }

        self.active.retain(|entity, _| seen.contains(entity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    /// Records every `play` call so the engine's orchestration can be
    /// asserted without a real audio device.
    struct MockBackend {
        plays: Vec<MixInput>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self { plays: Vec::new() }
        }
    }

    impl AudioBackendTrait for MockBackend {
        fn play(&mut self, input: MixInput) {
            self.plays.push(input);
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn sample_clip() -> AudioClip {
        // One second of 440 Hz mono at 48 kHz.
        let len = 48_000;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push((i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48_000.0).sin());
        }
        AudioClip {
            sample_rate: 48_000,
            channels: 1,
            samples: std::sync::Arc::new(data),
        }
    }

    fn playing_source(clip_id: usize, spatial: bool) -> AudioSource {
        AudioSource {
            clip_id: Some(clip_id),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial,
            state: AudioState::Playing,
        }
    }

    fn stopped_source(clip_id: usize) -> AudioSource {
        AudioSource {
            clip_id: Some(clip_id),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: false,
            state: AudioState::Stopped,
        }
    }

    #[test]
    fn step_plays_sources_marked_playing() {
        let clip = sample_clip();
        let mut engine = AudioEngine::with_backend(Box::new(MockBackend::new()));
        let clip_id = engine.register_clip(clip);

        let mut store = SmartStore::new();
        store.register::<AudioSource>();
        store.register::<Vec3>();
        let e = store.create_entity();
        store.insert(e, playing_source(clip_id, false));

        engine.step(&store);

        // One play call for the single playing source.
        let mock = engine.backend().as_any();
        let mock = mock.downcast_ref::<MockBackend>().expect("backend is mock");
        assert_eq!(mock.plays.len(), 1);
        // Non-spatial source -> no spatial params in the mix input.
        assert!(mock.plays[0].spatial.is_none());
    }

    #[test]
    fn step_computes_spatial_params_from_listener() {
        let clip = sample_clip();
        let mut engine = AudioEngine::with_backend(Box::new(MockBackend::new()));
        let clip_id = engine.register_clip(clip);

        // Listener is 3 units to the -X of the source at (2, 0, 0), so the
        // source sits 5 units to the +X of the listener.
        engine.set_listener(Vec3::new(-3.0, 0.0, 0.0), 1.0);

        let mut store = SmartStore::new();
        store.register::<AudioSource>();
        store.register::<Vec3>();
        let e = store.create_entity();
        store.insert(e, Vec3::new(2.0, 0.0, 0.0));
        store.insert(e, playing_source(clip_id, true));

        engine.step(&store);

        let mock = engine.backend().as_any();
        let mock = mock.downcast_ref::<MockBackend>().expect("backend is mock");
        assert_eq!(mock.plays.len(), 1);
        let sp = mock.plays[0].spatial.as_ref().expect("spatial source");
        assert!(
            (sp.distance - 5.0).abs() < 1e-3,
            "distance = {}",
            sp.distance
        );
        assert!((sp.azimuth - std::f32::consts::FRAC_PI_2).abs() < 1e-3);
    }

    #[test]
    fn listener_gain_ducks_volume() {
        let clip = sample_clip();
        let mut engine = AudioEngine::with_backend(Box::new(MockBackend::new()));
        let clip_id = engine.register_clip(clip);
        engine.set_listener(Vec3::ZERO, 0.5);

        let mut store = SmartStore::new();
        store.register::<AudioSource>();
        store.register::<Vec3>();
        let e = store.create_entity();
        store.insert(e, playing_source(clip_id, false));

        engine.step(&store);

        let mock = engine.backend().as_any();
        let mock = mock.downcast_ref::<MockBackend>().expect("backend is mock");
        assert_eq!(mock.plays.len(), 1);
        // Source volume is 1.0, listener gain 0.5 -> mixed volume 0.5.
        assert!(
            (mock.plays[0].volume - 0.5).abs() < 1e-6,
            "volume = {}",
            mock.plays[0].volume
        );
    }

    #[test]
    fn step_skips_already_active_and_stopped() {
        let clip = sample_clip();
        let mut engine = AudioEngine::with_backend(Box::new(MockBackend::new()));
        let clip_id = engine.register_clip(clip);

        let mut store = SmartStore::new();
        store.register::<AudioSource>();
        store.register::<Vec3>();
        let playing = store.create_entity();
        store.insert(playing, playing_source(clip_id, false));
        let stopped = store.create_entity();
        store.insert(stopped, stopped_source(clip_id));

        engine.step(&store);
        // Only the playing source produced a play call.
        {
            let mock = engine.backend().as_any();
            let mock = mock.downcast_ref::<MockBackend>().expect("backend is mock");
            assert_eq!(mock.plays.len(), 1);
        }

        // Second step: playing source is now active, so no new play.
        engine.step(&store);
        let mock = engine.backend().as_any();
        let mock = mock.downcast_ref::<MockBackend>().expect("backend is mock");
        assert_eq!(mock.plays.len(), 1);
    }
}
