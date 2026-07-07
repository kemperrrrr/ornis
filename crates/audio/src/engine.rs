use std::collections::HashMap;

use glam::Vec3;
use ornis_core::{Entity, SmartStore};

use crate::backend::AudioBackend;
use crate::source::{AudioClip, AudioListener, AudioSource, AudioState, MixInput, SpatialParams};

pub struct AudioEngine {
    backend: AudioBackend,
    clips: Vec<AudioClip>,
    active: HashMap<Entity, usize>,
    _listener_pos: Vec3,
    _listener_gain: f32,
}

impl AudioEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            backend: AudioBackend::new()?,
            clips: Vec::new(),
            active: HashMap::new(),
            _listener_pos: Vec3::ZERO,
            _listener_gain: 1.0,
        })
    }

    pub fn register_clip(&mut self, clip: AudioClip) -> usize {
        let id = self.clips.len();
        self.clips.push(clip);
        id
    }

    pub fn backend(&self) -> &AudioBackend {
        &self.backend
    }

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
                                    (diff.x / distance).asin().clamp(-1.0, 1.0)
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

                    let input = MixInput::new(&self.clips[clip_id], source, spatial);
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
