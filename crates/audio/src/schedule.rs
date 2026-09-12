//! Per-frame audio driver for the unified [`Engine`](ornis_core::Engine) schedule.
//!
//! [`AudioEngine`](crate::engine::AudioEngine) already scans the ECS
//! [`AudioSource`](crate::source::AudioSource) lane; what was missing was the
//! schedule half — a system that steps it once per frame after motion, so
//! spatialization observes fresh positions. [`AudioPlugin`] installs both the
//! [`AudioHost`] resource and the `audio_step` system, mirroring
//! `GameplayPlugin`/`ScriptPlugin` conventions.
//!
//! Platform split: schedule systems and resources must be `Send + Sync`, but
//! the wasm backend is single-threaded (`Rc<RefCell>`). The host therefore
//! only accepts backends that statically prove `Send + Sync` (native, mocks);
//! wasm keeps driving the engine directly from its render loop.

use std::sync::Mutex;

use glam::Vec3;
use ornis_core::{Engine, Position, Resources, SmartStore, System, SystemAccess};

use crate::backend::AudioBackendTrait;
use crate::engine::AudioEngine;
use crate::source::{AudioClip, AudioSource};

/// Schedule-owned [`AudioEngine`].
///
/// Systems only receive `&Resources`, hence the mutex (same pattern as
/// `ScriptHost` and `Mutex<PhysicsRuntime>`).
///
/// # Safety
///
/// The type carries manual `Send`/`Sync` impls below. They are sound because
/// the only constructor takes `Box<dyn AudioBackendTrait + Send + Sync>` —
/// the concrete backend is thread-safe — and every access holds the mutex.
pub struct AudioHost {
    engine: Mutex<AudioEngine>,
}

// SAFETY: constructed solely from `Send + Sync` backends (see
// [`AudioHost::with_backend`]); all engine access is mutex-guarded.
unsafe impl Send for AudioHost {}
// SAFETY: same as above — exclusive access through the mutex.
unsafe impl Sync for AudioHost {}

impl AudioHost {
    /// Wrap an engine built around a thread-safe backend (native, test mocks).
    pub fn with_backend(backend: Box<dyn AudioBackendTrait + Send + Sync>) -> Self {
        Self {
            engine: Mutex::new(AudioEngine::with_backend(backend)),
        }
    }

    /// Store a clip and return its stable id for [`AudioSource::clip_id`].
    pub fn register_clip(&self, clip: AudioClip) -> usize {
        self.engine
            .lock()
            .expect("audio host lock")
            .register_clip(clip)
    }

    /// One frame of orchestration against `store`.
    pub fn step(&self, store: &SmartStore) {
        self.engine.lock().expect("audio host lock").step(store);
    }

    /// Move/listen the virtual listener (schedule-safe counterpart of
    /// [`AudioEngine::set_listener`]); used by the world↔audio bridge.
    pub fn set_listener(&self, pos: Vec3, gain: f32) {
        self.engine
            .lock()
            .expect("audio host lock")
            .set_listener(pos, gain);
    }

    /// Duck the master gain without moving the listener (schedule-safe
    /// counterpart of [`AudioEngine::set_gain`]).
    pub fn set_gain(&self, gain: f32) {
        self.engine.lock().expect("audio host lock").set_gain(gain);
    }
}

/// Steps the [`AudioHost`] once per frame. Declared after `transform_update`
/// (best-effort) so spatial parameters derive from this frame's positions.
struct AudioStepSystem;

impl System for AudioStepSystem {
    fn name(&self) -> &'static str {
        "audio_step"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .writes::<AudioHost>()
            .reads_lane::<AudioSource>()
            .reads_lane::<Position>()
            .reads_lane::<Vec3>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(host) = resources.get::<AudioHost>() else {
            return;
        };
        host.step(store);
    }
}

/// Installs audio stepping into an [`Engine`]'s variable schedule.
///
/// Install after gameplay systems so the `transform_update → audio_step`
/// ordering applies; without them the system still runs, unordered.
pub struct AudioPlugin {
    host: AudioHost,
}

impl AudioPlugin {
    /// Build around a thread-safe backend (native device, test mocks).
    pub fn with_backend(backend: Box<dyn AudioBackendTrait + Send + Sync>) -> Self {
        Self {
            host: AudioHost::with_backend(backend),
        }
    }

    /// Build around the platform's real audio output (native only).
    ///
    /// Returns `None` when the mixer thread cannot spawn; the caller then
    /// runs silently without audio.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_default() -> Option<Self> {
        crate::backend::AudioBackend::new()
            .ok()
            .map(|backend| Self::with_backend(Box::new(backend)))
    }

    /// Register the [`AudioSource`] lane, insert the host, and add `audio_step`.
    pub fn install(self, engine: &mut Engine) {
        if let Some(store) = engine.world_mut().store_mut() {
            store.register::<AudioSource>();
        }
        engine.world_mut().insert(self.host);
        engine.schedule_mut().add_system(AudioStepSystem);
        let _ = engine
            .schedule_mut()
            .try_order_before("transform_update", "audio_step");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{AudioState, MixInput};
    use std::sync::{Arc, Mutex as StdMutex};

    /// Thread-safe recording backend: proves the DAG path without hardware.
    struct MockBackend {
        plays: Arc<StdMutex<Vec<MixInput>>>,
    }

    impl MockBackend {
        fn shared() -> (Self, Arc<StdMutex<Vec<MixInput>>>) {
            let plays = Arc::new(StdMutex::new(Vec::new()));
            (
                Self {
                    plays: plays.clone(),
                },
                plays,
            )
        }
    }

    impl AudioBackendTrait for MockBackend {
        fn play(&mut self, input: MixInput) {
            self.plays.lock().expect("mock lock").push(input);
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // SAFETY: test-only; `plays` is mutex-guarded, no other shared state.
    unsafe impl Send for MockBackend {}
    // SAFETY: same as above.
    unsafe impl Sync for MockBackend {}

    fn sample_clip() -> AudioClip {
        AudioClip {
            sample_rate: 48_000,
            channels: 1,
            samples: std::sync::Arc::new(vec![0.0f32; 48]),
        }
    }

    #[test]
    fn schedule_step_plays_ecs_sources_once() {
        let (backend, plays) = MockBackend::shared();
        let plugin = AudioPlugin::with_backend(Box::new(backend));
        // Register the clip through the host before installing.
        let clip_id = plugin.host.register_clip(sample_clip());

        let mut engine = Engine::new();
        plugin.install(&mut engine);
        {
            let store = engine.world_mut().store_mut().expect("store");
            let e = store.create_entity();
            store.insert(
                e,
                AudioSource {
                    clip_id: Some(clip_id),
                    volume: 1.0,
                    pitch: 1.0,
                    looping: false,
                    spatial: false,
                    state: AudioState::Playing,
                },
            );
        }

        engine.run_frame(1.0 / 60.0);
        assert_eq!(plays.lock().expect("plays lock").len(), 1);
        // Second frame: source already active, no replay.
        engine.run_frame(1.0 / 60.0);
        assert_eq!(plays.lock().expect("plays lock").len(), 1);
    }
}
