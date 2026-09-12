//! World↔audio bridge for the unified [`Engine`](ornis_core::Engine) schedule.
//!
//! Mirrors [`install_gameplay_physics_bridge`](ornis_app) (if that helper is
//! unavailable, see `ornis-app/src/lib.rs`): gameplay writes authoritative
//! [`Position`](ornis_core::Position), the bridge propagates it into the
//! audio domain, and the [`AudioPlugin`](crate::schedule::AudioPlugin)
//! stepping system consumes the result in the same frame DAG:
//!
//! ```text
//! frame:  transform_update / body_to_transform → audio_listener_sync → audio_step
//! ```
//!
//! Two halves, both frame-rate (audio has no fixed-step solver state):
//!
//! * source positions — [`AudioEngine`](crate::engine::AudioEngine) reads the
//!   gameplay [`Position`](ornis_core::Position) lane first, raw `Vec3` only
//!   as back-compat, so no copy system is needed;
//! * listener pose/gain — [`AudioListenerSyncSystem`] (below) pushes the
//!   world listener into the [`AudioHost`](crate::schedule::AudioHost) once
//!   per frame, after motion and before `audio_step`.

use glam::Vec3;
use ornis_core::{Engine, Position, Resources, SmartStore, System, SystemAccess};

use crate::schedule::AudioHost;
use crate::source::{AudioListener, AudioSource};

/// Pushes the world listener into the [`AudioHost`] once per frame.
///
/// Listener entity = first entity carrying an [`AudioListener`] component;
/// its pose is read from gameplay [`Position`](ornis_core::Position) (raw
/// `Vec3` as back-compat), its gain from the component. Without a marked
/// entity the [`AudioListener`] *resource* supplies the master gain and the
/// pose stays wherever [`AudioEngine::set_listener`](crate::engine::AudioEngine::set_listener)
/// (or a previous sync) left it. Without either, or without a host, this is
/// a no-op — manually driven listeners keep working.
struct AudioListenerSyncSystem;

impl System for AudioListenerSyncSystem {
    fn name(&self) -> &'static str {
        "audio_listener_sync"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .writes::<AudioHost>()
            .reads::<AudioListener>()
            .reads_lane::<AudioListener>()
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
        let listener_lane = store.read_lane::<AudioListener>();
        let pos_lane = store.read_lane::<Position>();
        let vec_lane = store.read_lane::<Vec3>();

        if let Some(lane) = listener_lane
            .as_ref()
            .and_then(|l| l.entities.first().and_then(|&e| l.get(e).map(|c| (e, *c))))
        {
            let (entity, marker) = lane;
            let pos = pos_lane
                .as_ref()
                .and_then(|l| l.get(entity))
                .map(|p| p.0)
                .or_else(|| vec_lane.as_ref().and_then(|l| l.get(entity)).copied())
                .unwrap_or(Vec3::ZERO);
            host.set_listener(pos, marker.gain);
        } else if let Some(global) = resources.get::<AudioListener>() {
            // Gain-only path: no marked entity, so only the master gain can
            // be forwarded statelessly; the pose stays where `set_listener`
            // (or a previous sync) left it.
            host.set_gain(global.gain);
        }
    }
}

/// Installs the world↔audio bridge into `engine`'s variable schedule.
///
/// Adds `audio_listener_sync` (world listener pose/gain → audio engine)
/// ordered after motion (`transform_update`, `body_to_transform`,
/// best-effort) and before `audio_step` (best-effort), so spatialization
/// observes this frame's positions. Source positions need no copy system:
/// the engine reads [`Position`](ornis_core::Position) directly.
///
/// Idempotent: a second call leaves a single `audio_listener_sync` system.
/// Safe without [`AudioPlugin`](crate::schedule::AudioPlugin): the sync is
/// a no-op until a host is installed.
pub fn install_gameplay_audio_bridge(engine: &mut Engine) {
    if engine.schedule().mermaid().contains("audio_listener_sync") {
        return;
    }
    engine
        .schedule_mut()
        .prepend_system(AudioListenerSyncSystem);
    let _ = engine
        .schedule_mut()
        .try_order_before("transform_update", "audio_listener_sync");
    let _ = engine
        .schedule_mut()
        .try_order_before("body_to_transform", "audio_listener_sync");
    let _ = engine
        .schedule_mut()
        .try_order_before("audio_listener_sync", "audio_step");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AudioBackendTrait;
    use crate::source::{AudioClip, AudioState, MixInput};
    use ornis_core::Entity;
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

    fn engine_with_audio() -> (Engine, Arc<StdMutex<Vec<MixInput>>>) {
        let (backend, plays) = MockBackend::shared();
        let plugin = crate::schedule::AudioPlugin::with_backend(Box::new(backend));
        let mut engine = Engine::new();
        plugin.install(&mut engine);
        install_gameplay_audio_bridge(&mut engine);
        (engine, plays)
    }

    #[test]
    fn source_position_reaches_spatial_params_through_run_frame() {
        let (mut engine, plays) = engine_with_audio();
        let clip_id = engine
            .world()
            .resources()
            .get::<AudioHost>()
            .expect("audio host")
            .register_clip(sample_clip());
        let store = engine.world_mut().store_mut().expect("store");
        let e: Entity = store.create_entity();
        store.insert(e, Position(Vec3::new(2.0, 0.0, 0.0)));
        store.insert(e, playing_source(clip_id, true));

        engine.run_frame(1.0 / 60.0);

        let plays = plays.lock().expect("plays lock");
        assert_eq!(plays.len(), 1);
        let sp = plays[0].spatial.as_ref().expect("spatial source");
        assert!(
            (sp.distance - 2.0).abs() < 1e-3,
            "distance = {}",
            sp.distance
        );
    }

    #[test]
    fn listener_entity_drives_spatial_relative() {
        let (mut engine, plays) = engine_with_audio();
        let clip_id = engine
            .world()
            .resources()
            .get::<AudioHost>()
            .expect("audio host")
            .register_clip(sample_clip());
        {
            let store = engine.world_mut().store_mut().expect("store");
            let listener: Entity = store.create_entity();
            store.insert(listener, AudioListener { gain: 1.0 });
            store.insert(listener, Position(Vec3::new(-3.0, 0.0, 0.0)));
            let src: Entity = store.create_entity();
            store.insert(src, Position(Vec3::new(2.0, 0.0, 0.0)));
            store.insert(src, playing_source(clip_id, true));
        }

        engine.run_frame(1.0 / 60.0);

        let plays = plays.lock().expect("plays lock");
        assert_eq!(plays.len(), 1);
        let sp = plays[0].spatial.as_ref().expect("spatial source");
        assert!(
            (sp.distance - 5.0).abs() < 1e-3,
            "distance = {}",
            sp.distance
        );
        assert!((sp.azimuth - std::f32::consts::FRAC_PI_2).abs() < 1e-3);
    }

    #[test]
    fn listener_component_gain_ducks_volume() {
        let (mut engine, plays) = engine_with_audio();
        let clip_id = engine
            .world()
            .resources()
            .get::<AudioHost>()
            .expect("audio host")
            .register_clip(sample_clip());
        {
            let store = engine.world_mut().store_mut().expect("store");
            let listener: Entity = store.create_entity();
            store.insert(listener, AudioListener { gain: 0.25 });
            store.insert(listener, Position(Vec3::ZERO));
            let src: Entity = store.create_entity();
            store.insert(src, Position(Vec3::ZERO));
            store.insert(src, playing_source(clip_id, false));
        }

        engine.run_frame(1.0 / 60.0);

        let plays = plays.lock().expect("plays lock");
        assert_eq!(plays.len(), 1);
        assert!(
            (plays[0].volume - 0.25).abs() < 1e-6,
            "volume = {}",
            plays[0].volume
        );
    }

    #[test]
    fn bridge_without_host_is_noop() {
        let mut engine = Engine::new();
        install_gameplay_audio_bridge(&mut engine);
        let store = engine.world_mut().store_mut().expect("store");
        let e: Entity = store.create_entity();
        store.insert(e, Position(Vec3::new(1.0, 0.0, 0.0)));
        // No AudioHost installed: sync and (absent) step must not panic.
        engine.run_frame(1.0 / 60.0);
    }

    #[test]
    fn install_is_idempotent() {
        let (mut engine, _) = engine_with_audio();
        install_gameplay_audio_bridge(&mut engine);
        install_gameplay_audio_bridge(&mut engine);
        let mermaid = engine.schedule().mermaid();
        assert_eq!(mermaid.matches("audio_listener_sync").count(), 1);
    }
}
