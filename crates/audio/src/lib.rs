#![warn(missing_docs)]
//! Ornis audio: clip decoding, an ECS-driven [`engine::AudioEngine`] and
//! platform backends (cpal on native, Web Audio on wasm).
//!
//! Pipeline: decoded [`source::AudioClip`]s are registered by id; per-entity
//! [`source::AudioSource`] components reference a clip and carry playback
//! state. Each frame [`engine::AudioEngine::step`] scans the world, derives
//! spatialization from the listener position and submits mix inputs to the
//! active [`backend::AudioBackendTrait`] implementation.
pub mod backend;
/// Symphonia-based decoders producing clips from files or in-memory bytes.
pub mod decoder;
/// The ECS-facing playback orchestrator ([`engine::AudioEngine`]).
pub mod engine;
/// Core data types: clips, sources, listener/spatial parameters.
pub mod source;

pub use backend::AudioBackend;
pub use decoder::{DecodeError, decode_bytes, decode_file};
pub use engine::AudioEngine;
pub use source::{
    AudioClip, AudioListener, AudioSource, AudioState, MixInput, SampleBuffer, SpatialParams,
};
