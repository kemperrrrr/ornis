use crate::source::MixInput;

/// Backend abstraction so `AudioEngine` can run headless in tests. The real
/// `AudioBackend` (cpal/native or wasm) implements this; tests inject a mock.
pub trait AudioBackendTrait: std::any::Any {
    fn play(&mut self, input: MixInput);
    /// Ergonomic downcast helper for tests inspecting a concrete backend.
    fn as_any(&self) -> &dyn std::any::Any;
}

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::AudioBackend;

#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::AudioBackend;

#[cfg(not(target_arch = "wasm32"))]
impl AudioBackendTrait for AudioBackend {
    fn play(&mut self, input: MixInput) {
        AudioBackend::play(self, input)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(target_arch = "wasm32")]
impl AudioBackendTrait for AudioBackend {
    fn play(&mut self, input: MixInput) {
        AudioBackend::play(self, input)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
