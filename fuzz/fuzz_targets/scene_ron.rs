#![no_main]

//! Fuzzing of scene deserialization from RON — external input (scene
//! files from the editor and disk). The parser must not panic on any
//! input: garbage → Ok or Err, never panic/abort.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = ornis_render::scene::Scene::from_ron(s);
    }
});
