#![no_main]

//! Fuzzing of the MaterialX XML parser — external input (.mtlx assets).
//! Invariant: any input → Ok/Err, no panics and no infinite loops.

use libfuzzer_sys::fuzz_target;
use ornis_materialx::MaterialXParser;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let parser = MaterialXParser::new();
        let _ = parser.parse(s);
    }
});
