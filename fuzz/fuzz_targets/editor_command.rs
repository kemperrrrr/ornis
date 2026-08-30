#![no_main]

//! Fuzzing of the HTTP `/api/command` payload parser — external input
//! (browser posts to the remote editor). Invariant: arbitrary bytes never
//! panic; malformed payloads return `None`, never a partial command.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = editor_backend::remote::parse_command_payload(&text);
});
