//! Contract test for the editor → WASM `/api/scene` boundary.
//!
//! Boots the real `RemoteEditor` HTTP server and pins the transport shape
//! the WASM client parses (`crates/wasm/src/scene_api.rs`): `version` and
//! `sequence` as `u64`, `entities`/`lights` arrays, plus the placeholder
//! fallback signal (`camera: null` — the WASM parser rejects it and keeps
//! the last good scene). This is the server half of the headless e2e; the
//! client half lives in `ornis-wasm` (`FULL_CONTRACT` golden).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crossbeam_channel::unbounded;
use editor_backend::{GameEvent, RemoteEditor, UiCommand};

/// Pick a free ephemeral port (same helper as `http_integration.rs`).
fn free_port() -> u16 {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

fn http_get(port: u16, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .expect("write");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("timeout");
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw).to_string();
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let body = match text.find("\r\n\r\n") {
        Some(idx) => text[idx + 4..].to_string(),
        None => String::new(),
    };
    (status, body)
}

#[test]
fn scene_endpoint_serves_wasm_contract_shape() {
    let port = free_port();
    let (cmd_tx, _cmd_rx) = unbounded::<UiCommand>();
    let (_ev_tx, ev_rx) = unbounded::<GameEvent>();
    let mut editor = RemoteEditor::start(port, cmd_tx, ev_rx);
    std::thread::sleep(Duration::from_millis(300));

    let (status, body) = http_get(port, "/api/scene");
    assert_eq!(status, 200);
    let scene: serde_json::Value = serde_json::from_str(&body).expect("scene JSON");

    // Transport metadata the WASM client relies on: authoritative
    // `version` plus the independent transport `sequence`.
    assert!(scene["version"].is_u64(), "version u64: {body}");
    assert!(scene["sequence"].is_u64(), "sequence u64: {body}");
    assert!(scene["entities"].is_array(), "entities array: {body}");
    assert!(scene["lights"].is_array(), "lights array: {body}");

    // Placeholder (no live world yet): `camera: null` is the documented
    // fallback signal — the WASM `parse_scene_json` rejects it and keeps
    // the last good scene instead of rendering garbage.
    assert!(
        scene["camera"].is_null(),
        "placeholder camera must be null: {body}"
    );

    // `/api/status` carries the same transport metadata.
    let (status, body) = http_get(port, "/api/status");
    assert_eq!(status, 200);
    let status_json: serde_json::Value = serde_json::from_str(&body).expect("status JSON");
    assert!(status_json["version"].is_u64(), "status version: {body}");
    assert!(status_json["sequence"].is_u64(), "status sequence: {body}");

    editor.stop();
}
