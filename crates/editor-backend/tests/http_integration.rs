//! Integration test: boot the real `RemoteEditor` HTTP server and exercise
//! its endpoints over a live TCP socket. Covers the server thread, request
//! routing, and the tiny_http bindings that unit tests mock around.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crossbeam_channel::unbounded;
use editor_backend::{GameEvent, RemoteEditor, UiCommand};

/// Pick a free ephemeral port. We bind to port 0 (OS assigns one), read it
/// back, then drop the listener. There is a small race between releasing the
/// port and the server binding it, but for single-process tests this is
/// reliable enough.
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

    // Parse status code and body (after the blank line).
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

fn http_post(port: u16, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(req.as_bytes()).expect("write");
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
fn remote_editor_http_endpoints() {
    let port = free_port();
    let (cmd_tx, _cmd_rx) = unbounded::<UiCommand>();
    let (_ev_tx, ev_rx) = unbounded::<GameEvent>();

    let mut editor = RemoteEditor::start(port, cmd_tx, ev_rx);

    // Give the server thread a moment to bind and start accepting.
    std::thread::sleep(Duration::from_millis(300));

    // /api/status returns the empty-status placeholder.
    let (status, body) = http_get(port, "/api/status");
    assert_eq!(status, 200);
    assert!(body.contains("Ornis Engine"), "status body: {body}");

    // /api/scene returns the empty-scene placeholder.
    let (status, body) = http_get(port, "/api/scene");
    assert_eq!(status, 200);
    assert!(body.contains("entities"), "scene body: {body}");

    // /api/events initially empty array.
    let (status, body) = http_get(port, "/api/events");
    assert_eq!(status, 200);
    assert_eq!(body, "[]");

    // POST /api/command returns an explicit queue acknowledgement while the
    // game side remains asynchronous.
    let (code, body) = http_post(
        port,
        "/api/command",
        r#"{"type":"set_component","request_id":99,"data":{"id":1,"component":"T","value":{}}}"#,
    );
    assert_eq!(code, 200);
    let ack: serde_json::Value = serde_json::from_str(&body).expect("ack JSON");
    assert_eq!(ack["accepted"], true);
    assert_eq!(ack["request_id"], 99);

    let (code, body) = http_post(port, "/api/command", "not-json");
    assert_eq!(code, 200);
    let ack: serde_json::Value = serde_json::from_str(&body).expect("rejection JSON");
    assert_eq!(ack["accepted"], false);
    assert_eq!(ack["request_id"], 100);
    assert_eq!(ack["error"], "request body is not valid JSON");

    // Unknown path → 404.
    let (status, _) = http_get(port, "/no-such-thing");
    assert_eq!(status, 404);

    editor.stop();
}

#[test]
fn remote_editor_replays_events_after_cursor() {
    let port = free_port();
    let (cmd_tx, _cmd_rx) = unbounded::<UiCommand>();
    let (ev_tx, ev_rx) = unbounded::<GameEvent>();
    let mut editor = RemoteEditor::start(port, cmd_tx, ev_rx);
    std::thread::sleep(Duration::from_millis(200));

    ev_tx
        .send(GameEvent::CommandCompleted {
            request_id: 7,
            command: "create_entity".into(),
            success: true,
            error: None,
        })
        .expect("send event");
    std::thread::sleep(Duration::from_millis(200));

    let (status, body) = http_get(port, "/api/events?after=0");
    assert_eq!(status, 200);
    let events: serde_json::Value = serde_json::from_str(&body).expect("replay JSON");
    assert_eq!(events[0]["sequence"], 1);
    assert_eq!(events[0]["CommandCompleted"]["request_id"], 7);

    let (status, body) = http_get(port, "/api/events?after=1");
    assert_eq!(status, 200);
    assert_eq!(body, "[]");
    editor.stop();
}

#[test]
fn remote_editor_bind_failure_is_safe() {
    // Starting two editors on the same port: the second must not panic,
    // it returns a stopped (handle=None) instance.
    let port = free_port();
    let (cmd_tx, _cmd_rx) = unbounded::<UiCommand>();
    let (_ev_tx, ev_rx) = unbounded::<GameEvent>();

    let mut first = RemoteEditor::start(port, cmd_tx.clone(), ev_rx.clone());
    std::thread::sleep(Duration::from_millis(100));
    let mut second = RemoteEditor::start(port, cmd_tx, ev_rx);

    // Both should be droppable without panic.
    second.stop();
    first.stop();
}
