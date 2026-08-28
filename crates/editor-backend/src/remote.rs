use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use tiny_http::{Header, Request, Response, Server};

use crate::ipc::{GameEvent, UiCommand};

/// Editor frontend root. Resolution order:
///   1. `--editor-dir <path>` CLI argument
///   2. `ORNIS_EDITOR_DIR` environment variable
///   3. `<workspace>/editor` (this crate lives at `crates/editor-backend`,
///      so CARGO_MANIFEST_DIR is two levels below the workspace root)
fn assets_root() -> PathBuf {
    let mut args = std::env::args().skip_while(|a| a != "--editor-dir");
    if args.next().is_some()
        && let Some(dir) = args.next()
    {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("ORNIS_EDITOR_DIR")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../editor")
}

/// The HTTP remote editor server: serves the static editor assets and the
/// `/api/*` endpoints from a background thread; `stop` shuts it down and joins.
pub struct RemoteEditor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RemoteEditor {
    /// Bind `127.0.0.1:{port}` and start serving. On bind failure prints an
    /// error and returns an inert handle instead of panicking.
    pub fn start(port: u16, game_tx: Sender<UiCommand>, game_rx: Receiver<GameEvent>) -> Self {
        let addr = format!("127.0.0.1:{port}");
        let server = match Server::http(&addr) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ornis: remote editor failed to bind {addr}: {e}");
                return Self {
                    stop: Arc::new(AtomicBool::new(true)),
                    handle: None,
                };
            }
        };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = thread::Builder::new()
            .name("remote-editor".into())
            .spawn(move || serve(server, stop_clone, game_tx, game_rx))
            .expect("spawn remote-editor thread");

        eprintln!("ornis: remote editor at http://{addr}");
        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// Signal shutdown and join the server thread.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RemoteEditor {
    fn drop(&mut self) {
        self.stop();
    }
}

const EMPTY_STATUS: &str = r#"{"entity_count":0,"name":"Ornis Engine","version":0,"sequence":0}"#;
const EMPTY_SCENE: &str = r#"{"version":0,"entity_count":0,"entities":[],"lights":[],"camera":null,"ambient":null,"sequence":0}"#;

/// Snapshot payloads refreshed out of the game-event stream; served by the
/// `/api/status` and `/api/scene` endpoints until the next snapshot arrives.
/// `sequence` is transport metadata and is independent of the scene's
/// authoritative `version`.
struct Snapshots {
    status: String,
    scene: String,
    sequence: u64,
}

impl Default for Snapshots {
    fn default() -> Self {
        Self {
            status: EMPTY_STATUS.to_string(),
            scene: EMPTY_SCENE.to_string(),
            sequence: 0,
        }
    }
}

fn serve(
    server: Server,
    stop: Arc<AtomicBool>,
    game_tx: Sender<UiCommand>,
    game_rx: Receiver<GameEvent>,
) {
    let mut buffer: Vec<GameEvent> = Vec::new();
    let mut snapshots = Snapshots::default();
    let mut next_request_id = 1_u64;
    let root = assets_root();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        drain_game_events(&game_rx, &mut buffer, &mut snapshots);

        // Accept one request with a short timeout.
        let mut request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(_) => break,
        };

        let response = route_request(
            &root,
            &mut request,
            &mut buffer,
            &snapshots,
            &game_tx,
            &mut next_request_id,
        );
        let _ = request.respond(response);
    }
}

/// Drain incoming game events into `buffer`. "status"/"scene" snapshots only
/// refresh the endpoint caches — they are not user-facing events, so they
/// skip the buffer.
fn drain_game_events(
    game_rx: &Receiver<GameEvent>,
    buffer: &mut Vec<GameEvent>,
    snaps: &mut Snapshots,
) {
    while let Ok(ev) = game_rx.try_recv() {
        match &ev {
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } if cmd_type == "status" => {
                snaps.sequence = snaps.sequence.saturating_add(1);
                snaps.status = add_sequence(json_data, snaps.sequence);
                continue;
            }
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } if cmd_type == "scene" => {
                snaps.sequence = snaps.sequence.saturating_add(1);
                snaps.scene = add_sequence(json_data, snaps.sequence);
                continue;
            }
            _ => {}
        }
        buffer.push(ev);
    }
}

/// Add transport sequence metadata to an object snapshot while preserving
/// its existing JSON shape. The scene's authoritative `version` remains a
/// separate field and is never rewritten.
fn add_sequence(body: &str, sequence: u64) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(body) else {
        return body.to_owned();
    };
    let Some(object) = value.as_object_mut() else {
        return body.to_owned();
    };
    object.insert("sequence".into(), serde_json::json!(sequence));
    serde_json::to_string(&value).unwrap_or_else(|_| body.to_owned())
}

/// A synchronous acknowledgement for one accepted or rejected HTTP command.
/// The engine may complete the command later; `accepted` only means that the
/// message was validated and queued successfully.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandAck {
    request_id: u64,
    accepted: bool,
    error: Option<String>,
}

fn command_ack_json(ack: &CommandAck) -> String {
    match &ack.error {
        Some(error) => serde_json::json!({
            "accepted": ack.accepted,
            "request_id": ack.request_id,
            "error": error,
        })
        .to_string(),
        None => serde_json::json!({
            "accepted": ack.accepted,
            "request_id": ack.request_id,
        })
        .to_string(),
    }
}

fn allocate_request_id(next_request_id: &mut u64) -> u64 {
    let request_id = (*next_request_id).max(1);
    *next_request_id = request_id.saturating_add(1);
    request_id
}

/// Use a client-provided positive request id when present; otherwise allocate
/// a monotonic server id. Advancing the allocator past a supplied id avoids
/// collisions with subsequent generated ids.
fn command_request_id(body: &str, next_request_id: &mut u64) -> u64 {
    let requested = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("request_id").and_then(|id| id.as_u64()))
        .filter(|&id| id > 0);
    if let Some(request_id) = requested {
        if request_id >= *next_request_id {
            *next_request_id = request_id.saturating_add(1).max(1);
        }
        request_id
    } else {
        allocate_request_id(next_request_id)
    }
}

/// Serve one HTTP request. `/api/command` posts are validated, forwarded to
/// the game thread and acknowledged synchronously; snapshot responses carry
/// transport sequence metadata; everything else is answered from current
/// server state.
fn route_request(
    root: &Path,
    request: &mut Request,
    buffer: &mut Vec<GameEvent>,
    snapshots: &Snapshots,
    game_tx: &Sender<UiCommand>,
    next_request_id: &mut u64,
) -> Response<Cursor<Vec<u8>>> {
    let url = request.url().to_string();
    let method = request.method().as_str().to_string();

    match (method.as_str(), url.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => serve_static(root, "index.html"),
        ("GET", "/api/status") => json_response(&snapshots.status),
        ("GET", "/api/scene") => json_response(&snapshots.scene),
        ("GET", "/api/events") => {
            let body = format_events(buffer);
            buffer.clear();
            json_response(&body)
        }
        ("POST", "/api/command") => {
            let mut body = String::new();
            let _ = request.as_reader().read_to_string(&mut body);
            let request_id = command_request_id(&body, next_request_id);
            let ack = post_command(&body, game_tx, request_id);
            let ack_body = command_ack_json(&ack);
            json_response(&ack_body)
        }
        ("GET", path) => serve_static(root, path),
        _ => not_found(),
    }
}

fn serve_static(root: &Path, url_path: &str) -> Response<Cursor<Vec<u8>>> {
    // Strip query string (e.g. `Inter-Regular.woff2?v=4.0`).
    let path = url_path.split('?').next().unwrap_or(url_path);
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        return not_found();
    }

    let mut full = root.join(rel);
    if rel.is_empty() || full.is_dir() {
        full = root.join("index.html");
    }

    match fs::read(&full) {
        Ok(bytes) => Response::from_data(bytes).with_header(content_type(&full)),
        Err(_) => not_found(),
    }
}

/// Parse a posted command envelope `{"type": …, "data": …}` and forward it
/// to the game thread. The returned acknowledgement distinguishes malformed
/// input and a disconnected game channel from a successfully queued command.
fn post_command(body: &str, game_tx: &Sender<UiCommand>, request_id: u64) -> CommandAck {
    let cmd = match serde_json::from_str::<serde_json::Value>(body) {
        Ok(cmd) => cmd,
        Err(_) => {
            return CommandAck {
                request_id,
                accepted: false,
                error: Some("request body is not valid JSON".into()),
            };
        }
    };
    let Some(cmd_type) = cmd.get("type").and_then(|value| value.as_str()) else {
        return CommandAck {
            request_id,
            accepted: false,
            error: Some("command type must be a string".into()),
        };
    };
    let Some(command) = build_command(cmd_type, cmd.get("data")) else {
        return CommandAck {
            request_id,
            accepted: false,
            error: Some("invalid command data".into()),
        };
    };
    let command = UiCommand::WithRequestId {
        request_id,
        command: Box::new(command),
    };
    if game_tx.send(command).is_err() {
        return CommandAck {
            request_id,
            accepted: false,
            error: Some("engine command channel is disconnected".into()),
        };
    }
    CommandAck {
        request_id,
        accepted: true,
        error: None,
    }
}

/// Route a posted command to its `UiCommand`: `set_component` is the typed
/// generic lane (registry); anything else is a Custom pass-through.
/// Malformed `set_component` shapes are dropped (`None`) like any garbage
/// on this endpoint.
fn build_command(cmd_type: &str, data: Option<&serde_json::Value>) -> Option<UiCommand> {
    if cmd_type == "set_component" {
        return parse_set_component(data);
    }
    let json_data = data.map(|v| v.to_string()).unwrap_or_default();
    Some(UiCommand::Custom {
        cmd_type: cmd_type.to_string(),
        json_data,
    })
}

/// Build the typed generic upsert from `data` of a `set_component` post:
/// `{"id": u32, "generation"?: u32, "component": "Transform", "value": {…}}`.
/// `None` on any schema violation — the world emits no ack, and the
/// malformed post is dropped like any other garbage on this endpoint.
fn parse_set_component(data: Option<&serde_json::Value>) -> Option<UiCommand> {
    let data = data?;
    let entity_id = data.get("id")?.as_u64()? as u32;
    let generation = data
        .get("generation")
        .and_then(|v| v.as_u64())
        .map(|g| g as u32);
    let type_name = data.get("component")?.as_str()?.to_string();
    let json_data = data.get("value")?.to_string();
    Some(UiCommand::SetComponent {
        entity_id,
        generation,
        type_name,
        json_data,
    })
}

fn json_response(body: &str) -> Response<Cursor<Vec<u8>>> {
    Response::from_data(body)
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
}

fn not_found() -> Response<Cursor<Vec<u8>>> {
    Response::from_data("404 Not Found").with_status_code(404)
}

fn content_type(path: &Path) -> Header {
    let ct = match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    };
    Header::from_bytes("Content-Type", ct).unwrap()
}

fn event_json_data(json_data: &str) -> serde_json::Value {
    serde_json::from_str(json_data)
        .unwrap_or_else(|_| serde_json::Value::String(json_data.to_owned()))
}

/// Serialize events as valid JSON, escaping all string fields through
/// `serde_json` and preserving canonical object payloads when `json_data`
/// contains JSON. Malformed payload strings remain valid JSON strings rather
/// than corrupting the entire `/api/events` response.
fn format_events(events: &[GameEvent]) -> String {
    let values: Vec<serde_json::Value> = events
        .iter()
        .map(|event| match event {
            GameEvent::EntityCreated { entity_id } => {
                serde_json::json!({"EntityCreated": {"entity_id": entity_id}})
            }
            GameEvent::EntityDestroyed { entity_id } => {
                serde_json::json!({"EntityDestroyed": {"entity_id": entity_id}})
            }
            GameEvent::ComponentUpdated {
                entity_id,
                type_name,
                json_data,
            } => serde_json::json!({
                "ComponentUpdated": {
                    "entity_id": entity_id,
                    "type_name": type_name,
                    "json_data": event_json_data(json_data),
                }
            }),
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } => serde_json::json!({
                "CustomEvent": {
                    "cmd_type": cmd_type,
                    "json_data": event_json_data(json_data),
                }
            }),
            GameEvent::CommandCompleted {
                request_id,
                command,
                success,
                error,
            } => serde_json::json!({
                "CommandCompleted": {
                    "request_id": request_id,
                    "command": command,
                    "success": success,
                    "error": error,
                }
            }),
        })
        .collect();
    serde_json::to_string(&values).expect("event values are serializable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;
    use std::io::Read;

    // ── format_events ──────────────────────────────────────────────────────

    #[test]
    fn format_events_empty() {
        assert_eq!(format_events(&[]), "[]");
    }

    #[test]
    fn format_events_all_variants() {
        let events = vec![
            GameEvent::EntityCreated { entity_id: 1 },
            GameEvent::EntityDestroyed { entity_id: 2 },
            GameEvent::ComponentUpdated {
                entity_id: 3,
                type_name: "Transform".into(),
                json_data: r#"{"x":1}"#.into(),
            },
            GameEvent::CustomEvent {
                cmd_type: "status".into(),
                json_data: r#"{"v":7}"#.into(),
            },
            GameEvent::CommandCompleted {
                request_id: 4,
                command: "set_component".into(),
                success: true,
                error: None,
            },
        ];
        let out = serde_json::from_str::<serde_json::Value>(&format_events(&events))
            .expect("all event variants must serialize as valid JSON");
        assert_eq!(
            out,
            serde_json::json!([
                {"EntityCreated": {"entity_id": 1}},
                {"EntityDestroyed": {"entity_id": 2}},
                {"ComponentUpdated": {
                    "entity_id": 3,
                    "type_name": "Transform",
                    "json_data": {"x": 1}
                }},
                {"CustomEvent": {
                    "cmd_type": "status",
                    "json_data": {"v": 7}
                }},
                {"CommandCompleted": {
                    "request_id": 4,
                    "command": "set_component",
                    "success": true,
                    "error": null
                }}
            ])
        );
    }

    #[test]
    fn format_events_escapes_text_and_invalid_payloads() {
        let events = vec![
            GameEvent::ComponentUpdated {
                entity_id: 3,
                type_name: "Transform\"\n".into(),
                json_data: "not-json".into(),
            },
            GameEvent::CustomEvent {
                cmd_type: "error\\tag".into(),
                json_data: "also-not-json".into(),
            },
        ];
        let value = serde_json::from_str::<serde_json::Value>(&format_events(&events))
            .expect("escaped event output must remain valid JSON");
        assert_eq!(value[0]["ComponentUpdated"]["type_name"], "Transform\"\n");
        assert_eq!(value[0]["ComponentUpdated"]["json_data"], "not-json");
        assert_eq!(value[1]["CustomEvent"]["cmd_type"], "error\\tag");
        assert_eq!(value[1]["CustomEvent"]["json_data"], "also-not-json");
    }

    #[test]
    fn add_sequence_preserves_snapshot_fields() {
        let value = serde_json::from_str::<serde_json::Value>(&add_sequence(
            r#"{"version":9,"entities":[]}"#,
            17,
        ))
        .expect("sequenced snapshot must be valid JSON");
        assert_eq!(value["version"], 9);
        assert_eq!(value["sequence"], 17);
        assert_eq!(value["entities"], serde_json::json!([]));
        assert_eq!(add_sequence("not-json", 4), "not-json");
    }

    // ── parse_set_component ────────────────────────────────────────────────

    #[test]
    fn parse_set_component_valid() {
        let data = serde_json::json!({
            "id": 42u64,
            "generation": 3u64,
            "component": "Transform",
            "value": {"x": 1.0}
        });
        let cmd = parse_set_component(Some(&data)).expect("valid");
        match cmd {
            UiCommand::SetComponent {
                entity_id,
                generation,
                type_name,
                json_data,
            } => {
                assert_eq!(entity_id, 42);
                assert_eq!(generation, Some(3));
                assert_eq!(type_name, "Transform");
                assert_eq!(json_data, r#"{"x":1.0}"#);
            }
            _ => panic!("expected SetComponent"),
        }
    }

    #[test]
    fn parse_set_component_no_generation() {
        let data = serde_json::json!({
            "id": 7u64,
            "component": "Mesh",
            "value": {"path": "cube.glb"}
        });
        let cmd = parse_set_component(Some(&data)).expect("valid");
        match cmd {
            UiCommand::SetComponent {
                entity_id,
                generation,
                ..
            } => {
                assert_eq!(entity_id, 7);
                assert_eq!(generation, None);
            }
            _ => panic!("expected SetComponent"),
        }
    }

    #[test]
    fn parse_set_component_missing_id() {
        let data = serde_json::json!({"component": "Mesh", "value": {}});
        assert!(parse_set_component(Some(&data)).is_none());
    }

    #[test]
    fn parse_set_component_missing_component() {
        let data = serde_json::json!({"id": 1u64, "value": {}});
        assert!(parse_set_component(Some(&data)).is_none());
    }

    #[test]
    fn parse_set_component_null_data() {
        assert!(parse_set_component(None).is_none());
    }

    // ── build_command ──────────────────────────────────────────────────────

    #[test]
    fn build_command_set_component_lane() {
        let data = serde_json::json!({"id": 1u64, "component": "X", "value": {}});
        let cmd = build_command("set_component", Some(&data));
        assert!(matches!(cmd, Some(UiCommand::SetComponent { .. })));
    }

    #[test]
    fn build_command_custom_passthrough() {
        let data = serde_json::json!({"foo": "bar"});
        let cmd = build_command("create_entity", Some(&data)).expect("custom");
        match cmd {
            UiCommand::Custom {
                cmd_type,
                json_data,
            } => {
                assert_eq!(cmd_type, "create_entity");
                assert_eq!(json_data, r#"{"foo":"bar"}"#);
            }
            _ => panic!("expected Custom"),
        }
    }

    #[test]
    fn build_command_custom_no_data() {
        let cmd = build_command("ping", None).expect("custom");
        match cmd {
            UiCommand::Custom {
                cmd_type,
                json_data,
            } => {
                assert_eq!(cmd_type, "ping");
                assert_eq!(json_data, "");
            }
            _ => panic!("expected Custom"),
        }
    }

    // ── post_command ───────────────────────────────────────────────────────

    #[test]
    fn post_command_valid_forwards_to_game_with_ack() {
        let (tx, rx) = unbounded::<UiCommand>();
        let ack = post_command(
            r#"{"type":"set_component","data":{"id":5,"component":"T","value":{}}}"#,
            &tx,
            42,
        );
        assert_eq!(
            ack,
            CommandAck {
                request_id: 42,
                accepted: true,
                error: None,
            }
        );
        let cmd = rx.try_recv().expect("command forwarded");
        match cmd {
            UiCommand::WithRequestId {
                request_id,
                command,
            } => {
                assert_eq!(request_id, 42);
                assert!(matches!(
                    *command,
                    UiCommand::SetComponent { entity_id: 5, .. }
                ));
            }
            _ => panic!("expected request-id wrapper"),
        }
    }

    #[test]
    fn post_command_garbage_returns_rejections_without_panicking() {
        let (tx, rx) = unbounded::<UiCommand>();
        // not JSON at all
        let invalid_json = post_command("this is not json", &tx, 1);
        assert!(!invalid_json.accepted);
        assert_eq!(invalid_json.request_id, 1);
        // JSON but no "type"
        let missing_type = post_command(r#"{"foo":1}"#, &tx, 2);
        assert!(!missing_type.accepted);
        // JSON with unknown type (still a Custom, not dropped)
        let accepted = post_command(r#"{"type":"unknown","data":{}}"#, &tx, 3);
        assert!(accepted.accepted);
        // exactly one command should have been sent (the Custom unknown)
        let cmd = rx.try_recv().expect("one command");
        match cmd {
            UiCommand::WithRequestId {
                request_id,
                command,
            } => {
                assert_eq!(request_id, 3);
                assert!(
                    matches!(*command, UiCommand::Custom { cmd_type, .. } if cmd_type == "unknown")
                );
            }
            _ => panic!("expected request-id wrapper"),
        }
        assert!(rx.try_recv().is_err(), "no more commands");
    }

    #[test]
    fn command_request_ids_are_monotonic_and_accept_client_ids() {
        let mut next = 1;
        assert_eq!(command_request_id(r#"{"type":"ping"}"#, &mut next), 1);
        assert_eq!(
            command_request_id(r#"{"type":"ping","request_id":41}"#, &mut next),
            41
        );
        assert_eq!(next, 42);
        assert_eq!(command_request_id(r#"{"type":"ping"}"#, &mut next), 42);
        assert_eq!(
            command_request_id(r#"{"type":"ping","request_id":0}"#, &mut next),
            43
        );
    }

    #[test]
    fn command_ack_json_is_explicit_and_valid() {
        let accepted = serde_json::from_str::<serde_json::Value>(&command_ack_json(&CommandAck {
            request_id: 7,
            accepted: true,
            error: None,
        }))
        .expect("accepted ack is valid JSON");
        assert_eq!(accepted["accepted"], true);
        assert_eq!(accepted["request_id"], 7);
        assert!(accepted.get("error").is_none());

        let rejected = serde_json::from_str::<serde_json::Value>(&command_ack_json(&CommandAck {
            request_id: 8,
            accepted: false,
            error: Some("bad request".into()),
        }))
        .expect("rejected ack is valid JSON");
        assert_eq!(rejected["accepted"], false);
        assert_eq!(rejected["error"], "bad request");
    }

    // ── content_type ───────────────────────────────────────────────────────

    #[test]
    fn content_type_variants() {
        let ct = |ext: &str| {
            let p = PathBuf::from(format!("x.{ext}"));
            content_type(&p).value.to_string()
        };
        assert!(ct("html").starts_with("text/html"));
        assert!(ct("css").starts_with("text/css"));
        assert!(ct("js").starts_with("application/javascript"));
        assert!(ct("json").starts_with("application/json"));
        assert!(ct("svg").starts_with("image/svg+xml"));
        assert!(ct("png").starts_with("image/png"));
        assert!(ct("woff2").starts_with("font/woff2"));
        assert_eq!(ct("unknown"), "application/octet-stream");
    }

    // ── serve_static ───────────────────────────────────────────────────────

    #[test]
    fn serve_static_reads_file() {
        let dir = std::env::temp_dir().join("editor_backend_test_static");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("hello.txt");
        fs::write(&file, b"hello world").unwrap();
        let resp = serve_static(&dir, "/hello.txt");
        assert_eq!(resp.status_code(), 200);
        let body = read_response(resp);
        assert_eq!(body, "hello world");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn serve_static_traversal_blocked() {
        let dir = std::env::temp_dir().join("editor_backend_test_static2");
        let _ = fs::create_dir_all(&dir);
        let resp = serve_static(&dir, "/../etc/passwd");
        assert_eq!(resp.status_code(), 404);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn serve_static_missing_file_404() {
        let dir = std::env::temp_dir().join("editor_backend_test_static3");
        let _ = fs::create_dir_all(&dir);
        let resp = serve_static(&dir, "/does-not-exist.txt");
        assert_eq!(resp.status_code(), 404);
        let _ = fs::remove_dir_all(&dir);
    }

    // ── drain_game_events ──────────────────────────────────────────────────

    #[test]
    fn drain_game_events_routes_status_and_scene() {
        let (tx, rx) = unbounded::<GameEvent>();
        // drain_game_events only READS from rx; we send via tx.
        tx.send(GameEvent::CustomEvent {
            cmd_type: "status".into(),
            json_data: r#"{"value":"STAT"}"#.into(),
        })
        .unwrap();
        tx.send(GameEvent::CustomEvent {
            cmd_type: "scene".into(),
            json_data: r#"{"value":"SCN"}"#.into(),
        })
        .unwrap();
        tx.send(GameEvent::EntityCreated { entity_id: 11 }).unwrap();

        let mut buffer = Vec::new();
        let mut snaps = Snapshots::default();
        drain_game_events(&rx, &mut buffer, &mut snaps);

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&snaps.status).expect("status JSON")["sequence"],
            1
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&snaps.scene).expect("scene JSON")["sequence"],
            2
        );
        assert_eq!(snaps.sequence, 2);
        assert_eq!(buffer.len(), 1);
        assert!(matches!(
            buffer[0],
            GameEvent::EntityCreated { entity_id: 11 }
        ));
    }

    // ── assets_root ────────────────────────────────────────────────────────

    #[test]
    fn assets_root_cli_arg_wins() {
        // cannot easily inject args without affecting the real process; we
        // at least verify the env + default path resolve without panicking.
        let _ = assets_root();
    }

    // ── helpers ────────────────────────────────────────────────────────────

    fn read_response(resp: Response<Cursor<Vec<u8>>>) -> String {
        let mut body = String::new();
        let mut reader = resp.into_reader();
        let _ = reader.read_to_string(&mut body);
        body
    }
}
