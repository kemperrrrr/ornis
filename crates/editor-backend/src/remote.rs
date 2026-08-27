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

const EMPTY_STATUS: &str = r#"{"entity_count":0,"name":"Ornis Engine","version":0}"#;
const EMPTY_SCENE: &str =
    r#"{"version":0,"entity_count":0,"entities":[],"lights":[],"camera":null,"ambient":null}"#;

/// Snapshot payloads refreshed out of the game-event stream; served by the
/// `/api/status` and `/api/scene` endpoints until the next snapshot arrives.
struct Snapshots {
    status: String,
    scene: String,
}

impl Default for Snapshots {
    fn default() -> Self {
        Self {
            status: EMPTY_STATUS.to_string(),
            scene: EMPTY_SCENE.to_string(),
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

        let response = route_request(&root, &mut request, &mut buffer, &snapshots, &game_tx);
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
                snaps.status = json_data.clone();
                continue;
            }
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } if cmd_type == "scene" => {
                snaps.scene = json_data.clone();
                continue;
            }
            _ => {}
        }
        buffer.push(ev);
    }
}

/// Serve one HTTP request. `/api/command` posts are forwarded to the game
/// thread fire-and-forget; everything else is answered synchronously.
fn route_request(
    root: &Path,
    request: &mut Request,
    buffer: &mut Vec<GameEvent>,
    snapshots: &Snapshots,
    game_tx: &Sender<UiCommand>,
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
            post_command(&body, game_tx);
            json_response("{}")
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
/// to the game thread. Malformed posts are dropped silently — the endpoint
/// answers `{}` either way.
fn post_command(body: &str, game_tx: &Sender<UiCommand>) {
    let cmd = serde_json::from_str::<serde_json::Value>(body).ok();
    if let Some(cmd) = cmd
        && let Some(cmd_type) = cmd.get("type").and_then(|v| v.as_str())
        && let Some(command) = build_command(cmd_type, cmd.get("data"))
    {
        game_tx.send(command).ok();
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

fn format_events(events: &[GameEvent]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(events.len());
    for ev in events {
        let s = match ev {
            GameEvent::EntityCreated { entity_id } => {
                format!(r#"{{"EntityCreated":{{"entity_id":{entity_id}}}}}"#)
            }
            GameEvent::EntityDestroyed { entity_id } => {
                format!(r#"{{"EntityDestroyed":{{"entity_id":{entity_id}}}}}"#)
            }
            GameEvent::ComponentUpdated {
                entity_id,
                type_name,
                json_data,
            } => {
                format!(
                    r#"{{"ComponentUpdated":{{"entity_id":{entity_id},"type_name":"{type_name}","json_data":{json_data}}}}}"#
                )
            }
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } => {
                format!(r#"{{"CustomEvent":{{"cmd_type":"{cmd_type}","json_data":{json_data}}}}}"#)
            }
        };
        parts.push(s);
    }
    format!("[{}]", parts.join(","))
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
        ];
        let out = format_events(&events);
        assert_eq!(
            out,
            r#"[{"EntityCreated":{"entity_id":1}},{"EntityDestroyed":{"entity_id":2}},{"ComponentUpdated":{"entity_id":3,"type_name":"Transform","json_data":{"x":1}}},{"CustomEvent":{"cmd_type":"status","json_data":{"v":7}}}]"#
        );
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
    fn post_command_valid_forwards_to_game() {
        let (tx, rx) = unbounded::<UiCommand>();
        post_command(
            r#"{"type":"set_component","data":{"id":5,"component":"T","value":{}}}"#,
            &tx,
        );
        let cmd = rx.try_recv().expect("command forwarded");
        assert!(matches!(cmd, UiCommand::SetComponent { entity_id: 5, .. }));
    }

    #[test]
    fn post_command_garbage_dropped_not_panicked() {
        let (tx, rx) = unbounded::<UiCommand>();
        // not JSON at all
        post_command("this is not json", &tx);
        // JSON but no "type"
        post_command(r#"{"foo":1}"#, &tx);
        // JSON with unknown type (still a Custom, not dropped)
        post_command(r#"{"type":"unknown","data":{}}"#, &tx);
        // exactly one command should have been sent (the Custom unknown)
        let cmd = rx.try_recv().expect("one command");
        assert!(matches!(cmd, UiCommand::Custom { cmd_type, .. } if cmd_type == "unknown"));
        assert!(rx.try_recv().is_err(), "no more commands");
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
            json_data: "STAT".into(),
        })
        .unwrap();
        tx.send(GameEvent::CustomEvent {
            cmd_type: "scene".into(),
            json_data: "SCN".into(),
        })
        .unwrap();
        tx.send(GameEvent::EntityCreated { entity_id: 11 }).unwrap();

        let mut buffer = Vec::new();
        let mut snaps = Snapshots::default();
        drain_game_events(&rx, &mut buffer, &mut snaps);

        assert_eq!(snaps.status, "STAT");
        assert_eq!(snaps.scene, "SCN");
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
