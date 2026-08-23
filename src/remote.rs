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
///   3. `<workspace>/editor` (CARGO_MANIFEST_DIR for the `ornis` binary
///      points at the workspace root)
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("editor")
}

pub struct RemoteEditor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RemoteEditor {
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
