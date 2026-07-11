use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use ornis_ui::ipc::{GameEvent, UiCommand};
use tiny_http::{Header, Response, Server};

const HTML: &str = include_str!("remote_editor.html");

pub struct RemoteEditor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RemoteEditor {
    pub fn start(
        port: u16,
        game_tx: Sender<UiCommand>,
        game_rx: Receiver<GameEvent>,
    ) -> Self {
        let addr = format!("127.0.0.1:{port}");
        let server = match Server::http(&addr) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ornis: remote editor failed to bind {addr}: {e}");
                return Self { stop: Arc::new(AtomicBool::new(true)), handle: None };
            }
        };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = thread::Builder::new()
            .name("remote-editor".into())
            .spawn(move || serve(server, stop_clone, game_tx, game_rx))
            .expect("spawn remote-editor thread");

        eprintln!("ornis: remote editor at http://{addr}");
        Self { stop, handle: Some(handle) }
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

fn serve(
    server: Server,
    stop: Arc<AtomicBool>,
    game_tx: Sender<UiCommand>,
    game_rx: Receiver<GameEvent>,
) {
    let mut events_buffer: Vec<GameEvent> = Vec::new();
    let mut cached_status: String = r#"{"entity_count":0,"name":"Ornis Engine"}"#.to_string();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Drain incoming game events into buffer
        while let Ok(ev) = game_rx.try_recv() {
            match &ev {
                GameEvent::CustomEvent { cmd_type, json_data } if cmd_type == "status" => {
                    cached_status = json_data.clone();
                }
                _ => {}
            }
            events_buffer.push(ev);
        }

        // Accept one request with a short timeout
        let mut request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(_) => break,
        };

        let url = request.url().to_string();
        let method = request.method().as_str().to_string();

        let response = match (method.as_str(), url.as_str()) {
            ("GET", "/") | ("GET", "/index.html") => {
                Response::from_string(HTML)
                    .with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap())
            }
            ("GET", "/api/status") => {
                Response::from_string(&cached_status)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
            }
            ("GET", "/api/events") => {
                let body = format_events(&events_buffer);
                events_buffer.clear();
                Response::from_string(body)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
            }
            ("POST", "/api/command") => {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                let cmd = serde_json::from_str::<serde_json::Value>(&body).ok();
                if let Some(cmd) = cmd {
                    if let Some(cmd_type) = cmd.get("type").and_then(|v| v.as_str()) {
                        let json_data = cmd.get("data")
                            .map(|v| v.to_string())
                            .unwrap_or_default();
                        game_tx.send(UiCommand::Custom {
                            cmd_type: cmd_type.to_string(),
                            json_data,
                        }).ok();
                    }
                }
                Response::from_string("{}")
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
            }
            _ => {
                Response::from_string("404 Not Found")
                    .with_status_code(404)
            }
        };

        let _ = request.respond(response);
    }
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
            GameEvent::ComponentUpdated { entity_id, type_name, json_data } => {
                format!(r#"{{"ComponentUpdated":{{"entity_id":{entity_id},"type_name":"{type_name}","json_data":{json_data}}}}}"#)
            }
            GameEvent::CustomEvent { cmd_type, json_data } => {
                format!(r#"{{"CustomEvent":{{"cmd_type":"{cmd_type}","json_data":{json_data}}}}}"#)
            }
        };
        parts.push(s);
    }
    format!("[{}]", parts.join(","))
}
