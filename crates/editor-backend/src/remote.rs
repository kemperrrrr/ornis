//! Remote editor transport: HTTP server and asset sync for live editing.
//!
//! Mutating endpoints (`POST /api/*`) are loopback-only hardening targets:
//! every POST re-checks the `Host` header (DNS-rebinding), the `Origin`
//! header (CSRF), and the `Content-Type` (JSON only), and bodies are read
//! bounded ([`MAX_COMMAND_BYTES`]). Scene-file commands additionally resolve
//! their `path` into a [`ScenePath`] sandboxed to `<workspace>/assets` or
//! `<workspace>/editor`.

use std::collections::VecDeque;
use std::fs;
use std::io::{Cursor, Read};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use tiny_http::{Header, Request, Response, Server};
use tungstenite::handshake::server::{ErrorResponse, Request as WsRequest, Response as WsResponse};
use tungstenite::http::StatusCode;
use tungstenite::protocol::frame::CloseFrame;
use tungstenite::protocol::frame::coding::CloseCode;
use tungstenite::{Bytes, Message, Utf8Bytes, WebSocket, accept_hdr_with_config};

use crate::ipc::{EditorCommand, EventSeq, GameEvent, RequestId, SetComponentPayload, UiCommand};

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
/// `/api/*` endpoints from background threads; `stop` shuts them down and joins.
///
/// Transport layout: the public port is owned by a plain `TcpListener` that
/// peek-dispatches each connection — `/api/events` WebSocket upgrades are
/// framed by tungstenite right here (over our own `TcpStream`, so read
/// timeouts are legitimate socket options), everything else is byte-proxied
/// to an internal tiny_http server on an ephemeral port.
pub struct RemoteEditor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    accept_handle: Option<JoinHandle<()>>,
    websocket_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl RemoteEditor {
    /// Bind `127.0.0.1:{port}` and start serving. On bind failure prints an
    /// error and returns an inert handle instead of panicking.
    pub fn start(port: u16, game_tx: Sender<UiCommand>, game_rx: Receiver<GameEvent>) -> Self {
        let inert = || Self {
            stop: Arc::new(AtomicBool::new(true)),
            handle: None,
            accept_handle: None,
            websocket_handles: Arc::new(Mutex::new(Vec::new())),
        };
        let addr = format!("127.0.0.1:{port}");
        let listener = match TcpListener::bind(&addr) {
            Ok(listener) => listener,
            Err(e) => {
                eprintln!("ornis: remote editor failed to bind {addr}: {e}");
                return inert();
            }
        };
        let internal = match Server::http("127.0.0.1:0") {
            Ok(server) => server,
            Err(e) => {
                eprintln!("ornis: remote editor failed to bind internal HTTP server: {e}");
                return inert();
            }
        };
        let Some(internal_addr) = internal.server_addr().to_ip() else {
            eprintln!("ornis: remote editor internal HTTP server has no IP address");
            return inert();
        };
        let internal_port = internal_addr.port();

        let stop = Arc::new(AtomicBool::new(false));
        // Replay log shared between the internal HTTP server thread (which
        // drains `game_rx` into it) and the `/api/events` socket handlers.
        let event_log = Arc::new(Mutex::new(EventLog::default()));
        let stop_clone = stop.clone();
        let websocket_handles = Arc::new(Mutex::new(Vec::new()));
        let websocket_handles_for_accept = Arc::clone(&websocket_handles);
        let game_tx_for_accept = game_tx.clone();
        let event_log_for_accept = Arc::clone(&event_log);
        let accept_handle = thread::Builder::new()
            .name("remote-editor-accept".into())
            .spawn(move || {
                accept_loop(
                    listener,
                    stop_clone,
                    game_tx_for_accept,
                    event_log_for_accept,
                    websocket_handles_for_accept,
                    internal_port,
                    port,
                )
            })
            .expect("spawn remote-editor-accept thread");
        let stop_clone = stop.clone();

        let handle = thread::Builder::new()
            .name("remote-editor".into())
            .spawn(move || serve(internal, stop_clone, game_tx, game_rx, event_log, port))
            .expect("spawn remote-editor thread");

        eprintln!("ornis: remote editor at http://{addr}");
        Self {
            stop,
            handle: Some(handle),
            accept_handle: Some(accept_handle),
            websocket_handles,
        }
    }

    /// Signal shutdown and join the server threads.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.accept_handle.take() {
            let _ = handle.join();
        }
        let handles = std::mem::take(
            &mut *self
                .websocket_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        for handle in handles {
            let _ = handle.join();
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
    sequence: EventSeq,
}

impl Default for Snapshots {
    fn default() -> Self {
        Self {
            status: EMPTY_STATUS.to_string(),
            scene: EMPTY_SCENE.to_string(),
            sequence: EventSeq::new(0),
        }
    }
}

const EVENT_HISTORY_CAPACITY: usize = 256;

/// One user-facing event plus its server-side replay sequence.
#[derive(Debug, Clone)]
struct EventRecord {
    sequence: EventSeq,
    event: GameEvent,
}

/// Bounded replay log for `/api/events?after=<sequence>`.
///
/// Snapshot cache updates use their own sequence field; this cursor covers
/// command/completion/domain events that are returned by `/api/events`.
#[derive(Debug)]
struct EventLog {
    records: VecDeque<EventRecord>,
    next_sequence: EventSeq,
}

impl Default for EventLog {
    fn default() -> Self {
        Self {
            records: VecDeque::new(),
            next_sequence: EventSeq::new(1),
        }
    }
}

impl EventLog {
    fn push(&mut self, event: GameEvent) {
        let sequence = EventSeq::new(self.next_sequence.get().max(1));
        self.next_sequence = EventSeq::new(sequence.get().saturating_add(1));
        if self.records.len() == EVENT_HISTORY_CAPACITY {
            self.records.pop_front();
        }
        self.records.push_back(EventRecord { sequence, event });
    }

    fn after(&self, cursor: EventSeq) -> Vec<EventRecord> {
        let mut events = Vec::new();
        if let Some(first) = self.records.front()
            && cursor.get().saturating_add(1) < first.sequence.get()
        {
            events.push(EventRecord {
                sequence: EventSeq::new(first.sequence.get().saturating_sub(1)),
                event: GameEvent::EventGap {
                    after: cursor,
                    oldest: first.sequence,
                },
            });
        }
        events.extend(
            self.records
                .iter()
                .filter(|record| record.sequence > cursor)
                .cloned(),
        );
        events
    }
}

/// Internal HTTP loop: drains `game_rx` into the shared replay log and
/// answers proxied plain-HTTP requests. WebSocket upgrades never reach this
/// loop — the public accept loop dispatches them to tungstenite beforehand.
fn serve(
    server: Server,
    stop: Arc<AtomicBool>,
    game_tx: Sender<UiCommand>,
    game_rx: Receiver<GameEvent>,
    event_log: Arc<Mutex<EventLog>>,
    server_port: u16,
) {
    let mut snapshots = Snapshots::default();
    let mut next_request_id = RequestId::new(1);
    let root = assets_root();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        {
            let mut buffer = event_log
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            drain_game_events(&game_rx, &mut buffer, &mut snapshots);
        }

        // Accept one request with a short timeout.
        let mut request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(_) => break,
        };

        let response = route_request(
            &root,
            &mut request,
            &event_log,
            &snapshots,
            &game_tx,
            &mut next_request_id,
            server_port,
        );
        let _ = request.respond(response);
    }
}

fn header_value(request: &Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.to_string())
}

// ── `/api/events` transport ─────────────────────────────────────────────
// The public port is a plain `TcpListener` (see `accept_loop`): each
// connection is peek-dispatched by `classify_connection`. WebSocket upgrades
// for `/api/events` are framed by tungstenite over our own `TcpStream`, so
// read timeouts are legitimate socket options — this replaces the
// hand-rolled handshake/SHA-1/base64/framing and the raw-fd timeout hack.
// Everything else is byte-proxied to the internal tiny_http server.
//
// Wire protocol (unchanged, byte-compatible): the server sends unmasked
// text frames carrying JSON event batches, empty pings every
// `WS_HEARTBEAT_INTERVAL`, and close code 1001 (`Away`) on shutdown; the
// client sends masked text/binary frames carrying `BrowserInput` snapshots.

/// Read timeout for `/api/events` sockets: bounds one poll iteration so the
/// push loop stays responsive. Set via the public `TcpStream` API.
const WS_READ_TIMEOUT: Duration = Duration::from_millis(10);
/// Idle interval between heartbeat pings.
const WS_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
/// Push-loop cadence for newly appended replay records.
const WS_PUSH_INTERVAL: Duration = Duration::from_millis(100);
/// Max HTTP head bytes peeked for routing; larger heads fall through to
/// plain HTTP (the client then uses the polling fallback).
const SNIFF_HEAD_LIMIT: usize = 8192;
/// Total budget for completing the routing peek of one connection.
const SNIFF_TIMEOUT: Duration = Duration::from_secs(5);

/// Server side of one `/api/events` stream: tungstenite over our own socket.
type EventsSocket = WebSocket<TcpStream>;

/// Routing outcome for one accepted TCP connection. Carries the replay
/// cursor for event streams so the handshake path never re-parses the URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConnectionRoute {
    /// RFC 6455 upgrade for exactly `/api/events`, with the `?after=` cursor.
    EventsStream { cursor: EventSeq },
    /// Anything else: byte-proxy to the internal HTTP server.
    Http,
}

/// What the connection handler should do after one inbound poll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServeAction {
    /// Keep serving: a frame was handled or none was available.
    KeepServing,
    /// Terminate: the peer closed or the transport died.
    StopServing,
}

/// Accept loop for the public port: peek-dispatch each connection to the
/// tungstenite event stream or the internal HTTP proxy. Runs until `stop`.
/// `server_port` is the public port: it feeds the WebSocket `Origin` gate
/// (proxied HTTP keeps its bytes, so the internal server checks the same
/// port independently — see [`check_post_guards`]).
fn accept_loop(
    listener: TcpListener,
    stop: Arc<AtomicBool>,
    game_tx: Sender<UiCommand>,
    event_log: Arc<Mutex<EventLog>>,
    websocket_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    internal_port: u16,
    server_port: u16,
) {
    if listener.set_nonblocking(true).is_err() {
        return;
    }
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                dispatch_connection(
                    stream,
                    &stop,
                    &game_tx,
                    &event_log,
                    &websocket_handles,
                    internal_port,
                    server_port,
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(_) => thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// Sniff one connection (inline) and either spawn a joined event-stream
/// handler or proxy it to the internal HTTP server.
fn dispatch_connection(
    stream: TcpStream,
    stop: &Arc<AtomicBool>,
    game_tx: &Sender<UiCommand>,
    event_log: &Arc<Mutex<EventLog>>,
    websocket_handles: &Arc<Mutex<Vec<JoinHandle<()>>>>,
    internal_port: u16,
    server_port: u16,
) {
    if stream.set_read_timeout(Some(SNIFF_TIMEOUT)).is_err() {
        return;
    }
    match classify_connection(&stream) {
        ConnectionRoute::EventsStream { cursor } => {
            let event_log = Arc::clone(event_log);
            let stop = Arc::clone(stop);
            let game_tx = game_tx.clone();
            if let Ok(handle) = thread::Builder::new()
                .name("remote-editor-websocket".into())
                .spawn(move || {
                    serve_events_stream(stream, event_log, stop, cursor, game_tx, server_port)
                })
            {
                websocket_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(handle);
            }
        }
        ConnectionRoute::Http => proxy_http(stream, internal_port),
    }
}

/// Peek at the pending HTTP head without consuming it and decide where the
/// connection goes. Anything that is not a complete, well-formed
/// `/api/events` WebSocket upgrade — including timeouts, oversize heads and
/// undecodable bytes — routes to plain HTTP, where tiny_http answers (and
/// the editor falls back to cursor polling).
fn classify_connection(stream: &TcpStream) -> ConnectionRoute {
    let mut buf = vec![0_u8; SNIFF_HEAD_LIMIT];
    let deadline = Instant::now() + SNIFF_TIMEOUT;
    loop {
        match stream.peek(&mut buf) {
            Ok(0) => return ConnectionRoute::Http,
            Ok(n) => match sniff_route(&buf[..n]) {
                Some(route) => return route,
                None if n >= SNIFF_HEAD_LIMIT => return ConnectionRoute::Http,
                None => {
                    if Instant::now() >= deadline {
                        return ConnectionRoute::Http;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            },
            Err(_) => return ConnectionRoute::Http,
        }
    }
}

/// Route one peeked head: `Some` when the head is complete, `None` when more
/// bytes are needed. Mirrors the old upgrade gate exactly — path
/// `/api/events` plus `Upgrade: websocket` (case-insensitive) — without
/// tightening it: tungstenite itself enforces the remaining RFC 6455
/// requirements (method, version, key) during the handshake.
fn sniff_route(head: &[u8]) -> Option<ConnectionRoute> {
    let text = std::str::from_utf8(head).ok()?;
    let end = text.find("\r\n\r\n")?;
    let mut lines = text[..end].split("\r\n");
    let target = lines.next()?.split_whitespace().nth(1)?;
    if target.split('?').next() != Some("/api/events") {
        return Some(ConnectionRoute::Http);
    }
    let upgrade = lines
        .filter_map(|line| line.split_once(':'))
        .any(|(name, value)| {
            name.trim().eq_ignore_ascii_case("upgrade")
                && value.trim().eq_ignore_ascii_case("websocket")
        });
    if !upgrade {
        return Some(ConnectionRoute::Http);
    }
    Some(ConnectionRoute::EventsStream {
        cursor: event_cursor(target),
    })
}

/// Serve one WebSocket `/api/events` connection over our own `TcpStream`.
/// Bidirectional: the server pushes replay records past the initial cursor
/// and newly appended ones; client text/binary frames carry `BrowserInput`
/// snapshots forwarded to the game thread. Idle connections get periodic
/// pings; server shutdown sends close code 1001 (`Away`).
///
/// `#[allow]` below: tungstenite's `Callback` fixes the handshake error
/// type to `ErrorResponse` (a full HTTP response, inherently large), so no
/// smaller `Err` spelling exists for the Origin gate.
#[allow(clippy::result_large_err)]
fn serve_events_stream(
    stream: TcpStream,
    event_log: Arc<Mutex<EventLog>>,
    stop: Arc<AtomicBool>,
    mut cursor: EventSeq,
    game_tx: Sender<UiCommand>,
    server_port: u16,
) {
    if stream.set_read_timeout(Some(WS_READ_TIMEOUT)).is_err() {
        return;
    }
    // `accept_unmasked_frames` preserves the previous transport's tolerance:
    // it unmasked when present and accepted frames without a mask instead of
    // failing the connection (browsers always mask; the old code tolerated).
    let config = tungstenite::protocol::WebSocketConfig::default().accept_unmasked_frames(true);
    // Same trust roots as `POST /api/*` (see [`decide_origin`]): a foreign
    // page must not read the event stream or inject input frames through a
    // socket — browsers always send `Origin` on WebSocket handshakes, so a
    // present-but-foreign value is denied while absent stays allowed for
    // non-browser clients.
    let mut ws = match accept_hdr_with_config(
        stream,
        |request: &WsRequest, response: WsResponse| {
            let origin = request
                .headers()
                .get("origin")
                .and_then(|value| value.to_str().ok());
            match decide_origin(origin, server_port) {
                OriginDecision::Allow => Ok(response),
                OriginDecision::Deny => {
                    let mut denied: ErrorResponse = response.map(|_| None);
                    *denied.status_mut() = StatusCode::FORBIDDEN;
                    Err(denied)
                }
            }
        },
        Some(config),
    ) {
        Ok(ws) => ws,
        // Tungstenite already wrote the rejection response.
        Err(_) => return,
    };
    let mut last_heartbeat = Instant::now();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        match poll_inbound(&mut ws, &game_tx) {
            ServeAction::KeepServing => {}
            ServeAction::StopServing => return,
        }
        if push_pending(&mut ws, &event_log, &mut cursor).is_err() {
            return;
        }
        if last_heartbeat.elapsed() >= WS_HEARTBEAT_INTERVAL {
            if ws.send(Message::Ping(Bytes::new())).is_err() {
                return;
            }
            last_heartbeat = Instant::now();
        }
        thread::sleep(WS_PUSH_INTERVAL);
    }
    let _ = ws.close(Some(CloseFrame {
        code: CloseCode::Away,
        reason: Utf8Bytes::from_static(""),
    }));
}

/// Read one inbound message without blocking the push loop (the socket has
/// `WS_READ_TIMEOUT`): forward data frames as `BrowserInput`, drive
/// tungstenite's automatic pong/close replies, and report whether serving
/// should continue.
fn poll_inbound(ws: &mut EventsSocket, game_tx: &Sender<UiCommand>) -> ServeAction {
    match ws.read() {
        Ok(Message::Text(payload)) => {
            forward_browser_input(payload.as_bytes(), game_tx);
            ServeAction::KeepServing
        }
        Ok(Message::Binary(payload)) => {
            forward_browser_input(&payload, game_tx);
            ServeAction::KeepServing
        }
        // Pings are answered with a queued pong by tungstenite itself; the
        // reply flushes on the next read/write. Pongs need no handling.
        // `Frame` never surfaces from `read`.
        Ok(Message::Ping(_)) | Ok(Message::Pong(_)) | Ok(Message::Frame(_)) => {
            ServeAction::KeepServing
        }
        Ok(Message::Close(_)) => {
            // Flush the queued close echo before dropping the socket.
            let _ = ws.flush();
            ServeAction::StopServing
        }
        Err(tungstenite::Error::Io(err))
            if err.kind() == std::io::ErrorKind::WouldBlock
                || err.kind() == std::io::ErrorKind::TimedOut =>
        {
            // No frame available — the push loop continues. Partial frames
            // stay buffered inside tungstenite and resume on the next read.
            ServeAction::KeepServing
        }
        Err(_) => ServeAction::StopServing,
    }
}

/// Forward one client data frame as a `BrowserInput` snapshot. Garbage
/// payloads are dropped like on the `POST /api/input` endpoint.
fn forward_browser_input(payload: &[u8], game_tx: &Sender<UiCommand>) {
    if let Some(input) = parse_browser_input(payload) {
        let _ = game_tx.send(UiCommand::Input { input });
    }
}

/// Push replay records past `cursor` as one text frame. `Err` means the
/// connection died; the caller terminates the handler (the client is
/// expected to reconnect with its last cursor).
fn push_pending(
    ws: &mut EventsSocket,
    event_log: &Arc<Mutex<EventLog>>,
    cursor: &mut EventSeq,
) -> Result<(), tungstenite::Error> {
    let records = event_log
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .after(*cursor);
    if let Some(last) = records.last() {
        *cursor = last.sequence;
        ws.send(Message::text(format_event_records(&records)))?;
    }
    Ok(())
}

/// Proxy one plain-HTTP connection to the internal tiny_http server.
/// Byte-transparent in both directions, so HTTP semantics (keep-alive,
/// framing, guards) stay exactly tiny_http's. Pump threads are transient:
/// each exits on EOF or error, and a read-idle inherited from the sniff
/// timeout reaps lingering keep-alive connections.
fn proxy_http(client: TcpStream, internal_port: u16) {
    let upstream = match TcpStream::connect(("127.0.0.1", internal_port)) {
        Ok(upstream) => upstream,
        Err(_) => return,
    };
    let (client_reader, client_writer) = match (client.try_clone(), client.try_clone()) {
        (Ok(reader), Ok(writer)) => (reader, writer),
        _ => return,
    };
    let (upstream_reader, upstream_writer) = match (upstream.try_clone(), upstream.try_clone()) {
        (Ok(reader), Ok(writer)) => (reader, writer),
        _ => return,
    };
    drop(client);
    drop(upstream);
    for (name, from, to) in [
        ("remote-editor-proxy-up", client_reader, upstream_writer),
        ("remote-editor-proxy-down", upstream_reader, client_writer),
    ] {
        let _ = thread::Builder::new()
            .name(name.into())
            .spawn(move || pump_one(from, to));
    }
}

/// Copy bytes one direction until EOF or error. The sibling pump notices the
/// closed socket and exits as well, so no joining is needed.
fn pump_one(mut from: TcpStream, mut to: TcpStream) {
    let _ = std::io::copy(&mut from, &mut to);
}

/// Poll the upgraded stream for any available client frames. Returns
/// `Ok(true)` if the connection should be closed (client sent close or
/// EOF), `Ok(false)` if no close was seen, or `Err` on I/O error.
/// Handles control frames and forwards text/binary data frames as
/// `BrowserInput` snapshots to `game_tx` (WS input channel, no polling).
/// Drain incoming game events into `buffer`. "status"/"scene" snapshots only
/// refresh the endpoint caches — they are not user-facing events, so they
/// skip the buffer.
fn drain_game_events(game_rx: &Receiver<GameEvent>, buffer: &mut EventLog, snaps: &mut Snapshots) {
    while let Ok(ev) = game_rx.try_recv() {
        match &ev {
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } if cmd_type.as_str() == "status" => {
                snaps.sequence = snaps.sequence.next();
                snaps.status = add_sequence(json_data, snaps.sequence);
                continue;
            }
            GameEvent::CustomEvent {
                cmd_type,
                json_data,
            } if cmd_type.as_str() == "scene" => {
                snaps.sequence = snaps.sequence.next();
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
fn add_sequence(body: &str, sequence: EventSeq) -> String {
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
    request_id: RequestId,
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

fn allocate_request_id(next_request_id: &mut RequestId) -> RequestId {
    let request_id = RequestId::new(next_request_id.get().max(1));
    *next_request_id = request_id.next();
    request_id
}

/// Use a client-provided positive request id when present; otherwise allocate
/// a monotonic server id. Advancing the allocator past a supplied id avoids
/// collisions with subsequent generated ids.
fn command_request_id(body: &str, next_request_id: &mut RequestId) -> RequestId {
    let requested = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("request_id").and_then(|id| id.as_u64()))
        .filter(|&id| id > 0)
        .map(RequestId::new);
    if let Some(request_id) = requested {
        if request_id >= *next_request_id {
            *next_request_id = request_id.next();
        }
        request_id
    } else {
        allocate_request_id(next_request_id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// POST /api/* request guards: CSRF + DNS-rebinding + payload limits.
// ═══════════════════════════════════════════════════════════════════════════
// The editor server binds `127.0.0.1` without authentication, so any page
// open in the browser could otherwise POST to it. Every mutating endpoint
// re-checks three trust roots before touching the body: the `Host` header
// (DNS-rebinding — the socket must be addressed as this loopback server,
// not as `evil.com` resolving to 127.0.0.1), the `Origin` header (CSRF —
// only the served editor page may drive the API from a browser;
// non-browser clients send no `Origin` and stay allowed), and the
// `Content-Type` (JSON only — blocks `text/plain` simple-request CSRF).
// Bodies are read bounded ([`MAX_COMMAND_BYTES`]). GET endpoints are
// read-only and intentionally unguarded.

/// Maximum accepted body size for `POST /api/*` (1 MiB): command and input
/// payloads are small JSON snapshots, so anything larger is rejected with
/// 413 before parsing.
pub const MAX_COMMAND_BYTES: usize = 1 << 20;

/// Verdict of a loopback header check — an enum, never a bare `bool`, so
/// call sites read as `decide_host(..) == OriginDecision::Deny`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginDecision {
    /// The header is absent (non-browser client) or names this server.
    Allow,
    /// The header names a foreign origin/host: reject with 403.
    Deny,
}

/// CSRF gate for `POST /api/*`: an absent `Origin` (curl, raw-socket
/// clients) is allowed; a present one must exactly match the served editor
/// page (`http://127.0.0.1:{port}` or `http://localhost:{port}`).
#[must_use]
pub fn decide_origin(origin: Option<&str>, server_port: u16) -> OriginDecision {
    let Some(origin) = origin else {
        return OriginDecision::Allow;
    };
    let origin = origin.trim().trim_end_matches('/').to_ascii_lowercase();
    if origin == format!("http://127.0.0.1:{server_port}")
        || origin == format!("http://localhost:{server_port}")
    {
        OriginDecision::Allow
    } else {
        OriginDecision::Deny
    }
}

/// DNS-rebinding gate for `POST /api/*`: an absent `Host` is allowed
/// (plain HTTP/1.0 clients); a present one must be this loopback server
/// (`127.0.0.1:{port}` or `localhost:{port}`).
#[must_use]
pub fn decide_host(host: Option<&str>, server_port: u16) -> OriginDecision {
    let Some(host) = host else {
        return OriginDecision::Allow;
    };
    let host = host.trim().to_ascii_lowercase();
    if host == format!("127.0.0.1:{server_port}") || host == format!("localhost:{server_port}") {
        OriginDecision::Allow
    } else {
        OriginDecision::Deny
    }
}

/// Why a `POST /api/*` request was rejected before reaching the engine:
/// typed HTTP-guard failures with fixed status codes (never strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ApiGuardError {
    /// `Origin` names a foreign page (CSRF).
    #[error("forbidden origin")]
    ForbiddenOrigin,
    /// `Host` is not this loopback server (DNS-rebinding).
    #[error("forbidden host")]
    ForbiddenHost,
    /// `Content-Type` is not `application/json`.
    #[error("unsupported media type: expected application/json")]
    UnsupportedMediaType,
    /// The body exceeds [`MAX_COMMAND_BYTES`].
    #[error("request body exceeds 1048576 bytes")]
    PayloadTooLarge,
}

impl ApiGuardError {
    /// HTTP status code for the rejection: 403 for trust roots, 415 for a
    /// wrong media type, 413 for an over-limit body.
    #[must_use]
    pub fn status_code(self) -> u16 {
        match self {
            Self::ForbiddenOrigin | Self::ForbiddenHost => 403,
            Self::UnsupportedMediaType => 415,
            Self::PayloadTooLarge => 413,
        }
    }
}

/// `Content-Type` gate for `POST /api/*`: requires `application/json`,
/// tolerating parameters (`application/json; charset=utf-8`) and case
/// differences. Absent or foreign types are rejected (415).
pub fn check_content_type(value: Option<&str>) -> Result<(), ApiGuardError> {
    let Some(value) = value else {
        return Err(ApiGuardError::UnsupportedMediaType);
    };
    let mime = value
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if mime == "application/json" {
        Ok(())
    } else {
        Err(ApiGuardError::UnsupportedMediaType)
    }
}

/// Read a request body bounded by `limit` bytes: longer bodies are rejected
/// with [`ApiGuardError::PayloadTooLarge`] without buffering the excess.
/// Best-effort like the previous unbounded read — a torn stream yields
/// whatever arrived (later rejected as invalid JSON), and non-UTF-8 bytes
/// surface the same way.
pub fn read_limited_body(reader: &mut dyn Read, limit: usize) -> Result<String, ApiGuardError> {
    let mut body = String::new();
    let _ = reader.take(limit as u64 + 1).read_to_string(&mut body);
    if body.len() > limit {
        Err(ApiGuardError::PayloadTooLarge)
    } else {
        Ok(body)
    }
}

/// The three header/body gates for `POST /api/*`, in rejection order:
/// `Host` (403), `Origin` (403), `Content-Type` (415). The body limit (413)
/// applies when the body is actually read.
fn check_post_guards(request: &Request, server_port: u16) -> Result<(), ApiGuardError> {
    if decide_host(header_value(request, "Host").as_deref(), server_port) == OriginDecision::Deny {
        return Err(ApiGuardError::ForbiddenHost);
    }
    if decide_origin(header_value(request, "Origin").as_deref(), server_port)
        == OriginDecision::Deny
    {
        return Err(ApiGuardError::ForbiddenOrigin);
    }
    check_content_type(header_value(request, "Content-Type").as_deref())
}

/// JSON rejection body for a guard failure, carrying its status code.
fn guard_response(error: ApiGuardError) -> Response<Cursor<Vec<u8>>> {
    let body = serde_json::json!({"accepted": false, "error": error.to_string()}).to_string();
    Response::from_string(body)
        .with_status_code(error.status_code())
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene-path sandbox: `save_scene`/`load_scene` stay in assets/ or editor/.
// ═══════════════════════════════════════════════════════════════════════════
// Validation lives in two places, deliberately: the HTTP backend drops
// escaping paths fail-fast in [`build_command`] (never queued), and the
// session re-resolves every path on execution (defense in depth — direct
// `UiCommand` senders bypass the HTTP edge). Both sides share [`ScenePath`].

/// Validated scene-file roots: `<workspace>/assets` plus
/// `<workspace>/editor`. The only directories `save_scene`/`load_scene`
/// may read or write.
#[derive(Debug, Clone)]
pub struct SceneRoots {
    root: PathBuf,
    assets: PathBuf,
    editor: PathBuf,
}

impl SceneRoots {
    /// Roots anchored at `workspace_root`.
    #[must_use]
    pub fn new(workspace_root: &Path) -> Self {
        Self {
            root: workspace_root.to_path_buf(),
            assets: workspace_root.join("assets"),
            editor: workspace_root.join("editor"),
        }
    }

    /// Production roots: the workspace this crate was compiled in (the same
    /// `../../` convention as [`assets_root`]).
    #[must_use]
    pub fn workspace_defaults() -> Self {
        Self::new(&workspace_root())
    }
}

/// Workspace root anchor: this crate lives at `crates/editor-backend`, so
/// its manifest dir is two levels below the root.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// Why a scene `path` was rejected: typed, never a plain string or panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScenePathError {
    /// The `path` field was empty.
    #[error("scene path is empty")]
    Empty,
    /// The resolved path leaves `<workspace>/assets` and `<workspace>/editor`.
    #[error("scene path escapes the sandbox (<workspace>/assets, <workspace>/editor): {path}")]
    OutsideSandbox {
        /// The raw user-supplied path (for diagnostics, never joined blindly).
        path: String,
    },
}

/// A scene-file path validated to stay inside [`SceneRoots`].
///
/// Construction is the check: [`ScenePath::resolve`] joins a user-supplied
/// `save_scene`/`load_scene` `path` onto the workspace root (absolute inputs
/// are kept as-is), normalizes `.`/`..` lexically, and requires the result
/// to sit under `<workspace>/assets` or `<workspace>/editor` — a
/// component-wise prefix comparison, so `editor-evil/` never matches
/// `editor/`. Existing path prefixes are additionally canonicalized
/// (symlink hardening); not-yet-existing files keep the lexical verdict so
/// `save_scene` can create them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenePath(PathBuf);

impl ScenePath {
    /// Validate `raw` against `roots`. See the type docs for the checks.
    ///
    /// # Errors
    ///
    /// [`ScenePathError::Empty`] for an empty input, [`ScenePathError::OutsideSandbox`]
    /// when the resolved path is not under either sandbox root.
    pub fn resolve(roots: &SceneRoots, raw: &str) -> Result<Self, ScenePathError> {
        if raw.is_empty() {
            return Err(ScenePathError::Empty);
        }
        let candidate = if Path::new(raw).is_absolute() {
            PathBuf::from(raw)
        } else {
            roots.root.join(raw)
        };
        let normalized = lexical_normalize(&candidate);
        let assets = lexical_normalize(&roots.assets);
        let editor = lexical_normalize(&roots.editor);
        if !normalized.starts_with(&assets) && !normalized.starts_with(&editor) {
            return Err(ScenePathError::OutsideSandbox {
                path: raw.to_owned(),
            });
        }
        if canonical_escapes(&assets, &editor, &normalized) {
            return Err(ScenePathError::OutsideSandbox {
                path: raw.to_owned(),
            });
        }
        Ok(Self(normalized))
    }

    /// The validated absolute path, for filesystem use.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

/// Collapse `.`/`..`/duplicate separators without touching the filesystem
/// (the target may not exist yet for `save_scene`). A `..` that would climb
/// past the filesystem root is preserved, so the sandbox prefix check
/// below rejects it.
fn lexical_normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Symlink hardening for a lexically approved candidate: canonicalize the
/// nearest existing ancestor and require it under a canonical root.
/// Layouts that cannot be verified (missing roots or ancestors) keep the
/// lexical verdict instead of failing closed on ordinary new files.
fn canonical_escapes(assets: &Path, editor: &Path, candidate: &Path) -> bool {
    let canonical_roots: Vec<PathBuf> = [assets, editor]
        .iter()
        .filter_map(|root| fs::canonicalize(root).ok())
        .collect();
    if canonical_roots.is_empty() {
        return false;
    }
    let mut current = candidate.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if current.exists() {
            let Ok(canonical) = fs::canonicalize(&current) else {
                return false;
            };
            let mut full = canonical;
            for component in tail.iter().rev() {
                full.push(component);
            }
            return !canonical_roots.iter().any(|root| full.starts_with(root));
        }
        let Some(name) = current.file_name() else {
            return false;
        };
        tail.push(name.to_os_string());
        current.pop();
    }
}

/// Serve one HTTP request. `/api/command` posts are validated, forwarded to
/// the game thread and acknowledged synchronously; snapshot responses carry
/// transport sequence metadata; everything else is answered from current
/// server state.
///
/// `server_port` is the bound loopback port: `POST /api/*` requests are
/// gated on it (see [`decide_host`]/[`decide_origin`]).
fn route_request(
    root: &Path,
    request: &mut Request,
    buffer: &Arc<Mutex<EventLog>>,
    snapshots: &Snapshots,
    game_tx: &Sender<UiCommand>,
    next_request_id: &mut RequestId,
    server_port: u16,
) -> Response<Cursor<Vec<u8>>> {
    let url = request.url().to_string();
    let method = request.method().as_str().to_string();
    let path = url.split('?').next().unwrap_or(url.as_str());

    match (method.as_str(), path) {
        ("GET", "/") | ("GET", "/index.html") => serve_static(root, "index.html"),
        ("GET", "/api/status") => json_response(&snapshots.status),
        ("GET", "/api/scene") => json_response(&snapshots.scene),
        ("GET", "/api/events") => {
            let cursor = event_cursor(&url);
            let records = buffer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .after(cursor);
            let body = format_event_records(&records);
            json_response(&body)
        }
        ("POST", "/api/command") => {
            if let Err(guard) = check_post_guards(request, server_port) {
                return guard_response(guard);
            }
            let body = match read_limited_body(request.as_reader(), MAX_COMMAND_BYTES) {
                Ok(body) => body,
                Err(guard) => return guard_response(guard),
            };
            let request_id = command_request_id(&body, next_request_id);
            let ack = post_command(&body, game_tx, request_id);
            let ack_body = command_ack_json(&ack);
            json_response(&ack_body)
        }
        ("POST", "/api/input") => {
            if let Err(guard) = check_post_guards(request, server_port) {
                return guard_response(guard);
            }
            let body = match read_limited_body(request.as_reader(), MAX_COMMAND_BYTES) {
                Ok(body) => body,
                Err(guard) => return guard_response(guard),
            };
            if let Some(input) = parse_browser_input(body.as_bytes()) {
                let _ = game_tx.send(UiCommand::Input { input });
                json_response(r#"{"accepted":true}"#)
            } else {
                Response::from_string(r#"{"accepted":false,"error":"invalid input"}"#)
                    .with_status_code(400)
                    .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
            }
        }
        ("GET", _) => serve_static(root, &url),
        _ => not_found(),
    }
}

fn event_cursor(url: &str) -> EventSeq {
    url.split_once('?')
        .and_then(|(_, query)| {
            query.split('&').find_map(|part| {
                let (key, value) = part.split_once('=')?;
                (key == "after")
                    .then(|| value.parse::<u64>().ok())
                    .flatten()
            })
        })
        .map(EventSeq::new)
        .unwrap_or(EventSeq::new(0))
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
fn post_command(body: &str, game_tx: &Sender<UiCommand>, request_id: RequestId) -> CommandAck {
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

/// Parse a raw `/api/command` request body into a validated `UiCommand`.
///
/// Pure protocol-parser entry point: it never touches the game channel and
/// never panics on arbitrary input. Returns `Some` for a well-formed
/// `{"type": …, "data": …}` envelope and `None` for malformed JSON, a
/// missing/non-string `type`, or schema-violating `set_component` data.
/// `post_command` keeps its finer-grained ack errors by performing the same
/// steps itself; this function exists for fuzzing and other callers that
/// only need the valid/garbage distinction.
#[must_use]
pub fn parse_command_payload(body: &str) -> Option<UiCommand> {
    let cmd = serde_json::from_str::<serde_json::Value>(body).ok()?;
    let cmd_type = cmd.get("type").and_then(|value| value.as_str())?;
    build_command(cmd_type, cmd.get("data"))
}

/// Route a posted command to its `UiCommand`: `set_component` is the typed
/// generic lane (registry, via [`SetComponentPayload`]); anything else is a
/// Custom pass-through keyed by the typed [`EditorCommand`] tag.
/// Malformed `set_component` shapes are dropped (`None`) like any garbage
/// on this endpoint.
///
/// `save_scene`/`load_scene` additionally fail fast on sandbox violations:
/// a `path` escaping `<workspace>/assets` or `<workspace>/editor` is
/// dropped here and never queued; the session re-validates on execution
/// (defense in depth). An absent or non-string `path` falls through to the
/// session default.
fn build_command(cmd_type: &str, data: Option<&serde_json::Value>) -> Option<UiCommand> {
    if cmd_type == EditorCommand::SetComponent.as_str() {
        return parse_set_component(data);
    }
    if cmd_type == "save_scene" || cmd_type == "load_scene" {
        let raw = data
            .and_then(|value| value.get("path"))
            .and_then(|path| path.as_str());
        if let Some(raw) = raw
            && ScenePath::resolve(&SceneRoots::workspace_defaults(), raw).is_err()
        {
            return None;
        }
    }
    let json_data = data.map(|v| v.to_string()).unwrap_or_default();
    Some(UiCommand::Custom {
        cmd_type: EditorCommand::from(cmd_type),
        json_data,
    })
}

/// Build the typed generic upsert from `data` of a `set_component` post:
/// `{"id": u32, "generation"?: u32, "component": "Transform", "value": {…}}`.
/// Deserializes through [`SetComponentPayload`] (out-of-range integers are
/// rejected, never truncated) and converts via `TryFrom`.
/// `None` on any schema violation — the world emits no ack, and the
/// malformed post is dropped like any other garbage on this endpoint.
fn parse_set_component(data: Option<&serde_json::Value>) -> Option<UiCommand> {
    let data = data?;
    let payload: SetComponentPayload = serde_json::from_value(data.clone()).ok()?;
    UiCommand::try_from(payload).ok()
}

/// Lenient wire shape of a browser input snapshot: integer codes arrive as
/// `u64` and are narrowed with `TryFrom` (out-of-range codes are dropped,
/// never truncated); pointer pairs keep the first two entries so older
/// payloads with trailing fields still parse.
#[derive(Debug, Default, serde::Deserialize)]
struct BrowserInputWire {
    #[serde(default)]
    pressed_keys: Vec<u64>,
    #[serde(default)]
    pressed_mouse_buttons: Vec<u64>,
    #[serde(default)]
    pointer_position: Vec<f64>,
    #[serde(default)]
    pointer_delta: Vec<f64>,
    #[serde(default)]
    wheel_delta: f64,
}

impl BrowserInputWire {
    fn into_input(self) -> crate::ipc::BrowserInput {
        fn pair(values: &[f64]) -> [f32; 2] {
            [
                values.first().copied().unwrap_or(0.0) as f32,
                values.get(1).copied().unwrap_or(0.0) as f32,
            ]
        }
        crate::ipc::BrowserInput {
            pressed_keys: self
                .pressed_keys
                .into_iter()
                .filter_map(|code| u32::try_from(code).ok())
                .collect(),
            pressed_mouse_buttons: self
                .pressed_mouse_buttons
                .into_iter()
                .filter_map(|code| u8::try_from(code).ok())
                .collect(),
            pointer_position: pair(&self.pointer_position),
            pointer_delta: pair(&self.pointer_delta),
            wheel_delta: self.wheel_delta as f32,
        }
    }
}

fn parse_browser_input(bytes: &[u8]) -> Option<crate::ipc::BrowserInput> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    // WS clients may double-encode the snapshot as a JSON string.
    let value = if let Some(inner) = value.as_str() {
        serde_json::from_str::<serde_json::Value>(inner).ok()?
    } else {
        value
    };
    serde_json::from_value::<BrowserInputWire>(value)
        .ok()
        .map(BrowserInputWire::into_input)
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

/// Convert one event to its canonical externally-tagged JSON value.
fn event_value(event: &GameEvent) -> serde_json::Value {
    match event {
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
        GameEvent::EventGap { after, oldest } => serde_json::json!({
            "EventGap": {
                "after": after,
                "oldest": oldest,
            }
        }),
    }
}

/// Serialize events as valid JSON, escaping all string fields through
/// `serde_json` and preserving canonical object payloads when `json_data`
/// contains JSON. Malformed payload strings remain valid JSON strings rather
/// than corrupting the entire `/api/events` response.
#[cfg(test)]
fn format_events(events: &[GameEvent]) -> String {
    let values: Vec<serde_json::Value> = events.iter().map(event_value).collect();
    serde_json::to_string(&values).expect("event values are serializable")
}

/// Serialize replay records with a transport `sequence` sibling next to the
/// canonical event variant, keeping existing consumers' `ev.CustomEvent`
/// shape intact.
fn format_event_records(records: &[EventRecord]) -> String {
    let values: Vec<serde_json::Value> = records
        .iter()
        .map(|record| {
            let mut value = event_value(&record.event);
            if let Some(object) = value.as_object_mut() {
                object.insert("sequence".into(), serde_json::json!(record.sequence));
            }
            value
        })
        .collect();
    serde_json::to_string(&values).expect("event record values are serializable")
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
                request_id: RequestId::new(4),
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
            EventSeq::new(17),
        ))
        .expect("sequenced snapshot must be valid JSON");
        assert_eq!(value["version"], 9);
        assert_eq!(value["sequence"], 17);
        assert_eq!(value["entities"], serde_json::json!([]));
        assert_eq!(add_sequence("not-json", EventSeq::new(4)), "not-json");
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

    #[test]
    fn parse_set_component_rejects_u32_overflow_without_truncation() {
        // `4_294_967_296` truncated with `as u32` would silently become `0`;
        // the typed payload rejects it instead.
        let data = serde_json::json!({
            "id": 4_294_967_296u64,
            "component": "T",
            "value": {}
        });
        assert!(parse_set_component(Some(&data)).is_none());
        let data = serde_json::json!({
            "id": 1u64,
            "generation": 4_294_967_296u64,
            "component": "T",
            "value": {}
        });
        assert!(parse_set_component(Some(&data)).is_none());
    }

    #[test]
    fn parse_browser_input_drops_out_of_range_codes_without_truncation() {
        // `4_294_967_296` truncated with `as u32` would silently become `0`
        // (a held Left mouse button); the typed wire drops it instead, while
        // valid codes and pointer pairs survive.
        let input = parse_browser_input(
            br#"{"pressed_keys":[17,4294967296],"pressed_mouse_buttons":[1,256],
                "pointer_position":[3.5,4.5],"pointer_delta":[0.5,-0.5],"wheel_delta":1.25}"#,
        )
        .expect("valid shape parses");
        assert_eq!(input.pressed_keys, vec![17]);
        assert_eq!(input.pressed_mouse_buttons, vec![1]);
        assert_eq!(input.pointer_position, [3.5, 4.5]);
        assert_eq!(input.pointer_delta, [0.5, -0.5]);
        assert_eq!(input.wheel_delta, 1.25);
        // Missing fields default; double-encoded WS strings still parse.
        let minimal = parse_browser_input(br#"{}"#).expect("empty object parses");
        assert_eq!(minimal, crate::ipc::BrowserInput::default());
        let encoded = parse_browser_input(br#""{\"pressed_keys\":[87]}""#)
            .expect("string-wrapped snapshot parses");
        assert_eq!(encoded.pressed_keys, vec![87]);
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

    // ── parse_command_payload ──────────────────────────────────────────────

    #[test]
    fn parse_command_payload_accepts_valid_and_rejects_garbage() {
        let valid = parse_command_payload(
            r#"{"type":"set_component","data":{"id":5,"component":"T","value":{}}}"#,
        );
        assert!(matches!(
            valid,
            Some(UiCommand::SetComponent { entity_id: 5, .. })
        ));

        let custom = parse_command_payload(r#"{"type":"ping"}"#);
        assert!(matches!(custom, Some(UiCommand::Custom { .. })));

        assert!(parse_command_payload("this is not json").is_none());
        assert!(parse_command_payload(r#"{"foo":1}"#).is_none());
        assert!(parse_command_payload(r#"{"type":7}"#).is_none());
        assert!(parse_command_payload(r#"{"type":"set_component","data":{"id":1}}"#).is_none());
    }

    // ── post_command ───────────────────────────────────────────────────────

    #[test]
    fn post_command_valid_forwards_to_game_with_ack() {
        let (tx, rx) = unbounded::<UiCommand>();
        let ack = post_command(
            r#"{"type":"set_component","data":{"id":5,"component":"T","value":{}}}"#,
            &tx,
            RequestId::new(42),
        );
        assert_eq!(
            ack,
            CommandAck {
                request_id: RequestId::new(42),
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
        let invalid_json = post_command("this is not json", &tx, RequestId::new(1));
        assert!(!invalid_json.accepted);
        assert_eq!(invalid_json.request_id, 1);
        // JSON but no "type"
        let missing_type = post_command(r#"{"foo":1}"#, &tx, RequestId::new(2));
        assert!(!missing_type.accepted);
        // JSON with unknown type (still a Custom, not dropped)
        let accepted = post_command(r#"{"type":"unknown","data":{}}"#, &tx, RequestId::new(3));
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
        let mut next = RequestId::new(1);
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
            request_id: RequestId::new(7),
            accepted: true,
            error: None,
        }))
        .expect("accepted ack is valid JSON");
        assert_eq!(accepted["accepted"], true);
        assert_eq!(accepted["request_id"], 7);
        assert!(accepted.get("error").is_none());

        let rejected = serde_json::from_str::<serde_json::Value>(&command_ack_json(&CommandAck {
            request_id: RequestId::new(8),
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

    #[test]
    fn event_cursor_reads_optional_after_query() {
        assert_eq!(event_cursor("/api/events"), EventSeq::new(0));
        assert_eq!(event_cursor("/api/events?after=41"), EventSeq::new(41));
        assert_eq!(event_cursor("/api/events?foo=1&after=9"), EventSeq::new(9));
        assert_eq!(event_cursor("/api/events?after=bad"), EventSeq::new(0));
    }

    #[test]
    fn event_log_replays_after_cursor_without_consuming_history() {
        let mut log = EventLog::default();
        log.push(GameEvent::EntityCreated { entity_id: 1 });
        log.push(GameEvent::CommandCompleted {
            request_id: RequestId::new(7),
            command: "create_entity".into(),
            success: true,
            error: None,
        });

        let first = log.after(EventSeq::new(0));
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].sequence, EventSeq::new(1));
        assert_eq!(first[1].sequence, EventSeq::new(2));
        assert!(matches!(
            log.after(EventSeq::new(1)).as_slice(),
            [record] if record.sequence.get() == 2
                && matches!(
                    record.event,
                    GameEvent::CommandCompleted { request_id, .. } if request_id.get() == 7
                )
        ));
        assert!(log.after(EventSeq::new(2)).is_empty());
        assert_eq!(log.records.len(), 2, "replay must not drain history");
    }

    #[test]
    fn event_log_reports_gap_after_bounded_history_rollover() {
        let mut log = EventLog::default();
        for entity_id in 0..(EVENT_HISTORY_CAPACITY as u32 + 2) {
            log.push(GameEvent::EntityCreated { entity_id });
        }

        let replay = log.after(EventSeq::new(0));
        assert_eq!(replay.len(), EVENT_HISTORY_CAPACITY + 1);
        assert!(matches!(
            replay[0],
            EventRecord {
                event: GameEvent::EventGap { .. },
                ..
            } if replay[0].sequence.get() == 2
                && matches!(
                    replay[0].event,
                    GameEvent::EventGap { after, oldest }
                        if after.get() == 0 && oldest.get() == 3
                )
        ));
        assert_eq!(replay[1].sequence, EventSeq::new(3));
    }

    #[test]
    fn event_records_keep_legacy_event_shape_and_add_cursor_metadata() {
        let records = vec![EventRecord {
            sequence: EventSeq::new(9),
            event: GameEvent::EntityCreated { entity_id: 4 },
        }];
        let value = serde_json::from_str::<serde_json::Value>(&format_event_records(&records))
            .expect("event records must be valid JSON");
        assert_eq!(value[0]["sequence"], 9);
        assert_eq!(value[0]["EntityCreated"]["entity_id"], 4);
    }

    // ── sniff_route ────────────────────────────────────────────────────────

    #[test]
    fn sniff_route_splits_events_stream_from_plain_http() {
        let stream = b"GET /api/events?after=41 HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
        assert_eq!(
            sniff_route(stream),
            Some(ConnectionRoute::EventsStream {
                cursor: EventSeq::new(41)
            })
        );
        // Same path without the upgrade header → plain HTTP (the polling
        // fallback serves it).
        let http =
            b"GET /api/events?after=41 HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n";
        assert_eq!(sniff_route(http), Some(ConnectionRoute::Http));
        // Other paths never upgrade, even with the header present.
        let other = b"GET /api/scene HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n\r\n";
        assert_eq!(sniff_route(other), Some(ConnectionRoute::Http));
        // A lookalike path prefix must not upgrade either.
        let prefix = b"GET /api/events2 HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n\r\n";
        assert_eq!(sniff_route(prefix), Some(ConnectionRoute::Http));
        // Header names and values are case-insensitive; no cursor → zero.
        let mixed = b"GET /api/events HTTP/1.1\r\nHost: x\r\nuPgRaDe: WeBsOcKeT\r\n\r\n";
        assert_eq!(
            sniff_route(mixed),
            Some(ConnectionRoute::EventsStream {
                cursor: EventSeq::new(0)
            })
        );
        // Incomplete head → None (the caller peeks for more bytes).
        assert_eq!(
            sniff_route(b"GET /api/events?after=1 HTTP/1.1\r\nHost: x\r\n"),
            None
        );
        // Complete garbage → None as well (bounded by the caller's timeout).
        assert_eq!(sniff_route(b"\r\n\r\n"), None);
    }

    // ── tungstenite wire bytes ───────────────────────────────────────────
    // The transport changed, the wire format must not: assert tungstenite
    // emits byte-identical frames for the text/ping/close messages the
    // server sends.

    /// Write-only byte sink: tungstenite's `send` path never reads, so
    /// `Read` always reports no data.
    struct VecSink(Vec<u8>);

    impl std::io::Read for VecSink {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "write-only sink",
            ))
        }
    }

    impl std::io::Write for VecSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn tungstenite_bytes(message: Message) -> Vec<u8> {
        let mut ws = WebSocket::from_raw_socket(
            VecSink(Vec::new()),
            tungstenite::protocol::Role::Server,
            None,
        );
        ws.send(message).expect("send into sink");
        ws.into_inner().0
    }

    #[test]
    fn tungstenite_text_frame_encodes_short_and_extended_lengths() {
        assert_eq!(
            tungstenite_bytes(Message::text("hello")),
            vec![0x81, 5, b'h', b'e', b'l', b'l', b'o']
        );

        let medium = tungstenite_bytes(Message::text("x".repeat(126)));
        assert_eq!(&medium[..4], &[0x81, 126, 0, 126]);
        assert_eq!(medium.len(), 4 + 126);
    }

    #[test]
    fn tungstenite_control_frames_encode_ping_and_away_close() {
        assert_eq!(
            tungstenite_bytes(Message::Ping(Bytes::new())),
            vec![0x89, 0]
        );

        assert_eq!(
            tungstenite_bytes(Message::Close(Some(CloseFrame {
                code: CloseCode::Away,
                reason: Utf8Bytes::from_static(""),
            }))),
            vec![0x88, 2, 0x03, 0xe9]
        );
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

        let mut buffer = EventLog::default();
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
        assert_eq!(snaps.sequence, EventSeq::new(2));
        assert_eq!(buffer.records.len(), 1);
        assert_eq!(buffer.records[0].sequence, EventSeq::new(1));
        assert!(matches!(
            buffer.records[0].event,
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

    // ── POST guards: Origin / Host / Content-Type / body limit ────────────

    #[test]
    fn origin_decision_allows_editor_and_non_browser_clients() {
        assert_eq!(decide_origin(None, 3420), OriginDecision::Allow);
        assert_eq!(
            decide_origin(Some("http://127.0.0.1:3420"), 3420),
            OriginDecision::Allow
        );
        assert_eq!(
            decide_origin(Some("http://localhost:3420"), 3420),
            OriginDecision::Allow
        );
    }

    #[test]
    fn origin_decision_denies_foreign_pages() {
        assert_eq!(
            decide_origin(Some("http://evil.com"), 3420),
            OriginDecision::Deny
        );
        assert_eq!(
            decide_origin(Some("http://127.0.0.1:9999"), 3420),
            OriginDecision::Deny
        );
        assert_eq!(
            decide_origin(Some("https://localhost:3420"), 3420),
            OriginDecision::Deny
        );
        assert_eq!(decide_origin(Some("null"), 3420), OriginDecision::Deny);
    }

    #[test]
    fn host_decision_denies_rebinding() {
        assert_eq!(decide_host(None, 3420), OriginDecision::Allow);
        assert_eq!(
            decide_host(Some("127.0.0.1:3420"), 3420),
            OriginDecision::Allow
        );
        assert_eq!(
            decide_host(Some("localhost:3420"), 3420),
            OriginDecision::Allow
        );
        assert_eq!(decide_host(Some("evil.com"), 3420), OriginDecision::Deny);
        assert_eq!(
            decide_host(Some("127.0.0.1:9999"), 3420),
            OriginDecision::Deny
        );
    }

    #[test]
    fn content_type_requires_json() {
        assert!(check_content_type(Some("application/json")).is_ok());
        assert!(check_content_type(Some("application/json; charset=utf-8")).is_ok());
        assert!(check_content_type(Some("Application/JSON")).is_ok());
        assert_eq!(
            check_content_type(Some("text/plain")),
            Err(ApiGuardError::UnsupportedMediaType)
        );
        assert_eq!(
            check_content_type(Some("application/x-www-form-urlencoded")),
            Err(ApiGuardError::UnsupportedMediaType)
        );
        assert_eq!(
            check_content_type(None),
            Err(ApiGuardError::UnsupportedMediaType)
        );
    }

    #[test]
    fn guard_errors_map_to_fixed_status_codes() {
        assert_eq!(ApiGuardError::ForbiddenOrigin.status_code(), 403);
        assert_eq!(ApiGuardError::ForbiddenHost.status_code(), 403);
        assert_eq!(ApiGuardError::UnsupportedMediaType.status_code(), 415);
        assert_eq!(ApiGuardError::PayloadTooLarge.status_code(), 413);
    }

    #[test]
    fn limited_body_read_accepts_small_and_rejects_oversized() {
        let mut small = std::io::Cursor::new(b"{}".to_vec());
        assert_eq!(
            read_limited_body(&mut small, MAX_COMMAND_BYTES).expect("small body"),
            "{}"
        );
        let big = vec![b'a'; MAX_COMMAND_BYTES + 1];
        let mut cursor = std::io::Cursor::new(big);
        assert_eq!(
            read_limited_body(&mut cursor, MAX_COMMAND_BYTES),
            Err(ApiGuardError::PayloadTooLarge)
        );
    }

    // ── ScenePath sandbox ────────────────────────────────────────────────

    fn sandbox_roots(tag: &str) -> (SceneRoots, PathBuf) {
        let dir = std::env::temp_dir().join(format!("ornis-sandbox-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("assets")).expect("sandbox assets");
        fs::create_dir_all(dir.join("editor")).expect("sandbox editor");
        (SceneRoots::new(&dir), dir)
    }

    #[test]
    fn scene_path_accepts_assets_and_editor_paths() {
        let (roots, dir) = sandbox_roots("accept");
        for raw in [
            "editor/scene.ron",
            "assets/scene.ron",
            "assets/sub/nested.ron",
        ] {
            let resolved = ScenePath::resolve(&roots, raw).expect("legit path resolves");
            assert!(resolved.as_path().is_absolute(), "{raw}");
        }
        let absolute = dir.join("editor").join("scene.ron");
        let resolved =
            ScenePath::resolve(&roots, &absolute.to_string_lossy()).expect("absolute inside");
        assert_eq!(resolved.as_path(), absolute);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scene_path_rejects_traversal_and_outside_absolute_paths() {
        let (roots, dir) = sandbox_roots("reject");
        for raw in [
            "../secret.ron",
            "../../etc/passwd",
            "editor/../../secret.ron",
            "assets/../editor/../../secret.ron",
            "/etc/passwd",
            "",
        ] {
            assert!(
                matches!(
                    ScenePath::resolve(&roots, raw),
                    Err(ScenePathError::OutsideSandbox { .. } | ScenePathError::Empty)
                ),
                "{raw} must be rejected"
            );
        }
        let outside = std::env::temp_dir().join("ornis-sandbox-outside.ron");
        assert!(matches!(
            ScenePath::resolve(&roots, &outside.to_string_lossy()),
            Err(ScenePathError::OutsideSandbox { .. })
        ));
        // A sibling that merely shares a name prefix is not inside.
        let evil = format!("{}-evil/x.ron", dir.join("editor").display());
        assert!(matches!(
            ScenePath::resolve(&roots, &evil),
            Err(ScenePathError::OutsideSandbox { .. })
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_command_drops_escaping_scene_paths_fail_fast() {
        let traversal = serde_json::json!({"path": "../../secret.ron"});
        assert!(build_command("save_scene", Some(&traversal)).is_none());
        let absolute = serde_json::json!({"path": "/etc/passwd"});
        assert!(build_command("load_scene", Some(&absolute)).is_none());
        // Legit and default paths still pass through to the session check.
        let legit = serde_json::json!({"path": "editor/scene.ron"});
        assert!(build_command("save_scene", Some(&legit)).is_some());
        assert!(build_command("load_scene", None).is_some());
        // Unrelated commands are untouched by the sandbox.
        let other = serde_json::json!({"path": "../../secret.ron"});
        assert!(build_command("ping", Some(&other)).is_some());
    }

    // ── helpers ────────────────────────────────────────────────────────────

    fn read_response(resp: Response<Cursor<Vec<u8>>>) -> String {
        let mut body = String::new();
        let mut reader = resp.into_reader();
        let _ = reader.read_to_string(&mut body);
        body
    }
}
