//! Remote editor transport: HTTP server and asset sync for live editing.

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use tiny_http::{Header, ReadWrite, Request, Response, Server};

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
/// `/api/*` endpoints from background threads; `stop` shuts them down and joins.
pub struct RemoteEditor {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    websocket_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
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
                    websocket_handles: Arc::new(Mutex::new(Vec::new())),
                };
            }
        };

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let websocket_handles = Arc::new(Mutex::new(Vec::new()));
        let ws_port = port;
        let websocket_handles_for_server = Arc::clone(&websocket_handles);
        let handle = thread::Builder::new()
            .name("remote-editor".into())
            .spawn(move || {
                serve(
                    server,
                    stop_clone,
                    game_tx,
                    game_rx,
                    websocket_handles_for_server,
                    ws_port,
                )
            })
            .expect("spawn remote-editor thread");

        eprintln!("ornis: remote editor at http://{addr}");
        Self {
            stop,
            handle: Some(handle),
            websocket_handles,
        }
    }

    /// Signal shutdown and join the server thread.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
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

const EVENT_HISTORY_CAPACITY: usize = 256;

/// One user-facing event plus its server-side replay sequence.
#[derive(Debug, Clone)]
struct EventRecord {
    sequence: u64,
    event: GameEvent,
}

/// Bounded replay log for `/api/events?after=<sequence>`.
///
/// Snapshot cache updates use their own sequence field; this cursor covers
/// command/completion/domain events that are returned by `/api/events`.
#[derive(Debug)]
struct EventLog {
    records: VecDeque<EventRecord>,
    next_sequence: u64,
}

impl Default for EventLog {
    fn default() -> Self {
        Self {
            records: VecDeque::new(),
            next_sequence: 1,
        }
    }
}

impl EventLog {
    fn push(&mut self, event: GameEvent) {
        let sequence = self.next_sequence.max(1);
        self.next_sequence = sequence.saturating_add(1);
        if self.records.len() == EVENT_HISTORY_CAPACITY {
            self.records.pop_front();
        }
        self.records.push_back(EventRecord { sequence, event });
    }

    fn after(&self, cursor: u64) -> Vec<EventRecord> {
        let mut events = Vec::new();
        if let Some(first) = self.records.front()
            && cursor.saturating_add(1) < first.sequence
        {
            events.push(EventRecord {
                sequence: first.sequence.saturating_sub(1),
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

fn serve(
    server: Server,
    stop: Arc<AtomicBool>,
    game_tx: Sender<UiCommand>,
    game_rx: Receiver<GameEvent>,
    websocket_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    server_port: u16,
) {
    let event_log = Arc::new(Mutex::new(EventLog::default()));
    let mut snapshots = Snapshots::default();
    let mut next_request_id = 1_u64;
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

        if is_websocket_request(&request) && request.url().split('?').next() == Some("/api/events")
        {
            let cursor = event_cursor(request.url());
            let event_log = Arc::clone(&event_log);
            let stop = Arc::clone(&stop);
            let game_tx_ws = game_tx.clone();
            if let Ok(handle) = thread::Builder::new()
                .name("remote-editor-websocket".into())
                .spawn(move || {
                    serve_websocket(request, event_log, stop, cursor, server_port, game_tx_ws)
                })
            {
                websocket_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(handle);
            }
            continue;
        }

        let response = route_request(
            &root,
            &mut request,
            &event_log,
            &snapshots,
            &game_tx,
            &mut next_request_id,
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

fn is_websocket_request(request: &Request) -> bool {
    header_value(request, "Upgrade").is_some_and(|value| value.eq_ignore_ascii_case("websocket"))
}

fn websocket_bad_request(request: Request) {
    let _ =
        request.respond(Response::from_string("WebSocket upgrade required").with_status_code(400));
}

/// Serve a WebSocket `/api/events` connection. The endpoint is bidirectional:
/// server pushes replay records after the initial cursor and then newly
/// appended records; the browser may also push `InputState` snapshots as text
/// frames that are forwarded to the game thread via `game_tx`. Idle
/// connections receive periodic ping frames, and server shutdown sends a
/// normal close frame.
fn serve_websocket(
    request: Request,
    event_log: Arc<Mutex<EventLog>>,
    stop: Arc<AtomicBool>,
    mut cursor: u64,
    server_port: u16,
    game_tx: Sender<UiCommand>,
) {
    let Some(key) = header_value(&request, "Sec-WebSocket-Key") else {
        websocket_bad_request(request);
        return;
    };
    if header_value(&request, "Sec-WebSocket-Version").as_deref() != Some("13") {
        websocket_bad_request(request);
        return;
    }
    let accept = websocket_accept(&key);
    let response = Response::new_empty(tiny_http::StatusCode(101))
        .with_header(Header::from_bytes("Upgrade", "websocket").unwrap())
        .with_header(Header::from_bytes("Connection", "Upgrade").unwrap())
        .with_header(Header::from_bytes("Sec-WebSocket-Accept", accept).unwrap());
    let mut stream = request.upgrade("websocket", response);
    #[cfg(unix)]
    ws_set_read_timeout(&mut *stream, Duration::from_millis(10), server_port);
    let mut last_heartbeat = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        // Poll for client frames without blocking the push loop. Data frames
        // (opcode 0x1/0x2) carry browser `InputState` snapshots that are
        // forwarded to the game thread.
        match poll_client_frames(&mut *stream, &game_tx) {
            Ok(true) => return,
            Ok(false) => {}
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                // No frame available — continue to push.
            }
            Err(_) => return,
        }
        let records = event_log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .after(cursor);
        if let Some(last) = records.last() {
            cursor = last.sequence;
            let payload = format_event_records(&records);
            if write_websocket_text(&mut stream, &payload).is_err() {
                return;
            }
        }
        if last_heartbeat.elapsed() >= Duration::from_secs(15) {
            if write_websocket_ping(&mut stream).is_err() {
                return;
            }
            last_heartbeat = Instant::now();
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = write_websocket_close(&mut stream, 1001);
}

/// Poll the upgraded stream for any available client frames. Returns
/// `Ok(true)` if the connection should be closed (client sent close or
/// EOF), `Ok(false)` if no close was seen, or `Err` on I/O error.
/// Handles control frames and forwards text/binary data frames as
/// `BrowserInput` snapshots to `game_tx` (WS input channel, no polling).
fn poll_client_frames(stream: &mut dyn ReadWrite, game_tx: &Sender<UiCommand>) -> io::Result<bool> {
    loop {
        match poll_one_frame(stream, game_tx) {
            Ok(Some(true)) => return Ok(true),
            Ok(Some(false)) => continue,
            Ok(None) => return Ok(false),
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                return Ok(false);
            }
            Err(e) => return Err(e),
        }
    }
}

fn poll_one_frame(
    stream: &mut dyn ReadWrite,
    game_tx: &Sender<UiCommand>,
) -> io::Result<Option<bool>> {
    let Some((opcode, masked, len_byte)) = ws_read_header(stream)? else {
        return Ok(None);
    };
    // ponytail: helper returns Err(WouldBlock) on timeout during ext reads — bubbled as Ok(false) above
    let payload_len = ws_decode_len(stream, len_byte)?;
    if payload_len == usize::MAX {
        return Ok(Some(true));
    }
    let mask = ws_read_mask(stream, masked)?;
    let payload = ws_read_payload(stream, payload_len, mask)?;
    // Browser input channel: text/binary frames carry BrowserInput JSON.
    if matches!(opcode, 0x1 | 0x2) {
        if let Some(input) = parse_browser_input(&payload) {
            let _ = game_tx.send(UiCommand::Input { input });
        }
        return Ok(Some(false));
    }
    Ok(Some(ws_dispatch_frame(stream, opcode, &payload)?))
}

fn ws_read_header(stream: &mut dyn ReadWrite) -> io::Result<Option<(u8, bool, u8)>> {
    let mut hdr = [0u8; 2];
    match stream.read(&mut hdr) {
        Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof")),
        Ok(1) => {
            let mut second = [0u8; 1];
            match stream.read(&mut second) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof")),
                Ok(1) => hdr[1] = second[0],
                Ok(_) => unreachable!(),
                Err(e)
                    if e.kind() == io::ErrorKind::WouldBlock
                        || e.kind() == io::ErrorKind::TimedOut =>
                {
                    return Ok(None);
                }
                Err(e) => return Err(e),
            }
        }
        Ok(2) => {}
        Ok(_) => unreachable!(),
        Err(e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut => {
            return Ok(None);
        }
        Err(e) => return Err(e),
    }
    Ok(Some((hdr[0] & 0x0F, hdr[1] & 0x80 != 0, hdr[1] & 0x7F)))
}

fn ws_decode_len(stream: &mut dyn ReadWrite, len_byte: u8) -> io::Result<usize> {
    match len_byte {
        126 => {
            let mut ext = [0u8; 2];
            read_exact_with_timeout(stream, &mut ext)?;
            Ok(u16::from_be_bytes(ext) as usize)
        }
        127 => {
            let mut ext = [0u8; 8];
            read_exact_with_timeout(stream, &mut ext)?;
            let len = u64::from_be_bytes(ext) as usize;
            if len > 1_048_576 {
                Ok(usize::MAX)
            } else {
                Ok(len)
            }
        }
        n => Ok(n as usize),
    }
}

fn ws_read_mask(stream: &mut dyn ReadWrite, masked: bool) -> io::Result<Option<[u8; 4]>> {
    if !masked {
        return Ok(None);
    }
    let mut key = [0u8; 4];
    read_exact_with_timeout(stream, &mut key)?;
    Ok(Some(key))
}

fn ws_read_payload(
    stream: &mut dyn ReadWrite,
    len: usize,
    mask: Option<[u8; 4]>,
) -> io::Result<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut buf = vec![0u8; len];
    read_exact_with_timeout(stream, &mut buf)?;
    if let Some(key) = mask {
        for (i, b) in buf.iter_mut().enumerate() {
            *b ^= key[i % 4];
        }
    }
    Ok(buf)
}

fn ws_dispatch_frame(stream: &mut dyn ReadWrite, opcode: u8, payload: &[u8]) -> io::Result<bool> {
    match opcode {
        0x8 => {
            let code = if payload.len() >= 2 {
                u16::from_be_bytes([payload[0], payload[1]])
            } else {
                1000
            };
            if payload.len() >= 2 {
                let _ = write_websocket_frame(stream, 0x88, payload);
            } else {
                let _ = write_websocket_close(stream, code);
            }
            Ok(true)
        }
        0x9 => {
            let _ = write_websocket_frame(stream, 0x8A, payload);
            Ok(false)
        }
        0xA => Ok(false),
        // 0x1/0x2 handled in poll_one_frame as BrowserInput
        _ => Ok(false),
    }
}

#[cfg(unix)]
fn ws_set_read_timeout(stream: &mut dyn ReadWrite, timeout: Duration, server_port: u16) {
    // ponytail: only touch fds whose local port == server port — avoids poisoning client socket in tests
    unsafe {
        use std::os::unix::io::FromRawFd;
        let ptr = &mut *stream as *mut dyn ReadWrite as *mut *mut u8;
        let data = *ptr;
        let mut seen = std::collections::HashSet::new();
        for offset in (0..256).step_by(4) {
            let fd = *(data.add(offset) as *const i32);
            if !(3..512).contains(&fd) || !seen.insert(fd) {
                continue;
            }
            let tcp = std::net::TcpStream::from_raw_fd(fd);
            let ok = tcp
                .local_addr()
                .map(|a| a.port() == server_port)
                .unwrap_or(false)
                && tcp.peer_addr().is_ok();
            if ok {
                let _ = tcp.set_read_timeout(Some(timeout));
            }
            std::mem::forget(tcp);
        }
    }
}
#[cfg(not(unix))]
fn ws_set_read_timeout(_stream: &mut dyn ReadWrite, _timeout: Duration, _server_port: u16) {}

fn read_exact_with_timeout(stream: &mut dyn ReadWrite, buf: &mut [u8]) -> io::Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        match stream.read(&mut buf[offset..]) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof")),
            Ok(n) => offset += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn write_websocket_text<W: Write + ?Sized>(stream: &mut W, payload: &str) -> io::Result<()> {
    write_websocket_frame(stream, 0x81, payload.as_bytes())
}

fn write_websocket_ping<W: Write + ?Sized>(stream: &mut W) -> io::Result<()> {
    write_websocket_frame(stream, 0x89, &[])
}

fn write_websocket_close<W: Write + ?Sized>(stream: &mut W, code: u16) -> io::Result<()> {
    write_websocket_frame(stream, 0x88, &code.to_be_bytes())
}

fn write_websocket_frame<W: Write + ?Sized>(
    stream: &mut W,
    first_byte: u8,
    payload: &[u8],
) -> io::Result<()> {
    let length = payload.len();
    let mut frame = Vec::with_capacity(length + 10);
    frame.push(first_byte);
    if length <= 125 {
        frame.push(length as u8);
    } else if length <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(length as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(length as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    stream.write_all(&frame)?;
    stream.flush()
}

fn websocket_accept(key: &str) -> String {
    let mut input = key.as_bytes().to_vec();
    input.extend_from_slice(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64_encode(&sha1_digest(&input))
}

fn sha1_digest(input: &[u8]) -> [u8; 20] {
    let mut message = input.to_vec();
    let bit_length = (message.len() as u64).saturating_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = [
        0x67452301_u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];
    for chunk in message.chunks_exact(64) {
        let mut words = [0_u32; 80];
        for (index, word) in words[..16].iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }

        let [mut a, mut b, mut c, mut d, mut e] = state;
        for (index, &word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a827999),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let temporary = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temporary;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
    }

    let mut digest = [0_u8; 20];
    for (index, word) in state.into_iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut index = 0;
    while index < bytes.len() {
        let first = bytes[index];
        let second = bytes.get(index + 1).copied();
        let third = bytes.get(index + 2).copied();
        output.push(TABLE[(first >> 2) as usize] as char);
        output.push(TABLE[((first & 0x03) << 4 | second.unwrap_or(0) >> 4) as usize] as char);
        output.push(match second {
            Some(second) => {
                TABLE[((second & 0x0f) << 2 | third.unwrap_or(0) >> 6) as usize] as char
            }
            None => '=',
        });
        output.push(match third {
            Some(third) => TABLE[(third & 0x3f) as usize] as char,
            None => '=',
        });
        index += 3;
    }
    output
}

/// Drain incoming game events into `buffer`. "status"/"scene" snapshots only
/// refresh the endpoint caches — they are not user-facing events, so they
/// skip the buffer.
fn drain_game_events(game_rx: &Receiver<GameEvent>, buffer: &mut EventLog, snaps: &mut Snapshots) {
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
    buffer: &Arc<Mutex<EventLog>>,
    snapshots: &Snapshots,
    game_tx: &Sender<UiCommand>,
    next_request_id: &mut u64,
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
            let mut body = String::new();
            let _ = request.as_reader().read_to_string(&mut body);
            let request_id = command_request_id(&body, next_request_id);
            let ack = post_command(&body, game_tx, request_id);
            let ack_body = command_ack_json(&ack);
            json_response(&ack_body)
        }
        ("POST", "/api/input") => {
            let mut body = String::new();
            let _ = request.as_reader().read_to_string(&mut body);
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

fn event_cursor(url: &str) -> u64 {
    url.split_once('?')
        .and_then(|(_, query)| {
            query.split('&').find_map(|part| {
                let (key, value) = part.split_once('=')?;
                (key == "after")
                    .then(|| value.parse::<u64>().ok())
                    .flatten()
            })
        })
        .unwrap_or(0)
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

fn parse_browser_input(bytes: &[u8]) -> Option<crate::ipc::BrowserInput> {
    let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let obj = if let Some(s) = v.as_str() {
        serde_json::from_str::<serde_json::Value>(s).ok()?
    } else {
        v
    };
    let o = obj.as_object()?;
    let pressed_keys = o
        .get("pressed_keys")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();
    let pressed_mouse_buttons = o
        .get("pressed_mouse_buttons")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u8))
                .collect()
        })
        .unwrap_or_default();
    let pointer_position = o
        .get("pointer_position")
        .and_then(|v| v.as_array())
        .map(|a| {
            let x = a.first().and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            let y = a.get(1).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            [x, y]
        })
        .unwrap_or([0.0, 0.0]);
    let pointer_delta = o
        .get("pointer_delta")
        .and_then(|v| v.as_array())
        .map(|a| {
            let x = a.first().and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            let y = a.get(1).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
            [x, y]
        })
        .unwrap_or([0.0, 0.0]);
    let wheel_delta = o.get("wheel_delta").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    Some(crate::ipc::BrowserInput {
        pressed_keys,
        pressed_mouse_buttons,
        pointer_position,
        pointer_delta,
        wheel_delta,
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

    #[test]
    fn event_cursor_reads_optional_after_query() {
        assert_eq!(event_cursor("/api/events"), 0);
        assert_eq!(event_cursor("/api/events?after=41"), 41);
        assert_eq!(event_cursor("/api/events?foo=1&after=9"), 9);
        assert_eq!(event_cursor("/api/events?after=bad"), 0);
    }

    #[test]
    fn event_log_replays_after_cursor_without_consuming_history() {
        let mut log = EventLog::default();
        log.push(GameEvent::EntityCreated { entity_id: 1 });
        log.push(GameEvent::CommandCompleted {
            request_id: 7,
            command: "create_entity".into(),
            success: true,
            error: None,
        });

        let first = log.after(0);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].sequence, 1);
        assert_eq!(first[1].sequence, 2);
        assert!(matches!(
            log.after(1).as_slice(),
            [EventRecord {
                sequence: 2,
                event: GameEvent::CommandCompleted { request_id: 7, .. }
            }]
        ));
        assert!(log.after(2).is_empty());
        assert_eq!(log.records.len(), 2, "replay must not drain history");
    }

    #[test]
    fn event_log_reports_gap_after_bounded_history_rollover() {
        let mut log = EventLog::default();
        for entity_id in 0..(EVENT_HISTORY_CAPACITY as u32 + 2) {
            log.push(GameEvent::EntityCreated { entity_id });
        }

        let replay = log.after(0);
        assert_eq!(replay.len(), EVENT_HISTORY_CAPACITY + 1);
        assert!(matches!(
            replay[0],
            EventRecord {
                sequence: 2,
                event: GameEvent::EventGap {
                    after: 0,
                    oldest: 3
                }
            }
        ));
        assert_eq!(replay[1].sequence, 3);
    }

    #[test]
    fn event_records_keep_legacy_event_shape_and_add_cursor_metadata() {
        let records = vec![EventRecord {
            sequence: 9,
            event: GameEvent::EntityCreated { entity_id: 4 },
        }];
        let value = serde_json::from_str::<serde_json::Value>(&format_event_records(&records))
            .expect("event records must be valid JSON");
        assert_eq!(value[0]["sequence"], 9);
        assert_eq!(value[0]["EntityCreated"]["entity_id"], 4);
    }

    #[test]
    fn websocket_accept_matches_rfc6455_example() {
        assert_eq!(
            websocket_accept("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn websocket_text_frame_encodes_short_and_extended_lengths() {
        let mut short = Vec::new();
        write_websocket_text(&mut short, "hello").expect("short frame");
        assert_eq!(short, vec![0x81, 5, b'h', b'e', b'l', b'l', b'o']);

        let mut medium = Vec::new();
        write_websocket_text(&mut medium, &"x".repeat(126)).expect("medium frame");
        assert_eq!(&medium[..4], &[0x81, 126, 0, 126]);
        assert_eq!(medium.len(), 4 + 126);
    }

    #[test]
    fn websocket_control_frames_encode_ping_and_normal_close() {
        let mut ping = Vec::new();
        write_websocket_ping(&mut ping).expect("ping frame");
        assert_eq!(ping, vec![0x89, 0]);

        let mut close = Vec::new();
        write_websocket_close(&mut close, 1001).expect("close frame");
        assert_eq!(close, vec![0x88, 2, 0x03, 0xe9]);
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
        assert_eq!(snaps.sequence, 2);
        assert_eq!(buffer.records.len(), 1);
        assert_eq!(buffer.records[0].sequence, 1);
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

    // ── helpers ────────────────────────────────────────────────────────────

    fn read_response(resp: Response<Cursor<Vec<u8>>>) -> String {
        let mut body = String::new();
        let mut reader = resp.into_reader();
        let _ = reader.read_to_string(&mut body);
        body
    }
}
