//! `ornis-shell` binary: native desktop window hosting the editor frontend.
//!
//! Production mode next to `editor-only` (`cargo run --features editor-only`
//! serves the editor over loopback HTTP on port 3420 for browser
//! development). Here no localhost server is started: the frontend loads
//! from the `ornis://app/…` custom scheme mapped onto the editor directory
//! on disk, and talks to the engine thread through the [`ornis_shell`]
//! IPC bridge (`window.ipc.postMessage` in, `ornis-ipc` custom events out).
//!
//! Run with:
//!
//! ```sh
//! cargo run -p ornis-shell [--editor-dir <path>]   # editor/ or editor-v2/
//! ORNIS_EDITOR_DIR=editor-v2 cargo run -p ornis-shell
//! cargo run -p ornis-shell -- --webgpu-probe       # WebGPU capability check
//! ```

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::Sender;
use editor_backend::ipc::{EventSeq, GameEvent, RequestId, UiCommand};
use ornis_shell::{
    AckOutcome, AssetError, BridgeError, ack_message_json, dispatch_script, events_message_json,
    init_script, parse_incoming, probe_page, read_asset, resolve_editor_dir, start_url,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};
use wry::{WebView, WebViewBuilder};

/// Initial window size (px).
const WINDOW_WIDTH: u32 = 1280;
/// Initial window size (px).
const WINDOW_HEIGHT: u32 = 800;

/// Cross-thread messages delivered to the event loop (all `WebView` calls
/// stay on the main thread).
#[derive(Debug)]
enum ShellEvent {
    /// Raw `window.ipc.postMessage` payload from the page.
    Ipc(String),
    /// Engine event drained from `ev_rx` by the pump thread.
    Engine(GameEvent),
}

/// Parsed CLI options (see the module docs for usage).
struct Options {
    editor_dir: Option<String>,
    webgpu_probe: bool,
}

fn parse_options() -> Options {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        editor_dir: None,
        webgpu_probe: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            // Skip the conventional separator (`cargo run -p … -- <flags>`).
            "--" => {}
            "--editor-dir" => {
                options.editor_dir = args.next();
            }
            "--webgpu-probe" => options.webgpu_probe = true,
            "--help" | "-h" => {
                println!(
                    "ornis-shell: native editor window\n\n  --editor-dir <path>  frontend root (default: ORNIS_EDITOR_DIR or <workspace>/editor)\n  --webgpu-probe       load the inline WebGPU capability page instead of the editor"
                );
                std::process::exit(0);
            }
            _ => eprintln!("ornis-shell: ignoring unknown argument {arg}"),
        }
    }
    options
}

fn main() {
    let options = parse_options();
    let editor_root = resolve_editor_dir(
        options.editor_dir.as_deref(),
        std::env::var("ORNIS_EDITOR_DIR").ok().as_deref(),
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );

    // Live ECS world on a dedicated thread (same entry point as
    // `editor-only`): executes queued commands, publishes events.
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<UiCommand>();
    let (ev_tx, ev_rx) = crossbeam_channel::unbounded::<GameEvent>();
    let _world = ornis_app::session::run(cmd_rx, ev_tx);

    if !options.webgpu_probe && !editor_root.is_dir() {
        eprintln!(
            "ornis-shell: editor root {} is not a directory \
             (pass --editor-dir or set ORNIS_EDITOR_DIR)",
            editor_root.display()
        );
        std::process::exit(2);
    }

    eprintln!(
        "ornis-shell: webview engine {}",
        wry::webview_version().unwrap_or_else(|_| "unknown".to_owned())
    );
    if options.webgpu_probe {
        eprintln!("ornis-shell: WebGPU probe mode (inline page, no editor files)");
    } else {
        eprintln!(
            "ornis-shell: serving {} as ornis://app/",
            editor_root.display()
        );
    }

    let mut loop_builder = EventLoop::<ShellEvent>::with_user_event();
    let event_loop = loop_builder.build().expect("shell event loop");
    let pump = event_loop.create_proxy();
    std::thread::Builder::new()
        .name("shell-event-pump".into())
        .spawn(move || {
            while let Ok(event) = ev_rx.recv() {
                if pump.send_event(ShellEvent::Engine(event)).is_err() {
                    break;
                }
            }
        })
        .expect("shell event pump thread");

    let mut app = ShellApp {
        proxy: event_loop.create_proxy(),
        window: None,
        webview: None,
        cmd_tx,
        next_request: RequestId::new(1),
        next_event: EventSeq::new(1),
        editor_root: Arc::new(editor_root),
        webgpu_probe: options.webgpu_probe,
    };
    event_loop.run_app(&mut app).expect("shell event loop runs");
}

/// Winit application owning the native window and the embedded webview.
struct ShellApp {
    /// Loop proxy; the IPC handler clones it to forward page posts.
    proxy: winit::event_loop::EventLoopProxy<ShellEvent>,
    window: Option<Window>,
    webview: Option<WebView>,
    cmd_tx: Sender<UiCommand>,
    next_request: RequestId,
    next_event: EventSeq,
    editor_root: Arc<PathBuf>,
    webgpu_probe: bool,
}

impl ShellApp {
    /// Forwards one page post to the engine and acks it back to the page.
    fn on_ipc(&mut self, body: String) {
        let post = parse_incoming(&body, &mut self.next_request);
        let (request_id, queued) = post.into_queued();
        let outcome = match queued {
            Ok(command) => match self.cmd_tx.send(command) {
                Ok(()) => AckOutcome::Accepted { request_id },
                Err(_) => AckOutcome::Rejected {
                    request_id,
                    error: BridgeError::Disconnected.to_string(),
                },
            },
            Err(error) => AckOutcome::Rejected {
                request_id,
                error: error.to_string(),
            },
        };
        self.eval(&dispatch_script(&ack_message_json(&outcome)));
    }

    /// Forwards one engine event to the page with a transport sequence.
    fn on_engine_event(&mut self, event: GameEvent) {
        let sequence = EventSeq::new(self.next_event.get().max(1));
        self.next_event = EventSeq::new(sequence.get().saturating_add(1));
        self.eval(&dispatch_script(&events_message_json(&[(sequence, event)])));
    }

    /// Runs JS in the page; failures (page gone) are ignored — the loop
    /// keeps serving the engine either way.
    fn eval(&self, script: &str) {
        if let Some(webview) = &self.webview {
            let _ = webview.evaluate_script(script);
        }
    }
}

/// Full-window bounds for the child webview (position origin, initial
/// size; kept in sync with the window in `window_event` below).
fn full_window_bounds() -> wry::Rect {
    wry::Rect {
        position: wry::dpi::LogicalPosition::new(0, 0).into(),
        size: wry::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT).into(),
    }
}

impl ApplicationHandler<ShellEvent> for ShellApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Ornis Editor")
                    .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            )
            .expect("shell window");
        let root = Arc::clone(&self.editor_root);
        let mut builder = WebViewBuilder::new()
            .with_initialization_script(init_script())
            .with_custom_protocol("ornis".into(), move |_, request| {
                serve_asset(&root, request.uri().path())
            });
        if self.webgpu_probe {
            builder = builder.with_html(probe_page());
        } else {
            builder = builder.with_url(start_url());
        }
        let ipc = self.proxy.clone();
        builder = builder.with_ipc_handler(move |request| {
            let _ = ipc.send_event(ShellEvent::Ipc(request.body().clone()));
        });
        // Child view, not a content-view takeover: `build` replaces the
        // window's content view with its own, which breaks winit's
        // `view()` accessor on focus loss (`windowDidResignKey` abort,
        // winit#4203 / wry#1477). As a child, winit keeps its own view and
        // the crash path is gone; bounds are synced on resize below.
        let webview = builder
            .with_bounds(full_window_bounds())
            .build_as_child(&window)
            .expect("shell webview");
        self.window = Some(window);
        self.webview = Some(webview);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: ShellEvent) {
        match event {
            ShellEvent::Ipc(body) => self.on_ipc(body),
            ShellEvent::Engine(event) => self.on_engine_event(event),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            // Keep the child webview filling the window (child views do
            // not track resizes by themselves).
            WindowEvent::Resized(size) => {
                if let Some(webview) = &self.webview {
                    let _ = webview.set_bounds(wry::Rect {
                        position: wry::dpi::LogicalPosition::new(0, 0).into(),
                        size: wry::dpi::LogicalSize::new(size.width, size.height).into(),
                    });
                }
            }
            _ => {}
        }
    }
}

/// Serves one `ornis://` request from the editor root on disk.
///
/// Missing or escaping paths answer 404 (the HTTP static server likewise
/// never distinguishes them), unreadable files 500.
fn serve_asset(root: &Path, url_path: &str) -> wry::http::Response<Cow<'static, [u8]>> {
    fn response(
        status: u16,
        content_type: &'static str,
        body: Vec<u8>,
    ) -> wry::http::Response<Cow<'static, [u8]>> {
        wry::http::Response::builder()
            .status(status)
            .header("Content-Type", content_type)
            .header("Access-Control-Allow-Origin", "*")
            .body(Cow::Owned(body))
            .unwrap_or_else(|_| {
                wry::http::Response::builder()
                    .status(500)
                    .body(Cow::Borrowed(b"internal error".as_slice()))
                    .expect("fallback error response builds")
            })
    }

    match read_asset(root, url_path) {
        Ok(asset) => response(200, asset.content_type, asset.bytes),
        Err(AssetError::NotFound { .. } | AssetError::Forbidden { .. }) => {
            response(404, "text/plain; charset=utf-8", b"404 Not Found".to_vec())
        }
        Err(AssetError::Io { .. }) => response(
            500,
            "text/plain; charset=utf-8",
            b"500 Cannot Read Asset".to_vec(),
        ),
    }
}
