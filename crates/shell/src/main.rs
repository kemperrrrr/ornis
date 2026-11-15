//! `ornis-shell` binary: native desktop window hosting the editor frontend.
//!
//! Production mode next to `editor-only` (`cargo run --features editor-only`
//! serves the editor over loopback HTTP on port 3420 for browser
//! development). Here no localhost server is started: the frontend loads
//! from the `ornis://app/…` custom scheme mapped onto the editor directory
//! on disk, and talks to the engine thread through the [`ornis_shell`]
//! IPC bridge (`window.ipc.postMessage` in, `ornis-ipc` custom events out).
//!
//! # The viewport
//!
//! WKWebView has no `navigator.gpu` (PLAN.md §j), so the WASM viewport cannot
//! run here. Two paths, one page:
//!
//! * default — the page boots the WASM renderer in Chrome and, inside this
//!   shell, paints a written explanation instead of a dead red canvas;
//! * `--native-viewport` — the shell renders the scene itself with wgpu into a
//!   layer *under* a transparent viewport region, and the page only reports
//!   where that region is and forwards input over it.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p ornis-shell [--editor-dir <path>]   # editor-v2/ (default) or editor/
//! ORNIS_EDITOR_DIR=editor cargo run -p ornis-shell
//! cargo run -p ornis-shell -- --webgpu-probe       # WebGPU capability check
//! cargo run -p ornis-shell -- --native-viewport    # native wgpu viewport prototype
//! ```
#![warn(missing_docs)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::Instant;

use crossbeam_channel::Sender;
use editor_backend::ipc::{EventSeq, GameEvent, RequestId, UiCommand};
#[cfg(target_os = "macos")]
use ornis_shell::native_viewport::ViewportHost;
use ornis_shell::options::{OptionAction, USAGE};
#[cfg(target_os = "macos")]
use ornis_shell::scene_snapshot_json;
use ornis_shell::{
    AckOutcome, BridgeError, ShellCapabilities, ViewportMessage, WINDOW_HEIGHT, WINDOW_WIDTH,
    ack_message_json, dispatch_script, events_message_json, full_window_bounds, init_script_for,
    next_event_sequence, parse_args, parse_incoming, parse_viewport_message, probe_page,
    resolve_editor_dir, start_url,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};
use wry::{WebView, WebViewBuilder};

/// Cross-thread messages delivered to the event loop (all `WebView` calls
/// stay on the main thread).
#[derive(Debug)]
enum ShellEvent {
    /// Raw `window.ipc.postMessage` payload from the page.
    Ipc(String),
    /// Engine event drained from `ev_rx` by the pump thread.
    Engine(GameEvent),
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let outcome = parse_args(&argv);
    for warning in &outcome.warnings {
        eprintln!("ornis-shell: {warning}");
    }
    let options = match outcome.action {
        OptionAction::Run(options) => options,
        OptionAction::Help => {
            println!("{USAGE}");
            std::process::exit(0);
        }
    };
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
        #[cfg(target_os = "macos")]
        viewport: None,
        cmd_tx,
        next_request: RequestId::new(1),
        next_event: EventSeq::new(1),
        editor_root: Arc::new(editor_root),
        webgpu_probe: options.webgpu_probe,
        native_viewport: options.native_viewport && !options.webgpu_probe,
        #[cfg(target_os = "macos")]
        last_frame: Instant::now(),
    };
    event_loop.run_app(&mut app).expect("shell event loop runs");
}

/// Winit application owning the native window and the embedded webview.
struct ShellApp {
    /// Loop proxy; the IPC handler clones it to forward page posts.
    proxy: winit::event_loop::EventLoopProxy<ShellEvent>,
    window: Option<Window>,
    webview: Option<WebView>,
    /// Native wgpu viewport, when `--native-viewport` got that far. Absent on
    /// hosts without the AppKit/Metal surface path, which is why it is a
    /// `cfg`-gated field rather than an always-`None` one.
    #[cfg(target_os = "macos")]
    viewport: Option<ViewportHost>,
    cmd_tx: Sender<UiCommand>,
    next_request: RequestId,
    next_event: EventSeq,
    editor_root: Arc<PathBuf>,
    webgpu_probe: bool,
    native_viewport: bool,
    /// Wall clock of the previous viewport frame.
    #[cfg(target_os = "macos")]
    last_frame: Instant,
}

impl ShellApp {
    /// Forwards one page post to the engine and acks it back to the page.
    ///
    /// Shell-level messages (the viewport rectangle, forwarded input) never
    /// reach the engine queue: they are consumed here, because they describe
    /// this window rather than the scene.
    fn on_ipc(&mut self, body: String) {
        if let Some(message) = parse_viewport_message(&body) {
            self.on_viewport_message(message);
            return;
        }
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

    /// Applies a viewport message to the native viewport.
    ///
    /// Without one (a plain embedded shell, or a host without the AppKit/Metal
    /// surface path) the page's posts are simply not needed — it only sends
    /// them because the shell advertised `ornisShell.nativeViewport`.
    fn on_viewport_message(&mut self, message: ViewportMessage) {
        #[cfg(not(target_os = "macos"))]
        let _ = message;
        #[cfg(target_os = "macos")]
        {
            let (scale, height) = (self.scale_factor(), self.content_height());
            let Some(viewport) = self.viewport.as_mut() else {
                return;
            };
            match message {
                ViewportMessage::Bounds(rect) => {
                    viewport.set_bounds(rect, scale, height);
                }
                ViewportMessage::Input(input) => viewport.apply_input(&input),
            }
        }
    }

    /// Current device-pixel ratio of the window.
    #[cfg(target_os = "macos")]
    fn scale_factor(&self) -> f32 {
        self.window.as_ref().map_or(1.0, Window::scale_factor) as f32
    }

    /// Window content height in points, for clipping stale viewport rectangles.
    #[cfg(target_os = "macos")]
    fn content_height(&self) -> f64 {
        self.window.as_ref().map_or(0.0, |window| {
            let size = window.inner_size();
            f64::from(size.height) / window.scale_factor()
        })
    }

    /// Forwards one engine event to the page with a transport sequence.
    ///
    /// Scene snapshots also feed the native viewport: the same body the page
    /// polls is what the shell renders, so both viewports always agree.
    fn on_engine_event(&mut self, event: GameEvent) {
        // The viewport renders the same snapshot body the page receives, so it
        // is fed before the event is dispatched (no second snapshot channel).
        #[cfg(target_os = "macos")]
        if let (Some(json), Some(viewport)) = (scene_snapshot_json(&event), self.viewport.as_mut())
        {
            viewport.apply_scene_json(json);
        }
        let sequence = next_event_sequence(&mut self.next_event);
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
        let capabilities = if self.native_viewport {
            ShellCapabilities::NATIVE_VIEWPORT
        } else {
            ShellCapabilities::WEBVIEW_ONLY
        };
        let mut builder = WebViewBuilder::new()
            .with_initialization_script(init_script_for(capabilities))
            .with_custom_protocol("ornis".into(), move |_, request| {
                // The handler is a pure library function (status, content
                // type, CORS header), tested without a window.
                ornis_shell::serve_response(&root, request.uri().path())
            });
        // The viewport layer shows through the page's transparent viewport
        // box, so the webview itself must not paint an opaque background.
        if self.native_viewport {
            builder = builder.with_transparent(true);
        }
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

        #[cfg(target_os = "macos")]
        if self.native_viewport {
            // A failed viewport is not fatal: the page falls back to its
            // written notice, which is strictly better than no shell at all.
            match ViewportHost::attach(&webview, window.scale_factor()) {
                Ok(viewport) => self.viewport = Some(viewport),
                Err(error) => {
                    eprintln!("ornis-shell: native viewport unavailable: {error}");
                    eprintln!("ornis-shell: the viewport stays in Chrome — cargo xtask editor");
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        if self.native_viewport {
            eprintln!(
                "ornis-shell: --native-viewport is macOS-only in this prototype;                  the viewport stays in Chrome — cargo xtask editor"
            );
        }
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
            // Only the native viewport redraws on demand; the webview paints
            // itself. Without this the layer would only update on events.
            #[cfg(target_os = "macos")]
            WindowEvent::RedrawRequested => {
                let elapsed = self.last_frame.elapsed().as_secs_f32();
                self.last_frame = Instant::now();
                if let Some(viewport) = self.viewport.as_mut() {
                    viewport.render(ornis_shell::native_viewport::clamp_frame_step(elapsed));
                }
            }
            _ => {}
        }
    }

    #[cfg(target_os = "macos")]
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // A continuous frame loop: the viewport owns its own redraw cadence, so
        // it asks for the next frame as soon as the previous one presented.
        if self.viewport.is_some()
            && let Some(window) = &self.window
        {
            window.request_redraw();
        }
    }
}
