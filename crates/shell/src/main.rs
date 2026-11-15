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
    BridgeError, ShellCapabilities, ViewportMessage, WINDOW_HEIGHT, WINDOW_WIDTH, ack_message_json,
    acknowledge, dispatch_script, events_message_json, full_window_bounds, init_script_for,
    next_event_sequence, parse_args, parse_incoming, parse_viewport_message, probe_page,
    resolve_editor_dir, start_url,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};
use wry::{WebView, WebViewBuilder};

/// Exit code for an unusable environment: no event loop, no window, no
/// webview. Distinct from `2` (bad usage / missing editor dir) so a wrapper
/// script can tell "the user pointed me at nothing" from "this host cannot
/// run a window".
const EXIT_ENVIRONMENT: i32 = 3;

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

    let (cmd_tx, ev_rx) = ornis_shell::start_engine();

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
    // A shell that cannot open its event loop has nothing to fall back to, so
    // this reports the cause and exits non-zero instead of panicking: a panic
    // here prints a backtrace and, on release builds, an unhelpful message.
    let Some(event_loop) = loop_builder.build().ok() else {
        eprintln!("ornis-shell: cannot create the winit event loop");
        std::process::exit(EXIT_ENVIRONMENT);
    };
    let proxy = event_loop.create_proxy();
    let pumped = ornis_shell::spawn_event_pump(ev_rx, move |event| {
        proxy.send_event(ShellEvent::Engine(event)).is_ok()
    });
    if let Err(error) = pumped {
        eprintln!("ornis-shell: cannot start the event pump thread: {error}");
        std::process::exit(EXIT_ENVIRONMENT);
    }

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
    if let Err(error) = event_loop.run_app(&mut app) {
        eprintln!("ornis-shell: event loop stopped: {error}");
        std::process::exit(EXIT_ENVIRONMENT);
    }
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
    // qual:allow(test_quality, untested) reason: "webview IPC edge; the routing (parse_viewport_message, scene_snapshot_json) and the ack decision (acknowledge) are host-tested, the eval/queue handoff needs a live webview"
    fn on_ipc(&mut self, body: String) {
        if let Some(message) = parse_viewport_message(&body) {
            self.on_viewport_message(message);
            return;
        }
        let post = parse_incoming(&body, &mut self.next_request);
        // The ack decision itself is `ornis_shell::acknowledge`, tested without
        // an event loop; only the channel handoff crosses a thread here.
        let outcome = acknowledge(post, |command| {
            self.cmd_tx
                .send(command)
                .map_err(|_| BridgeError::Disconnected)
        });
        self.eval(&dispatch_script(&ack_message_json(&outcome)));
    }

    /// Applies a viewport message to the native viewport.
    ///
    /// Without one (a plain embedded shell, or a host without the AppKit/Metal
    /// surface path) the page's posts are simply not needed — it only sends
    /// them because the shell advertised `ornisShell.nativeViewport`.
    // qual:allow(test_quality, untested) reason: "window/AppKit edge; the bounds decision (BoundsUpdate::decide) and every geometry path are host-tested, this only reads the window's scale"
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
    // qual:allow(test_quality, untested) reason: "event-loop edge; the scene routing (scene_snapshot_json) and the sequence allocator (next_event_sequence) are host-tested, the eval into the webview needs a window"
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

    /// Builds the child webview and wires the `ornis://` handler and IPC.
    ///
    /// `None` when `wry` refuses the webview: without one there is no shell,
    /// so the caller leaves the loop cleanly instead of panicking inside
    /// `resumed`.
    // qual:allow(test_quality, untested) reason: "wry webview construction; the init script, the response handler and the IPC parse it wires are all host-tested"
    fn build_webview(&self, window: &Window) -> Option<WebView> {
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
        match builder
            .with_bounds(full_window_bounds())
            .build_as_child(window)
        {
            Ok(webview) => Some(webview),
            Err(error) => {
                eprintln!("ornis-shell: cannot build the webview: {error}");
                None
            }
        }
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
        // Window creation failing is fatal but must not panic: the message
        // names the cause and the exit code is distinct from a usage error.
        let Ok(window) = event_loop.create_window(
            WindowAttributes::default()
                .with_title("Ornis Editor")
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
        ) else {
            eprintln!("ornis-shell: cannot create the editor window");
            event_loop.exit();
            return;
        };
        let Some(webview) = self.build_webview(&window) else {
            event_loop.exit();
            return;
        };

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
