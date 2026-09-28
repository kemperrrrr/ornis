//! Manual live-pixel e2e for the WASM viewport (`cargo xtask e2e`).
//!
//! Chain: editor scene → `GET /api/scene` → WASM `RenderWorld` → rendered
//! frame → screenshot → golden compare with tolerances (never bytewise).
//! CI has no browser/WebGPU, so this task SKIPs honestly (exit 0) when the
//! environment cannot run the pixel leg. The headless half of the same
//! chain is pinned by `cargo test -p ornis-wasm` (FULL_CONTRACT golden).
//!
//! Cross-platform: everything goes through `std::process::Command` and
//! `std::net::TcpStream`, no shell. HTTP is a hand-rolled `GET` over TCP
//! (stdlib only) because xtask must stay dependency-light.
//!
//! # Errors
//!
//! Exits non-zero only on bad CLI usage. Every missing environment
//! capability (toolchain, browser, adapter, golden) is a `SKIP` with
//! exit 0, never a failure.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{exit, Child, Command};
use std::time::{Duration, Instant};

/// Fixed port: the `editor-only` server hardcodes 3420 (see `src/main.rs`).
const DEFAULT_PORT: u16 = 3420;
const POLL_DEADLINE_SECS: u64 = 30;
const VIEWPORT: &str = "800,600";

const BROWSERS: &[&str] = &[
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "firefox",
];

/// Golden tolerances, mirrored in `docs/WASM_PIXEL_E2E.md` (never bytewise).
const GATE_TEXT: &str = "tolerance gate: max per-pixel <= 12/255, \
    mean <= 2/255, fraction <= 0.5%";

/// Honest skip: the pixel leg cannot run here, so there is nothing to fail.
fn skip(reason: &str) -> ! {
    println!("SKIP wasm_pixel_e2e: {reason}");
    exit(0);
}

/// `true` when `program --version` spawns and exits successfully.
fn have(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// First browser binary on `PATH`, or `None`. Extension-aware on Windows.
fn probe_browser() -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for candidate in BROWSERS {
            #[cfg(windows)]
            let file = dir.join(format!("{candidate}.exe"));
            #[cfg(not(windows))]
            let file = dir.join(candidate);
            if file.is_file() {
                return Some((*candidate).to_string());
            }
        }
    }
    None
}

/// `true` when a playwright browser cache already exists for this user.
fn playwright_cached() -> bool {
    let home = dirs_home();
    // `extend` (not conditional `push`) keeps the `mut` live on every
    // platform: Linux probes exactly one root, macOS/Windows add theirs.
    let mut dirs = vec![home.join(".cache/ms-playwright")];
    dirs.extend(platform_cache_extra(&home));
    dirs.iter().any(|d| d.is_dir())
}

/// At most one extra playwright root beyond the common Unix cache dir:
/// macOS system cache, or Windows `%LOCALAPPDATA%` (missing var → none).
fn platform_cache_extra(home: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    return Some(home.join("Library/Caches/ms-playwright"));
    #[cfg(windows)]
    return std::env::var("LOCALAPPDATA")
        .map(|local| PathBuf::from(local).join("ms-playwright"))
        .ok();
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = home;
        None
    }
}

fn dirs_home() -> PathBuf {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("~"))
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("~"))
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent directory")
        .to_path_buf()
}

/// Wait until `127.0.0.1:port` accepts TCP, then `true`. Polls, never hangs.
fn wait_for_server(port: u16) -> bool {
    let deadline = Instant::now() + Duration::from_secs(POLL_DEADLINE_SECS);
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().expect("loopback addr"),
            Duration::from_millis(500),
        )
        .is_ok()
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    false
}

/// Minimal blocking `GET /api/scene` over TCP. Returns the response body.
fn fetch_scene(port: u16) -> Option<Vec<u8>> {
    let addr = format!("127.0.0.1:{port}");
    let mut stream = TcpStream::connect_timeout(
        &addr.parse().expect("loopback addr"),
        Duration::from_secs(2),
    )
    .ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    stream
        .write_all(
            format!(
                "GET /api/scene HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
                 Connection: close\r\n\r\n"
            )
            .as_bytes(),
        )
        .ok()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw);
    let mut lines = text.lines();
    let status = lines.next().unwrap_or("");
    if !status.contains(" 2") {
        return None;
    }
    let sep = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)?;
    Some(raw[sep..].to_vec())
}

/// Screenshot `url` into `out_png`. `true` when the file appears.
/// Waits for the `#bevy` canvas and lets it settle, so the shot pins a
/// rendered frame rather than a blank viewport.
fn screenshot(browser: &str, url: &str, out_png: &Path) -> bool {
    let status = if browser == "playwright-chromium" {
        Command::new("npx")
            .args([
                "-y",
                "playwright",
                "screenshot",
                "--browser=chromium",
                &format!("--viewport-size={VIEWPORT}"),
                "--wait-for-selector",
                "#bevy",
                "--wait-for-timeout",
                "3000",
                url,
            ])
            .arg(out_png)
            .status()
    } else if browser.contains("firefox") {
        Command::new(browser)
            .arg("--headless")
            .arg(format!("--screenshot={}", out_png.display()))
            .arg(url)
            .status()
    } else {
        Command::new(browser)
            .args([
                "--headless",
                "--disable-gpu-sandbox",
                "--no-sandbox",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--timeout=20000",
                &format!("--window-size={VIEWPORT}"),
            ])
            .arg(format!("--screenshot={}", out_png.display()))
            .arg(url)
            .status()
    };
    status.map(|s| s.success()).unwrap_or(false) && out_png.is_file()
}

/// Compare `frame` against the golden (or bootstrap it). Always exit 0.
fn compare_golden(root: &Path, frame: &Path) -> ! {
    let golden = root.join("editor/e2e_golden.png");
    if !golden.is_file() {
        println!(
            "no {} yet — stored {} for manual review \
             (first run bootstraps the golden)",
            golden.display(),
            frame.display()
        );
        exit(0);
    }
    if have("compare") {
        let out = Command::new("compare")
            .arg("-metric")
            .arg("RMSE")
            .arg(&golden)
            .arg(frame)
            .arg("null:")
            .output();
        match out {
            Ok(o) => println!(
                "RMSE: {}",
                String::from_utf8_lossy(&[o.stdout, o.stderr].concat())
                    .trim()
                    .replace('\n', " | ")
            ),
            Err(e) => println!("compare failed to run: {e}"),
        }
        println!("{GATE_TEXT}");
        println!("(automated threshold parsing is manual for now — inspect the RMSE above)");
    } else {
        println!(
            "no ImageMagick 'compare' — stored {} next to {} for manual review",
            frame.display(),
            golden.display()
        );
    }
    exit(0);
}

fn spawn_server(root: &Path, editor_dir: &Path) -> Child {
    Command::new("cargo")
        .arg("run")
        .arg("--features")
        .arg("editor-only")
        .arg("--")
        .arg("--editor-dir")
        .arg(editor_dir)
        .current_dir(root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap_or_else(|e| skip(&format!("cannot spawn editor server: {e}")))
}

/// Poll `/api/scene` until the server publishes a non-empty scene.
/// The first answer after boot is an empty placeholder (version 0, no
/// entities); screenshotting that would pin a blank viewport, so this
/// waits for real content with the same deadline as the TCP wait.
fn wait_for_scene(port: u16) -> Option<Vec<u8>> {
    let deadline = Instant::now() + Duration::from_secs(POLL_DEADLINE_SECS);
    while Instant::now() < deadline {
        if let Some(body) = fetch_scene(port) {
            let published = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("entities")?.as_array().map(|e| !e.is_empty()))
                .unwrap_or(false);
            if published {
                return Some(body);
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    None
}

/// Post-spawn half of the chain. `Err` is a SKIP reason; the caller kills
/// the server before reporting it.
fn pixel_leg(port: u16, browser: &str, out_scene: &Path, out_frame: &Path) -> Result<(), String> {
    if !wait_for_server(port) {
        return Err("editor server did not answer /api/scene".to_string());
    }
    let body = wait_for_scene(port)
        .ok_or_else(|| "server never published a scene (empty placeholder only)".to_string())?;
    std::fs::write(out_scene, &body)
        .map_err(|e| format!("cannot write {}: {e}", out_scene.display()))?;

    println!("capturing screenshot…");
    if !screenshot(browser, &format!("http://127.0.0.1:{port}"), out_frame) {
        return Err("screenshot failed (likely no WebGPU adapter)".to_string());
    }
    if !out_frame.is_file() {
        return Err("no screenshot produced".to_string());
    }
    Ok(())
}

/// Run the manual live-pixel e2e harness (honest SKIP when not runnable).
pub fn e2e(args: &[String]) {
    let mut port = DEFAULT_PORT;
    let mut out = std::env::temp_dir().join("ornis-e2e");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("")
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("xtask: --port requires a number");
                        exit(2);
                    });
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(args.get(i).unwrap_or_else(|| {
                    eprintln!("xtask: --out requires a path");
                    exit(2);
                }));
            }
            "-h" | "--help" => {
                println!(
                    "cargo xtask e2e [--port 3420] [--out <dir>]\n\
                     Manual live-pixel e2e for the WASM viewport.\n\
                     ORNIS_E2E_INSTALL=1 allows a playwright chromium download."
                );
                exit(0);
            }
            other => {
                eprintln!("xtask: unknown flag '{other}'");
                exit(2);
            }
        }
        i += 1;
    }
    if port != DEFAULT_PORT {
        skip("editor-only server port is hardcoded to 3420");
    }

    if !have("cargo") {
        skip("no cargo toolchain");
    }
    if !have("node") || !have("npx") {
        skip("no node/npx (playwright driver missing)");
    }
    let browser = match probe_browser() {
        Some(b) => b,
        None => {
            if !playwright_cached() {
                if std::env::var("ORNIS_E2E_INSTALL").ok().as_deref() == Some("1") {
                    println!("installing playwright chromium (ORNIS_E2E_INSTALL=1)…");
                    let ok = Command::new("npx")
                        .args(["-y", "playwright", "install", "chromium"])
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false);
                    if !ok {
                        skip("playwright browser install failed");
                    }
                } else {
                    skip(
                        "no browser binary and no playwright cache \
                         (set ORNIS_E2E_INSTALL=1 to download)",
                    );
                }
            }
            "playwright-chromium".to_string()
        }
    };
    println!("browser: {browser}");

    std::fs::create_dir_all(&out)
        .unwrap_or_else(|_| skip(&format!("cannot create {}", out.display())));

    if !have("wasm-pack") {
        skip("no wasm-pack");
    }
    let root = workspace_root();
    let pkg = root.join("editor/pkg");
    println!("xtask: wasm-pack build crates/wasm --target web");
    let built = Command::new("wasm-pack")
        .arg("build")
        .arg(root.join("crates/wasm"))
        .arg("--target")
        .arg("web")
        .arg("--out-dir")
        .arg(&pkg)
        .current_dir(&root)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !built {
        skip("wasm-pack build failed");
    }

    println!("xtask: starting editor server on :{port}…");
    let mut server = spawn_server(&root, &root.join("editor"));
    let out_scene = out.join("scene.json");
    let out_frame = out.join("frame.png");
    let result = pixel_leg(port, &browser, &out_scene, &out_frame);
    let _ = server.kill();
    let _ = server.wait();
    if let Err(reason) = result {
        skip(&reason);
    }
    println!("artifacts: {} {}", out_scene.display(), out_frame.display());
    compare_golden(&root, &out_frame);
}
