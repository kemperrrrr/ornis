//! Command-line options of the shell binary.
//!
//! The binary used to parse `std::env::args` inline, which made the flags —
//! including the one that selects the viewport path — untestable. The parser
//! is pure here (`&[String]` in, [`ShellOptions`] out) and the binary only
//! reads the real argv and interprets [`OptionAction`].
//!
//! Unknown flags are reported, never fatal: this is a development shell and a
//! stale alias in a script should print a warning rather than refuse to start.

/// Parsed CLI options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShellOptions {
    /// `--editor-dir <path>`: frontend root; falls back to `ORNIS_EDITOR_DIR`
    /// and then the workspace `editor-v2/`, like the HTTP transport.
    pub editor_dir: Option<String>,
    /// `--webgpu-probe`: load the inline capability page instead of the editor.
    pub webgpu_probe: bool,
    /// `--native-viewport`: render the viewport with wgpu under the webview.
    pub native_viewport: bool,
}

/// What the binary should do after parsing argv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionAction {
    /// Start the window with these options.
    Run(ShellOptions),
    /// Print [`USAGE`] and exit successfully (`--help` / `-h`).
    Help,
}

/// Args the parser could not use, in argv order (reported, never fatal).
///
/// Returned separately from [`OptionAction`] so the binary decides how to log
/// them — a library writing to stderr on its own would be a hidden side effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseOutcome {
    /// What to do next.
    pub action: OptionAction,
    /// Unknown flags and orphaned values.
    pub warnings: Vec<String>,
}

/// The `--help` text, including the viewport paths this binary offers.
pub const USAGE: &str = "ornis-shell: native editor window\n\
     \n\
     \x20 --editor-dir <path>   frontend root (default: ORNIS_EDITOR_DIR or <workspace>/editor-v2)\n\
     \x20 --webgpu-probe        load the inline WebGPU capability page instead of the editor\n\
     \x20 --native-viewport     render the viewport natively with wgpu under the webview (macOS)";

/// Parses argv (without the program name).
///
/// `--` is accepted as the conventional `cargo run -- <flags>` separator and
/// ignored. A flag that takes a value consumes the next argument; a missing
/// value is a warning, not an error, so the shell still starts with defaults.
pub fn parse_args(args: &[String]) -> ParseOutcome {
    let mut options = ShellOptions::default();
    let mut warnings = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--" => {}
            "--editor-dir" => match iter.next() {
                Some(value) => options.editor_dir = Some(value.clone()),
                None => warnings.push("--editor-dir requires a path".into()),
            },
            "--webgpu-probe" => options.webgpu_probe = true,
            "--native-viewport" => options.native_viewport = true,
            "--help" | "-h" => {
                return ParseOutcome {
                    action: OptionAction::Help,
                    warnings,
                };
            }
            other => warnings.push(format!("ignoring unknown argument {other}")),
        }
    }
    ParseOutcome {
        action: OptionAction::Run(options),
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    fn run(list: &[&str]) -> ShellOptions {
        match parse_args(&args(list)).action {
            OptionAction::Run(options) => options,
            OptionAction::Help => panic!("expected a run outcome"),
        }
    }

    #[test]
    fn defaults_enable_neither_probe_nor_native_viewport() {
        let options = run(&[]);
        assert_eq!(options, ShellOptions::default());
        assert!(!options.native_viewport);
        assert!(!options.webgpu_probe);
        assert_eq!(options.editor_dir, None);
    }

    #[test]
    fn every_flag_is_recognized() {
        let options = run(&[
            "--editor-dir",
            "/tmp/editor",
            "--webgpu-probe",
            "--native-viewport",
        ]);
        assert_eq!(options.editor_dir.as_deref(), Some("/tmp/editor"));
        assert!(options.webgpu_probe);
        assert!(options.native_viewport);
    }

    #[test]
    fn the_cargo_separator_is_ignored() {
        // `cargo run -p ornis-shell -- --native-viewport` passes `--` through.
        let options = run(&["--", "--native-viewport"]);
        assert!(options.native_viewport);
        assert_eq!(options.editor_dir, None);
    }

    #[test]
    fn a_missing_path_and_unknown_flags_warn_but_still_run() {
        let outcome = parse_args(&args(&["--editor-dir"]));
        assert_eq!(outcome.action, OptionAction::Run(ShellOptions::default()));
        assert_eq!(outcome.warnings, vec!["--editor-dir requires a path"]);

        let outcome = parse_args(&args(&["--frobnicate", "--webgpu-probe"]));
        assert_eq!(
            outcome.warnings,
            vec!["ignoring unknown argument --frobnicate"]
        );
        match outcome.action {
            OptionAction::Run(options) => assert!(options.webgpu_probe, "later flags still apply"),
            OptionAction::Help => panic!("unknown flags must not stop parsing"),
        }
    }

    #[test]
    fn help_wins_and_stops_parsing() {
        let outcome = parse_args(&args(&["--help", "--native-viewport"]));
        assert_eq!(outcome.action, OptionAction::Help);
        assert!(outcome.warnings.is_empty());
        assert_eq!(parse_args(&args(&["-h"])).action, OptionAction::Help,);
        // The usage text must mention every flag, or `--help` starts lying.
        for flag in ["--editor-dir", "--webgpu-probe", "--native-viewport"] {
            assert!(USAGE.contains(flag), "{flag} missing from usage");
        }
    }
}
