//! Inline WebGPU capability probe page.
//!
//! Risk check for the `wry` track: whether the system webview on this
//! machine exposes WebGPU (`navigator.gpu`). The page is fully
//! self-contained (no network, no editor files) and reports its verdict in
//! three redundant places — `document.title`, the page body, and the dev
//! console — so both a human screenshot and automation can read it.
//!
//! Run it with `cargo run -p ornis-shell -- --webgpu-probe`. It deliberately
//! does not touch `editor/` or `editor-v2/`.

/// Self-contained HTML probing `navigator.gpu`.
///
/// On load it requests an adapter and writes `ORNIS_WEBGPU:PASS …` or
/// `ORNIS_WEBGPU:FAIL …` (plus the reason) into `document.title` and a
/// full-window verdict line, and mirrors the same line to the console.
pub fn probe_page() -> &'static str {
    concat!(
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        "<title>ORNIS_WEBGPU:PENDING</title>",
        "<style>body{margin:0;display:flex;min-height:100vh;align-items:center;",
        "justify-content:center;background:#101418;color:#e8eef4;",
        "font:28px/1.4 system-ui,sans-serif;text-align:center;padding:2em}",
        ".pass{color:#5fd08a}.fail{color:#ff7a7a}.small{font-size:15px;opacity:.75}</style>",
        "</head><body><div id=\"v\">probing…</div>",
        "<script>(function(){",
        "function verdict(ok,detail){",
        "var line=\"ORNIS_WEBGPU:\"+(ok?\"PASS\":\"FAIL\")+\" \"+detail;",
        "document.title=line;",
        "var v=document.getElementById(\"v\");",
        "v.innerHTML=\"<div class='\"+(ok?\"pass\":\"fail\")+\"'>\"+line+\"</div>\"+",
        "\"<div class='small'>\"+navigator.userAgent+\"</div>\";",
        "if(window.console&&console.log)console.log(line);}",
        "try{",
        "if(!(\"gpu\" in navigator)){verdict(false,\"navigator.gpu missing\");return;}",
        "navigator.gpu.requestAdapter().then(function(a){",
        "if(!a){verdict(false,\"requestAdapter() resolved null\");return;}",
        "a.requestDevice().then(function(){verdict(true,\"adapter+device ok\");},",
        "function(e){verdict(false,\"requestDevice: \"+(e&&e.message||e));});",
        "},function(e){verdict(false,\"requestAdapter: \"+(e&&e.message||e));});",
        "}catch(e){verdict(false,\"threw: \"+(e&&e.message||e));}",
        "})();</script></body></html>",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_page_is_self_contained_and_reports_a_verdict() {
        let page = probe_page();
        assert!(page.contains("navigator.gpu"));
        assert!(page.contains("requestAdapter"));
        assert!(page.contains("ORNIS_WEBGPU:"));
        assert!(page.contains("document.title"));
        // No external fetches: the verdict must not depend on the network.
        assert!(!page.contains("http://") && !page.contains("https://"));
        assert!(!page.contains("<script src"));
    }
}
