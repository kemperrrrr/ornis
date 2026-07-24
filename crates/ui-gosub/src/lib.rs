//! Ornis UI - Gosub Backend
//!
//! Alternative UI rendering engine using Gosub browser engine.
//! Provides full HTML/CSS/JS support with parley text shaping.
//!
//! # Architecture
//!
//! ```text
//! HTML → gosub_html5 → DOM → gosub_css3 → Styles → gosub_taffy → Layout
//! JS: gosub_v8 (full React/Vue support)
//! Render: gosub_renderer_vello (GPU)
//! ```
//!
//! # Usage
//!
//! ```rust
//! use ornis_ui_gosub::GosubUI;
//!
//! let ui = GosubUI::new()?;
//! ui.load_html("<html>...</html>")?;
//! ui.render()?;
//! ```

mod engine;
mod renderer;

pub use engine::GosubUI;
pub use renderer::GosubRenderer;
