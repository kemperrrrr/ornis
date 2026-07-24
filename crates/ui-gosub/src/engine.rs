//! Gosub UI Engine wrapper
//!
//! Provides high-level API for HTML/CSS/JS rendering using Gosub engine.

use anyhow::Result;
use gosub_engine::GosubEngine;
use gosub_engine::config::EngineConfig;

pub struct GosubUI {
    engine: GosubEngine,
}

impl GosubUI {
    /// Create new Gosub UI instance
    pub fn new() -> Result<Self> {
        let config = EngineConfig::default();
        let engine = GosubEngine::new(config)?;
        Ok(Self { engine })
    }

    /// Load HTML content
    pub fn load_html(&mut self, html: &str) -> Result<()> {
        // TODO: Implement HTML loading via Gosub engine
        // This will:
        // 1. Parse HTML with gosub_html5
        // 2. Apply CSS with gosub_css3
        // 3. Calculate layout with gosub_taffy
        // 4. Prepare render commands
        Ok(())
    }

    /// Render to Vello scene
    pub fn render(&self) -> Result<()> {
        // TODO: Implement rendering via gosub_renderer_vello
        // This will:
        // 1. Build Vello Scene from paint commands
        // 2. Use parley for text shaping
        // 3. Apply layers, scroll, etc.
        Ok(())
    }

    /// Handle window resize
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        // TODO: Recalculate layout for new size
        Ok(())
    }
}
