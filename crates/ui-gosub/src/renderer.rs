//! Gosub Vello Renderer
//!
//! Wrapper around gosub_renderer_vello for integration with Ornis.

use anyhow::Result;
use vello::Scene;
use wgpu::{Device, Queue};

pub struct GosubRenderer {
    // TODO: Add gosub_renderer_vello backend
}

impl GosubRenderer {
    /// Create new renderer
    pub fn new(device: &Device, queue: &Queue) -> Result<Self> {
        // TODO: Initialize gosub_renderer_vello::VelloBackend
        Ok(Self {})
    }

    /// Render scene to texture
    pub fn render(&self, scene: &Scene, device: &Device, queue: &Queue) -> Result<()> {
        // TODO: Use gosub_renderer_vello to render
        Ok(())
    }
}
