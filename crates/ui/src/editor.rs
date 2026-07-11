use vello::peniko::{Color, FontData};

use crate::render::UIRenderer;

/// In-game editor overlay rendered directly with Vello primitives.
/// Provides a heads-up display for debugging/editing game state.
pub struct EditorOverlay {
    visible: bool,
    /// Entity count for display
    entity_count: u32,
}

impl EditorOverlay {
    pub fn new() -> Self {
        Self {
            visible: false,
            entity_count: 0,
        }
    }

    pub fn set_entity_count(&mut self, count: u32) {
        self.entity_count = count;
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn handle_key(&mut self, key: &str) -> bool {
        if key == "`" || key == "Backtick" {
            self.toggle();
            true
        } else if key == "Escape" && self.visible {
            self.visible = false;
            true
        } else {
            false
        }
    }

    /// Paint the editor overlay on top of the current Vello scene.
    pub fn paint(&self, renderer: &mut UIRenderer, viewport_w: f64, viewport_h: f64, font: &FontData) {
        if !self.visible {
            return;
        }

        let panel_w = 280.0;
        let panel_x = viewport_w - panel_w;

        // Semi-transparent panel background
        renderer.fill_rounded_rect(panel_x, 0.0, panel_w, viewport_h, 0.0, Color::new([0.12, 0.12, 0.14, 0.92]));

        // Left border
        renderer.stroke_rect(panel_x, 0.0, 1.0, viewport_h, 1.0, Color::new([0.27, 0.27, 0.27, 1.0]));

        // Title
        renderer.fill_text(panel_x + 16.0, 18.0, "Editor", 18.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);

        // Separator
        renderer.fill_rect(panel_x + 12.0, 40.0, panel_w - 24.0, 1.0, Color::new([0.33, 0.33, 0.33, 1.0]));

        // -- Scene Stats section --
        renderer.fill_rounded_rect(panel_x + 12.0, 52.0, panel_w - 24.0, 60.0, 6.0, Color::new([1.0, 1.0, 1.0, 0.05]));

        renderer.fill_text(panel_x + 24.0, 64.0, "Scene Stats", 12.0, Color::new([0.67, 0.67, 0.67, 1.0]), font);
        renderer.fill_text(panel_x + 24.0, 82.0, &format!("Entities: {}", self.entity_count), 14.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);

        // -- UIStyle Editor section --
        renderer.fill_rounded_rect(panel_x + 12.0, 122.0, panel_w - 24.0, 80.0, 6.0, Color::new([1.0, 1.0, 1.0, 0.05]));

        renderer.fill_text(panel_x + 24.0, 134.0, "UIStyle Editor", 12.0, Color::new([0.67, 0.67, 0.67, 1.0]), font);

        // Color swatch
        renderer.fill_text(panel_x + 24.0, 154.0, "Color", 12.0, Color::new([0.67, 0.67, 0.67, 1.0]), font);
        renderer.fill_rounded_rect(panel_x + 100.0, 152.0, 24.0, 16.0, 3.0, Color::new([0.23, 0.43, 0.94, 1.0]));

        // Font size label
        renderer.fill_text(panel_x + 24.0, 178.0, "Font Size: 16", 12.0, Color::new([0.67, 0.67, 0.67, 1.0]), font);

        // -- Help text at the bottom --
        let help_y = viewport_h - 30.0;
        renderer.fill_rect(panel_x + 12.0, help_y - 8.0, panel_w - 24.0, 1.0, Color::new([0.2, 0.2, 0.2, 1.0]));
        renderer.fill_text(panel_x + 16.0, help_y, "~ toggle  ·  Esc close", 11.0, Color::new([0.4, 0.4, 0.4, 1.0]), font);
    }
}

impl Default for EditorOverlay {
    fn default() -> Self {
        Self::new()
    }
}
