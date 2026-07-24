//! Unified Editor - HTML/CSS-based editor that runs both in-game (Vello) and remote (HTTP)
//!
//! This replaces the old Vello-based editor with a unified HTML/CSS-based editor
//! that uses our Rust ECS bridge instead of JavaScript.

#[cfg(feature = "js-engine")]
use crate::components::{EcsBridge, UIStyle};
use crate::layout::LayoutTree;
use crate::paint::paint_layout;
use crate::render::UIRenderer;
use ornis_core::{Entity, SmartStore};
use std::sync::Arc;
use vello::peniko::{Color, FontData};

/// Unified Editor configuration
#[derive(Debug, Clone)]
pub struct UnifiedEditorConfig {
    pub font_size: f32,
    pub theme: EditorTheme,
    pub show_hierarchy: bool,
    pub show_inspector: bool,
    pub show_assets: bool,
    pub show_console: bool,
}

impl Default for UnifiedEditorConfig {
    fn default() -> Self {
        Self {
            font_size: 12.0,
            theme: EditorTheme::default(),
            show_hierarchy: true,
            show_inspector: true,
            show_assets: true,
            show_console: false,
        }
    }
}

/// Editor theme colors (matching the mockup CSS variables)
#[derive(Debug, Clone)]
pub struct EditorTheme {
    pub window_background: Color,
    pub panel_background: Color,
    pub panel_text: Color,
    pub icons_background: Color,
    pub input_background: Color,
    pub input_text: Color,
    pub border: Color,
    pub accent: Color,
    pub error: Color,
    pub warning: Color,
    pub success: Color,
}

impl Default for EditorTheme {
    fn default() -> Self {
        // Values mirror the asset `index.css` `:root` block so the in-game
        // (Vello) editor matches the browser-rendered one exactly. The asset
        // backgrounds are fully opaque, so alpha is 1.0 here (a translucent
        // panel would tint differently over the 3D scene behind it).
        Self {
            window_background: Color::new([0.2235, 0.2235, 0.2431, 1.0]), // #39393e
            panel_background: Color::new([0.1373, 0.1373, 0.1490, 1.0]),  // #232326
            panel_text: Color::new([1.0, 1.0, 1.0, 1.0]),                 // #ffffff
            icons_background: Color::new([0.4863, 0.4863, 0.5294, 1.0]),  // #7c7c87
            input_background: Color::new([0.0941, 0.0941, 0.1020, 1.0]),  // #18181a
            input_text: Color::new([1.0, 1.0, 1.0, 1.0]),
            border: Color::new([0.27, 0.27, 0.27, 1.0]),
            accent: Color::new([0.23, 0.43, 0.94, 1.0]),
            error: Color::new([0.80, 0.23, 0.25, 1.0]),
            warning: Color::new([0.97, 0.64, 0.31, 1.0]),
            success: Color::new([0.29, 0.68, 0.25, 1.0]),
        }
    }
}

/// Editor state for the unified editor
#[derive(Debug)]
pub struct UnifiedEditor {
    pub config: UnifiedEditorConfig,
    pub visible: bool,
    pub selected_entity: Option<Entity>,
    pub hierarchy_scroll: f64,
    pub inspector_scroll: f64,
    pub search_filter: String,
    pub show_add_component: bool,
    pub add_component_type: String,

    // Gizmo state
    gizmo_mode: GizmoMode,
    gizmo_active: bool,
    gizmo_axis: Option<GizmoAxis>,
    gizmo_start_pos: Option<(f64, f64)>,

    // Layout tree for UI rendering
    layout_tree: Option<LayoutTree>,

    // Editor theme CSS variables
    theme_css: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    XY,
    XZ,
    YZ,
    All,
}

impl Default for UnifiedEditor {
    fn default() -> Self {
        let mut editor = Self {
            config: UnifiedEditorConfig::default(),
            visible: false,
            selected_entity: None,
            hierarchy_scroll: 0.0,
            inspector_scroll: 0.0,
            search_filter: String::new(),
            show_add_component: false,
            add_component_type: String::new(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_active: false,
            gizmo_axis: None,
            gizmo_start_pos: None,
            layout_tree: None,
            theme_css: String::new(),
        };
        editor.generate_theme_css();
        editor
    }
}

impl UnifiedEditor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: UnifiedEditorConfig) -> Self {
        let mut editor = Self {
            config,
            visible: false,
            selected_entity: None,
            hierarchy_scroll: 0.0,
            inspector_scroll: 0.0,
            search_filter: String::new(),
            show_add_component: false,
            add_component_type: String::new(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_active: false,
            gizmo_axis: None,
            gizmo_start_pos: None,
            layout_tree: None,
            theme_css: String::new(),
        };
        editor.generate_theme_css();
        editor
    }

    fn generate_theme_css(&mut self) {
        self.theme_css = Self::theme_root_css(&self.config.theme);
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Builds the `:root` custom-property block that themes the shared editor
    /// CSS (darker backgrounds, proper text color, accents). Shared with the
    /// in-game `EditorTemplate` so the Vello editor matches the browser one.
    pub fn theme_root_css(theme: &EditorTheme) -> String {
        let wb = theme.window_background.components;
        let pb = theme.panel_background.components;
        let pt = theme.panel_text.components;
        let ib = theme.icons_background.components;
        let inb = theme.input_background.components;
        let it = theme.input_text.components;
        let b = theme.border.components;
        let a = theme.accent.components;
        let e = theme.error.components;
        let w = theme.warning.components;
        let s = theme.success.components;
        format!(
            r#":root {{
    --window-background: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --panel-background: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --panel-text: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --icons-background: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --input-background: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --input-text: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --border: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --accent: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --error: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --warning: rgba({:.0}, {:.0}, {:.0}, {:.2});
    --success: rgba({:.0}, {:.0}, {:.0}, {:.2});
}}"#,
            wb[0] * 255.0,
            wb[1] * 255.0,
            wb[2] * 255.0,
            wb[3],
            pb[0] * 255.0,
            pb[1] * 255.0,
            pb[2] * 255.0,
            pb[3],
            pt[0] * 255.0,
            pt[1] * 255.0,
            pt[2] * 255.0,
            pt[3],
            ib[0] * 255.0,
            ib[1] * 255.0,
            ib[2] * 255.0,
            ib[3],
            inb[0] * 255.0,
            inb[1] * 255.0,
            inb[2] * 255.0,
            inb[3],
            it[0] * 255.0,
            it[1] * 255.0,
            it[2] * 255.0,
            it[3],
            b[0] * 255.0,
            b[1] * 255.0,
            b[2] * 255.0,
            b[3],
            a[0] * 255.0,
            a[1] * 255.0,
            a[2] * 255.0,
            a[3],
            e[0] * 255.0,
            e[1] * 255.0,
            e[2] * 255.0,
            e[3],
            w[0] * 255.0,
            w[1] * 255.0,
            w[2] * 255.0,
            w[3],
            s[0] * 255.0,
            s[1] * 255.0,
            s[2] * 255.0,
            s[3],
        )
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn handle_key(&mut self, key: &str) -> bool {
        if key == "`" || key == "Backtick" || key == "F1" {
            self.toggle();
            true
        } else if key == "Escape" && self.visible {
            self.visible = false;
            self.selected_entity = None;
            true
        } else if key == "1" && self.visible {
            self.gizmo_mode = GizmoMode::Translate;
            true
        } else if key == "2" && self.visible {
            self.gizmo_mode = GizmoMode::Rotate;
            true
        } else if key == "3" && self.visible {
            self.gizmo_mode = GizmoMode::Scale;
            true
        } else {
            false
        }
    }

    pub fn select_entity(&mut self, entity: Entity) {
        self.selected_entity = Some(entity);
    }

    pub fn deselect(&mut self) {
        self.selected_entity = None;
    }

    pub fn set_search_filter(&mut self, filter: String) {
        self.search_filter = filter;
    }

    pub fn show_add_component(&mut self, entity: Entity) {
        self.selected_entity = Some(entity);
        self.show_add_component = true;
    }

    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode = mode;
    }

    pub fn gizmo_mode(&self) -> GizmoMode {
        self.gizmo_mode
    }

    #[cfg(feature = "js-engine")]
    pub fn entity_count(&self, store: &SmartStore) -> u32 {
        store
            .read_lane::<UIStyle>()
            .map(|lane| lane.len() as u32)
            .unwrap_or(0)
    }
    #[cfg(not(feature = "js-engine"))]
    pub fn entity_count(&self, _store: &SmartStore) -> u32 { 0 }

    pub fn build_layout(
        &mut self,
        _store: &SmartStore,
        font: &FontData,
        viewport_w: f32,
        viewport_h: f32,
    ) {
        let html = crate::editor_template::read_asset("index.html");
        let document = crate::html::parse_html(&html);

        // Load the real editor CSS (base rules + inline `<style>` blocks from
        // the shared asset + the themed `:root` overrides) so the layout engine
        // actually applies grid/flex/absolute positioning and panel colors.
        // Without this the tree is built with an empty stylesheet and the
        // editor collapses into a default block flow — nothing matches the
        // browser-rendered mockup.
        let css = crate::editor_template::EditorTemplate::generate_css_with_theme(&self.config);
        let stylesheet = crate::css::Stylesheet::parse(&css).unwrap_or_else(|e| {
            eprintln!("editor CSS parse failed: {e}");
            crate::css::Stylesheet {
                rules: Vec::new(),
                custom_properties: std::collections::HashMap::new(),
            }
        });

        self.layout_tree = Some(
            LayoutTree::build_with_viewport(&document, &[stylesheet], viewport_w, viewport_h, font, &[], None)
                .unwrap_or_default(),
        );
    }

    pub fn paint(
        &self,
        renderer: &mut UIRenderer,
        _viewport_w: f64,
        _viewport_h: f64,
        font: &FontData,
    ) {
        if !self.visible {
            return;
        }

        if let Some(ref layout_tree) = self.layout_tree {
            paint_layout(layout_tree, renderer, font, None);
        }
    }

    /// Generate the full HTML for the editor
    fn generate_html(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ornis Editor</title>
    <link rel="stylesheet" href="index.css">
    <style>
        {}
    </style>
</head>
<body>
    <div class="app">
        <!-- Header -->
        <div class="panel header">
            <img src="favicon.png" class="logo" alt="Ornis Logo" />
            <span>File</span>
            <span>Edit</span>
            <span>View</span>
            <span>Window</span>
            <span>Help</span>
            <div class="play-options">
                <button id="play-btn" title="Play">
                    <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24">
                        <path d="M320-200v-560l440 280-440 280Z" />
                    </svg>
                </button>
                <button id="pause-btn" title="Pause" style="display:none;">
                    <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24">
                        <path d="M560-200v-560h160v560H560Zm-320 0v-560h160v560H240Z" />
                    </svg>
                </button>
                <button id="step-btn" title="Step">
                    <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24">
                        <path d="M660-240v-480h80v480h-80Zm-440 0v-480l360 240-360 240Z" />
                    </svg>
                </button>
                <div class="slider time-scale" data-suffix="x" data-min="0.05" data-max="2" data-step="0.1" data-value="1">
                    <div class="fill"></div>
                    <span class="value">1.0x</span>
                </div>
            </div>
            <div class="logs">
                <div class="errors">
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M13 13H11V7H13M11 15H13V17H11M15.73 3H8.27L3 8.27V15.73L8.27 21H15.73L21 15.73V8.27L15.73 3Z" />
                        </svg>
                    </div>
                    <span id="error-count">0</span>
                </div>
                <div class="warnings">
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M13 14H11V9H13M13 18H11V16H13M1 21H23L12 2L1 21Z" />
                        </svg>
                    </div>
                    <span id="warning-count">0</span>
                </div>
            </div>
            <div class="command-palette-container">
                <input type="text" placeholder="Command Palette (Ctrl+P)" class="command-palette" />
            </div>
        </div>
        
        <!-- Left Panel: Hierarchy + Search -->
        <div class="panel left-upper">
            <div class="tab-list">
                <div class="tab active">
                    Hierarchy
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="tab">
                    Resources
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="icon new-tab">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                    </svg>
                </div>
            </div>
            <div class="search">
                <input type="text" id="hierarchy-search" placeholder="Search entities..." />
            </div>
            <div class="hierarchy" id="hierarchy-panel">
                <details open class="active">
                    <summary>
                        <div class="left">
                            <div class="arrow"></div>
                            <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,1C10.89,1 10,1.9 10,3C10,4.11 10.89,5 12,5C13.11,5 14,4.11 14,3A2,2 0 0,0 12,1M10,6C9.73,6 9.5,6.11 9.31,6.28H9.3L4,11.59L5.42,13L9,9.41V22H11V15H13V22H15V9.41L18.58,13L20,11.59L14.7,6.28C14.5,6.11 14.27,6 14,6" /></svg></div>Player
                        </div>
                    </summary>
                    <div class="content">
                        <details open>
                            <summary>
                                <div class="left">
                                    <div class="arrow" style="opacity: 0;"></div>
                                    <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h480q33 0 56.5 23.5T720-720v180l160-160v440L720-420v180q0 33-23.5 56.5T640-160H160Z" /></svg></div>Camera3d
                                </div>
                            </summary>
                        </details>
                    </div>
                </details>
                <details open>
                    <summary>
                        <div class="left">
                            <div class="arrow" style="opacity: 0;"></div>
                            <div class="icon custom"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg></div>Floor
                        </div>
                    </summary>
                </details>
                <details>
                    <summary>
                        <div class="left">
                            <div class="arrow"></div>
                            <div class="icon custom"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg></div>Wall
                        </div>
                    </summary>
                    <div class="content">
                        <details open>
                            <summary>
                                <div class="left">
                                    <div class="arrow" style="opacity: 0;"></div>
                                    <div class="icon custom"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg></div>Particle Emitter
                                </div>
                            </summary>
                        </details>
                    </div>
                </details>
            </div>
        </div>
        
        <!-- Center Panel: Viewport -->
        <div class="panel center">
            <div class="tab-list">
                <div class="tab active">
                    Viewport
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="icon new-tab">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                    </svg>
                </div>
            </div>
            <div class="viewport" id="viewport">
                <canvas id="viewport-canvas" class="viewport-canvas"></canvas>
            </div>
        </div>
        
        <!-- Right Panel: Inspector -->
        <div class="panel right">
            <div class="tab-list">
                <div class="tab active">
                    Inspector
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="icon new-tab">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                    </svg>
                </div>
            </div>
            <div class="name">
                <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,1C10.89,1 10,1.9 10,3C10,4.11 10.89,5 12,5C13.11,5 14,4.11 14,3A2,2 0 0,0 12,1M10,6C9.73,6 9.5,6.11 9.31,6.28H9.3L4,11.59L5.42,13L9,9.41V22H11V15H13V22H15V9.41L18.58,13L20,11.59L14.7,6.28C14.5,6.11 14.27,6 14,6" /></svg></div>
                <input type="text" id="entity-name" placeholder="Name" value="Entity" />
            </div>
            <details>
                <summary>
                    <div class="left">
                        <div class="arrow"></div>
                        <div class="icon" style="fill: #5796e8;"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2L16,6H13V13.85L19.53,17.61L21,15.03L22.5,20.5L17,21.96L18.53,19.35L12,15.58L5.47,19.35L7,21.96L1.5,20.5L3,15.03L4.47,17.61L11,13.85V6H8L12,2Z" /></svg></div>Global Transform
                    </div>
                </summary>
                <div class="content">
                    <details open>
                        <summary>
                            <div class="left">
                                <div class="arrow" style="opacity: 0;"></div>
                                <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h480q33 0 56.5 23.5T720-720v180l160-160v440L720-420v180q0 33-23.5 56.5T640-160H160Z" /></svg></div>Transform
                            </div>
                        </summary>
                        <div class="content">
                            <div class="item">
                                <span>Translation</span>
                                <div class="vec3">
                                    <input type="number" class="transform-input" data-field="translation.x" value="0" step="0.1" />
                                    <input type="number" class="transform-input" data-field="translation.y" value="0" step="0.1" />
                                    <input type="number" class="transform-input" data-field="translation.z" value="0" step="0.1" />
                                </div>
                            </div>
                        </div>
                    </details>
                </details>
                <details open>
                    <summary>
                        <div class="left">
                            <div class="arrow"></div>
                            <div class="icon" style="fill: #a156d6;"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg></div>Sprite
                        </div>
                    </summary>
                    <div class="content">
                        <div class="item">
                            <span>Color</span>
                            <div class="color-picker" style="background-color: #ffffff;">
                                <div class="color-picker-dot"></div>
                            </div>
                        </div>
                    </div>
                </details>
            </details>
            <details open>
                <summary>
                    <div class="left">
                        <div class="arrow"></div>
                        <div class="icon custom"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg></div>Point Light
                    </div>
                </summary>
                <div class="content">
                    <div class="item">
                        <span>Color</span>
                        <div class="color-picker" style="background-color: #ff6e6e;">
                            <div class="color-picker-dot"></div>
                        </div>
                    </div>
                    <div class="item">
                        <span>Intensity</span>
                        <div class="slider">
                            <div class="fill"></div>
                            <span class="value">64</span>
                        </div>
                    </div>
                </div>
            </details>
            <details open>
                <summary>
                    <div class="left">
                        <div class="arrow" style="opacity: 0;"></div>
                        <div class="icon custom"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" /></svg></div>Custom Component
                    </div>
                </summary>
            </details>
            <button class="bottom-button" id="add-component-btn">Add Component</button>
        </div>
        
        <!-- Bottom Panel: Assets -->
        <div class="panel bottom">
            <div class="tab-list">
                <div class="tab active">
                    Assets
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="tab">
                    Console
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="tab">
                    Project Settings
                    <div class="icon close">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                        </svg>
                    </div>
                </div>
                <div class="icon new-tab">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
                    </svg>
                </div>
            </div>
            <div class="breadcrumbs">
                <div class="icon"><img src="folder.svg" /></div>
                <span>my_game</span>
                <div class="arrow"></div>
                <div class="icon"><img src="folder.svg" /></div>
                <span>assets</span>
                <div class="arrow"></div>
            </div>
            <div class="files" id="assets-panel">
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">dialogue</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">levels</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">sounds</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">textures</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">dialogue</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">levels</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">sounds</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">textures</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">dialogue</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">levels</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">sounds</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">textures</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="folder.svg" />
                    <span class="file-name">photo.png</span>
                </div>
                <div class="file">
                    <img class="file-icon" src="midi.svg" />
                    <span class="file-name">song.midi</span>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <div class="left"><span>Ornis Engine v0.1.0</span><span>Rust 1.88</span></div>
        <div class="center-container"><span>my_game v0.1.0</span></div>
        <div class="right"><a href="../">Home</a></div>
    </footer>

    <script src="sliders.js"></script>
    <script src="editor.js"></script>
</body>
</html>"#,
            self.theme_css
        )
    }

    fn get_full_css(&self) -> String {
        // Read the base CSS file and append theme CSS
        // Note: This is a placeholder - in production, the CSS would be loaded differently
        let base_css = "";
        format!("{}\n{}", base_css, self.theme_css)
    }

    /// Handle UI events from the Rust side (called from JS via EcsBridge)
    pub fn handle_event(&mut self, event: EditorEvent, store: &mut SmartStore) {
        match event {
            EditorEvent::Toggle => self.toggle(),
            EditorEvent::SelectEntity(entity) => self.select_entity(entity),
            EditorEvent::Deselect => self.deselect(),
            EditorEvent::SetSearchFilter(filter) => self.set_search_filter(filter),
            EditorEvent::ShowAddComponent(entity) => self.show_add_component(entity),
            EditorEvent::AddComponent(component_type) => self.add_component(store, component_type),
            EditorEvent::SetGizmoMode(mode) => self.set_gizmo_mode(mode),
            EditorEvent::SetEntityName(entity, name) => self.set_entity_name(store, entity, name),
            EditorEvent::SetTransform(entity, transform) => {
                self.set_entity_transform(store, entity, transform)
            }
        }
    }

    fn add_component(&mut self, store: &mut SmartStore, component_type: String) {
        if let Some(entity) = self.selected_entity {
            match component_type.as_str() {
                "Transform" => {
                    // Transform is added by default
                }
                #[cfg(feature = "js-engine")]
                "UIStyle" => {
                    store.insert(entity, UIStyle::default());
                }
                "RigidBody" => {
                    // Would need physics integration
                }
                _ => {}
            }
            self.show_add_component = false;
        }
    }

    fn set_entity_name(&mut self, _store: &mut SmartStore, _entity: Entity, _name: String) {
        // Would update entity name in a Name component
    }

    fn set_entity_transform(
        &mut self,
        _store: &mut SmartStore,
        _entity: Entity,
        _transform: Transform,
    ) {
        // Would update Transform component
    }
}

/// Events that can be sent to the editor
#[derive(Debug, Clone)]
pub enum EditorEvent {
    Toggle,
    SelectEntity(Entity),
    Deselect,
    SetSearchFilter(String),
    ShowAddComponent(Entity),
    AddComponent(String),
    SetGizmoMode(GizmoMode),
    SetEntityName(Entity, String),
    SetTransform(Entity, Transform),
}

/// Transform data for inspector
#[derive(Debug, Clone, Copy, Default)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4], // quaternion
    pub scale: [f32; 3],
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}
