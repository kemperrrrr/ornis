use vello::peniko::{Color, FontData};
use crate::render::UIRenderer;
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use ornis_core::Entity;

/// In-game editor overlay rendered directly with Vello primitives.
/// Provides entity hierarchy, component inspector, and gizmo tools.
pub struct EditorOverlay {
    visible: bool,
    entity_count: u32,
    
    /// Selected entity for inspection
    selected_entity: Option<Entity>,
    
    /// Editor mode
    mode: EditorMode,
    
    /// Scroll offsets for panels
    hierarchy_scroll: f64,
    inspector_scroll: f64,
    
    /// Search filter for entities
    search_filter: String,
    
    /// UI state
    show_add_component: bool,
    add_component_type: String,
    
    /// Gizmo state
    gizmo_mode: GizmoMode,
    gizmo_active: bool,
    gizmo_axis: Option<GizmoAxis>,
    gizmo_start_pos: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    Hierarchy,
    Inspector,
    AddComponent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
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

impl EditorOverlay {
    pub fn new() -> Self {
        Self {
            visible: false,
            entity_count: 0,
            selected_entity: None,
            mode: EditorMode::Hierarchy,
            hierarchy_scroll: 0.0,
            inspector_scroll: 0.0,
            search_filter: String::new(),
            show_add_component: false,
            add_component_type: String::new(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_active: false,
            gizmo_axis: None,
            gizmo_start_pos: None,
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
        self.mode = EditorMode::Inspector;
    }

    pub fn deselect(&mut self) {
        self.selected_entity = None;
        self.mode = EditorMode::Hierarchy;
    }

    pub fn set_search_filter(&mut self, filter: String) {
        self.search_filter = filter;
    }

    pub fn show_add_component(&mut self, entity: Entity) {
        self.selected_entity = Some(entity);
        self.show_add_component = true;
        self.mode = EditorMode::AddComponent;
    }

    pub fn get_gizmo_mode(&self) -> GizmoMode {
        self.gizmo_mode
    }

    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode = mode;
    }

    /// Paint the editor overlay on top of the current Vello scene.
    pub fn paint(&self, renderer: &mut UIRenderer, viewport_w: f64, viewport_h: f64, font: &FontData) {
        if !self.visible {
            return;
        }

        self.paint_hierarchy_panel(renderer, viewport_w, viewport_h, font);
        self.paint_inspector_panel(renderer, viewport_w, viewport_h, font);
        self.paint_gizmo(renderer, viewport_w, viewport_h, font);
        self.paint_toolbar(renderer, viewport_w, viewport_h, font);
    }

    fn paint_hierarchy_panel(&self, renderer: &mut UIRenderer, viewport_w: f64, viewport_h: f64, font: &FontData) {
        let panel_w = 280.0;
        let panel_x = 0.0;

        // Panel background
        renderer.fill_rounded_rect(panel_x, 0.0, panel_w, viewport_h, 0.0, Color::new([0.12, 0.12, 0.14, 0.92]));
        renderer.stroke_rect(panel_w, 0.0, 1.0, viewport_h, 1.0, Color::new([0.27, 0.27, 0.27, 1.0]));

        // Title bar
        renderer.fill_rect(panel_x, 0.0, panel_w, 40.0, Color::new([0.18, 0.18, 0.20, 1.0]));
        renderer.fill_text(panel_x + 16.0, 18.0, "Hierarchy", 16.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
        
        // Search box
        let search_y = 48.0;
        renderer.fill_rounded_rect(panel_x + 12.0, search_y, panel_w - 24.0, 28.0, 4.0, Color::new([1.0, 1.0, 1.0, 0.08]));
        renderer.fill_text(panel_x + 20.0, search_y + 20.0, &format!("Search: {}", self.search_filter), 12.0, Color::new([0.7, 0.7, 0.7, 1.0]), font);

        // Entity list
        let list_y = search_y + 44.0;
        let list_h = viewport_h - list_y - 120.0;
        
        // Clip rect for scrolling
        let entities_per_row = 1;
        let item_h = 24.0;
        let visible_count = (list_h / item_h).floor() as usize + 1;
        
        // TODO: Draw actual entity list from ECS
        // For now show placeholder
        renderer.fill_text(panel_x + 16.0, list_y + 20.0, "Entities:", 12.0, Color::new([0.67, 0.67, 0.67, 1.0]), font);
        
        if self.search_filter.is_empty() {
            renderer.fill_text(panel_x + 24.0, list_y + 50.0, "(No entities)", 13.0, Color::new([0.5, 0.5, 0.5, 1.0]), font);
        }
    }

    fn paint_inspector_panel(&self, renderer: &mut UIRenderer, viewport_w: f64, viewport_h: f64, font: &FontData) {
        if self.selected_entity.is_none() {
            return;
        }

        let panel_w = 320.0;
        let panel_x = viewport_w - panel_w;

        // Panel background
        renderer.fill_rounded_rect(panel_x, 0.0, panel_w, viewport_h, 0.0, Color::new([0.12, 0.12, 0.14, 0.92]));
        renderer.stroke_rect(panel_x, 0.0, 1.0, viewport_h, 1.0, Color::new([0.27, 0.27, 0.27, 1.0]));

        // Title bar
        let entity = self.selected_entity.unwrap();
        renderer.fill_rect(panel_x, 0.0, panel_w, 40.0, Color::new([0.18, 0.18, 0.20, 1.0]));
        renderer.fill_text(panel_x + 16.0, 18.0, &format!("Inspector: Entity {}", entity.id()), 14.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
        
        // Component type badge
        renderer.fill_rounded_rect(panel_x + 16.0, 52.0, 100.0, 24.0, 4.0, Color::new([0.23, 0.43, 0.94, 0.3]));
        renderer.fill_text(panel_x + 20.0, 64.0, "Transform", 12.0, Color::new([0.5, 0.7, 1.0, 1.0]), font);
        renderer.fill_rounded_rect(panel_x + 120.0, 52.0, 100.0, 24.0, 4.0, Color::new([0.23, 0.94, 0.43, 0.3]));
        renderer.fill_text(panel_x + 124.0, 64.0, "UIStyle", 12.0, Color::new([0.5, 1.0, 0.7, 1.0]), font);

        // Add component button
        let btn_y = 90.0;
        let btn_hover = self.show_add_component;
        renderer.fill_rounded_rect(panel_x + 12.0, btn_y, panel_w - 24.0, 32.0, 4.0, 
            if btn_hover { Color::new([0.23, 0.43, 0.94, 0.5]) } else { Color::new([1.0, 1.0, 1.0, 0.08]) });
        renderer.fill_text(panel_x + 24.0, btn_y + 18.0, "+ Add Component", 13.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);

        // Component details
        if self.show_add_component {
            self.paint_add_component_modal(renderer, viewport_w, viewport_h, font);
        }
    }

    fn paint_add_component_modal(&self, renderer: &mut UIRenderer, viewport_w: f64, viewport_h: f64, font: &FontData) {
        // Semi-transparent overlay
        renderer.fill_rect(0.0, 0.0, viewport_w, viewport_h, Color::new([0.0, 0.0, 0.0, 0.5]));
        
        // Modal panel
        let modal_w = 400.0;
        let modal_h = 300.0;
        let modal_x = (viewport_w - modal_w) / 2.0;
        let modal_y = (viewport_h - modal_h) / 2.0;
        
        renderer.fill_rounded_rect(modal_x, modal_y, modal_w, modal_h, 8.0, Color::new([0.15, 0.15, 0.18, 0.98]));
        renderer.stroke_rect(modal_x, modal_y, modal_w, modal_h, 1.0, Color::new([0.3, 0.3, 0.3, 1.0]));
        
        renderer.fill_text(modal_x + 24.0, modal_y + 28.0, "Add Component", 18.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
        renderer.fill_text(modal_x + 24.0, modal_y + 58.0, "Select component type:", 13.0, Color::new([0.7, 0.7, 0.7, 1.0]), font);
        
        let types = ["Transform", "UIStyle", "RigidBody", "Mesh", "Light", "Camera", "AudioSource"];
        for (i, t) in types.iter().enumerate() {
            let y = modal_y + 88.0 + i as f64 * 32.0;
            let selected = self.add_component_type == *t;
            renderer.fill_rounded_rect(modal_x + 16.0, y, modal_w - 32.0, 28.0, 4.0,
                if selected { Color::new([0.23, 0.43, 0.94, 0.5]) } else { Color::new([1.0, 1.0, 1.0, 0.05]) });
            renderer.fill_text(modal_x + 28.0, y + 18.0, t, 13.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
        }
        
        // Buttons
        let btn_y = modal_y + modal_h - 52.0;
        renderer.fill_rounded_rect(modal_x + modal_w - 120.0, btn_y, 52.0, 32.0, 4.0, Color::new([0.5, 0.5, 0.5, 0.5]));
        renderer.fill_text(modal_x + modal_w - 104.0, btn_y + 16.0, "Cancel", 13.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
        
        renderer.fill_rounded_rect(modal_x + modal_w - 60.0, btn_y, 52.0, 32.0, 4.0, Color::new([0.23, 0.43, 0.94, 0.8]));
        renderer.fill_text(modal_x + modal_w - 44.0, btn_y + 16.0, "Add", 13.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
    }

    fn paint_gizmo(&self, _renderer: &mut UIRenderer, _viewport_w: f64, _viewport_h: f64, _font: &FontData) {
        // TODO: Implement 3D gizmo rendering
        // For now, just show gizmo mode indicator
    }

    fn paint_toolbar(&self, renderer: &mut UIRenderer, viewport_w: f64, viewport_h: f64, font: &FontData) {
        // Bottom toolbar
        let toolbar_h = 40.0;
        let toolbar_y = viewport_h - toolbar_h;
        
        renderer.fill_rect(0.0, toolbar_y, viewport_w, toolbar_h, Color::new([0.10, 0.10, 0.12, 0.95]));
        renderer.stroke_rect(0.0, toolbar_y, viewport_w, 1.0, 1.0, Color::new([0.2, 0.2, 0.2, 1.0]));
        
        // Mode buttons
        let modes = [("1", "Translate", GizmoMode::Translate), ("2", "Rotate", GizmoMode::Rotate), ("3", "Scale", GizmoMode::Scale)];
        for (i, (key, label, mode)) in modes.iter().enumerate() {
            let x = 20.0 + i as f64 * 120.0;
            let active = self.gizmo_mode == *mode;
            let color = if active { Color::new([0.23, 0.43, 0.94, 0.8]) } else { Color::new([1.0, 1.0, 1.0, 0.1]) };
            renderer.fill_rounded_rect(x, toolbar_y + 6.0, 100.0, 28.0, 4.0, color);
            renderer.fill_text(x + 12.0, toolbar_y + 20.0, &format!("{} {}", key, label), 12.0, Color::new([1.0, 1.0, 1.0, 1.0]), font);
        }
        
        // Status
        let status = if let Some(entity) = self.selected_entity {
            format!("Selected: Entity {} | Mode: {:?}", entity.id(), self.gizmo_mode)
        } else {
            "No selection".to_string()
        };
        renderer.fill_text(viewport_w - 300.0, toolbar_y + 14.0, &status, 12.0, Color::new([0.7, 0.7, 0.7, 1.0]), font);
    }
}

impl Default for EditorOverlay {
    fn default() -> Self {
        Self::new()
    }
}

/// Gizmo system for 3D transform manipulation
pub struct GizmoSystem {
    pub mode: GizmoMode,
    pub active: bool,
    pub axis: Option<GizmoAxis>,
    pub transform: Mat4,
    pub start_pos: Option<Vec3>,
    pub viewport_size: (f64, f64),
    pub camera_view: Mat4,
    pub camera_proj: Mat4,
}

impl GizmoSystem {
    pub fn new() -> Self {
        Self {
            mode: GizmoMode::Translate,
            active: false,
            axis: None,
            transform: Mat4::IDENTITY,
            start_pos: None,
            viewport_size: (800.0, 600.0),
            camera_view: Mat4::IDENTITY,
            camera_proj: Mat4::IDENTITY,
        }
    }

    pub fn set_mode(&mut self, mode: GizmoMode) {
        self.mode = mode;
    }

    pub fn set_transform(&mut self, transform: Mat4) {
        self.transform = transform;
    }

    pub fn set_camera(&mut self, view: Mat4, proj: Mat4) {
        self.camera_view = view;
        self.camera_proj = proj;
    }

    pub fn set_viewport(&mut self, size: (f64, f64)) {
        self.viewport_size = size;
    }

    /// Check if mouse ray hits gizmo axis
    pub fn raycast_gizmo(&self, mouse_pos: (f64, f64)) -> Option<GizmoAxis> {
        // Convert screen space to normalized device coordinates
        let (vx, vy) = self.viewport_size;
        let ndc_x = (mouse_pos.0 / vx) * 2.0 - 1.0;
        let ndc_y = 1.0 - (mouse_pos.1 / vy) * 2.0;
        
        // Ray in clip space
        let clip = Vec4::new(ndc_x as f32, ndc_y as f32, -1.0, 1.0);
        let inv_proj = self.camera_proj.inverse();
        let eye = inv_proj * clip;
        let eye = Vec4::new(eye.x, eye.y, -1.0, 0.0);
        let inv_view = self.camera_view.inverse();
        let ray_dir = (inv_view * eye).xyz().normalize();
        let ray_origin = (inv_view * Vec4::new(0.0, 0.0, 0.0, 1.0)).xyz();
        
        // Test against gizmo axes (simplified - axis-aligned lines from origin)
        let axes = [
            (GizmoAxis::X, Vec3::X),
            (GizmoAxis::Y, Vec3::Y),
            (GizmoAxis::Z, Vec3::Z),
        ];
        
        let mut best_dist = f32::MAX;
        let mut best_axis = None;
        
        for (axis, dir) in axes {
            // Distance from ray to axis line
            let origin_to_line = self.transform.w_axis.xyz() - ray_origin;
            let cross = origin_to_line.cross(ray_dir);
            let dist = cross.length() / ray_dir.length();
            
            if dist < 0.1 && dist < best_dist {
                best_dist = dist;
                best_axis = Some(axis);
            }
        }
        
        best_axis
    }

    /// Start gizmo manipulation
    pub fn begin_manipulation(&mut self, axis: GizmoAxis, mouse_pos: (f64, f64)) {
        self.active = true;
        self.axis = Some(axis);
        self.start_pos = Some(Vec3::new(mouse_pos.0 as f32, mouse_pos.1 as f32, 0.0));
    }

    /// Update manipulation based on mouse movement
    pub fn update_manipulation(&mut self, mouse_pos: (f64, f64)) -> Vec3 {
        let delta = Vec3::new(
            mouse_pos.0 as f32 - self.start_pos.unwrap().x,
            mouse_pos.1 as f32 - self.start_pos.unwrap().y,
            0.0,
        ) * 0.01; // Scale factor
        
        match self.mode {
            GizmoMode::Translate => {
                if let Some(axis) = self.axis {
                    let dir = match axis {
                        GizmoAxis::X => Vec3::X,
                        GizmoAxis::Y => Vec3::Y,
                        GizmoAxis::Z => Vec3::Z,
                        GizmoAxis::XY | GizmoAxis::XZ | GizmoAxis::YZ => Vec3::ONE,
                        GizmoAxis::All => Vec3::ONE,
                    };
                    delta * dir
                } else {
                    Vec3::ZERO
                }
            }
            GizmoMode::Rotate => {
                // Rotation around camera forward
                let axis = match self.axis {
                    Some(GizmoAxis::X) => Vec3::X,
                    Some(GizmoAxis::Y) => Vec3::Y,
                    Some(GizmoAxis::Z) => Vec3::Z,
                    Some(GizmoAxis::XY) => Vec3::X, // Primary axis for XY plane
                    Some(GizmoAxis::XZ) => Vec3::X, // Primary axis for XZ plane
                    Some(GizmoAxis::YZ) => Vec3::Y, // Primary axis for YZ plane
                    _ => Vec3::Z,
                };
                delta * axis
            }
            GizmoMode::Scale => {
                Vec3::ONE + delta * 0.1
            }
        }
    }
}

impl Default for GizmoSystem {
    fn default() -> Self {
        Self::new()
    }
}