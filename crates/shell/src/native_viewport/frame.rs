//! Scene → GPU frame state for the native viewport.
//!
//! One [`ViewportFrame`] owns everything a frame needs once the surface
//! exists: the replica world (the same [`ReplicaGameWorld`] the browser
//! viewport uses), the renderer, the frame plan and the scene version the GPU
//! data came from. [`ViewportFrame::apply_snapshot`] is the only way a scene
//! enters it, so "the GPU never draws a stale world" is a property of the
//! type rather than of the call order.
//!
//! The frame path is deliberately the *same* one the WASM viewport uses —
//! `ReplicaGameWorld::frame_upload` → `Renderer3D::upload_*` →
//! `RenderFrame3D::render` — so the prototype cannot introduce a rendering
//! difference that the perf budget would have to absorb. The shell is a new
//! *caller* of the render crate, not a new implementation inside it.
//!
//! macOS-only, like the surface it draws into (see the parent module).

use ornis_app::{ReplicaGameWorld, SceneSnapshot, SnapshotEntityMeta};
use ornis_core::InputState;
use ornis_editor::{Hovered, Selected, install_editor};
use ornis_render::{
    AaWiring, AntiAliasMode, Bloom, CameraView, FrameUpload, OrbitCamera, PointerDpr,
    RenderContext, RenderFrame3D, RenderLights, Renderer3D, Technique, camera_view_projection,
    read_orbit_camera,
};

/// Outline flag for a selected instance (the WASM viewport's mapping).
const OUTLINE_SELECTED: u32 = 2;
/// Outline flag for a hovered instance.
const OUTLINE_HOVERED: u32 = 1;
/// Placeholder sphere tessellation before the first snapshot arrives.
const PLACEHOLDER_MESH: (u32, u32) = (8, 6);

/// What the last applied frame holds, for the shell's status line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Instanced draws uploaded from the current snapshot.
    pub instances: u32,
    /// Scene snapshot version the GPU data came from (`0` = nothing yet).
    pub version: u64,
}

/// Surface description the frame is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceSpec {
    /// Swap-chain format.
    pub format: wgpu::TextureFormat,
    /// Drawable width, device pixels.
    pub width: u32,
    /// Drawable height, device pixels.
    pub height: u32,
}

/// GPU state of the native viewport.
pub struct ViewportFrame {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer3D,
    plan: RenderFrame3D,
    world: ReplicaGameWorld,
    mesh: ornis_render::Mesh,
    mesh_params: (u32, u32),
    instance_count: u32,
    version: u64,
    size: (u32, u32),
}

impl ViewportFrame {
    /// Builds the frame state for a surface, with an empty replica world.
    ///
    /// Anti-aliasing follows the WASM viewport (FXAA, single-sample): the
    /// shell renders the same picture as the browser for the same scene, which
    /// is what makes "the viewport behaves the same in both places" a
    /// testable claim. Hardware MSAA negotiation stays the native runner's
    /// policy for its own window; the shell does not silently differ.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, spec: SurfaceSpec) -> Self {
        let mut world = ReplicaGameWorld::new_replica();
        install_editor(world.engine_mut());
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: spec.format,
            width: spec.width.max(1),
            height: spec.height.max(1),
            present_mode: ornis_render::surface_present_mode(),
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let renderer = Renderer3D::new(&device, &config, ornis_render::SINGLE_SAMPLE_COUNT);
        renderer.set_anti_alias(AntiAliasMode::Fxaa);
        let plan = RenderFrame3D::new_with_aa_outline(
            spec.format,
            (config.width, config.height),
            Technique::Hybrid,
            Bloom::Off,
            AaWiring::new(AntiAliasMode::Fxaa),
        );
        Self {
            mesh: ornis_render::create_sphere(&device, 1.0, PLACEHOLDER_MESH.0, PLACEHOLDER_MESH.1),
            device,
            queue,
            renderer,
            plan,
            world,
            mesh_params: PLACEHOLDER_MESH,
            instance_count: 0,
            version: 0,
            size: (config.width, config.height),
        }
    }

    /// Resizes the drawable (the layer bounds changed).
    ///
    /// Renderer targets and the plan's `MatchSurface` resources move
    /// together, exactly as the native runner's `Resized` handler does — a
    /// renderer and plan at different sizes is the classic resize flicker.
    pub fn resize(&mut self, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size == self.size {
            return;
        }
        self.size = size;
        self.renderer.resize(&self.device, size.0, size.1);
        self.plan.set_surface_size(size.0, size.1);
    }

    /// Applies one `/api/scene` snapshot.
    ///
    /// Returns `false` for a reply older than what is already uploaded: the
    /// viewport polls and streams events on the same channel, so an
    /// out-of-order snapshot must not roll the scene back. The whole world is
    /// replaced (the contract's semantics — the server owns the scene) and the
    /// GPU data is re-uploaded in one pass.
    pub fn apply_snapshot(&mut self, snapshot: &SceneSnapshot) -> bool {
        if snapshot.version <= self.version {
            return false;
        }
        self.world.replace_scene(&snapshot.scene);
        mirror_marks(&mut self.world, &snapshot.meta);
        self.world
            .spawn(OrbitCamera::from_desc(&snapshot.scene.camera));
        self.world.frame(0.0);

        let upload = self.world.frame_upload();
        let lights = RenderLights::from_scene(&snapshot.scene);
        if upload.mesh_params != self.mesh_params {
            self.mesh = ornis_render::create_sphere(
                &self.device,
                1.0,
                upload.mesh_params.0,
                upload.mesh_params.1,
            );
            self.mesh_params = upload.mesh_params;
        }
        self.renderer
            .upload_materials(&self.device, &self.queue, &upload.materials);
        self.renderer
            .upload_instances(&self.device, &self.queue, &upload.instances);
        // `Color` stores linear RGB; the renderer's ambient argument is that
        // triple (alpha is not part of the lighting uniform).
        let ambient = lights.ambient.to_linear_rgb().as_array();
        self.renderer
            .set_lights(&self.queue, ambient, &lights.set_lights_args());
        self.plan
            .set_outline_flags(&outline_flags(&self.world, &upload));
        self.instance_count = u32::try_from(upload.instances.len()).unwrap_or(u32::MAX);
        self.version = snapshot.version;
        true
    }

    /// Advances the simulation by `delta` seconds and draws one frame into
    /// `target`.
    ///
    /// `delta` is the wall-clock interval since the previous frame; the caller
    /// clamps it (see `clamp_frame_dt`) so a hitch cannot fast-forward the
    /// simulation.
    pub fn render(&mut self, target: &wgpu::TextureView, delta: f32) {
        self.world.frame(delta);
        self.sync_camera();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shell viewport encoder"),
            });
        self.plan.render(
            RenderContext {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                target,
            },
            &self.renderer,
            &self.mesh,
            self.instance_count,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Publishes the orbit camera for the current drawable size.
    fn sync_camera(&mut self) {
        let Some(camera) = read_orbit_camera(self.world.engine()) else {
            return;
        };
        let (view_proj, eye) = camera_view_projection(&camera.view_parameters(), self.size);
        self.renderer
            .set_camera(&self.queue, &view_proj.to_cols_array_2d(), eye.to_array());
    }

    /// Applies one forwarded viewport input snapshot (see the crate's
    /// `viewport_ipc`): the pressed sets are authoritative for the frame, so a
    /// dropped post cannot leave a key or button stuck.
    pub fn apply_input(&mut self, input: &editor_backend::BrowserInput) {
        let world = self.world.engine_mut().world_mut();
        if world.resources().get::<InputState>().is_none() {
            world.resources_mut().insert(InputState::default());
        }
        let Some(state) = world.resources_mut().get_mut::<InputState>() else {
            return;
        };
        state.apply_snapshot(
            &input.pressed_keys,
            &input.pressed_mouse_buttons,
            input.pointer_position,
            input.pointer_delta,
            input.wheel_delta,
        );
    }

    /// Publishes the scale that turns pointer deltas into CSS-anchored camera
    /// motion (`PointerDpr`); non-finite or non-positive values reset to `1.0`.
    pub fn set_pointer_scale(&mut self, scale: f32) {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        self.world
            .engine_mut()
            .world_mut()
            .resources_mut()
            .insert(PointerDpr(scale));
    }

    /// The replica's current orbit camera view (diagnostics and tests).
    pub fn camera_view(&self) -> Option<CameraView> {
        read_orbit_camera(self.world.engine()).map(|camera| camera.view_parameters())
    }

    /// Diagnostics for the shell's status line.
    pub fn stats(&self) -> FrameStats {
        FrameStats {
            instances: self.instance_count,
            version: self.version,
        }
    }

    /// Current drawable size, device pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}

/// Outline flags parallel to `upload.instances`: `2` selected, `1` hovered.
///
/// The WASM viewport's mapping, so selection chrome looks the same in both.
fn outline_flags(world: &ReplicaGameWorld, upload: &FrameUpload) -> Vec<u32> {
    let Some(store) = world.engine().world().store() else {
        return Vec::new();
    };
    let selected = store.read_lane::<Selected>();
    let hovered = store.read_lane::<Hovered>();
    upload
        .instance_entities
        .iter()
        .map(|entity| {
            let is_selected = selected.as_ref().is_some_and(|lane| lane.contains(*entity));
            let is_hovered = hovered.as_ref().is_some_and(|lane| lane.contains(*entity));
            match (is_selected, is_hovered) {
                (true, _) => OUTLINE_SELECTED,
                (false, true) => OUTLINE_HOVERED,
                (false, false) => 0,
            }
        })
        .collect()
}

/// Mirrors the snapshot's editor marks onto the freshly rebuilt replica.
///
/// `replace_scene` destroys and recreates every scene entity, so marks cannot
/// survive a snapshot on their own. A length mismatch means the payload
/// changed shape between parse and insert: marks are cosmetic, so clearing
/// beats mislabelling the wrong entity.
fn mirror_marks(world: &mut ReplicaGameWorld, meta: &[SnapshotEntityMeta]) {
    let entities = world.entities().to_vec();
    let Some(store) = world.engine_mut().world_mut().store_mut() else {
        return;
    };
    if entities.len() != meta.len() {
        for entity in entities {
            let mut lane = match store.write_lane::<Selected>() {
                Some(lane) => lane,
                None => return,
            };
            lane.remove(entity);
        }
        return;
    }
    for (entity, meta) in entities.into_iter().zip(meta) {
        if meta.editor.selected {
            store.insert(entity, Selected);
        }
        if meta.editor.hovered {
            store.insert(entity, Hovered);
        }
    }
}
