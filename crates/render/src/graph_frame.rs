//! Phase 1: wgpu executor for the render graph.
//!
//! Maps pool slots to real `wgpu::Texture` objects (created lazily, reused
//! every frame) and resolves every resource's view during pass execution —
//! either from the slot pool or from externally provided views (swapchain).
//! Also wires the four existing `Renderer3D` passes (gbuffer → lighting →
//! forward → composite) as graph nodes ([`RenderGraph3D`]).
//!
//! Lifetimes are computed by the pure [`RenderGraph`] layout; this module
//! only owns GPU objects. On wgpu, barriers are handled by wgpu itself, so
//! the executor is small by design.

use crate::graph_passes::{
    Albedo, Bloom0, Bloom1, Bloom2, BloomBright, BloomDown1Pass, BloomDown2Pass, BloomUp0Pass,
    BloomUp1Pass, Composite, CompositeDeferred, CompositeDeferredBloom, CompositeForward,
    CompositeForwardBloom, CompositeHybrid, CompositeHybridBloom, Depth, Forward, FromDeferred,
    FromForward, GbufferPass, Hdr, HdrFwd, LightingPass, MaterialId, MaterialParams, Normal,
    OwnsDepth, SharedDepth, Target, WorldPosition,
};
use crate::mesh::Mesh;
use crate::render_graph::{GraphLayout, PassLayout, RenderGraph, ResourceId, ResourceLayout};
use crate::renderer::Renderer3D;
use crate::system::{Frame, SystemSet};
use std::collections::HashMap;

/// One pooled texture per render-graph slot.
#[derive(Debug)]
struct PooledTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bytes: u64,
}

/// Bytes per pixel for the texture formats used by the engine's renderer.
pub fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb
        | wgpu::TextureFormat::R32Uint
        | wgpu::TextureFormat::Rg16Float
        | wgpu::TextureFormat::Depth32Float
        | wgpu::TextureFormat::Depth24Plus => 4,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rg32Float => 8,
        wgpu::TextureFormat::Rgba32Float => 16,
        other => panic!("format_bytes_per_pixel: unsupported format {other:?}"),
    }
}

/// Executes a [`GraphLayout`] on wgpu: lazily creates one texture per pool
/// slot and hands every pass a [`PassViews`] resolver.
#[derive(Debug, Default)]
pub struct GraphExecutor {
    pool: Vec<Option<PooledTexture>>,
    external_views: HashMap<ResourceId, wgpu::TextureView>,
    surface_size: Option<(u32, u32)>,
}

impl GraphExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Provides the view backing an external resource (see
    /// `RenderGraph::external_output`). Call before `execute`.
    pub fn set_external_view(&mut self, id: ResourceId, view: wgpu::TextureView) {
        self.external_views.insert(id, view);
    }

    /// Number of pooled textures currently allocated.
    pub fn slots_len(&self) -> usize {
        self.pool.len()
    }

    /// Total bytes of the pooled GPU textures at the current surface size.
    /// The gap between this and the legacy path's persistent textures is
    /// the aliasing win (see `Renderer3D::texture_budget`).
    pub fn texture_budget(&self) -> u64 {
        self.pool.iter().flatten().map(|t| t.bytes).sum()
    }

    /// Executes `layout`: for each pass in order, `run` receives the encoder
    /// and a [`PassViews`] resolver for the live resources.
    pub fn execute<'a>(
        &'a mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &'a GraphLayout,
        run: impl FnMut(&mut wgpu::CommandEncoder, PassViews<'a>),
    ) {
        self.ensure_pool(device, layout);
        let pool = &self.pool;
        let externals = &self.external_views;
        let mut run = run;
        for index in 0..layout.passes.len() {
            run(
                encoder,
                PassViews {
                    layout,
                    pool,
                    externals,
                    index,
                },
            );
        }
    }

    fn ensure_pool(&mut self, device: &wgpu::Device, layout: &GraphLayout) {
        // Recreate all textures when the surface size changes.
        if self.surface_size != Some(layout.surface_size) {
            self.pool.clear();
            self.surface_size = Some(layout.surface_size);
        }
        self.pool.resize_with(layout.slots.len(), || None);
        for (i, slot) in layout.slots.iter().enumerate() {
            if self.pool[i].is_none() {
                self.pool[i] = Some(create_pooled_texture(device, slot, layout.surface_size, i));
            }
        }
    }
}

fn create_pooled_texture(
    device: &wgpu::Device,
    slot: &crate::render_graph::PoolSlot,
    surface_size: (u32, u32),
    index: usize,
) -> PooledTexture {
    let (width, height) = slot.spec.size.resolve(surface_size);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("render graph slot #{index}")),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: slot.spec.samples,
        dimension: wgpu::TextureDimension::D2,
        format: slot.spec.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bytes = format_bytes_per_pixel(slot.spec.format) as u64
        * width as u64
        * height as u64
        * slot.spec.samples as u64;
    PooledTexture {
        _texture: texture,
        view,
        bytes,
    }
}

/// Per-pass view resolver handed to pass callbacks during execution.
#[derive(Debug)]
pub struct PassViews<'a> {
    layout: &'a GraphLayout,
    pool: &'a [Option<PooledTexture>],
    externals: &'a HashMap<ResourceId, wgpu::TextureView>,
    index: usize,
}

impl<'a> PassViews<'a> {
    /// The pass in execution order.
    pub fn pass(&self) -> &'a PassLayout {
        &self.layout.passes[self.index]
    }

    /// Index of the pass within the layout.
    pub fn pass_index(&self) -> usize {
        self.index
    }

    /// Resources alive on this pass.
    pub fn alive(&self) -> &'a [ResourceId] {
        &self.layout.pass_alive[self.index]
    }

    /// Resource metadata.
    pub fn resource(&self, id: ResourceId) -> &'a ResourceLayout {
        &self.layout.resources[id.0 as usize]
    }

    /// Pool slot for the resource, if it is alive and pooled on this pass.
    pub fn slot_of(&self, id: ResourceId) -> Option<usize> {
        let rl = self.resource(id);
        if rl.alive_at(self.index) {
            rl.slot
        } else {
            None
        }
    }

    /// Texture view backing `id` at the current pass: the external view for
    /// external resources, otherwise the pooled slot texture.
    ///
    /// # Panics
    /// Panics if the resource is not alive on this pass, if its external
    /// view is not set, or if its slot texture was never created.
    pub fn view_of(&self, id: ResourceId) -> &'a wgpu::TextureView {
        let rl = self.resource(id);
        assert!(
            rl.alive_at(self.index),
            "resource {id:?} is not alive on pass {}",
            self.index
        );
        if rl.external {
            self.externals
                .get(&id)
                .unwrap_or_else(|| panic!("external view for {id:?} is not set"))
        } else {
            let slot = rl.slot.expect("resource has no pool slot");
            self.pool[slot]
                .as_ref()
                .expect("pool slot not created")
                .view_ref()
        }
    }
}

impl PooledTexture {
    fn view_ref(&self) -> &wgpu::TextureView {
        &self.view
    }
}

/// Total bytes the pool will allocate for this layout at its surface
/// size — the device-free counterpart of [`GraphExecutor::texture_budget`]
/// (golden tests, S0 metrics and the S4 budget groundwork).
pub fn planned_pool_bytes(layout: &GraphLayout) -> u64 {
    layout
        .slots
        .iter()
        .map(|slot| {
            let (w, h) = slot.spec.size.resolve(layout.surface_size);
            format_bytes_per_pixel(slot.spec.format) as u64 * w as u64 * h as u64
        })
        .sum()
}

/// Graph-driven frame over the `Renderer3D` passes:
/// gbuffer → lighting → forward → composite.
///
/// G-buffer textures are transient: albedo/normal/material_id/world_position/
/// material_params live only on the gbuffer pass, the depth spans
/// gbuffer..forward, and both HDR targets are pooled after gbuffer, so
/// non-overlapping resources with the same spec share one GPU texture.
pub struct RenderGraph3D {
    graph: RenderGraph,
    executor: GraphExecutor,
    ids: GraphIds,
    /// Typed S2 systems: `type → ResourceId` registry + type-erased runners
    /// for the passes declared as `GraphPass` implementations
    /// (see [`crate::graph_passes`] and [`crate::system`]).
    systems: SystemSet,
    bloom: bool,
    technique: Technique,
}

/// Which lighting technique the graph wires up. The choice is expressed
/// purely as which nodes get added — `gbuffer`/`lighting` appear for
/// deferred work, `forward` for forward work; the composite pass mixes
/// whichever layers exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Technique {
    /// Only the forward node: mesh → composite. No G-buffer, no lighting
    /// pass; best for cheap scenes or weak GPUs.
    Forward,
    /// Only the deferred chain: gbuffer → lighting → composite. The
    /// forward layer is dropped, so transparency needs another future node.
    Deferred,
    /// Both: opaque lighting through the G-buffer, extra forward work on
    /// top — the engine's classic path.
    Hybrid,
}

impl Technique {
    /// Whether the `gbuffer`/`lighting` nodes exist for this technique.
    pub fn has_deferred(&self) -> bool {
        !matches!(self, Self::Forward)
    }

    /// Whether the `forward` node exists for this technique.
    pub fn has_forward(&self) -> bool {
        !matches!(self, Self::Deferred)
    }

    /// Composite shader mode: 0 = deferred-only, 1 = forward-only,
    /// 2 = hybrid (deferred + forward over it).
    pub fn composite_mode(&self) -> u32 {
        match self {
            Self::Forward => 1,
            Self::Deferred => 0,
            Self::Hybrid => 2,
        }
    }
}

/// Resource handles of the [`RenderGraph3D`] graph.
#[derive(Debug, Clone, Copy)]
pub struct GraphIds {
    pub albedo: ResourceId,
    pub normal: ResourceId,
    pub material_id: ResourceId,
    pub world_position: ResourceId,
    pub material_params: ResourceId,
    pub depth: ResourceId,
    pub hdr: ResourceId,
    pub hdr_fwd: ResourceId,
    pub target: ResourceId,
    /// Bloom chain levels: 1/2, 1/4, 1/8 of the surface. Always declared so
    /// every graph shares one `GraphIds` shape; they only consume pool slots
    /// when the bloom passes exist (`new_with_bloom`).
    pub bloom0: ResourceId,
    pub bloom1: ResourceId,
    pub bloom2: ResourceId,
}

impl RenderGraph3D {
    /// Builds the pass graph. `surface_format` is the render target format
    /// (the lighting pass writes into it); `surface_size` seeds
    /// `SizePolicy::MatchSurface` resources.
    pub fn new(surface_format: wgpu::TextureFormat, surface_size: (u32, u32)) -> Self {
        Self::new_with(surface_format, surface_size, Technique::Hybrid, false)
    }

    /// Like [`new`](Self::new), plus the bloom cascade:
    ///
    /// `bloom_down0` (bright-pass at 1/2) → `bloom_down1` (1/4) →
    /// `bloom_down2` (1/8) → `bloom_up1` → `bloom_up0`, where each upsample
    /// adds its level back over the downsampled content (`LoadOp::Load`).
    /// The composite pass then mixes the final bloom level into the HDR
    /// result.
    pub fn new_with_bloom(surface_format: wgpu::TextureFormat, surface_size: (u32, u32)) -> Self {
        Self::new_with(surface_format, surface_size, Technique::Hybrid, true)
    }

    /// Builds the graph for a specific [`Technique`] with optional bloom.
    /// The technique decides which nodes are wired: deferred nodes
    /// (`gbuffer`, `lighting`) exist only when [`Technique::has_deferred`],
    /// the `forward` node only when [`Technique::has_forward`], and the
    /// bloom chain reads whichever HDR layer the technique produces.
    pub fn new_with(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: bool,
    ) -> Self {
        let mut graph = RenderGraph::new(surface_size);
        // S2: resources are registered by type; specs/names (and the
        // ResourceId order) mirror the imperative wiring exactly.
        let mut systems = SystemSet::new();
        let ids = GraphIds {
            albedo: systems.register_resource::<Albedo>(&mut graph, surface_format),
            normal: systems.register_resource::<Normal>(&mut graph, surface_format),
            material_id: systems.register_resource::<MaterialId>(&mut graph, surface_format),
            world_position: systems.register_resource::<WorldPosition>(&mut graph, surface_format),
            material_params: systems
                .register_resource::<MaterialParams>(&mut graph, surface_format),
            depth: systems.register_resource::<Depth>(&mut graph, surface_format),
            hdr: systems.register_resource::<Hdr>(&mut graph, surface_format),
            hdr_fwd: systems.register_resource::<HdrFwd>(&mut graph, surface_format),
            target: systems.register_resource::<Target>(&mut graph, surface_format),
            bloom0: systems.register_resource::<Bloom0>(&mut graph, surface_format),
            bloom1: systems.register_resource::<Bloom1>(&mut graph, surface_format),
            bloom2: systems.register_resource::<Bloom2>(&mut graph, surface_format),
        };
        if technique.has_deferred() {
            systems.add_system(&mut graph, GbufferPass);
            systems.add_system(&mut graph, LightingPass);
        }
        if technique.has_forward() {
            // In forward-only mode the pass owns the depth buffer; in
            // hybrid it was already filled by the gbuffer pass.
            if technique == Technique::Forward {
                systems.add_system(&mut graph, Forward::<OwnsDepth>::new());
            } else {
                systems.add_system(&mut graph, Forward::<SharedDepth>::new());
            }
        }
        if bloom {
            // The bright-pass input is the HDR layer the active technique
            // produced: `hdr` (deferred/hybrid) or `hdr_fwd` (forward-only).
            if technique.has_deferred() {
                systems.add_system(&mut graph, BloomBright::<FromDeferred>::new());
            } else {
                systems.add_system(&mut graph, BloomBright::<FromForward>::new());
            }
            systems.add_system(&mut graph, BloomDown1Pass);
            systems.add_system(&mut graph, BloomDown2Pass);
            systems.add_system(&mut graph, BloomUp1Pass);
            systems.add_system(&mut graph, BloomUp0Pass);
        }
        // The composite mode is a pure function of (technique, bloom):
        // which HDR layers exist and whether the bloom chain feeds the mix.
        match (technique, bloom) {
            (Technique::Deferred, true) => {
                systems.add_system(&mut graph, Composite::<CompositeDeferredBloom>::new());
            }
            (Technique::Deferred, false) => {
                systems.add_system(&mut graph, Composite::<CompositeDeferred>::new());
            }
            (Technique::Forward, true) => {
                systems.add_system(&mut graph, Composite::<CompositeForwardBloom>::new());
            }
            (Technique::Forward, false) => {
                systems.add_system(&mut graph, Composite::<CompositeForward>::new());
            }
            (Technique::Hybrid, true) => {
                systems.add_system(&mut graph, Composite::<CompositeHybridBloom>::new());
            }
            (Technique::Hybrid, false) => {
                systems.add_system(&mut graph, Composite::<CompositeHybrid>::new());
            }
        }
        Self {
            graph,
            executor: GraphExecutor::new(),
            ids,
            systems,
            bloom,
            technique,
        }
    }

    /// Resource handles of this graph.
    pub fn ids(&self) -> GraphIds {
        self.ids
    }

    /// The technique this graph was wired for.
    pub fn technique(&self) -> Technique {
        self.technique
    }

    /// Whether the bloom cascade is wired into this graph.
    pub fn bloom_enabled(&self) -> bool {
        self.bloom
    }

    /// Read access to the underlying graph (layout diagnostics, probes).
    pub fn graph(&self) -> &RenderGraph {
        &self.graph
    }

    /// Mutable access to the underlying graph. Any mutation invalidates
    /// the layout cache (see `RenderGraph::layout`); intended for
    /// benchmarks/tests that drive recomputation explicitly.
    pub fn graph_mut(&mut self) -> &mut RenderGraph {
        &mut self.graph
    }

    /// Updates the surface size before the next render (window resize).
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.graph.set_surface_size(width, height);
    }

    /// Textual layout dump for debugging/reporting (uses the layout cache).
    pub fn layout_dump(&mut self) -> String {
        self.graph.layout().debug_dump()
    }

    /// Number of pooled GPU textures (vs. declared resources — the
    /// difference is the aliasing win).
    pub fn pool_slots(&self) -> usize {
        self.executor.slots_len()
    }

    /// Bytes of the pooled GPU textures (see `GraphExecutor::texture_budget`).
    pub fn texture_budget(&self) -> u64 {
        self.executor.texture_budget()
    }

    /// Renders one frame through the graph: builds the layout, feeds the
    /// swapchain view, executes the passes against `renderer`. With bloom
    /// enabled, the composite pass mixes the bloom chain into the HDR
    /// result; without it, a zero-intensity stub keeps the composite
    /// pixel-identical to the legacy path.
    pub fn render(
        &mut self,
        context: crate::render_backend::RenderContext<'_>,
        renderer: &Renderer3D,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        let Self {
            graph,
            executor,
            ids,
            systems,
            ..
        } = self;
        let crate::render_backend::RenderContext {
            device,
            queue,
            encoder,
            target,
        } = context;
        // S1: the layout is cached — a steady-state frame (no resize/toggle)
        // is a cache hit, `compute_layout` stays off the hot path.
        let layout = graph.layout();
        executor.set_external_view(ids.target, target.clone());
        executor.execute(device, encoder, layout, |encoder, pass| {
            // S2b: every pass is a typed system (conditional passes are
            // mode families selected at registration); dispatch by PassId.
            let mut frame = Frame {
                device,
                queue,
                encoder,
                renderer,
                mesh,
                instance_count,
            };
            if !systems.run_pass(pass.pass().id, &pass, &mut frame) {
                unreachable!(
                    "render graph 3d: pass '{}' is not a typed system",
                    pass.pass().name
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::{SizePolicy, TextureSpec};

    #[test]
    fn bytes_per_pixel_table() {
        assert_eq!(format_bytes_per_pixel(wgpu::TextureFormat::Rgba8Unorm), 4);
        assert_eq!(
            format_bytes_per_pixel(wgpu::TextureFormat::Rgba8UnormSrgb),
            4
        );
        assert_eq!(format_bytes_per_pixel(wgpu::TextureFormat::Rg16Float), 4);
        assert_eq!(format_bytes_per_pixel(wgpu::TextureFormat::R32Uint), 4);
        assert_eq!(format_bytes_per_pixel(wgpu::TextureFormat::Rgba16Float), 8);
        assert_eq!(format_bytes_per_pixel(wgpu::TextureFormat::Depth32Float), 4);
        assert_eq!(format_bytes_per_pixel(wgpu::TextureFormat::Rgba32Float), 16);
    }

    #[test]
    fn technique_flag_matrix() {
        // Forward: no gbuffer/lighting, forward node present.
        assert!(!Technique::Forward.has_deferred());
        assert!(Technique::Forward.has_forward());
        assert_eq!(Technique::Forward.composite_mode(), 1);
        // Deferred: deferred chain only, no forward node.
        assert!(Technique::Deferred.has_deferred());
        assert!(!Technique::Deferred.has_forward());
        assert_eq!(Technique::Deferred.composite_mode(), 0);
        // Hybrid: both node sets.
        assert!(Technique::Hybrid.has_deferred());
        assert!(Technique::Hybrid.has_forward());
        assert_eq!(Technique::Hybrid.composite_mode(), 2);
    }

    #[test]
    fn technique_wires_expected_passes() {
        let pass_names = |technique: Technique| {
            let mut graph =
                RenderGraph3D::new_with(wgpu::TextureFormat::Rgba8Unorm, (32, 32), technique, true);
            graph
                .graph
                .build()
                .passes
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            pass_names(Technique::Forward),
            vec![
                "forward",
                "bloom_down0",
                "bloom_down1",
                "bloom_down2",
                "bloom_up1",
                "bloom_up0",
                "composite"
            ]
        );
        assert_eq!(
            pass_names(Technique::Deferred),
            vec![
                "gbuffer",
                "lighting",
                "bloom_down0",
                "bloom_down1",
                "bloom_down2",
                "bloom_up1",
                "bloom_up0",
                "composite"
            ]
        );
        assert_eq!(
            pass_names(Technique::Hybrid),
            vec![
                "gbuffer",
                "lighting",
                "forward",
                "bloom_down0",
                "bloom_down1",
                "bloom_down2",
                "bloom_up1",
                "bloom_up0",
                "composite"
            ]
        );
    }

    #[test]
    fn technique_bloom_input_follows_hdr_layer() {
        // Forward-only: bloom's bright-pass input is `hdr_fwd` (hdr dead).
        let mut forward = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            true,
        );
        let layout = forward.graph.build();
        let down0 = layout
            .passes
            .iter()
            .find(|p| p.name == "bloom_down0")
            .expect("bloom_down0 exists");
        assert_eq!(down0.reads, vec![forward.ids.hdr_fwd]);
        // Hybrid: bright-pass input is `hdr`.
        let mut hybrid = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Hybrid,
            true,
        );
        let layout = hybrid.graph.build();
        let down0 = layout
            .passes
            .iter()
            .find(|p| p.name == "bloom_down0")
            .expect("bloom_down0 exists");
        assert_eq!(down0.reads, vec![hybrid.ids.hdr]);
    }

    #[test]
    fn technique_forward_owns_depth_in_forward_mode() {
        let mut forward = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            false,
        );
        let layout = forward.graph.build();
        let fwd = layout
            .passes
            .iter()
            .find(|p| p.name == "forward")
            .expect("forward pass exists");
        // Depth is cleared (owned) by the forward pass itself.
        assert!(
            fwd.writes
                .iter()
                .any(|(id, clear)| *id == forward.ids.depth && clear.is_some())
        );
        // Composite reads only the forward layer — hdr stays dead/unpooled.
        let composite = layout
            .passes
            .iter()
            .find(|p| p.name == "composite")
            .expect("composite exists");
        assert_eq!(
            composite.reads,
            vec![forward.ids.hdr_fwd, forward.ids.target]
                .into_iter()
                .filter(|id| *id != forward.ids.target)
                .collect::<Vec<_>>(),
            "composite reads the live layers only"
        );
        assert!(!composite.reads.contains(&forward.ids.hdr));
        // hdr and the gbuffer targets are never touched → not pooled.
        let dead = layout
            .resources
            .iter()
            .filter(|r| r.first_use == usize::MAX)
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>();
        assert!(dead.contains(&"hdr"));
        assert!(dead.contains(&"albedo"));
    }

    #[test]
    fn texture_budget_matches_spec() {
        // 1280x720: Rgba16Float = 8 B/px * 921_600 px = 7_372_800 B.
        let spec = TextureSpec {
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
            size: SizePolicy::Fixed {
                width: 1280,
                height: 720,
            },
        };
        let mut graph = RenderGraph::new((1280, 720));
        let a = graph.create_resource("a", spec);
        let b = graph.create_resource("b", spec);
        graph.add_pass("p0").write(a);
        graph.add_pass("p2").read(a);
        graph.add_pass("p3").write(b);
        graph.add_pass("p4").read(b);
        // a [0,1], b [2,3]: same spec, non-overlapping → one slot.
        let layout = graph.build();
        assert_eq!(layout.slots.len(), 1);
        // Budget math is exercised on the wgpu side (needs a device); here
        // we only pin the per-slot byte formula via the layout spec.
        assert_eq!(
            format_bytes_per_pixel(spec.format) as u64 * 1280 * 720,
            7_372_800
        );
    }

    // ── S1: layout cache on the RenderGraph3D level ──────────────────

    #[test]
    fn layout_cache_reused_across_frames() {
        let mut g3 = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            true,
        );
        // Two "frames" without mutations → one computation.
        let _ = g3.graph_mut().layout();
        let dump_a = g3.layout_dump();
        let _ = g3.graph_mut().layout();
        let dump_b = g3.layout_dump();
        assert_eq!(g3.graph().layout_computations(), 1);
        assert_eq!(dump_a, dump_b, "layout must not change between frames");

        // Window resize → recompute once, then cache again.
        g3.set_surface_size(1920, 1080);
        let _ = g3.layout_dump();
        assert_eq!(g3.graph().layout_computations(), 2);
        let _ = g3.layout_dump();
        assert_eq!(g3.graph().layout_computations(), 2, "cached after resize");
    }

    // ── S2: typed systems must reproduce the imperative wiring ─────────

    /// Verbatim pre-S2 resource registration — the reference the typed
    /// registration (`register_resource`) has to match bit-for-bit.
    fn imperative_resources(
        graph: &mut RenderGraph,
        surface_format: wgpu::TextureFormat,
    ) -> GraphIds {
        let spec = |format| TextureSpec {
            format,
            samples: 1,
            size: SizePolicy::MatchSurface,
        };
        let frac = |format, divisor| TextureSpec {
            format,
            samples: 1,
            size: SizePolicy::Fraction(divisor),
        };
        GraphIds {
            albedo: graph.create_resource("albedo", spec(wgpu::TextureFormat::Rgba8Unorm)),
            normal: graph.create_resource("normal", spec(wgpu::TextureFormat::Rg16Float)),
            material_id: graph.create_resource("material_id", spec(wgpu::TextureFormat::R32Uint)),
            world_position: graph
                .create_resource("world_position", spec(wgpu::TextureFormat::Rg16Float)),
            material_params: graph
                .create_resource("material_params", spec(wgpu::TextureFormat::Rgba16Float)),
            depth: graph.create_resource("depth", spec(wgpu::TextureFormat::Depth32Float)),
            hdr: graph.create_resource("hdr", spec(surface_format)),
            hdr_fwd: graph.create_resource("hdr_fwd", spec(wgpu::TextureFormat::Rgba16Float)),
            target: graph.external_output("target"),
            bloom0: graph.create_resource("bloom0", frac(wgpu::TextureFormat::Rgba16Float, 2)),
            bloom1: graph.create_resource("bloom1", frac(wgpu::TextureFormat::Rgba16Float, 4)),
            bloom2: graph.create_resource("bloom2", frac(wgpu::TextureFormat::Rgba16Float, 8)),
        }
    }

    /// Verbatim pre-S2 pass wiring — the reference the typed systems
    /// (`add_system`) and the conditional passes have to match.
    fn imperative_passes(
        graph: &mut RenderGraph,
        ids: &GraphIds,
        technique: Technique,
        bloom: bool,
    ) {
        if technique.has_deferred() {
            graph
                .add_pass("gbuffer")
                .write(ids.albedo)
                .write(ids.normal)
                .write(ids.material_id)
                .write(ids.world_position)
                .write(ids.material_params)
                .write(ids.depth);
            graph
                .add_pass("lighting")
                .read(ids.albedo)
                .read(ids.normal)
                .read(ids.material_id)
                .read(ids.world_position)
                .read(ids.material_params)
                .read(ids.depth)
                .write_clear(ids.hdr, wgpu::Color::BLACK);
        }
        if technique.has_forward() {
            let pass = graph.add_pass("forward");
            let pass = if technique == Technique::Forward {
                pass.write_clear(ids.depth, wgpu::Color::WHITE)
            } else {
                pass.read(ids.depth)
            };
            pass.write_clear(ids.hdr_fwd, wgpu::Color::TRANSPARENT);
        }
        if bloom {
            let bloom_input = if technique.has_deferred() {
                ids.hdr
            } else {
                ids.hdr_fwd
            };
            graph
                .add_pass("bloom_down0")
                .read(bloom_input)
                .write_clear(ids.bloom0, wgpu::Color::BLACK);
            graph
                .add_pass("bloom_down1")
                .read(ids.bloom0)
                .write_clear(ids.bloom1, wgpu::Color::BLACK);
            graph
                .add_pass("bloom_down2")
                .read(ids.bloom1)
                .write_clear(ids.bloom2, wgpu::Color::BLACK);
            graph
                .add_pass("bloom_up1")
                .read(ids.bloom2)
                .write(ids.bloom1);
            graph
                .add_pass("bloom_up0")
                .read(ids.bloom1)
                .write(ids.bloom0);
        }
        let mut composite = graph.add_pass("composite").write(ids.target);
        if technique.has_deferred() {
            composite = composite.read(ids.hdr);
        }
        if technique.has_forward() {
            composite = composite.read(ids.hdr_fwd);
        }
        if bloom {
            composite.read(ids.bloom0);
        }
    }

    /// The pre-S2 wiring, verbatim, as one graph: resources then passes.
    fn imperative_wiring(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: bool,
    ) -> RenderGraph {
        let mut graph = RenderGraph::new(surface_size);
        let ids = imperative_resources(&mut graph, surface_format);
        imperative_passes(&mut graph, &ids, technique, bloom);
        graph
    }

    #[test]
    fn typed_wiring_matches_imperative_reference() {
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [false, true] {
                let mut typed = RenderGraph3D::new_with(fmt, (1280, 720), technique, bloom);
                let mut reference = imperative_wiring(fmt, (1280, 720), technique, bloom);
                assert_eq!(
                    typed.graph.build().debug_dump(),
                    reference.build().debug_dump(),
                    "typed wiring diverged: {technique:?} bloom={bloom}"
                );
            }
        }
    }\n

    // ── S3: golden layout tests — the pool must not change silently ────

    fn slots_for(technique: Technique, bloom: bool) -> usize {
        let mut g3 = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            technique,
            bloom,
        );
        g3.graph_mut().layout().slots.len()
    }

    #[test]
    fn golden_pool_slots_per_technique() {
        // Pinned against B1-R7 measurements (surface format Rgba8Unorm so
        // `hdr` shares the albedo spec group): 9 resources → 7 slots on the
        // deferred/hybrid path; the bloom cascade adds exactly its three
        // fraction levels (bloom0/1/2 have distinct TextureSpec keys).
        assert_eq!(slots_for(Technique::Forward, false), 2);
        assert_eq!(slots_for(Technique::Forward, true), 5);
        assert_eq!(slots_for(Technique::Deferred, false), 7);
        assert_eq!(slots_for(Technique::Deferred, true), 10);
        assert_eq!(slots_for(Technique::Hybrid, false), 7);
        assert_eq!(slots_for(Technique::Hybrid, true), 10);
    }

    #[test]
    fn golden_bloom_adds_exactly_three_slots() {
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            assert_eq!(
                slots_for(technique, true) - slots_for(technique, false),
                3,
                "bloom cascade must add exactly its three fraction levels"
            );
        }
    }

    #[test]
    fn golden_dead_layers_are_unpooled() {
        // Forward-only: the deferred HDR layer and the gbuffer targets are
        // never touched → no lifetime window, no pool slot.
        let mut fwd = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Forward,
            true,
        );
        let ids = fwd.ids();
        let layout = fwd.graph_mut().layout().clone();
        for id in [ids.hdr, ids.albedo, ids.normal] {
            let rl = &layout.resources[id.0 as usize];
            assert_eq!(rl.first_use, usize::MAX, "{} must be dead", rl.name);
            assert_eq!(rl.slot, None);
        }
        // No bloom → the cascade levels are dead.
        let mut plain = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            false,
        );
        let ids = plain.ids();
        let layout = plain.graph_mut().layout();
        assert_eq!(layout.resources[ids.bloom0.0 as usize].slot, None);
        assert_eq!(layout.resources[ids.bloom1.0 as usize].slot, None);
        assert_eq!(layout.resources[ids.bloom2.0 as usize].slot, None);
    }

    #[test]
    fn golden_hybrid_lifetimes() {
        let mut g3 = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            true,
        );
        let ids = g3.ids();
        let layout = g3.graph_mut().layout();
        let window = |id: ResourceId| {
            let rl = &layout.resources[id.0 as usize];
            (rl.first_use, rl.last_use)
        };
        // gbuffer=0, lighting=1, forward=2, bloom chain 3..8, composite=8.
        assert_eq!(window(ids.depth), (0, 2), "depth: gbuffer → forward");
        assert_eq!(window(ids.hdr), (1, 8));
        assert_eq!(window(ids.hdr_fwd), (2, 8));
        assert_eq!(window(ids.bloom0), (3, 8));
    }

    #[test]
    fn golden_planned_pool_bytes() {
        // Forward, no bloom, 1280×720: depth (D32, 4 B/px) + hdr_fwd
        // (Rgba16, 8 B/px) = 12 B/px over the surface.
        let mut g3 = RenderGraph3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Forward,
            false,
        );
        let layout = g3.graph_mut().layout();
        assert_eq!(planned_pool_bytes(layout), 12 * 1280 * 720);
    }
\n}\n