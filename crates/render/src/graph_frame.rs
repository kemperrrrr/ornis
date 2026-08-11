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

use crate::mesh::Mesh;
use crate::render_graph::{
    GraphLayout, PassLayout, RenderGraph, ResourceId, ResourceLayout, SizePolicy, TextureSpec,
};
use crate::renderer::{CompositeInputs, GbufferTargets, Renderer3D};
use std::collections::HashMap;

/// One pooled texture per render-graph slot.
#[derive(Debug)]
struct PooledTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bytes: u64,
}

/// Bright-pass threshold for the first bloom downsample (luminance gate:
/// only pixels brighter than this contribute to the bloom chain).
const BLOOM_BRIGHT_THRESHOLD: f32 = 0.7;

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
    bloom: bool,
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
        Self::build(surface_format, surface_size, false)
    }

    /// Like [`new`](Self::new), plus the bloom cascade:
    ///
    /// `bloom_down0` (bright-pass at 1/2) → `bloom_down1` (1/4) →
    /// `bloom_down2` (1/8) → `bloom_up1` → `bloom_up0`, where each upsample
    /// adds its level back over the downsampled content (`LoadOp::Load`).
    /// The composite pass then mixes the final bloom level into the HDR
    /// result.
    pub fn new_with_bloom(surface_format: wgpu::TextureFormat, surface_size: (u32, u32)) -> Self {
        Self::build(surface_format, surface_size, true)
    }

    fn build(surface_format: wgpu::TextureFormat, surface_size: (u32, u32), bloom: bool) -> Self {
        let mut graph = RenderGraph::new(surface_size);
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
        let ids = GraphIds {
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
        };
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
        graph
            .add_pass("forward")
            .read(ids.depth)
            .write_clear(ids.hdr_fwd, wgpu::Color::TRANSPARENT);
        if bloom {
            // Bright-pass threshold: only the brightest pixels survive the
            // first downsample; deeper levels pass everything (threshold 0).
            graph
                .add_pass("bloom_down0")
                .read(ids.hdr)
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
        let composite = graph
            .add_pass("composite")
            .read(ids.hdr)
            .read(ids.hdr_fwd)
            .write(ids.target);
        if bloom {
            composite.read(ids.bloom0);
        }
        Self {
            graph,
            executor: GraphExecutor::new(),
            ids,
            bloom,
        }
    }

    /// Resource handles of this graph.
    pub fn ids(&self) -> GraphIds {
        self.ids
    }

    /// Updates the surface size before the next render (window resize).
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.graph.set_surface_size(width, height);
    }

    /// Textual layout dump for debugging/reporting.
    pub fn layout_dump(&self) -> String {
        self.graph.build().debug_dump()
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
            bloom,
        } = self;
        let crate::render_backend::RenderContext {
            device,
            queue,
            encoder,
            target,
        } = context;
        let layout = graph.build();
        executor.set_external_view(ids.target, target.clone());
        executor.execute(device, encoder, &layout, |encoder, pass| {
            match pass.pass().name.as_str() {
                "gbuffer" | "lighting" => {
                    // G-buffer targets are alive on exactly these passes.
                    let g = GbufferTargets {
                        albedo: pass.view_of(ids.albedo),
                        normal: pass.view_of(ids.normal),
                        material_id: pass.view_of(ids.material_id),
                        world_position: pass.view_of(ids.world_position),
                        material_params: pass.view_of(ids.material_params),
                        depth: pass.view_of(ids.depth),
                    };
                    if pass.pass().name.as_str() == "gbuffer" {
                        renderer.render_gbuffer(encoder, &g, mesh, instance_count);
                    } else {
                        renderer.render_lighting(device, encoder, &g, pass.view_of(ids.hdr));
                    }
                }
                "forward" => renderer.render_forward(
                    encoder,
                    pass.view_of(ids.depth),
                    pass.view_of(ids.hdr_fwd),
                    mesh,
                    instance_count,
                ),
                "bloom_down0" => renderer.render_bloom_down(
                    device,
                    queue,
                    encoder,
                    pass.view_of(ids.hdr),
                    pass.view_of(ids.bloom0),
                    BLOOM_BRIGHT_THRESHOLD,
                ),
                "bloom_down1" => renderer.render_bloom_down(
                    device,
                    queue,
                    encoder,
                    pass.view_of(ids.bloom0),
                    pass.view_of(ids.bloom1),
                    0.0,
                ),
                "bloom_down2" => renderer.render_bloom_down(
                    device,
                    queue,
                    encoder,
                    pass.view_of(ids.bloom1),
                    pass.view_of(ids.bloom2),
                    0.0,
                ),
                "bloom_up1" => renderer.render_bloom_up(
                    device,
                    encoder,
                    pass.view_of(ids.bloom2),
                    pass.view_of(ids.bloom1),
                ),
                "bloom_up0" => renderer.render_bloom_up(
                    device,
                    encoder,
                    pass.view_of(ids.bloom1),
                    pass.view_of(ids.bloom0),
                ),
                "composite" => {
                    // With bloom culled (no bloom passes), `bloom0` is never
                    // written → not alive here; bind the forward target as a
                    // stub with zero intensity, exactly like `render_scene`.
                    let (bloom, bloom_intensity) = if *bloom {
                        (pass.view_of(ids.bloom0), 1.0)
                    } else {
                        (pass.view_of(ids.hdr_fwd), 0.0)
                    };
                    renderer.render_composite(
                        device,
                        queue,
                        encoder,
                        CompositeInputs {
                            target: pass.view_of(ids.target),
                            hdr: pass.view_of(ids.hdr),
                            hdr_fwd: pass.view_of(ids.hdr_fwd),
                            bloom,
                            bloom_intensity,
                        },
                    );
                }
                other => unreachable!("render graph 3d: unknown pass '{other}'"),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
