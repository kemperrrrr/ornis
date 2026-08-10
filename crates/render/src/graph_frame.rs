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
use crate::renderer::{GbufferTargets, Renderer3D};
use std::collections::HashMap;

/// One pooled texture per render-graph slot.
#[derive(Debug)]
struct PooledTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
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
    let (width, height) = match slot.spec.size {
        SizePolicy::MatchSurface => surface_size,
        SizePolicy::Fixed { width, height } => (width, height),
    };
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
    PooledTexture {
        _texture: texture,
        view,
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
}

impl RenderGraph3D {
    /// Builds the pass graph. `surface_format` is the render target format
    /// (the lighting pass writes into it); `surface_size` seeds
    /// `SizePolicy::MatchSurface` resources.
    pub fn new(surface_format: wgpu::TextureFormat, surface_size: (u32, u32)) -> Self {
        let mut graph = RenderGraph::new(surface_size);
        let spec = |format| TextureSpec {
            format,
            samples: 1,
            size: SizePolicy::MatchSurface,
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
        graph
            .add_pass("composite")
            .read(ids.hdr)
            .read(ids.hdr_fwd)
            .write(ids.target);
        Self {
            graph,
            executor: GraphExecutor::new(),
            ids,
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

    /// Renders one frame through the graph: builds the layout, feeds the
    /// swapchain view, executes the four passes against `renderer`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        renderer: &Renderer3D,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        let Self {
            graph,
            executor,
            ids,
        } = self;
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
                "composite" => renderer.render_composite(
                    device,
                    encoder,
                    pass.view_of(ids.target),
                    pass.view_of(ids.hdr),
                    pass.view_of(ids.hdr_fwd),
                ),
                other => unreachable!("render graph 3d: unknown pass '{other}'"),
            }
        });
    }
}
