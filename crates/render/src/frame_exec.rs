//! Phase 1: wgpu executor for the frame plan (formerly "render graph").
//!
//! Maps pool slots to real `wgpu::Texture` objects (created lazily, reused
//! every frame) and resolves every resource's view during pass execution —
//! either from the slot pool or from externally provided views (swapchain).
//! Also wires the four existing `Renderer3D` passes (gbuffer → lighting →
//! forward → composite) as plan nodes ([`RenderFrame3D`]).
//!
//! Lifetimes are computed by the pure [`transient_pool`] layout; this module
//! only owns GPU objects. On wgpu, barriers are handled by wgpu itself, so
//! the executor is small by design.

use crate::antialias::{AaWiring, AntiAliasMode};
use crate::flags::Bloom;
use crate::frame_passes::{
    Albedo, Ao, AoBlurPass, AoRaw, AoRawPass, Bloom0, Bloom1, Bloom2, BloomBright, BloomDown1Pass,
    BloomDown2Pass, BloomUp0Pass, BloomUp1Pass, Composite, CompositeDeferred,
    CompositeDeferredBloom, CompositeForward, CompositeForwardBloom, CompositeHybrid,
    CompositeHybridBloom, Depth, EdgeFilterPass, FogPass, FogPlacement, FogWiring, Forward,
    FromDeferred, FromForward, GbufferMotionPass, GbufferPass, Hdr, HdrFwd, LdrLayer,
    LightingAoPass, LightingPass, MaterialId, MaterialParams, Motion, Normal, OwnsDepth,
    PreviousDepth, SharedDepth, TaaPass, Target, WorldPosition,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu_resources::FrameCommandBuffers;
use crate::mesh::Mesh;
use crate::renderer::{CustomGbufferDraw, Renderer3D};
use crate::schedule_bridge::ProjectionError;
use crate::system::{Frame, SystemSet};
use crate::transient_pool::{
    Budget, FrameLayout, PassLayout, ResourceId, ResourceLayout, TransientPool,
    format_bytes_per_pixel,
};
use ornis_schedule::run_levels;
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

/// One pooled texture per render-plan slot.
#[derive(Debug)]
struct PooledTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bytes: u64,
    /// Spec the texture was created for: a layout with a different spec
    /// (sample count, size, format) recreates the slot instead of reusing
    /// a stale texture.
    spec: crate::transient_pool::TextureSpec,
}

/// Executes a [`FrameLayout`] on wgpu: lazily creates one texture per pool
/// slot and hands every pass a [`PassViews`] resolver.
///
/// Besides the GPU object pool, the executor owns a [`TransientPool`]
/// compiling the registry's declarations into a shared layout snapshot (see
/// [`FrameExecutor::ensure_layout`]): declarations stay in [`SystemSet`],
/// but the hot-path compiler instance lives here, keyed by the registry's
/// declaration generation.
#[derive(Debug, Default)]
pub struct FrameExecutor {
    pool: Vec<Option<PooledTexture>>,
    external_views: HashMap<ResourceId, wgpu::TextureView>,
    surface_size: Option<(u32, u32)>,
    /// Executor-owned transient allocator: compiles declaration snapshots
    /// into shared layouts for the frame hot path. Separate from the
    /// registry-owned instance serving `SystemSet::layout()` cold paths;
    /// both memoize against the same declaration generation.
    layout_pool: TransientPool,
}

impl FrameExecutor {
    /// Create an executor with an empty texture pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared snapshot of the registry's compiled layout for the frame hot path.
    ///
    /// In steady state (no declaration mutations since the last call) this
    /// is a cache hit: no compilation and no vector clone, just an `Arc`
    /// clone. Any registry mutation bumps [`SystemSet::generation`], which
    /// refreshes the memoized snapshot on the next call. Budget violations
    /// keep the previous snapshot when one exists; otherwise return an empty plan.
    pub fn ensure_layout(&mut self, set: &SystemSet) -> Arc<FrameLayout> {
        let generation = set.generation();
        let input = set.pool_input();
        match self.layout_pool.ensure(generation, &input) {
            Ok(layout) => layout,
            Err(_) => self.layout_pool.cached().cloned().unwrap_or_else(|| {
                Arc::new(FrameLayout {
                    surface_size: input.surface_size,
                    passes: Vec::new(),
                    resources: Vec::new(),
                    slots: Vec::new(),
                    pass_alive: Vec::new(),
                    levels: Vec::new(),
                })
            }),
        }
    }

    /// Drops the memoized layout snapshot without touching the GPU pool.
    pub fn invalidate_layout(&mut self) {
        self.layout_pool.invalidate();
    }

    /// Provides the view backing an external resource (see
    /// `SystemSet::external_output`). Call before `execute`.
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

    /// Executes `layout` in DAG level order: level by level from the shared
    /// `ornis-schedule` plan (`FrameLayout::levels`), `run` receives the
    /// encoder and a [`PassViews`] resolver for the live resources.
    ///
    /// The sequential sibling of [`execute_parallel`](Self::execute_parallel):
    /// levels come from the same [`bitset_level_plan`](ornis_schedule::bitset_level_plan)
    /// engine that levels `core::Schedule` systems, and passes within a
    /// level are independent by construction — so the flattened order is a
    /// valid topological order. On every production wiring
    /// ([`RenderFrame3D`]) it coincides with registration order (pinned by
    /// `sequential_execute_matches_registration_on_production_plans`), making
    /// the default path structurally identical to the E1 schedule-driven
    /// [`render_schedule`](RenderFrame3D::render_schedule) order.
    pub fn execute<'a>(
        &'a mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &'a FrameLayout,
        run: impl FnMut(&mut wgpu::CommandEncoder, PassViews<'a>),
    ) {
        self.ensure_pool(device, layout);
        let pool = &self.pool;
        let externals = &self.external_views;
        let mut run = run;
        for level in layout.levels() {
            for index in level {
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
    }

    /// E1 (S5e): sequential execution in an explicitly given pass order —
    /// the flattened levels of a projected `core::Schedule`
    /// ([`crate::schedule_bridge`]) — recording onto the caller's encoder.
    /// The ordered sibling of [`execute`](Self::execute).
    pub fn execute_in_order<'a>(
        &'a mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &'a FrameLayout,
        order: &[usize],
        run: impl FnMut(&mut wgpu::CommandEncoder, PassViews<'a>),
    ) {
        self.ensure_pool(device, layout);
        let pool = &self.pool;
        let externals = &self.external_views;
        let mut run = run;
        for &index in order {
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

    /// E2 (S5e): per-pass encoders in an explicitly given pass order —
    /// no caller encoder and no submit: every pass records into its own
    /// encoder and the finished buffers are pushed into `sink` in that
    /// order, ready for one ordered submit by the caller's flush step
    /// (the encoder-as-frame-resource handover; the mechanics are the
    /// sequential form of [`execute_parallel`](Self::execute_parallel)).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn record_in_order<'a>(
        &'a mut self,
        device: &wgpu::Device,
        layout: &'a FrameLayout,
        order: &[usize],
        sink: &Mutex<Vec<wgpu::CommandBuffer>>,
        mut run: impl FnMut(usize, &PassViews<'a>, &mut wgpu::CommandEncoder),
    ) {
        self.ensure_pool(device, layout);
        let pool = &self.pool;
        let externals = &self.external_views;
        for &index in order {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            run(
                index,
                &PassViews {
                    layout,
                    pool,
                    externals,
                    index,
                },
                &mut encoder,
            );
            sink.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(encoder.finish());
        }
    }

    /// S5b: parallel command recording. The shared level executor
    /// (`ornis_schedule::run_levels`, backlog #19 — single executor with
    /// `core::Schedule`) runs levels sequentially and passes inside a
    /// level concurrently (rayon); every pass records into its own
    /// encoder; all buffers are submitted to `queue` in registration
    /// order with a single submit — `queue.write_buffer` calls made while
    /// recording keep the same before-submit semantics as the sequential
    /// path, so the pixel result is identical. Submission order is
    /// registration order on both targets (wasm twin below).
    ///
    /// Invariant (pass authors): passes that write the same queue-backed
    /// buffer (renderer-internal uniforms are not part of the declared
    /// texture accesses) must land in different levels. The texture side
    /// of the contract is now enforced — `PassViews::view_of` panics in
    /// debug on a `ResourceId` outside the pass's declared reads/writes
    /// (backlog #6); `queue.write_buffer` calls never flow through
    /// `view_of`, so this buffer-side invariant stays an author contract,
    /// like the rayon limitation of system enforcement (audit §3.3).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute_parallel<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &'a FrameLayout,
        run: impl Fn(usize, &PassViews<'a>, &mut wgpu::CommandEncoder) + Sync,
    ) {
        self.ensure_pool(device, layout);
        let pool = &self.pool;
        let externals = &self.external_views;
        let buffers = Mutex::new(Vec::new());
        let levels = layout.levels();
        run_levels(&levels, layout.passes.len(), true, |index| {
            let desc = wgpu::CommandEncoderDescriptor { label: None };
            let mut encoder = device.create_command_encoder(&desc);
            let views = PassViews {
                layout,
                pool,
                externals,
                index,
            };
            run(index, &views, &mut encoder);
            buffers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((index, encoder.finish()));
        });
        let mut buffers = buffers
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        buffers.sort_by_key(|(index, _)| *index);
        queue.submit(buffers.into_iter().map(|(_, buffer)| buffer));
    }

    /// wasm32 twin of [`FrameExecutor::execute_parallel`]: wgpu types are
    /// not `Sync` on the web backend (Rc inside) and there are no rayon
    /// threads — the shared level executor (`run_levels`) runs the same
    /// per-pass encoders sequentially in registration order (the layout
    /// holds enabled passes only, so 0..nodes is correct and matches the
    /// native submission order, backlog #19), submitting as they go.
    /// Signature drops the `Sync` bound so the shared `render()` call
    /// site compiles for both targets.
    #[cfg(target_arch = "wasm32")]
    pub fn execute_parallel<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &'a FrameLayout,
        run: impl Fn(usize, &PassViews<'a>, &mut wgpu::CommandEncoder),
    ) {
        self.ensure_pool(device, layout);
        let pool = &self.pool;
        let externals = &self.external_views;
        let levels = layout.levels();
        run_levels(&levels, layout.passes.len(), false, |index| {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let views = PassViews {
                layout,
                pool,
                externals,
                index,
            };
            run(index, &views, &mut encoder);
            queue.submit(std::iter::once(encoder.finish()));
        });
    }

    fn ensure_pool(&mut self, device: &wgpu::Device, layout: &FrameLayout) {
        // Recreate all textures when the surface size changes.
        if self.surface_size != Some(layout.surface_size) {
            self.pool.clear();
            self.surface_size = Some(layout.surface_size);
        }
        self.pool.resize_with(layout.slots.len(), || None);
        for (i, slot) in layout.slots.iter().enumerate() {
            let stale = self.pool[i].as_ref().is_some_and(|t| t.spec != slot.spec);
            if self.pool[i].is_none() || stale {
                self.pool[i] = Some(create_pooled_texture(device, slot, layout.surface_size, i));
            }
        }
    }
}

fn create_pooled_texture(
    device: &wgpu::Device,
    slot: &crate::transient_pool::PoolSlot,
    surface_size: (u32, u32),
    index: usize,
) -> PooledTexture {
    let (width, height) = slot.spec.size.resolve(surface_size);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("frame plan slot #{index}")),
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
        spec: slot.spec,
    }
}

/// Per-pass view resolver handed to pass callbacks during execution.
#[derive(Debug)]
pub struct PassViews<'a> {
    layout: &'a FrameLayout,
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
    /// Returns `None` when the resource is not alive on this pass, its
    /// external view is unset, or its pool slot was never created. Debug
    /// builds still panic when `id` sits outside the pass's declared
    /// reads/writes (backlog #6, `assert_pass_access_declared`) — the
    /// pass-level counterpart of the system TLS enforcement in
    /// `core::Schedule`.
    pub fn view_of(&self, id: ResourceId) -> Option<&'a wgpu::TextureView> {
        #[cfg(debug_assertions)]
        crate::transient_pool::assert_pass_access_declared(self.layout, self.index, id);
        let rl = self.resource(id);
        if !rl.alive_at(self.index) {
            return None;
        }
        if rl.backing().is_external() {
            self.externals.get(&id)
        } else {
            let slot = rl.slot?;
            self.pool
                .get(slot)
                .and_then(|slot| slot.as_ref())
                .map(|t| t.view_ref())
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
pub struct RenderFrame3D {
    executor: FrameExecutor,
    /// S5b: record independent passes in parallel (a level at a time,
    /// e.g. lighting ∥ forward); sequential single-encoder path by default.
    parallel_recording: bool,
    ids: FrameIds,
    /// Typed S2 systems (d3: also the single declaration registry):
    /// `type → ResourceId` map + type-erased runners for the passes
    /// declared as `FramePass` implementations
    /// (see [`crate::frame_passes`] and [`crate::system`]).
    systems: SystemSet,
    bloom: crate::flags::Bloom,
    technique: Technique,
    /// Selected anti-aliasing mode (hardware MSAA or a software resolve).
    anti_alias: AntiAliasMode,
    /// Negotiated MSAA sample count of the pooled geometry targets
    /// (`1` or `4`): must match the [`Renderer3D`] the plan renders
    /// against, whose pipelines and resolve targets are built for the same
    /// count. Single-sample resources stay 1x in all modes.
    sample_count: u32,
}

/// Which lighting technique the plan wires up. The choice is expressed
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
    ///
    /// Prefer [`Self::technique`] with [`crate::flags::CompositeTechnique`].
    pub fn composite_mode(&self) -> u32 {
        self.technique().shader_mode()
    }

    /// Which HDR layers this technique produces (typed replacement for
    /// [`Self::composite_mode`]).
    pub fn technique(&self) -> crate::flags::CompositeTechnique {
        match self {
            Self::Forward => crate::flags::CompositeTechnique::Forward,
            Self::Deferred => crate::flags::CompositeTechnique::Deferred,
            Self::Hybrid => crate::flags::CompositeTechnique::Hybrid,
        }
    }
}

/// Resource handles of the [`RenderFrame3D`] plan.
#[derive(Debug, Clone, Copy)]
pub struct FrameIds {
    /// Albedo/base-color gbuffer target.
    pub albedo: ResourceId,
    /// World-space normal target.
    pub normal: ResourceId,
    /// Material id target.
    pub material_id: ResourceId,
    /// World-space position target.
    pub world_position: ResourceId,
    /// Material parameter target.
    pub material_params: ResourceId,
    /// Depth buffer (gbuffer-owned or forward-owned per technique).
    pub depth: ResourceId,
    /// Deferred HDR color layer.
    pub hdr: ResourceId,
    /// Forward HDR color layer.
    pub hdr_fwd: ResourceId,
    /// External output (swapchain view).
    pub target: ResourceId,
    /// Bloom chain levels: 1/2, 1/4, 1/8 of the surface. Always declared so
    /// every plan shares one `FrameIds` shape; they only consume pool slots
    /// when the bloom passes exist (`new_with_bloom`).
    pub bloom0: ResourceId,
    /// Second bloom mip level (1/4 surface).
    pub bloom1: ResourceId,
    /// Third bloom mip level (1/8 surface).
    pub bloom2: ResourceId,
}

#[derive(Default)]
struct FramePassWiring {
    fog: Option<FogWiring>,
    ao: Option<crate::ao::AoWiring>,
    aa: Option<AaWiring>,
}

impl RenderFrame3D {
    /// Builds the pass plan. `surface_format` is the render target format
    /// (the lighting pass writes into it); `surface_size` seeds
    /// `SizePolicy::MatchSurface` resources.
    pub fn new(surface_format: wgpu::TextureFormat, surface_size: (u32, u32)) -> Self {
        Self::new_with(surface_format, surface_size, Technique::Hybrid, Bloom::Off)
    }

    /// Like [`new`](Self::new), plus the bloom cascade:
    ///
    /// `bloom_down0` (bright-pass at 1/2) → `bloom_down1` (1/4) →
    /// `bloom_down2` (1/8) → `bloom_up1` → `bloom_up0`, where each upsample
    /// adds its level back over the downsampled content (`LoadOp::Load`).
    /// The composite pass then mixes the final bloom level into the HDR
    /// result.
    pub fn new_with_bloom(surface_format: wgpu::TextureFormat, surface_size: (u32, u32)) -> Self {
        Self::new_with(surface_format, surface_size, Technique::Hybrid, Bloom::On)
    }

    /// Builds the plan for a specific [`Technique`] with optional bloom.
    /// The technique decides which nodes are wired: deferred nodes
    /// (`gbuffer`, `lighting`) exist only when [`Technique::has_deferred`],
    /// the `forward` node only when [`Technique::has_forward`], and the
    /// bloom chain reads whichever HDR layer the technique produces.
    ///
    /// No fog node is registered (default): opt in with
    /// [`new_with_fog`](Self::new_with_fog).
    ///
    /// Single-sample plan: pooled geometry targets are 1x. Pair with a
    /// 1x [`Renderer3D`]; for the negotiated 4x path use
    /// [`new_with_samples`](Self::new_with_samples).
    pub fn new_with(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
    ) -> Self {
        Self::build::<Target>(
            surface_format,
            surface_size,
            technique,
            bloom,
            FramePassWiring::default(),
            crate::renderer::SINGLE_SAMPLE_COUNT,
        )
    }

    /// Like [`new_with`](Self::new_with), with the pooled geometry targets
    /// at `sample_count` (normalized to `{1, 4}`): pass the negotiated
    /// renderer count (see
    /// [`negotiate_sample_count`](crate::renderer::negotiate_sample_count))
    /// so plan and renderer agree — a 4x plan against a 4x [`Renderer3D`]
    /// resolves through the renderer's single-sample targets, a 1x plan
    /// lays out exactly like [`new_with`](Self::new_with).
    pub fn new_with_samples(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        sample_count: u32,
    ) -> Self {
        Self::build::<Target>(
            surface_format,
            surface_size,
            technique,
            bloom,
            FramePassWiring::default(),
            sample_count,
        )
    }

    /// Like [`new_with`](Self::new_with), plus an opt-in [`FogPass`] node
    /// from `fog` ([`FogWiring`]: placement relative to `composite` plus
    /// fog state). A [`FogState::Disabled`](crate::frame_passes::FogState::Disabled)
    /// node records no commands, so the frame stays pixel-identical to
    /// [`new_with`](Self::new_with); fog needs the deferred HDR layer
    /// (see [`FogPlacement`]).
    ///
    /// Single-sample plan; for the negotiated 4x path use
    /// [`new_with_fog_samples`](Self::new_with_fog_samples).
    pub fn new_with_fog(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        fog: FogWiring,
    ) -> Self {
        Self::build::<Target>(
            surface_format,
            surface_size,
            technique,
            bloom,
            FramePassWiring {
                fog: Some(fog),
                ..FramePassWiring::default()
            },
            crate::renderer::SINGLE_SAMPLE_COUNT,
        )
    }

    /// Like [`new_with_fog`](Self::new_with_fog), with the pooled geometry
    /// targets at `sample_count` (see
    /// [`new_with_samples`](Self::new_with_samples)).
    pub fn new_with_fog_samples(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        fog: FogWiring,
        sample_count: u32,
    ) -> Self {
        Self::build::<Target>(
            surface_format,
            surface_size,
            technique,
            bloom,
            FramePassWiring {
                fog: Some(fog),
                ..FramePassWiring::default()
            },
            sample_count,
        )
    }

    /// Like [`new_with`](Self::new_with), plus opt-in screen-space ambient
    /// occlusion from `ao` ([`AoWiring`](crate::ao::AoWiring): SSAO/GTAO
    /// kernel plus settings). Registers the half-res `ao_raw`/`ao` targets
    /// and the `ao_raw` → `ao_blur` → `lighting` chain; the blurred factor
    /// modulates only the ambient and IBL terms. Without `ao` the plan is
    /// byte-identical to [`new_with`](Self::new_with). No fog node is
    /// registered (default).
    ///
    /// Single-sample plan: pooled geometry targets are 1x. Pair with a
    /// 1x [`Renderer3D`].
    // qual:api — public opt-in; in-tree callers are the AO gates until a sample wires it.
    pub fn new_with_ao(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        ao: crate::ao::AoWiring,
    ) -> Self {
        Self::build::<Target>(
            surface_format,
            surface_size,
            technique,
            bloom,
            FramePassWiring {
                ao: Some(ao),
                ..FramePassWiring::default()
            },
            crate::renderer::SINGLE_SAMPLE_COUNT,
        )
    }

    /// Like [`new_with`](Self::new_with), plus an opt-in anti-aliasing
    /// pass from `aa` ([`AaWiring`]: the mode plus the settings of its
    /// kernel).
    ///
    /// The anti-aliasing pass filters the composited image: the composite
    /// writes the intermediate `ldr_layer` and the pass presents the
    /// filtered image. Shader-based modes use single-sample geometry, so
    /// pair this plan with a 1x [`Renderer3D`]. TAA also jitters the
    /// projection, so the renderer must run in the same mode (see
    /// [`Renderer3D::set_anti_alias`]) — otherwise the resolve lacks the
    /// sub-pixel samples it accumulates. TAA requires a deferred or hybrid
    /// plan, which provides the motion-aware g-buffer.
    ///
    /// Without `aa` (or with the disabled mode) the render graph wiring is
    /// identical to [`new_with`](Self::new_with). No fog node is registered: the
    /// fog pass writes the presented target, which the anti-aliasing pass
    /// owns.
    pub fn new_with_aa(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        aa: AaWiring,
    ) -> Self {
        Self::new_with_aa_samples(
            surface_format,
            surface_size,
            technique,
            bloom,
            aa,
            crate::renderer::SINGLE_SAMPLE_COUNT,
        )
    }

    /// Like [`new_with_aa`](Self::new_with_aa), with a negotiated sample
    /// count for `Msaa`. `Off` and the shader-based modes (TAA/FXAA/SMAA)
    /// use single-sample geometry; the filters replace MSAA rather than
    /// compounding its cost.
    // qual:allow(iosp) reason: "the runtime AA mode selects distinct generic target types in the typed render graph"
    pub fn new_with_aa_samples(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        aa: AaWiring,
        sample_count: u32,
    ) -> Self {
        let sample_count = match aa.mode {
            AntiAliasMode::Off => crate::renderer::SINGLE_SAMPLE_COUNT,
            AntiAliasMode::Msaa => crate::renderer::normalize_sample_count(sample_count),
            AntiAliasMode::Taa | AntiAliasMode::Fxaa | AntiAliasMode::Smaa => {
                crate::renderer::SINGLE_SAMPLE_COUNT
            }
        };
        if aa.mode.is_enabled() {
            Self::build::<LdrLayer>(
                surface_format,
                surface_size,
                technique,
                bloom,
                FramePassWiring {
                    aa: Some(aa),
                    ..FramePassWiring::default()
                },
                sample_count,
            )
        } else {
            Self::build::<Target>(
                surface_format,
                surface_size,
                technique,
                bloom,
                FramePassWiring {
                    aa: Some(aa),
                    ..FramePassWiring::default()
                },
                sample_count,
            )
        }
    }

    /// Registers the opt-in AO chain: the raw kernel runs over the g-buffer
    /// depth/normals into half-res `ao_raw`, the bilateral blur smooths it
    /// into `ao`.
    ///
    /// Returns whether the occluded lighting pass must be wired: true when
    /// AO is on and the deferred chain exists to feed it. Like fog on
    /// forward plans, wiring AO without the deferred chain fails at layout
    /// time (read-before-write on the g-buffer layers).
    fn register_ao(
        systems: &mut SystemSet,
        surface_format: wgpu::TextureFormat,
        ao: Option<crate::ao::AoWiring>,
        technique: Technique,
    ) -> bool {
        let Some(wiring) = ao else {
            return false;
        };
        let settings = wiring.settings();
        systems.register_resource::<AoRaw>(surface_format);
        systems.register_resource::<Ao>(surface_format);
        systems.add_system(AoRawPass::new(settings));
        systems.add_system(AoBlurPass::new(settings));
        technique.has_deferred()
    }

    /// Registers the opt-in anti-aliasing chain: the intermediate layer
    /// the composite writes and the pass that filters it.
    ///
    /// The temporal mode wires the resolve (it reads the depth), the
    /// post-process modes the edge filter; the disabled mode wires
    /// nothing.
    fn register_anti_alias(systems: &mut SystemSet, aa: Option<AaWiring>) {
        let Some(wiring) = aa else {
            return;
        };
        if !wiring.mode.is_enabled() {
            return;
        }
        match wiring.mode {
            AntiAliasMode::Taa => {
                systems.add_system(TaaPass::new(wiring.taa_settings));
            }
            AntiAliasMode::Fxaa | AntiAliasMode::Smaa => {
                systems.add_system(EdgeFilterPass::new(wiring.mode, wiring.edge_settings));
            }
            AntiAliasMode::Off | AntiAliasMode::Msaa => {}
        }
    }

    /// Shared plan builder: registers resources, technique nodes, the
    /// optional AO chain (`ao_raw` → `ao_blur` → occluded `lighting`), the
    /// optional bloom chain, the optional fog node (at `fog` placement),
    /// then the composite pass.
    fn build<O: crate::system::FrameResource>(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: crate::flags::Bloom,
        passes: FramePassWiring,
        sample_count: u32,
    ) -> Self {
        let FramePassWiring { fog, ao, aa } = passes;
        // Register resources by type in the plan's stable declaration order.
        let mut systems = SystemSet::new();
        systems.set_sample_count(sample_count);
        let sample_count = systems.sample_count();
        systems.set_surface_size(surface_size);
        let ids = FrameIds {
            albedo: systems.register_resource::<Albedo>(surface_format),
            normal: systems.register_resource::<Normal>(surface_format),
            material_id: systems.register_resource::<MaterialId>(surface_format),
            world_position: systems.register_resource::<WorldPosition>(surface_format),
            material_params: systems.register_resource::<MaterialParams>(surface_format),
            depth: systems.register_resource::<Depth>(surface_format),
            hdr: systems.register_resource::<Hdr>(surface_format),
            hdr_fwd: systems.register_resource::<HdrFwd>(surface_format),
            target: systems.register_resource::<Target>(surface_format),
            bloom0: systems.register_resource::<Bloom0>(surface_format),
            bloom1: systems.register_resource::<Bloom1>(surface_format),
            bloom2: systems.register_resource::<Bloom2>(surface_format),
        };
        if aa.is_some_and(|wiring| wiring.mode.is_enabled()) {
            // The composite writes this layer before the filter reads it.
            systems.register_resource::<LdrLayer>(surface_format);
        }
        let taa = aa.is_some_and(|wiring| wiring.mode == AntiAliasMode::Taa);
        if taa {
            systems.register_resource::<Motion>(surface_format);
            systems.register_resource::<PreviousDepth>(surface_format);
        }
        if technique.has_deferred() {
            if taa {
                systems.add_system(GbufferMotionPass);
            } else {
                systems.add_system(GbufferPass);
            }
        }
        // Opt-in AO between gbuffer and lighting (see `register_ao`).
        let ao_lighting = Self::register_ao(&mut systems, surface_format, ao, technique);
        if technique.has_deferred() {
            if ao_lighting {
                systems.add_system(LightingAoPass);
            } else {
                systems.add_system(LightingPass);
            }
        }
        if technique.has_forward() {
            // Forward owns depth; hybrid shares the g-buffer depth.
            if technique == Technique::Forward {
                systems.add_system(Forward::<OwnsDepth>::new());
            } else {
                systems.add_system(Forward::<SharedDepth>::new());
            }
        }
        if bloom.is_on() {
            // Bright-pass reads the HDR layer produced by this technique.
            if technique.has_deferred() {
                systems.add_system(BloomBright::<FromDeferred>::new());
            } else {
                systems.add_system(BloomBright::<FromForward>::new());
            }
            systems.add_system(BloomDown1Pass);
            systems.add_system(BloomDown2Pass);
            systems.add_system(BloomUp1Pass);
            systems.add_system(BloomUp0Pass);
        }
        // Before-composite fog is a plan-shape node; composite clears target.
        if let Some(FogWiring {
            placement: FogPlacement::BeforeComposite,
            state,
        }) = fog
        {
            systems.add_system(FogPass::new(state));
        }
        // Select the composite variant from the technique and bloom state.
        match (technique, bloom) {
            (Technique::Deferred, Bloom::On) => {
                systems.add_system(Composite::<CompositeDeferredBloom, O>::new());
            }
            (Technique::Deferred, Bloom::Off) => {
                systems.add_system(Composite::<CompositeDeferred, O>::new());
            }
            (Technique::Forward, Bloom::On) => {
                systems.add_system(Composite::<CompositeForwardBloom, O>::new());
            }
            (Technique::Forward, Bloom::Off) => {
                systems.add_system(Composite::<CompositeForward, O>::new());
            }
            (Technique::Hybrid, Bloom::On) => {
                systems.add_system(Composite::<CompositeHybridBloom, O>::new());
            }
            (Technique::Hybrid, Bloom::Off) => {
                systems.add_system(Composite::<CompositeHybrid, O>::new());
            }
        }
        // The AA pass filters the composite and presents the result.
        Self::register_anti_alias(&mut systems, aa);
        // After-composite fog writes the presented frame.
        if let Some(FogWiring {
            placement: FogPlacement::AfterComposite,
            state,
        }) = fog
        {
            systems.add_system(FogPass::new(state));
        }
        Self {
            executor: FrameExecutor::new(),
            parallel_recording: false,
            ids,
            systems,
            bloom,
            technique,
            anti_alias: aa.map_or(AntiAliasMode::Msaa, |wiring| wiring.mode),
            sample_count,
        }
    }

    /// Resource handles of this plan.
    pub fn ids(&self) -> FrameIds {
        self.ids
    }

    /// Negotiated MSAA sample count of the pooled geometry targets (`1`
    /// or `4`): matches the [`Renderer3D`] this plan renders against.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The technique this plan was wired for.
    pub fn technique(&self) -> Technique {
        self.technique
    }

    /// Anti-aliasing mode this plan wires (the renderer must select the
    /// same mode through [`Renderer3D::set_anti_alias`]).
    // qual:api — public plan setting query for external frame orchestration.
    pub fn anti_alias_mode(&self) -> AntiAliasMode {
        self.anti_alias
    }

    /// Whether the bloom cascade is wired into this plan.
    pub fn bloom_enabled(&self) -> bool {
        self.bloom.is_on()
    }

    /// Bloom wiring as a [`crate::flags::Bloom`].
    pub fn bloom_state(&self) -> crate::flags::Bloom {
        self.bloom
    }

    /// Read access to the declaration registry (layout diagnostics, probes).
    pub fn systems(&self) -> &SystemSet {
        &self.systems
    }

    /// Mutable access to the declaration registry. Any mutation invalidates
    /// the layout cache (see `SystemSet::layout`); intended for
    /// benchmarks/tests that drive recomputation explicitly.
    pub fn systems_mut(&mut self) -> &mut SystemSet {
        &mut self.systems
    }

    /// Updates the surface size before the next render (window resize).
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.systems.set_surface_size((width, height));
    }

    /// Enables/disables parallel command recording (S5b). Off by
    /// default: the sequential path records into the caller's single
    /// encoder; with this on, each pass gets its own encoder, passes of
    /// one parallel level record concurrently and all buffers submit in
    /// pass order — pixel-identical to the sequential path.
    pub fn set_parallel_recording(&mut self, parallel: bool) {
        self.parallel_recording = parallel;
    }

    /// Whether parallel recording is on.
    pub fn parallel_recording(&self) -> bool {
        self.parallel_recording
    }

    /// Sets the S4 GPU memory budget for the transient pool; the next
    /// layout computation refuses (panic via `render`/`layout`, or a
    /// `BudgetExceeded` from `systems_mut().try_layout()`) if exceeded.
    pub fn set_budget(&mut self, budget: Budget) {
        self.systems.set_budget(budget);
    }

    /// Textual layout dump for debugging/reporting (uses the layout cache).
    pub fn layout_dump(&mut self) -> String {
        self.systems.layout().debug_dump()
    }

    /// Number of pooled GPU textures (vs. declared resources — the
    /// difference is the aliasing win).
    pub fn pool_slots(&self) -> usize {
        self.executor.slots_len()
    }

    /// Bytes of the pooled GPU textures (see `FrameExecutor::texture_budget`).
    pub fn texture_budget(&self) -> u64 {
        self.executor.texture_budget()
    }

    /// Renders one frame through the plan: builds the layout, feeds the
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
        self.render_with_custom(context, renderer, mesh, instance_count, &[]);
    }

    /// Like [`render`](Self::render), plus the staged per-entity custom
    /// draws (`RenderSubmit` output via [`custom_draw_items`](crate::renderer::custom_draw_items)):
    /// sphere instances occupy `0..instance_count`, custom instances the
    /// slots above. Empty `customs` renders exactly like
    /// [`render`](Self::render).
    pub fn render_with_custom(
        &mut self,
        context: crate::render_backend::RenderContext<'_>,
        renderer: &Renderer3D,
        mesh: &Mesh,
        instance_count: u32,
        customs: &[CustomGbufferDraw<'_>],
    ) {
        let Self {
            executor,
            parallel_recording,
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
        // is a cache hit, `compute_layout` stays off the hot path. The
        // snapshot is memoized in the executor (`Arc`, keyed by the plan's
        // declaration generation), so frames share it without cloning.
        let layout = executor.ensure_layout(systems);
        executor.set_external_view(ids.target, target.clone());
        let dispatch = PassDispatch {
            systems,
            device,
            queue,
            renderer,
            mesh,
            instance_count,
            customs,
        };
        if *parallel_recording {
            executor.execute_parallel(device, queue, &layout, |_index, pass, enc| {
                dispatch_pass(&dispatch, enc, pass);
            });
        } else {
            executor.execute(device, encoder, &layout, |encoder, pass| {
                dispatch_pass(&dispatch, encoder, &pass);
            });
        }
    }

    /// E1 (S5e): renders one frame driven by the projected core
    /// `Schedule` — every pass is a `PassSystem` declaration twin
    /// ([`crate::schedule_bridge`]), levels come from the unified
    /// scheduler engine, and the dispatch records through the caller's
    /// borrowed encoder exactly like the sequential path. Pixel-identical
    /// to [`render`](Self::render) by construction: projected levels are
    /// pinned equal to `FrameLayout::levels()` (parity canon,
    /// `scheduler_parity`), and a debug assertion re-checks it per frame.
    ///
    /// The projection is built per call in E1; E2's
    /// [`render_to_buffers`](Self::render_to_buffers) supersedes this on
    /// the native runtime path — this method stays the borrowed-encoder
    /// contract for callers that own the submit themselves.
    ///
    /// # Errors
    /// Returns the [`ProjectionError`] of
    /// [`schedule_bridge::try_project_schedule`](crate::schedule_bridge::try_project_schedule)
    /// when a pass touches a resource without a typed registry identity.
    pub fn render_schedule(
        &mut self,
        context: crate::render_backend::RenderContext<'_>,
        renderer: &Renderer3D,
        mesh: &Mesh,
        instance_count: u32,
    ) -> Result<(), ProjectionError> {
        self.render_schedule_with_custom(context, renderer, mesh, instance_count, &[])
    }

    /// Like [`render_schedule`](Self::render_schedule), plus the staged
    /// per-entity custom draws (see
    /// [`render_with_custom`](Self::render_with_custom)). Empty `customs`
    /// renders exactly like [`render_schedule`](Self::render_schedule).
    ///
    /// # Errors
    /// Returns the [`ProjectionError`] of
    /// [`schedule_bridge::try_project_schedule`](crate::schedule_bridge::try_project_schedule)
    /// when a pass touches a resource without a typed registry identity.
    pub fn render_schedule_with_custom(
        &mut self,
        context: crate::render_backend::RenderContext<'_>,
        renderer: &Renderer3D,
        mesh: &Mesh,
        instance_count: u32,
        customs: &[CustomGbufferDraw<'_>],
    ) -> Result<(), ProjectionError> {
        let Self {
            executor,
            ids,
            systems,
            ..
        } = self;
        let (layout, order) = Self::projected_order(executor, systems)?;
        executor.set_external_view(ids.target, context.target.clone());
        let dispatch = PassDispatch {
            systems,
            device: context.device,
            queue: context.queue,
            renderer,
            mesh,
            instance_count,
            customs,
        };
        executor.execute_in_order(
            context.device,
            context.encoder,
            &layout,
            &order,
            |encoder, pass| dispatch_pass(&dispatch, encoder, &pass),
        );
        Ok(())
    }

    /// E2 (S5e): renders one frame with the encoder context as frame
    /// data instead of a borrowed parameter: the passes, ordered by the
    /// projected `Schedule` levels, each record into their own encoder
    /// and the finished buffers land in `buffers` in registration order.
    /// No submit happens here — a separate flush step
    /// (`FrameCommandBuffers::flush`, the runtime's `RenderFlush` system)
    /// owns the ordered queue handover, so recording and submit compose
    /// inside one schedule.
    ///
    /// Pixel-identical to [`render`](Self::render): per-pass encoders +
    /// one ordered submit is the proven `execute_parallel` mechanics
    /// (pinned by the E2 gate in `tests/schedule_render.rs`).
    /// Native-only: the handover resource requires wgpu
    /// `CommandBuffer: Send` (the web backend's is not).
    ///
    /// # Errors
    /// Returns the [`ProjectionError`] of the schedule projection when a
    /// pass touches a resource without a typed registry identity.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_to_buffers(
        &mut self,
        context: BufferRenderContext<'_>,
    ) -> Result<(), ProjectionError> {
        let Self {
            executor,
            ids,
            systems,
            ..
        } = self;
        let (layout, order) = Self::projected_order(executor, systems)?;
        executor.set_external_view(ids.target, context.target.clone());
        let dispatch = PassDispatch {
            systems,
            device: context.device,
            queue: context.queue,
            renderer: context.renderer,
            mesh: context.mesh,
            instance_count: context.instance_count,
            customs: context.customs,
        };
        executor.record_in_order(
            context.device,
            &layout,
            &order,
            &context.buffers.0,
            |_index, pass, encoder| {
                dispatch_pass(&dispatch, encoder, pass);
            },
        );
        Ok(())
    }

    /// E1/E2 shared: the projected schedule order for the current
    /// declarations plus the memoized frame layout, with the parity
    /// invariant (projected levels == `FrameLayout::levels()`) re-checked
    /// per frame in debug builds.
    fn projected_order(
        executor: &mut FrameExecutor,
        systems: &SystemSet,
    ) -> Result<(Arc<FrameLayout>, Vec<usize>), ProjectionError> {
        let schedule = crate::schedule_bridge::try_project_schedule(systems)?;
        let layout = executor.ensure_layout(systems);
        let levels = schedule.levels();
        debug_assert_eq!(
            levels,
            layout.levels(),
            "projected Schedule levels != FrameLayout::levels()"
        );
        let order: Vec<usize> = levels.iter().flatten().copied().collect();
        Ok((layout, order))
    }
}

/// Borrowed per-frame pass dispatch context shared by all
/// `RenderFrame3D` execution paths (sequential, parallel,
/// schedule-ordered): the registry plus the frame inputs a
/// borrowed-encoder [`Frame`] is built from.
struct PassDispatch<'a> {
    systems: &'a SystemSet,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    renderer: &'a Renderer3D,
    mesh: &'a Mesh,
    instance_count: u32,
    customs: &'a [CustomGbufferDraw<'a>],
}

/// Frame inputs for [`RenderFrame3D::render_to_buffers`] (E2): GPU
/// handles, draw state and the handover sink, grouped to stay within the
/// argument budget. Native-only, like the call itself.
#[cfg(not(target_arch = "wasm32"))]
pub struct BufferRenderContext<'a> {
    /// Device used for the per-pass encoders.
    pub device: &'a wgpu::Device,
    /// Upload/submit queue.
    pub queue: &'a wgpu::Queue,
    /// Swapchain view of the frame.
    pub target: &'a wgpu::TextureView,
    /// Deferred renderer (pipelines + buffers).
    pub renderer: &'a Renderer3D,
    /// Instanced mesh drawn by the frame.
    pub mesh: &'a Mesh,
    /// Instances to draw.
    pub instance_count: u32,
    /// E2 handover sink for the per-pass command buffers.
    pub buffers: &'a FrameCommandBuffers,
    /// Staged per-entity custom draws (empty when the scene holds no
    /// custom geometry — spheres-unchanged).
    pub customs: &'a [CustomGbufferDraw<'a>],
}

/// Runs one pass through the registry dispatch: builds the
/// borrowed-encoder [`Frame`] and invokes the typed runner (S2b).
fn dispatch_pass(
    dispatch: &PassDispatch<'_>,
    encoder: &mut wgpu::CommandEncoder,
    pass: &PassViews<'_>,
) {
    let mut frame = Frame {
        device: dispatch.device,
        queue: dispatch.queue,
        encoder,
        renderer: dispatch.renderer,
        mesh: dispatch.mesh,
        instance_count: dispatch.instance_count,
        customs: dispatch.customs,
    };
    if !dispatch.systems.run_pass(pass.pass().id, pass, &mut frame) {
        unreachable!(
            "render frame 3d: pass '{}' is not a typed system",
            pass.pass().name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient_pool::{SizePolicy, TextureSpec};

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
        assert_eq!(
            Technique::Forward.technique(),
            crate::flags::CompositeTechnique::Forward
        );
        assert_eq!(Technique::Forward.composite_mode(), 1);
        // Deferred: deferred chain only, no forward node.
        assert!(Technique::Deferred.has_deferred());
        assert!(!Technique::Deferred.has_forward());
        assert_eq!(
            Technique::Deferred.technique(),
            crate::flags::CompositeTechnique::Deferred
        );
        assert_eq!(Technique::Deferred.composite_mode(), 0);
        // Hybrid: both node sets.
        assert!(Technique::Hybrid.has_deferred());
        assert!(Technique::Hybrid.has_forward());
        assert_eq!(
            Technique::Hybrid.technique(),
            crate::flags::CompositeTechnique::Hybrid
        );
        assert_eq!(Technique::Hybrid.composite_mode(), 2);
    }

    #[test]
    fn technique_wires_expected_passes() {
        let pass_names = |technique: Technique| {
            let mut plan = RenderFrame3D::new_with(
                wgpu::TextureFormat::Rgba8Unorm,
                (32, 32),
                technique,
                Bloom::On,
            );
            plan.systems
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
    fn default_plans_register_no_fog() {
        // Golden guard: fog is opt-in — no default plan wires a "fog"
        // node, so the default frame stays pixel-identical (see
        // `FogWiring`).
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [Bloom::Off, Bloom::On] {
                let mut plan = RenderFrame3D::new_with(
                    wgpu::TextureFormat::Rgba8Unorm,
                    (32, 32),
                    technique,
                    bloom,
                );
                let names = plan
                    .systems
                    .build()
                    .passes
                    .iter()
                    .map(|p| p.name.clone())
                    .collect::<Vec<_>>();
                assert!(
                    !names.iter().any(|name| name == "fog"),
                    "{technique:?}/{bloom:?} registers no fog: {names:?}"
                );
            }
        }
    }

    #[test]
    fn fog_wiring_places_node_per_placement() {
        use crate::frame_passes::{FogSettings, FogState};
        // Enabled fog on a deferred plan compiles; placement decides the
        // node position relative to `composite`.
        let names_with = |placement: FogPlacement| {
            let mut plan = RenderFrame3D::new_with_fog(
                wgpu::TextureFormat::Rgba8Unorm,
                (32, 32),
                Technique::Hybrid,
                Bloom::Off,
                FogWiring::new(placement, FogState::Enabled(FogSettings::default())),
            );
            plan.systems
                .build()
                .passes
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names_with(FogPlacement::BeforeComposite),
            vec!["gbuffer", "lighting", "forward", "fog", "composite"]
        );
        assert_eq!(
            names_with(FogPlacement::AfterComposite),
            vec!["gbuffer", "lighting", "forward", "composite", "fog"]
        );
    }

    #[test]
    fn fog_disabled_wiring_matches_no_fog_frame() {
        use crate::frame_passes::FogState;
        // Placement parity: a Disabled node records no commands
        // ([`FogPass`](crate::frame_passes::FogPass) no-ops), so both
        // placements present the same frame as no fog — pin the node
        // shape plus the identity math.
        for placement in [FogPlacement::BeforeComposite, FogPlacement::AfterComposite] {
            let wiring = FogWiring::new(placement, FogState::Disabled);
            assert!(wiring.state.is_disabled());
            let mut plan = RenderFrame3D::new_with_fog(
                wgpu::TextureFormat::Rgba8Unorm,
                (32, 32),
                Technique::Deferred,
                Bloom::Off,
                wiring,
            );
            let names = plan
                .systems
                .build()
                .passes
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>();
            assert_eq!(names.iter().filter(|name| *name == "fog").count(), 1);
            let last = match placement {
                FogPlacement::BeforeComposite => "composite",
                FogPlacement::AfterComposite => "fog",
            };
            assert_eq!(names.last().map(String::as_str), Some(last));
        }
        let color = [0.2, 0.4, 0.6];
        assert_eq!(
            crate::frame_passes::apply_fog(color, 42.0, FogState::Disabled),
            color
        );
    }

    #[test]
    #[should_panic(expected = "before any write")]
    fn fog_on_forward_plan_fails_without_hdr() {
        use crate::frame_passes::{FogSettings, FogState};
        // Fog reads the deferred `hdr` layer, which forward-only plans
        // never write: compiling the layout fails (read-before-write).
        let mut plan = RenderFrame3D::new_with_fog(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            Bloom::Off,
            FogWiring::new(
                FogPlacement::AfterComposite,
                FogState::Enabled(FogSettings::default()),
            ),
        );
        let _ = plan.systems.build();
    }

    #[test]
    fn default_plans_register_no_ao() {
        use crate::ao::{AoSettings, AoWiring};
        // Golden guard: AO is opt-in — no default plan wires `ao_raw` /
        // `ao_blur` nodes, so the default frame stays pixel-identical (see
        // `AoWiring`). The wired plan inserts both before `lighting`, and
        // `lighting` then reads the blurred `ao` factor.
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [Bloom::Off, Bloom::On] {
                let mut plan = RenderFrame3D::new_with(
                    wgpu::TextureFormat::Rgba8Unorm,
                    (32, 32),
                    technique,
                    bloom,
                );
                let names = plan
                    .systems
                    .build()
                    .passes
                    .iter()
                    .map(|p| p.name.clone())
                    .collect::<Vec<_>>();
                assert!(
                    !names
                        .iter()
                        .any(|name| name == "ao_raw" || name == "ao_blur"),
                    "{technique:?}/{bloom:?} registers no AO: {names:?}"
                );
            }
        }
        let wiring = AoWiring::new(AoSettings::default());
        assert_eq!(wiring.settings(), AoSettings::default());
        for method in [crate::ao::AoMethod::Ssao, crate::ao::AoMethod::Gtao] {
            let settings =
                crate::ao::AoSettings::new(method, 0.5, 1.0, 0.025).expect("valid AO settings");
            let mut plan = RenderFrame3D::new_with_ao(
                wgpu::TextureFormat::Rgba8Unorm,
                (32, 32),
                Technique::Deferred,
                Bloom::Off,
                AoWiring::new(settings),
            );
            let names = plan
                .systems
                .build()
                .passes
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>();
            assert_eq!(
                names,
                vec!["gbuffer", "ao_raw", "ao_blur", "lighting", "composite"]
            );
        }
    }

    #[test]
    fn ao_adds_two_half_res_slots_and_extends_hdr_lifetime() {
        use crate::ao::{AoSettings, AoWiring};
        // Deferred, no bloom, 1280×720: the AO chain adds exactly its two
        // half-res targets (same spec → shared only across non-overlapping
        // windows, and `ao_raw`/`ao` overlap on the blur pass).
        let mut plain = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Deferred,
            Bloom::Off,
        );
        let plain_slots = plain.systems.layout().slots.len();
        let mut ao_plan = RenderFrame3D::new_with_ao(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Deferred,
            Bloom::Off,
            AoWiring::new(AoSettings::default()),
        );
        let layout = ao_plan.systems.layout().clone();
        assert_eq!(
            layout.slots.len(),
            plain_slots + 2,
            "AO adds exactly its two half-res slots"
        );
        // `ao_raw` lives [ao_raw, ao_blur], `ao` lives [ao_blur, lighting]:
        // pass indices are gbuffer=0, ao_raw=1, ao_blur=2, lighting=3.
        let window = |name: &str| {
            let rl = layout
                .resources
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("resource '{name}' missing"));
            (rl.first_use, rl.last_use)
        };
        assert_eq!(window("ao_raw"), (1, 2));
        assert_eq!(window("ao"), (2, 3));
        assert_eq!(window("hdr"), (3, 4));
        // Pass cost ledger for the PR: two half-res fullscreen draws plus
        // one lighting sample — 2 × (640×360) AO texels per frame.
        let (w, h) = crate::ao::half_extent(1280, 720);
        assert_eq!((w, h), (640, 360));
        assert_eq!(
            layout.planned_pool_bytes(),
            plain.systems.layout().planned_pool_bytes() + 2 * u64::from(w) * u64::from(h) * 4,
            "AO pool cost is exactly two half-res Rgba8 targets"
        );
    }

    #[test]
    #[should_panic(expected = "before any write")]
    fn ao_on_forward_plan_fails_without_gbuffer() {
        use crate::ao::{AoSettings, AoWiring};
        // AO reads the g-buffer layers, which forward-only plans never
        // write: compiling the layout fails (read-before-write, like fog).
        let mut plan = RenderFrame3D::new_with_ao(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            Bloom::Off,
            AoWiring::new(AoSettings::default()),
        );
        let _ = plan.systems.build();
    }

    #[test]
    #[should_panic(expected = "before any write")]
    fn taa_on_forward_plan_fails_without_motion_vectors() {
        // TAA requires motion vectors from the deferred g-buffer. A
        // forward-only plan must fail layout validation rather than sample
        // an unwritten attachment.
        let mut plan = RenderFrame3D::new_with_aa(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            Bloom::Off,
            AaWiring::new(AntiAliasMode::Taa),
        );
        let _ = plan.systems.build();
    }

    // qual:allow(dry, fragment) reason: "The AA mode table intentionally repeats equivalent graph assertions per technique"
    #[test]
    fn anti_alias_plans_wire_the_matching_pass() {
        // The mode decides the pass: the temporal resolve reads the
        // depth, the post-process filters only the composited image. The
        // disabled mode keeps the plain plan (no filter pass at all).
        let names_of = |mode: AntiAliasMode| {
            let mut plan = RenderFrame3D::new_with_aa(
                wgpu::TextureFormat::Rgba8Unorm,
                (32, 32),
                Technique::Hybrid,
                Bloom::Off,
                AaWiring::new(mode),
            );
            plan.systems
                .build()
                .passes
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names_of(AntiAliasMode::Taa),
            vec!["gbuffer", "lighting", "forward", "composite", "taa"]
        );
        assert_eq!(
            names_of(AntiAliasMode::Fxaa),
            vec!["gbuffer", "lighting", "forward", "composite", "edge_filter"]
        );
        assert_eq!(
            names_of(AntiAliasMode::Smaa),
            vec!["gbuffer", "lighting", "forward", "composite", "edge_filter"]
        );
        assert_eq!(
            names_of(AntiAliasMode::Off),
            vec!["gbuffer", "lighting", "forward", "composite"]
        );
        assert_eq!(
            names_of(AntiAliasMode::Msaa),
            vec!["gbuffer", "lighting", "forward", "composite"]
        );
    }

    // qual:allow(dry, fragment) reason: "The resource graph assertions deliberately mirror the separate AA mode table"
    #[test]
    fn anti_alias_plan_filters_the_composited_layer() {
        // The composite writes the intermediate layer, the anti-aliasing
        // pass reads it and presents the target: the presented image is
        // the filtered one.
        let mut plan = RenderFrame3D::new_with_aa(
            wgpu::TextureFormat::Rgba8Unorm,
            (64, 64),
            Technique::Deferred,
            Bloom::Off,
            AaWiring::new(AntiAliasMode::Taa),
        );
        let layout = plan.systems.layout();
        let resource = |name: &str| {
            layout
                .resources
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("{name} is declared"))
        };
        let pass = |name: &str| {
            layout
                .passes
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} is registered"))
        };
        let ldr = resource("ldr_layer");
        let motion = resource("motion");
        let previous_depth = resource("previous_depth");
        let target = resource("target");
        let gbuffer = pass("gbuffer");
        let composite = pass("composite");
        let taa = pass("taa");
        assert_eq!(motion.spec.format, wgpu::TextureFormat::Rg16Float);
        assert_eq!(previous_depth.spec.format, wgpu::TextureFormat::R16Uint);
        assert!(gbuffer.writes.iter().any(|(id, _)| *id == motion.id));
        assert!(
            gbuffer
                .writes
                .iter()
                .any(|(id, _)| *id == previous_depth.id)
        );
        assert!(taa.reads.contains(&motion.id));
        assert!(taa.reads.contains(&previous_depth.id));
        assert!(
            composite.writes.iter().any(|(id, _)| *id == ldr.id),
            "the composite writes the intermediate layer"
        );
        assert!(taa.reads.contains(&ldr.id), "the pass reads it");
        assert!(
            taa.writes.iter().any(|(id, _)| *id == target.id),
            "the pass presents the target"
        );
        assert!(
            !composite.writes.iter().any(|(id, _)| *id == target.id),
            "the composite no longer presents the frame"
        );
    }

    #[test]
    fn technique_bloom_input_follows_hdr_layer() {
        // Forward-only: bloom's bright-pass input is `hdr_fwd` (hdr dead).
        let mut forward = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            Bloom::On,
        );
        let layout = forward.systems.build();
        let down0 = layout
            .passes
            .iter()
            .find(|p| p.name == "bloom_down0")
            .expect("bloom_down0 exists");
        assert_eq!(down0.reads, vec![forward.ids.hdr_fwd]);
        // Hybrid: bright-pass input is `hdr`.
        let mut hybrid = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Hybrid,
            Bloom::On,
        );
        let layout = hybrid.systems.build();
        let down0 = layout
            .passes
            .iter()
            .find(|p| p.name == "bloom_down0")
            .expect("bloom_down0 exists");
        assert_eq!(down0.reads, vec![hybrid.ids.hdr]);
    }

    #[test]
    fn technique_forward_owns_depth_in_forward_mode() {
        let mut forward = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Forward,
            Bloom::Off,
        );
        let layout = forward.systems.build();
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
        let mut plan = SystemSet::new();
        plan.set_surface_size((1280, 720));
        let a = plan.create_resource("a", spec);
        let b = plan.create_resource("b", spec);
        plan.add_pass("p0").write(a);
        plan.add_pass("p2").read(a);
        plan.add_pass("p3").write(b);
        plan.add_pass("p4").read(b);
        // a [0,1], b [2,3]: same spec, non-overlapping → one slot.
        let layout = plan.build();
        assert_eq!(layout.slots.len(), 1);
        // Budget math is exercised on the wgpu side (needs a device); here
        // we only pin the per-slot byte formula via the layout spec.
        assert_eq!(
            format_bytes_per_pixel(spec.format) as u64 * 1280 * 720,
            7_372_800
        );
    }

    // ── Unified scheduler: sequential execute follows DAG levels ──────

    #[test]
    fn sequential_execute_matches_registration_on_production_plans() {
        // `FrameExecutor::execute` iterates the flattened `layout.levels()`
        // from the shared `bitset_level_plan` engine. On every production
        // wiring that order must coincide with registration order —
        // otherwise the default `render()` path would reorder passes.
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [Bloom::Off, Bloom::On] {
                let mut plan = RenderFrame3D::new_with(fmt, (32, 32), technique, bloom);
                let layout = plan.systems.build();
                let flat: Vec<usize> = layout.levels().iter().flatten().copied().collect();
                let registration: Vec<usize> = (0..layout.passes.len()).collect();
                assert_eq!(
                    flat, registration,
                    "technique {technique:?} bloom {bloom:?}: DAG order != registration order"
                );
            }
        }
    }

    #[test]
    fn dag_levels_pull_independent_passes_forward() {
        // A diamond the shared level engine must expose: `p2` is independent
        // of the `p0 → p1` chain, so it shares level 0 and the flattened DAG
        // order — the order `execute` now follows — is `[0, 2, 1]`, not
        // registration order.
        let spec = TextureSpec {
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            size: SizePolicy::Fixed {
                width: 64,
                height: 64,
            },
        };
        let mut set = SystemSet::new();
        set.set_surface_size((64, 64));
        let a = set.create_resource("a", spec);
        let b = set.create_resource("b", spec);
        let c = set.create_resource("c", spec);
        set.add_pass("p0").write(a);
        set.add_pass("p1").read(a).write(b);
        set.add_pass("p2").write(c);
        let layout = set.build();
        assert_eq!(layout.levels(), vec![vec![0, 2], vec![1]]);
        let flat: Vec<usize> = layout.levels().iter().flatten().copied().collect();
        assert_eq!(flat, vec![0, 2, 1]);
    }

    // ── S1: layout cache on the RenderFrame3D level ──────────────────

    #[test]
    fn layout_cache_reused_across_frames() {
        let mut g3 = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            Bloom::On,
        );
        // Two "frames" without mutations → one computation.
        let _ = g3.systems.layout();
        let dump_a = g3.layout_dump();
        let _ = g3.systems.layout();
        let dump_b = g3.layout_dump();
        assert_eq!(g3.systems.layout_computations(), 1);
        assert_eq!(dump_a, dump_b, "layout must not change between frames");

        // Window resize → recompute once, then cache again.
        g3.set_surface_size(1920, 1080);
        let _ = g3.layout_dump();
        assert_eq!(g3.systems.layout_computations(), 2);
        let _ = g3.layout_dump();
        assert_eq!(g3.systems.layout_computations(), 2, "cached after resize");
    }

    // ── Dissolution: executor-owned transient pool snapshot ──────────────

    #[test]
    fn executor_memoizes_layout_across_frames() {
        // d3: the production registry is `SystemSet`; the executor's
        // `ensure_layout` borrows it immutably, so the registry-owned
        // pool (cold path: `SystemSet::layout()`) is untouched by the
        // frame hot path. The `scheduler_parity` integration test pins
        // the same memoization contract on the parity frontends.
        let spec = TextureSpec {
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            size: SizePolicy::Fixed {
                width: 64,
                height: 64,
            },
        };
        let mut set = SystemSet::new();
        let a = set.create_resource("a", spec);
        set.add_pass("p0").write(a);
        set.add_pass("p1").read(a);
        let generation = set.generation();

        let mut executor = FrameExecutor::new();
        let first = executor.ensure_layout(&set);
        let second = executor.ensure_layout(&set);
        assert!(Arc::ptr_eq(&first, &second), "steady state shares one Arc");
        assert_eq!(
            executor.layout_pool.layout_computations(),
            1,
            "hot path compiles once"
        );
        assert_eq!(
            set.layout_computations(),
            0,
            "cold-path registry pool untouched by the hot path"
        );
        assert_eq!(first.slots.len(), 1);

        // Mutation bumps the generation and refreshes the snapshot.
        set.add_pass("p2").read(a);
        assert_ne!(set.generation(), generation);
        let third = executor.ensure_layout(&set);
        assert!(!Arc::ptr_eq(&second, &third), "mutation refreshes snapshot");
        assert_eq!(executor.layout_pool.layout_computations(), 2);
        assert_eq!(third.passes.len(), 3);
    }

    // ── S2: typed systems must reproduce the imperative wiring ─────────

    /// Verbatim pre-S2 resource registration — the reference the typed
    /// registration (`register_resource`) has to match bit-for-bit.
    fn imperative_resources(
        plan: &mut SystemSet,
        _surface_format: wgpu::TextureFormat,
    ) -> FrameIds {
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
        FrameIds {
            albedo: plan.create_resource("albedo", spec(wgpu::TextureFormat::Rgba8Unorm)),
            normal: plan.create_resource("normal", spec(wgpu::TextureFormat::Rg16Float)),
            material_id: plan.create_resource("material_id", spec(wgpu::TextureFormat::R16Uint)),
            world_position: plan
                .create_resource("world_position", spec(wgpu::TextureFormat::Rg16Float)),
            material_params: plan
                .create_resource("material_params", spec(wgpu::TextureFormat::Rgba16Float)),
            depth: plan.create_resource("depth", spec(wgpu::TextureFormat::Depth32Float)),
            hdr: plan.create_resource("hdr", spec(crate::renderer::DEFERRED_HDR_FORMAT)),
            hdr_fwd: plan.create_resource("hdr_fwd", spec(wgpu::TextureFormat::Rgba16Float)),
            target: plan.external_output("target"),
            bloom0: plan.create_resource("bloom0", frac(wgpu::TextureFormat::Rgba16Float, 2)),
            bloom1: plan.create_resource("bloom1", frac(wgpu::TextureFormat::Rgba16Float, 4)),
            bloom2: plan.create_resource("bloom2", frac(wgpu::TextureFormat::Rgba16Float, 8)),
        }
    }

    /// Verbatim pre-S2 pass wiring — the reference the typed systems
    /// (`add_system`) and the conditional passes have to match.
    fn imperative_passes(plan: &mut SystemSet, ids: &FrameIds, technique: Technique, bloom: Bloom) {
        if technique.has_deferred() {
            plan.add_pass("gbuffer")
                .write(ids.albedo)
                .write(ids.normal)
                .write(ids.material_id)
                .write(ids.world_position)
                .write(ids.material_params)
                .write(ids.depth);
            plan.add_pass("lighting")
                .read(ids.albedo)
                .read(ids.normal)
                .read(ids.material_id)
                .read(ids.world_position)
                .read(ids.material_params)
                .read(ids.depth)
                .write_clear(ids.hdr, wgpu::Color::BLACK);
        }
        if technique.has_forward() {
            let pass = plan.add_pass("forward");
            let pass = if technique == Technique::Forward {
                pass.write_clear(ids.depth, wgpu::Color::WHITE)
            } else {
                pass.read(ids.depth)
            };
            pass.write_clear(ids.hdr_fwd, wgpu::Color::TRANSPARENT);
        }
        if bloom.is_on() {
            let bloom_input = if technique.has_deferred() {
                ids.hdr
            } else {
                ids.hdr_fwd
            };
            plan.add_pass("bloom_down0")
                .read(bloom_input)
                .write_clear(ids.bloom0, wgpu::Color::BLACK);
            plan.add_pass("bloom_down1")
                .read(ids.bloom0)
                .write_clear(ids.bloom1, wgpu::Color::BLACK);
            plan.add_pass("bloom_down2")
                .read(ids.bloom1)
                .write_clear(ids.bloom2, wgpu::Color::BLACK);
            plan.add_pass("bloom_up1")
                .read(ids.bloom2)
                .write(ids.bloom1);
            plan.add_pass("bloom_up0")
                .read(ids.bloom1)
                .write(ids.bloom0);
        }
        let mut composite = plan.add_pass("composite").write(ids.target);
        if technique.has_deferred() {
            composite = composite.read(ids.hdr);
        }
        if technique.has_forward() {
            composite = composite.read(ids.hdr_fwd);
        }
        if bloom.is_on() {
            composite.read(ids.bloom0);
        }
    }

    /// The pre-S2 wiring, verbatim, as one plan: resources then passes.
    fn imperative_wiring(
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        technique: Technique,
        bloom: Bloom,
    ) -> SystemSet {
        let mut plan = SystemSet::new();
        plan.set_surface_size(surface_size);
        let ids = imperative_resources(&mut plan, surface_format);
        imperative_passes(&mut plan, &ids, technique, bloom);
        plan
    }

    #[test]
    fn typed_wiring_matches_imperative_reference() {
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [Bloom::Off, Bloom::On] {
                let mut typed = RenderFrame3D::new_with(fmt, (1280, 720), technique, bloom);
                let mut reference = imperative_wiring(fmt, (1280, 720), technique, bloom);
                assert_eq!(
                    typed.systems.build().debug_dump(),
                    reference.build().debug_dump(),
                    "typed wiring diverged: {technique:?} bloom={bloom:?}"
                );
            }
        }
    }

    // ── S3: golden layout tests — the pool must not change silently ────

    fn slots_for(technique: Technique, bloom: Bloom) -> usize {
        let mut g3 = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            technique,
            bloom,
        );
        g3.systems.layout().slots.len()
    }

    #[test]
    fn golden_pool_slots_per_technique() {
        // `hdr` is Rgba16Float. Its lifetime overlaps `material_params`
        // (same spec), so it still takes its own slot: 9 resources → 7
        // slots on the deferred/hybrid path, same count as when `hdr`
        // mirrored an `Rgba8Unorm` surface and overlapped albedo. The bloom
        // cascade adds exactly its three fraction levels. Forward leaves
        // `hdr` dead.
        assert_eq!(slots_for(Technique::Forward, Bloom::Off), 2);
        assert_eq!(slots_for(Technique::Forward, Bloom::On), 5);
        assert_eq!(slots_for(Technique::Deferred, Bloom::Off), 7);
        assert_eq!(slots_for(Technique::Deferred, Bloom::On), 10);
        assert_eq!(slots_for(Technique::Hybrid, Bloom::Off), 7);
        assert_eq!(slots_for(Technique::Hybrid, Bloom::On), 10);
    }

    #[test]
    fn golden_bloom_adds_exactly_three_slots() {
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            assert_eq!(
                slots_for(technique, Bloom::On) - slots_for(technique, Bloom::Off),
                3,
                "bloom cascade must add exactly its three fraction levels"
            );
        }
    }

    #[test]
    fn golden_dead_layers_are_unpooled() {
        // Forward-only: the deferred HDR layer and the gbuffer targets are
        // never touched → no lifetime window, no pool slot.
        let mut fwd = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Forward,
            Bloom::On,
        );
        let ids = fwd.ids();
        let layout = fwd.systems.layout().clone();
        for id in [ids.hdr, ids.albedo, ids.normal] {
            let rl = &layout.resources[id.0 as usize];
            assert_eq!(rl.first_use, usize::MAX, "{} must be dead", rl.name);
            assert_eq!(rl.slot, None);
        }
        // No bloom → the cascade levels are dead.
        let mut plain = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            Bloom::Off,
        );
        let ids = plain.ids();
        let layout = plain.systems.layout();
        assert_eq!(layout.resources[ids.bloom0.0 as usize].slot, None);
        assert_eq!(layout.resources[ids.bloom1.0 as usize].slot, None);
        assert_eq!(layout.resources[ids.bloom2.0 as usize].slot, None);
    }

    #[test]
    fn golden_hybrid_lifetimes() {
        let mut g3 = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            Bloom::On,
        );
        let ids = g3.ids();
        let layout = g3.systems.layout();
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
    fn production_graph_levels() {
        // Hybrid+bloom levels (pass indices in registration order):
        // gbuffer → {lighting, forward} — deferred layers and the forward path
        // share NO resources, the first real pipeline parallelism —
        // then the bloom chain and composite. The initial "strict
        // chain" expectation is refuted by the test itself: lighting ∥ forward.
        let mut g3 = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            Bloom::On,
        );
        let levels = g3.systems.layout().levels();
        let pass_count = levels.iter().map(|l| l.len()).sum::<usize>();
        assert_eq!(pass_count, 9, "hybrid + bloom: 9 passes");
        assert_eq!(levels[0], vec![0], "gbuffer first");
        assert_eq!(
            levels[1],
            vec![1, 2],
            "lighting runs in parallel with forward"
        );
        for (expected_level, pass) in levels.iter().skip(2).zip(3..9) {
            assert_eq!(*expected_level, vec![pass], "bloom chain + composite");
        }
    }

    #[test]
    fn budget_exceeded_is_actionable() {
        let mut g3 = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Hybrid,
            Bloom::On,
        );
        let planned = g3.systems.layout().planned_pool_bytes();
        // Exact budget — fits.
        g3.set_budget(Budget::gpu_textures(planned));
        assert!(g3.systems.try_layout().is_ok());
        // One byte less — clear rejection with details.
        g3.set_budget(Budget::gpu_textures(planned - 1));
        let err = g3.systems.try_layout().unwrap_err();
        assert_eq!(err.required, planned);
        assert_eq!(err.budget, planned - 1);
        let msg = err.to_string();
        assert!(msg.contains("MiB"), "message: {msg}");
        assert!(
            msg.contains("bloom") || msg.contains("hdr"),
            "offenders named: {msg}"
        );
        // Removing the budget restores S3 behavior.
        g3.set_budget(Budget::unbounded());
        assert!(g3.systems.try_layout().is_ok());
    }

    #[test]
    fn golden_planned_pool_bytes() {
        // Forward, no bloom, 1280×720: depth (D32, 4 B/px) + hdr_fwd
        // (Rgba16, 8 B/px) = 12 B/px over the surface.
        let mut g3 = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (1280, 720),
            Technique::Forward,
            Bloom::Off,
        );
        let layout = g3.systems.layout();
        assert_eq!(layout.planned_pool_bytes(), 12 * 1280 * 720);
    }

    #[test]
    fn plan_1x_stays_single_sample() {
        // The default plan is all-1x: every pooled spec keeps samples 1,
        // so the existing golden dumps stay byte-identical.
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [Bloom::Off, Bloom::On] {
                let mut plan = RenderFrame3D::new_with(
                    wgpu::TextureFormat::Rgba8Unorm,
                    (64, 64),
                    technique,
                    bloom,
                );
                assert_eq!(plan.sample_count(), 1);
                for resource in &plan.systems.build().resources {
                    assert_eq!(
                        resource.spec.samples, 1,
                        "{technique:?}/{bloom:?} {}",
                        resource.name
                    );
                }
            }
        }
    }

    #[test]
    fn plan_4x_specs_follow_negotiated_count() {
        use crate::renderer::MSAA_SAMPLE_COUNT;
        let mut plan = RenderFrame3D::new_with_samples(
            wgpu::TextureFormat::Rgba8Unorm,
            (64, 64),
            Technique::Hybrid,
            Bloom::On,
            MSAA_SAMPLE_COUNT,
        );
        assert_eq!(plan.sample_count(), MSAA_SAMPLE_COUNT);
        let layout = plan.systems.build();
        let samples_of = |name: &str| {
            layout
                .resources
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("resource '{name}' missing"))
                .spec
                .samples
        };
        for msaa in [
            "albedo",
            "normal",
            "material_id",
            "world_position",
            "material_params",
            "depth",
            "hdr_fwd",
        ] {
            assert_eq!(samples_of(msaa), 4, "{msaa} must follow the plan count");
        }
        for single in ["hdr", "bloom0", "bloom1", "bloom2", "target"] {
            assert_eq!(samples_of(single), 1, "{single} must stay single-sample");
        }
        // Out-of-range requests normalize to single-sample (never a panic).
        let fallback = RenderFrame3D::new_with_samples(
            wgpu::TextureFormat::Rgba8Unorm,
            (64, 64),
            Technique::Hybrid,
            Bloom::Off,
            2,
        );
        assert_eq!(fallback.sample_count(), 1);
    }

    #[test]
    fn aa_modes_select_their_required_sample_count() {
        use crate::renderer::{MSAA_SAMPLE_COUNT, SINGLE_SAMPLE_COUNT};
        let make = |mode, samples| {
            RenderFrame3D::new_with_aa_samples(
                wgpu::TextureFormat::Rgba8Unorm,
                (32, 32),
                Technique::Hybrid,
                Bloom::Off,
                AaWiring::new(mode),
                samples,
            )
        };
        assert_eq!(
            make(AntiAliasMode::Off, MSAA_SAMPLE_COUNT).sample_count(),
            SINGLE_SAMPLE_COUNT,
            "Off must not silently run hardware MSAA"
        );
        assert_eq!(
            make(AntiAliasMode::Msaa, MSAA_SAMPLE_COUNT).sample_count(),
            MSAA_SAMPLE_COUNT
        );
        for mode in [AntiAliasMode::Taa, AntiAliasMode::Fxaa, AntiAliasMode::Smaa] {
            assert_eq!(
                make(mode, MSAA_SAMPLE_COUNT).sample_count(),
                SINGLE_SAMPLE_COUNT,
                "{mode:?} replaces MSAA"
            );
        }
    }

    #[test]
    fn msaa_plan_keeps_pool_shape() {
        // Every sharing pair upgrades together (matparams/hdr_fwd are both
        // multisampled), so 4x slot counts match the 1x goldens
        // technique-for-technique — only the specs' sample counts differ.
        for technique in [Technique::Forward, Technique::Deferred, Technique::Hybrid] {
            for bloom in [Bloom::Off, Bloom::On] {
                let mut one = RenderFrame3D::new_with(
                    wgpu::TextureFormat::Rgba8Unorm,
                    (1280, 720),
                    technique,
                    bloom,
                );
                let mut four = RenderFrame3D::new_with_samples(
                    wgpu::TextureFormat::Rgba8Unorm,
                    (1280, 720),
                    technique,
                    bloom,
                    crate::renderer::MSAA_SAMPLE_COUNT,
                );
                assert_eq!(
                    one.systems.layout().slots.len(),
                    four.systems.layout().slots.len(),
                    "slot shape must not change at 4x: {technique:?}/{bloom:?}"
                );
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "'sneaky' (index 1) accesses resource 'b'")]
    fn pass_views_undeclared_view_panics_in_debug() {
        // Backlog #6 / audit §4.1 — "sneaky pass" on the real view
        // dispatch path: `PassViews::view_of` for a ResourceId outside declared
        // reads/writes panics with pass and resource names before touching
        // the pool (empty pool — device-free ground truth).
        let mut plan = SystemSet::new();
        plan.set_surface_size((64, 64));
        let tex = TextureSpec {
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            size: SizePolicy::MatchSurface,
        };
        let a = plan.create_resource("a", tex);
        let b = plan.create_resource("b", tex);
        plan.add_pass("writer").write(a).write(b);
        plan.add_pass("sneaky").read(a);
        let layout = plan.build();
        let pool: Vec<Option<PooledTexture>> = Vec::new();
        let externals = HashMap::new();
        let views = PassViews {
            layout: &layout,
            pool: &pool,
            externals: &externals,
            index: 1,
        };
        // "sneaky" declared only read(a); view(b) is out of set.
        let _ = views.view_of(b);
    }
}
