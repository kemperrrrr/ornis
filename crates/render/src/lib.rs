//! Ornis render library: deferred [`renderer::Renderer3D`], the
//! frame-plan layer ([`transient_pool`]/[`system`]/[`frame_exec`]/[`frame_passes`]),
//! procedural meshes and the WGSL shader assembly. Scene description types
//! live in `ornis-assets`; this crate only projects them.
#![warn(missing_docs)]
/// Anti-aliasing: the mode selection, the temporal jitter and the CPU
/// reference of the resolve kernels (TAA, FXAA, SMAA).
pub mod antialias;
/// Screen-space ambient occlusion (classic SSAO plus GTAO).
pub mod ao;
/// GPU eye adaptation (auto-exposure) from a log-luminance histogram of
/// the lit HDR target: the histogram compute pass, the adapted exposure
/// the composite reads, and the manual/auto [`auto_exposure::ExposureMode`].
pub mod auto_exposure;
/// Client-side orbit camera and backend-neutral input consumer.
pub mod camera;
/// Final PBR/UI blend pass (legacy path).
pub mod composite;
/// Shared ECS-to-render extraction and logical render-world frame host.
pub mod extraction;
/// Typed replacements for `bool`/`u32` frame-plan flags.
pub mod flags;
/// wgpu executor mapping plan slots to textures and running passes.
pub mod frame_exec;
/// Typed pass implementations wired into the frame plan.
pub mod frame_passes;
/// GPU frustum culling with `draw_indexed_indirect` submits (CPU
/// reference compaction, indirect-args layout, compute-shader source,
/// [`gpu_culling::Culler`] buffer owner). The gbuffer pass exposes an
/// indirect entry point (`render_gbuffer_indirect`) selected per frame
/// by the opt-in flag (see `render_backend::RenderBackendConfig`);
/// shadows stay on the direct pre-pass (off-frame casters); without an
/// armed batch the frame draws through the direct `draw_indexed`
/// fallback.
pub mod gpu_culling;
/// GPU resources as ECS singletons for the unified scheduler (S7 design).
/// Native-only: stores wgpu `Device`/`Queue`/`Surface`/`CommandBuffer` as
/// `World` resources (`Send + Sync` bound), but wgpu's web backend types
/// are `!Send`/`!Sync` (Rc, JS callbacks). The wasm path renders through
/// `RenderWorld` extraction + `ornis-wasm`, not through this module.
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_resources;
/// Split-sum image-based lighting (BRDF LUT, prefilter, irradiance).
pub mod ibl;
/// Typed directional light spawned into a world light rig.
pub mod light;
/// GPU mesh representation and primitive generation.
pub mod mesh;
/// Upload of `ornis-mesh-editor` mesh data to the GPU.
pub mod mesh_upload;
/// Deferred MSAA edge mask (covered samples only; clear normal is not +Z-shaded).
pub mod msaa_edge;
/// Backend-neutral rendering trait plus its factory.
pub mod render_backend;
/// The deferred [`renderer::Renderer3D`] and its passes.
pub mod renderer;
/// E1 (S5e) bridge: frame passes projected as ordinary core `Schedule`
/// systems (declaration twins; level parity pinned by `scheduler_parity`).
pub mod schedule_bridge;
/// WGSL shader assembly and Rust-side BRDF math kernels.
pub mod shaders;
/// Phase D GPU skinning: joint-palette layout, skinned vertex stage and
/// palette upload bytes (CPU-skinned extraction stays the fallback).
pub mod skinning;
/// Typed plan systems + single declaration registry (d3).
pub mod system;
/// Compressed-format selection and material mip-chain upload.
pub mod texture_upload;
/// glTF texture image upload (`LoadedImage` → `wgpu` texture + sampler).
pub mod textures;
/// Shared CPU texture representations used by import and rendering.
pub use ornis_texture::{Ktx2Image, TextureEncoding, TextureImage};
/// Transient pool — the dynamic half of the dissolved frame-plan shell
/// (d2): declaration snapshots compile into shared layouts here.
pub mod transient_pool;

pub use antialias::{
    AaWiring, AntiAliasMode, EdgeFilterSettings, JitterSequence, MotionValidity, TaaSettings,
    halton, jittered_projection, neighbourhood_bounds, taa_resolve, ycocg_decode, ycocg_encode,
};
pub use ao::{
    AoMethod, AoSettings, AoWiring, ao_from_sum, apply_ao_to_ambient, bilateral_blur_center,
    bilateral_weight, gtao_clamp_horizon, gtao_falloff, gtao_sample, half_extent,
    linear_view_depth, ssao_sample,
};
/// Whether GPU-gated tests and benches must fail instead of skipping on
/// adapter-less hosts: set when `ORNIS_REQUIRE_GPU` is non-empty and not
/// `"0"` (the CI GPU lane exports it, so a missing adapter surfaces as red
/// rather than a silent skip). Public so the criterion bench — a separate
/// crate root that never compiles under `cfg(test)` — shares this one
/// definition with the in-crate tests. `#[doc(hidden)]`: a cross-target
/// implementation detail, not a stable API.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub fn gpu_required() -> bool {
    std::env::var("ORNIS_REQUIRE_GPU").is_ok_and(|value| !value.is_empty() && value != "0")
}
/// Headless adapter/device pair, or `None` on a host with no wgpu adapter.
///
/// This is the single bootstrap every GPU-gated test and bench calls, so
/// they share one code path instead of three near-copies. When
/// [`gpu_required`] holds (`ORNIS_REQUIRE_GPU` — non-empty, not `"0"`; the
/// CI `test` shard sets `=1`) a missing adapter or device panics instead of
/// returning `None`, so it can never turn a GPU gate into a pass; unset or
/// explicit `"0"` keeps the skip, which is logged here — the shared
/// bootstrap reports every skip, so no caller can skip silently.
///
/// `#[doc(hidden)]`: public only so the criterion bench — a separate crate
/// root that never compiles under `cfg(test)` — reaches it, exactly like
/// [`gpu_required`]. Not a stable API.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub fn try_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            Ok(adapter) => adapter,
            Err(err) => {
                assert!(
                    !gpu_required(),
                    "ORNIS_REQUIRE_GPU is set but no wgpu adapter: {err}"
                );
                eprintln!("SKIPPED (try_headless_device): no wgpu adapter — GPU gate skipped");
                return None;
            }
        };
        match adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
        {
            Ok(pair) => Some(pair),
            Err(err) => {
                assert!(
                    !gpu_required(),
                    "ORNIS_REQUIRE_GPU is set but no wgpu device: {err}"
                );
                eprintln!("SKIPPED (try_headless_device): no wgpu device — GPU gate skipped");
                None
            }
        }
    })
}

/// Copies `bytes` from `src` into a mapping buffer and reads them back
/// (`None` when the headless readback fails — the caller skips instead of
/// failing).
///
/// Shared by every GPU-gated test and bench that needs a buffer readback,
/// so they hold one copy instead of two near-identical ones. `#[doc(hidden)]`:
/// public only so the criterion bench — a separate crate root that never
/// compiles under `cfg(test)` — reaches it, exactly like [`gpu_required`]
/// and [`try_headless_device`]. Not a stable API.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub fn read_bytes_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    bytes: u64,
) -> Option<Vec<u8>> {
    let floor = std::mem::size_of::<u32>() as u64;
    let span = bytes.max(floor);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu test readback"),
        size: span,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu test readback encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = readback.slice(0..span);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return None;
    }
    let Ok(Ok(())) = rx.recv() else {
        return None;
    };
    let Ok(view) = slice.get_mapped_range() else {
        return None;
    };
    let out = view[..bytes as usize].to_vec();
    drop(view);
    readback.unmap();
    Some(out)
}
pub use camera::{
    CameraView, OrbitCamera, PointerDpr, camera_view_projection, install_orbit_camera,
    read_orbit_camera,
};
pub use composite::CompositePass as LegacyCompositePass;
pub use extraction::{
    ExtractionStats, FrameUpload, MeshPose, RenderLights, RenderWorld, SkinInfluences,
    extract_render_data, extract_render_data_with_stats, max_mesh_params, unlit_material,
};
pub use flags::{
    Access as AccessKind, AnisotropyLevel, Bloom, CompositeTechnique, DepthOwnership, PassState,
    ResourceBacking, SamplerKind, ShadowCast,
};
pub use frame_exec::{FrameExecutor, FrameIds, PassViews, RenderFrame3D, Technique};
pub use frame_passes::{
    Ao, AoBlurPass, AoRaw, AoRawPass, FogDensity, FogPass, FogPlacement, FogSettings, FogState,
    FogWiring, LightingAoPass, apply_fog, fog_factor,
};
pub use gpu_culling::{
    CULL_COMPUTE_WGSL, CULL_PARAMS_BYTE_SIZE, CULL_WORKGROUP_SIZE, CullParams, Culler, CullerError,
    DrawIndexedIndirectArgs, FRUSTUM_PLANE_COUNT, INDIRECT_ARGS_WORD_BYTES,
    INDIRECT_ARGS_WORD_COUNT, INSTANCE_COUNT_WORD, compact_visible_indices, cull_params,
    cull_workgroup_count, indirect_args_for_visible, reset_indirect_instance_count,
};
pub use ibl::{
    BrdfLut, CubeFace, EnvironmentCube, SplitSum, bake_brdf_lut, convolve_irradiance,
    convolve_specular, integrate_brdf,
};
pub use light::{DirectionalLight, StudioLights};
pub use mesh::{Mesh, SkinnedVertex, Vertex, create_box, create_sphere};
pub use mesh_upload::{
    ConvertedSoup, SoupCache, SoupHash, UploadCache, UploadError, to_vertices, upload_mesh_data,
};
pub use msaa_edge::{DeferredSample, SampleCoverage, shade_mask};
pub use ornis_core::{OPENPBR_MATERIAL_SIZE, OPENPBR_MATERIAL_VEC4_COUNT, OpenPBRMaterial};
/// Unified explicit-ordering edge error (Phase A, audit §4.2); the same type
/// `ornis_core` re-exports for systems.
pub use ornis_schedule::OrderError;
pub use render_backend::{
    RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};
pub use renderer::{
    AoBlurInputs, AoRawInputs, BlendMode, CameraUniform, CompositeInputs, CompositePass,
    CustomGbufferDraw, EdgeInputs, FogInputs, ForwardPass, GBufferTextures, GbufferTargets,
    InstanceData, LightingPass, MSAA_SAMPLE_COUNT, MaterialIdx, PerObjectGpu, Renderer3D,
    SINGLE_SAMPLE_COUNT, ShadingDebug, StagedCustomMesh, TaaInputs, TransparencyError,
    TransparencyOptions, custom_draw_items, forward_blend_state, negotiate_sample_count,
    normalize_sample_count, sort_by_depth, upload_vertex_rows,
};
pub use schedule_bridge::{ProjectionError, try_project_schedule};
pub use skinning::{
    GBUFFER_SKINNED_RESOURCES, JointPalette, PALETTE_BYTE_SIZE, PaletteHandle, SkinBindError,
    SkinJoint, SkinnedDraw, joint_palette_bytes, palette_upload_bytes, skinned_entry_point,
    wgsl_vertex_source_skinned,
};
pub use system::{
    Access, AccessSet, ClearBlack, ClearTransparent, ClearValue, ClearWhite, Frame, FramePass,
    FrameResource, Read, Resolver, ResourceKind, SystemSet, SystemViews, Write, WriteClear,
};
pub use textures::{
    CpuImage, GpuTexture, MaterialTextureSet, TextureCache, TextureHandle, TextureRole,
    TextureUploadError,
};
pub use transient_pool::{
    Budget, BudgetExceeded, FrameLayout, PassId, PassLayout, PassName, PoolSlot, ResourceId,
    ResourceLayout, ResourceName, SizePolicy, TextureSpec, TransientPool, format_bytes_per_pixel,
};
