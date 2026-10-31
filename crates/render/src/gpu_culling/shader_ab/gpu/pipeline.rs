//! Pipelines for the production cull shader and the flag ABI shared by the
//! lowered kernel and the hand twin.

use super::super::super::CULL_COMPUTE_WGSL;
use super::super::super::CULL_WORKGROUP_SIZE;

/// Binding 0: params uniform (production) or plane uniform (flag ABI).
pub(super) const BIND_PARAMS: u32 = 0;
/// Binding 1: source objects.
pub(super) const BIND_SRC: u32 = 1;
/// Binding 2: compacted records (production) or visibility flags.
pub(super) const BIND_DST: u32 = 2;
/// Binding 3: indirect args (production) or the guest count word.
pub(super) const BIND_TAIL: u32 = 3;

pub(super) const FLAG_ENTRY: &str = "cs_main";
pub(super) const PROD_ENTRY: &str = "cull_instances";

/// Hand-written flag kernel with the same ABI and workgroup as the lowered
/// `cull` guest. The e2e twin in `wasm2gpu-shader` is the same program; this
/// copy lives here so the bench crate does not reach into that test.
pub(super) const TWIN_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> planes: array<vec4<f32>, 24>;
@group(0) @binding(1) var<storage, read_write> spheres: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> sphere_count: array<u32, 4>;

@compute @workgroup_size(4, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id < sphere_count[0]) {
        let center = spheres[id];
        var visible = 1u;
        var plane = 0u;
        while (plane < 6u) {
            let at = plane * 4u;
            let px = planes[at].x;
            let py = planes[at + 1u].x;
            let pz = planes[at + 2u].x;
            let pw = planes[at + 3u].x;
            if (px * center.x + py * center.y + pz * center.z + pw + center.w < 0.0) {
                visible = 0u;
            }
            plane += 1u;
        }
        flags[id] = visible;
    }
}
"#;

pub(super) struct Pipe {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) layout: wgpu::BindGroupLayout,
}

pub(super) fn flag_layout_of(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    flag_layout(device)
}

pub(super) fn compile_with(
    device: &wgpu::Device,
    wgsl: &str,
    entry: &str,
    layout: wgpu::BindGroupLayout,
) -> Pipe {
    let pipeline = compile(device, wgsl, entry, &layout);
    Pipe { pipeline, layout }
}

pub(super) fn build(device: &wgpu::Device, wgsl: &str, entry: &str, production: bool) -> Pipe {
    let layout = if production {
        prod_layout(device)
    } else {
        flag_layout(device)
    };
    let pipeline = compile(device, wgsl, entry, &layout);
    Pipe { pipeline, layout }
}

pub(super) fn prod() -> &'static str {
    CULL_COMPUTE_WGSL
}

pub(super) fn groups(count: usize, wgsl: &str) -> u32 {
    (count as u32).div_ceil(workgroup_x(wgsl).max(1))
}

fn workgroup_x(wgsl: &str) -> u32 {
    parsed_workgroup(wgsl).unwrap_or(CULL_WORKGROUP_SIZE)
}

fn parsed_workgroup(wgsl: &str) -> Option<u32> {
    let rest = wgsl.split("@workgroup_size(").nth(1)?;
    let token = rest.split([',', ')']).next()?.trim();
    token.trim_end_matches('u').parse().ok()
}

fn compile(
    device: &wgpu::Device,
    wgsl: &str,
    entry: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn prod_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let readable = wgpu::BufferBindingType::Storage { read_only: true };
    let writable = wgpu::BufferBindingType::Storage { read_only: false };
    layout(device, "prod", readable, writable)
}

fn flag_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    layout(device, "flag", storage, storage)
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    src: wgpu::BufferBindingType,
    rest: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            entry(BIND_PARAMS, wgpu::BufferBindingType::Uniform),
            entry(BIND_SRC, src),
            entry(BIND_DST, rest),
            entry(BIND_TAIL, rest),
        ],
    })
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
