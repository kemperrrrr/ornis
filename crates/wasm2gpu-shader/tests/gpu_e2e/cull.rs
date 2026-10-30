//! Dispatch the lowered `cull` kernel and A/B it against a hand twin of
//! the production `sphere_visible` test.
//!
//! The kernel flags eight sphere slots — six guarded by `count` — against
//! six planes: a clear pass, a clear cut, a `dot + r == 0` boundary, a
//! NaN radius that fails open like the production WGSL, a cut by a second
//! plane, and two sentinel slots past `count`. Both shaders run on the
//! same bytes: the outputs must match [`EXPECTED`], [`derived_flags`]
//! re-derives that constant from the production expression
//! (`render/src/gpu_culling.rs`), and the line this test prints shows
//! both GPU times from the CI log.

use std::time::Duration;
use std::time::Instant;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;
use ornis_wasm2gpu_shader::COUNT_BINDING;
use ornis_wasm2gpu_shader::FLAG_BINDING;
use ornis_wasm2gpu_shader::PLANE_BINDING;
use ornis_wasm2gpu_shader::PLANE_COUNT;
use ornis_wasm2gpu_shader::PLANE_LEN;
use ornis_wasm2gpu_shader::SPHERE_BINDING;

const LANES: usize = symbols::CS_WORKGROUP[0] as usize;
/// Workgroups the test dispatches: ids run `0..LANES * WORKGROUPS`.
const WORKGROUPS: u32 = 2;
/// An untouched slot must keep this value.
const SENTINEL: u32 = 7;
const RUNS: u32 = 64;

/// Sphere centers and radii: four lanes each, packed storage order.
/// Slots 6 and 7 sit past [`ACTIVE_COUNT`] and are never read.
const SPHERES: [[f32; 4]; 8] = [
    // A clear pass against plane 0 (+x).
    [0.5, 0.0, 0.0, 0.25],
    // A clear cut by plane 0.
    [-1.0, 0.0, 0.0, 0.25],
    // A clear pass.
    [2.0, 0.0, 0.0, 0.1],
    // Boundary: `dot + w + radius == 0.0`, and the production
    // `dot + w < -radius` is `false` on the same value — visible.
    [-0.25, 0.0, 0.0, 0.25],
    // NaN radius: IEEE `<` fails open in the kernel and in the WGSL twin.
    [0.0, 0.0, 0.0, f32::NAN],
    // A clear cut by plane 3 (+y), not by plane 0.
    [0.0, -0.5, 0.0, 0.25],
    [9.0, 9.0, 9.0, 1.0],
    [9.0, 9.0, 9.0, 1.0],
];

/// Slots that do work: `id < count`; the rest must keep [`SENTINEL`].
const ACTIVE_COUNT: usize = 6;

/// Flag slots one dispatch covers: every sphere lane.
const SLOTS: usize = SPHERES.len();

/// Visible, culled, visible, boundary pass, NaN pass, cut by plane 3,
/// then two untouched slots past [`ACTIVE_COUNT`].
const EXPECTED: [u32; SPHERES.len()] = [1, 0, 1, 1, 1, 0, SENTINEL, SENTINEL];

/// The six clip planes in uniform slots: plane 0 is `+x`, plane 3 `+y`,
/// the rest pass every sphere the test feeds.
const PLANES: [[f32; 4]; PLANE_COUNT as usize] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// The hand twin runs the same guard, the same `dot + w + radius < 0.0`
/// order, and the same workgroup, so bitwise agreement is the contract.
const TWIN_WGSL: &str = r#"
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

#[test]
fn cull_kernel_matches_the_hand_twin() {
    assert_eq!(LANES * WORKGROUPS as usize, SPHERES.len());
    let wasm = super::compile_wasm();
    let wat = super::compute_lift::lift_wat(&wasm, "cull");
    let module = wat::parse_str(&wat).unwrap_or_else(|err| panic!("cull wat: {err}\n{wat}"));
    let lowered = match translate(&module) {
        Ok(lowered) => lowered,
        Err(err) => panic!("cull translate {err:?}\nWAT:\n{wat}"),
    };
    let wgsl = &lowered.wgsl;
    assert!(wgsl.contains("@compute"), "{wgsl}");
    assert!(wgsl.contains("@builtin(global_invocation_id)"), "{wgsl}");
    assert!(
        wgsl.contains(&format!("storage_0_{FLAG_BINDING}_")),
        "{wgsl}"
    );
    assert!(!lowered.compute_spirv.is_empty());
    let derived = derived_flags();
    assert_eq!(EXPECTED, derived, "production-formula cross-check");
    match super::device() {
        Some((device, queue)) => {
            let (flags, wasm_time) = run(&device, &queue, wgsl);
            assert_eq!(flags, EXPECTED, "wasm2gpu kernel output");
            let (twin, twin_time) = run(&device, &queue, TWIN_WGSL);
            assert_eq!(twin, EXPECTED, "hand twin output");
            assert_eq!(flags, twin, "bitwise A/B");
            eprintln!("cull {flags:?}: wasm {wasm_time:?} twin {twin_time:?} x{RUNS}");
        }
        None if std::env::var_os("ORNIS_REQUIRE_GPU").is_some() => {
            panic!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        }
        None => eprintln!("cull_kernel_matches_the_hand_twin: no wgpu adapter — skipped"),
    }
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str) -> ([u32; SLOTS], Duration) {
    let bgl = bind_layout(device);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wasm2gpu cull"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wasm2gpu cull"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wasm2gpu cull"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });
    let (flags, bind) = upload(device, queue, &bgl);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wasm2gpu cull"),
    });
    let start = Instant::now();
    for _ in 0..RUNS {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wasm2gpu cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(WORKGROUPS, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let time = start.elapsed();
    (map_u32s(device, queue, &flags), time)
}

fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wasm2gpu cull"),
        entries: &[
            buffer_entry(PLANE_BINDING, wgpu::BufferBindingType::Uniform),
            buffer_entry(SPHERE_BINDING, storage),
            buffer_entry(FLAG_BINDING, storage),
            buffer_entry(COUNT_BINDING, storage),
        ],
    })
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

/// Upload the inputs and return the flags buffer: the bind group keeps
/// the other buffers alive for as long as it lives.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let planes = super::gpu_buffer(device, "cull-planes", 384, uniform);
    let spheres = super::storage_buffer(device, "cull-spheres", 128);
    let flags = super::storage_buffer(device, "cull-flags", 32);
    let count = super::storage_buffer(device, "cull-count", 16);
    queue.write_buffer(&planes, 0, &pack_planes());
    queue.write_buffer(&spheres, 0, &pack_spheres());
    queue.write_buffer(&flags, 0, &pack_flags());
    queue.write_buffer(&count, 0, &pack_count());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wasm2gpu cull"),
        layout,
        entries: &[
            super::entry(PLANE_BINDING, &planes),
            super::entry(SPHERE_BINDING, &spheres),
            super::entry(FLAG_BINDING, &flags),
            super::entry(COUNT_BINDING, &count),
        ],
    });
    (flags, bind)
}

/// Plane components in uniform slots: one `f32` per 16-byte stride.
/// Plane 0 points along `+x`, plane 3 along `+y`; the rest pass everything.
fn pack_planes() -> Vec<u8> {
    let mut bytes = vec![0u8; (PLANE_LEN * 16) as usize];
    let mut plane = 0u32;
    while plane < PLANE_COUNT {
        let mut lane = 0u32;
        while lane < 4 {
            super::put_f32(
                &mut bytes,
                (plane * 4 + lane) * 16,
                PLANES[plane as usize][lane as usize],
            );
            lane += 1;
        }
        plane += 1;
    }
    bytes
}

fn pack_spheres() -> Vec<u8> {
    let mut bytes = vec![0u8; SPHERES.len() * 16];
    let mut sphere = 0usize;
    while sphere < SPHERES.len() {
        let mut lane = 0usize;
        while lane < 4 {
            let at = (sphere * 4 + lane) as u32 * 4;
            super::put_f32(&mut bytes, at, SPHERES[sphere][lane]);
            lane += 1;
        }
        sphere += 1;
    }
    bytes
}

/// Every flag starts at [`SENTINEL`]: an untouched slot keeps it, which
/// is how the test sees that the count guard wrote nothing.
fn pack_flags() -> [u8; 32] {
    let mut bytes = [0u8; 32];
    let mut slot = 0usize;
    while slot < SLOTS {
        super::put_u32(&mut bytes, (slot as u32) * 4, SENTINEL);
        slot += 1;
    }
    bytes
}

fn pack_count() -> Vec<u8> {
    let mut bytes = vec![0u8; 16];
    super::put_u32(&mut bytes, 0, ACTIVE_COUNT as u32);
    bytes
}

/// The production `sphere_visible` from `render/src/gpu_culling.rs`,
/// expression for expression: `dot(plane.xyz, center) + plane.w < -radius`
/// with the same early exit. The non-finite guards are CPU-only, so a NaN
/// radius fails open here — the semantics [`EXPECTED`] pins.
fn production_visible(center: [f32; 4]) -> bool {
    let radius = center[3];
    let mut plane = 0usize;
    while plane < PLANES.len() {
        let at = PLANES[plane];
        let dot = at[0] * center[0] + at[1] * center[1] + at[2] * center[2];
        if dot + at[3] < -radius {
            return false;
        }
        plane += 1;
    }
    true
}

/// Flags the production expression writes for each slot: visible up to
/// [`ACTIVE_COUNT`], then the sentinel the count guard must not touch.
fn derived_flags() -> [u32; SLOTS] {
    let mut flags = [SENTINEL; SLOTS];
    let mut slot = 0usize;
    while slot < ACTIVE_COUNT {
        flags[slot] = u32::from(production_visible(SPHERES[slot]));
        slot += 1;
    }
    flags
}

fn map_u32s(device: &wgpu::Device, queue: &wgpu::Queue, storage: &wgpu::Buffer) -> [u32; SLOTS] {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cull-flags-read"),
        size: 32,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cull-flags-read"),
    });
    encoder.copy_buffer_to_buffer(storage, 0, &read, 0, 32);
    queue.submit([encoder.finish()]);
    let bytes = super::map_bytes(device, &read);
    let mut values = [0u32; SLOTS];
    for (slot, word) in bytes.chunks_exact(4).take(SLOTS).enumerate() {
        values[slot] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    values
}
