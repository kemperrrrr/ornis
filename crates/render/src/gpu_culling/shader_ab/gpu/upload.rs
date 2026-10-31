//! Host uploads for the production record ABI and the flag ABI.
//!
//! Buffers are created once per scene, outside the timed loop. The
//! production indirect-args buffer starts uninitialized; every dispatch
//! zeroes the instance-count word before the pass.

use super::super::super::INDIRECT_ARGS_WORD_BYTES;
use super::super::super::INDIRECT_ARGS_WORD_COUNT;
use super::super::scene::Scene;
use super::super::scene::pack_flag_count;
use super::super::scene::pack_flag_planes;
use super::super::scene::pack_flag_spheres;
use super::super::scene::pack_objects;
use super::super::scene::pack_params;
use super::dispatch::FlagSet;
use super::dispatch::ProdSet;
use super::dispatch::flag_span;
use super::dispatch::object_span;
use super::pipeline::BIND_DST;
use super::pipeline::BIND_PARAMS;
use super::pipeline::BIND_SRC;
use super::pipeline::BIND_TAIL;
use super::pipeline::Pipe;
use super::session::Session;

/// Bytes of the production indirect-args buffer.
const ARGS_BYTES: u64 = INDIRECT_ARGS_WORD_COUNT as u64 * INDIRECT_ARGS_WORD_BYTES;

pub(super) fn upload_prod(session: &Session, scene: &Scene, pipe: &Pipe) -> ProdSet {
    let params = upload(
        session,
        "cull ab params",
        &pack_params(scene),
        uniform_usage(),
    );
    let src = upload(
        session,
        "cull ab src",
        &pack_objects(&scene.spheres),
        storage_src(),
    );
    let dst = make(
        session,
        "cull ab dst",
        object_span(scene.spheres.len()),
        storage_both(),
    );
    let args = make(session, "cull ab args", ARGS_BYTES, storage_both());
    let bind = bind4(
        session,
        "production cull",
        &pipe.layout,
        [&params, &src, &dst, &args],
    );
    ProdSet {
        bind,
        args,
        dst,
        _params: params,
        _src: src,
    }
}

pub(super) fn upload_flag(session: &Session, scene: &Scene, pipe: &Pipe) -> FlagSet {
    let planes = upload(
        session,
        "cull ab planes",
        &pack_flag_planes(&scene.planes),
        uniform_usage(),
    );
    let spheres = upload(
        session,
        "cull ab spheres",
        &pack_flag_spheres(&scene.spheres),
        storage_src(),
    );
    let flags = make(
        session,
        "cull ab flags",
        flag_span(scene.spheres.len()),
        storage_both(),
    );
    let count = upload(
        session,
        "cull ab count",
        &pack_flag_count(scene.spheres.len() as u32),
        storage_src(),
    );
    let bind = bind4(
        session,
        "flag cull",
        &pipe.layout,
        [&planes, &spheres, &flags, &count],
    );
    FlagSet {
        bind,
        flags,
        _planes: planes,
        _spheres: spheres,
        _count: count,
    }
}

fn upload(
    session: &Session,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = make(session, label, contents.len() as u64, usage);
    session.queue.write_buffer(&buffer, 0, contents);
    buffer
}

fn make(session: &Session, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    session.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn uniform_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
}

fn storage_src() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
}

fn storage_both() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

fn bind4(
    session: &Session,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 4],
) -> wgpu::BindGroup {
    session
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                resource(BIND_PARAMS, buffers[0]),
                resource(BIND_SRC, buffers[1]),
                resource(BIND_DST, buffers[2]),
                resource(BIND_TAIL, buffers[3]),
            ],
        })
}

fn resource(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }),
    }
}
