//! Upload both ABIs and dispatch one compute pass.
//!
//! The production kernel compacts `PerObject` records and bumps an atomic
//! counter, so that counter is zeroed on every dispatch. The flag kernels
//! only store a `u32` per slot. Timestamps bracket the compute pass; the
//! counter reset is queued into the same submit and shows up in wall time.

use std::time::Instant;

use super::super::super::INSTANCE_COUNT_WORD;
use super::super::scene::OBJECT_BYTES;
use super::pipeline::Pipe;
use super::session::QUERY_BEGIN;
use super::session::QUERY_END;
use super::session::QUERY_SLOTS;
use super::session::Session;

/// One recorded dispatch.
pub(super) struct Sample {
    /// Pass-boundary timestamp delta in nanoseconds, when the device has them.
    pub(super) gpu_ns: Option<f64>,
    /// Host time from the optional counter reset through the blocking poll.
    pub(super) wall_ns: f64,
}

/// Bytes of one resolved timestamp.
const STAMP_BYTES: usize = std::mem::size_of::<u64>();
/// Bytes of the whole resolve buffer.
const STAMP_SPAN: u64 = QUERY_SLOTS as u64 * STAMP_BYTES as u64;
/// Bytes of one `u32` lane.
const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Production buffers. The bind group is first so it drops before the
/// buffers it borrows.
pub(super) struct ProdSet {
    pub(super) bind: wgpu::BindGroup,
    pub(super) args: wgpu::Buffer,
    pub(super) dst: wgpu::Buffer,
    pub(super) _params: wgpu::Buffer,
    pub(super) _src: wgpu::Buffer,
}

/// Flag-ABI buffers, shared by the lowered kernel and the hand twin.
pub(super) struct FlagSet {
    pub(super) bind: wgpu::BindGroup,
    pub(super) flags: wgpu::Buffer,
    pub(super) _planes: wgpu::Buffer,
    pub(super) _spheres: wgpu::Buffer,
    pub(super) _count: wgpu::Buffer,
}

pub(super) fn submit(
    session: &Session,
    pipe: &Pipe,
    bind: &wgpu::BindGroup,
    groups: u32,
    clear: Option<&wgpu::Buffer>,
) -> Sample {
    let started = Instant::now();
    clear_args(session, clear);
    encode_and_submit(session, pipe, bind, groups);
    Sample {
        wall_ns: elapsed_ns(started),
        gpu_ns: gpu_ns(session),
    }
}

pub(super) fn prod_bind(set: &ProdSet) -> &wgpu::BindGroup {
    &set.bind
}

pub(super) fn prod_args(set: &ProdSet) -> &wgpu::Buffer {
    &set.args
}

pub(super) fn flag_bind(set: &FlagSet) -> &wgpu::BindGroup {
    &set.bind
}

pub(super) fn object_span(count: usize) -> u64 {
    (count * OBJECT_BYTES) as u64
}

pub(super) fn flag_span(count: usize) -> u64 {
    (count * U32_BYTES) as u64
}

pub(super) fn instance_at() -> usize {
    INSTANCE_COUNT_WORD * U32_BYTES
}

fn clear_args(session: &Session, clear: Option<&wgpu::Buffer>) {
    if let Some(args) = clear {
        let zero = 0u32.to_le_bytes();
        session
            .queue
            .write_buffer(args, instance_at() as u64, &zero);
    }
}

fn encode_and_submit(session: &Session, pipe: &Pipe, bind: &wgpu::BindGroup, groups: u32) {
    let mut encoder = session
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull ab"),
        });
    record_pass(&mut encoder, session, &pipe.pipeline, bind, groups);
    resolve_stamps(&mut encoder, session);
    session.queue.submit([encoder.finish()]);
    let _ = session.device.poll(wgpu::PollType::wait_indefinitely());
}

fn record_pass(
    encoder: &mut wgpu::CommandEncoder,
    session: &Session,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("cull ab"),
        timestamp_writes: stamp_writes(session),
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

fn stamp_writes(session: &Session) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    session
        .queries
        .as_ref()
        .map(|queries| wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(QUERY_BEGIN),
            end_of_pass_write_index: Some(QUERY_END),
        })
}

fn resolve_stamps(encoder: &mut wgpu::CommandEncoder, session: &Session) {
    if let Some(queries) = &session.queries {
        encoder.resolve_query_set(queries, 0..QUERY_SLOTS, &session.resolve, 0);
    }
}

fn elapsed_ns(started: Instant) -> f64 {
    started.elapsed().as_nanos() as f64
}

fn gpu_ns(session: &Session) -> Option<f64> {
    session.queries.as_ref()?;
    let bytes = super::readback::read_exact(session, &session.resolve, STAMP_SPAN);
    let begin = super::readback::load_u64(&bytes, 0);
    let end = super::readback::load_u64(&bytes, STAMP_BYTES);
    Some(scale(end.saturating_sub(begin), session.period_ns))
}

fn scale(ticks: u64, period_ns: f32) -> f64 {
    ticks as f64 * f64::from(period_ns)
}
