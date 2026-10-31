//! Read compacted material slots and visibility flags back to the host.
//!
//! Production order is the atomic slot order, so the ids are sorted before
//! the set comparison. Flag kernels write one `u32` per source slot.

use super::super::super::INDIRECT_ARGS_WORD_BYTES;
use super::super::super::INDIRECT_ARGS_WORD_COUNT;
use super::super::scene::MATERIAL_AT;
use super::super::scene::OBJECT_BYTES;
use super::dispatch::FlagSet;
use super::dispatch::ProdSet;
use super::dispatch::flag_span;
use super::dispatch::instance_at;
use super::dispatch::object_span;
use super::session::Session;

/// Bytes of one `u32` lane.
const U32_BYTES: usize = std::mem::size_of::<u32>();
/// Bytes of one resolved timestamp.
const STAMP_BYTES: usize = std::mem::size_of::<u64>();
/// Bytes of the production indirect-args buffer.
const ARGS_BYTES: u64 = INDIRECT_ARGS_WORD_COUNT as u64 * INDIRECT_ARGS_WORD_BYTES;
/// Flag value the guest writes for a visible sphere.
const FLAG_VISIBLE: u32 = 1;

pub(super) fn prod_ids(session: &Session, set: &ProdSet, count: usize) -> Vec<u32> {
    let take = written_count(session, set).min(count);
    if take == 0 {
        return Vec::new();
    }
    let bytes = read_exact(session, &set.dst, object_span(take));
    sorted_slots(&bytes, OBJECT_BYTES, MATERIAL_AT)
}

pub(super) fn flag_ids(session: &Session, set: &FlagSet, count: usize) -> Vec<u32> {
    let bytes = read_exact(session, &set.flags, flag_span(count));
    visible_flags(&bytes)
}

pub(super) fn read_exact(session: &Session, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    match crate::read_bytes_back(&session.device, &session.queue, src, bytes) {
        Some(data) => data,
        None => read_failed(),
    }
}

pub(super) fn load_u64(bytes: &[u8], at: usize) -> u64 {
    let mut word = [0u8; STAMP_BYTES];
    copy_lane(bytes, at, &mut word);
    u64::from_le_bytes(word)
}

fn written_count(session: &Session, set: &ProdSet) -> usize {
    let args = read_exact(session, &set.args, ARGS_BYTES);
    load_u32(&args, instance_at()) as usize
}

fn sorted_slots(bytes: &[u8], stride: usize, at: usize) -> Vec<u32> {
    let mut ids = slots(bytes, stride, at);
    ids.sort_unstable();
    ids
}

fn slots(bytes: &[u8], stride: usize, at: usize) -> Vec<u32> {
    bytes
        .chunks_exact(stride)
        .map(|chunk| load_u32(chunk, at))
        .collect()
}

fn visible_flags(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(U32_BYTES)
        .enumerate()
        .filter(|(_, word)| word_is_visible(word))
        .map(|(index, _)| index as u32)
        .collect()
}

fn word_is_visible(word: &[u8]) -> bool {
    load_u32(word, 0) == FLAG_VISIBLE
}

fn load_u32(bytes: &[u8], at: usize) -> u32 {
    let mut word = [0u8; U32_BYTES];
    copy_lane(bytes, at, &mut word);
    u32::from_le_bytes(word)
}

fn copy_lane(bytes: &[u8], at: usize, word: &mut [u8]) {
    let end = at + word.len();
    if end > bytes.len() {
        short_read();
    }
    word.copy_from_slice(&bytes[at..end]);
}

fn read_failed() -> ! {
    eprintln!("wasm2gpu cull A/B: buffer readback failed");
    std::process::exit(1)
}

fn short_read() -> ! {
    eprintln!("wasm2gpu cull A/B: buffer readback was shorter than the record");
    std::process::exit(1)
}
