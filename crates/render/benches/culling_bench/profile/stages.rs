//! Per-stage timers of one profiled frame: the record build for the
//! frame's path, the culler's dirty-run fill, and the uniform/args
//! pre-write. Split out of `profile.rs` so the harness loop (warm-up,
//! sampling, criterion registration) and the stage taps stay separate.
use std::time::{Duration, Instant};

use ornis_core::ComponentChanges;
use ornis_render::gpu_culling::profile;

use super::INDEX_COUNT;
use super::model::{BenchPath, Frame};

/// The CPU half of the frame's staging: the record build for this path.
/// `moved` is the mover list (the change feed's payload) and `first_frame`
/// marks the frame that has no tracking history yet, so the feed arrives
/// exhaustive exactly like the first production frame.
// qual:allow(test_quality, untested) reason: "bench-only stage tap, driven by the criterion harness, never under libtest"
pub(super) fn build_records(frame: &mut Frame, moved: &[u32], first_frame: bool) -> Duration {
    let clock = Instant::now();
    match frame.path {
        BenchPath::Compact => {
            frame.next.clear();
            profile::convert(&mut frame.next, &frame.instances);
        }
        BenchPath::Records => {
            frame.records = frame.renderer.profile_stage_records(
                frame.setup.device,
                &frame.instances,
                &[],
                &frame.entities,
            );
        }
        BenchPath::Changed => {
            let changes = ComponentChanges {
                ids: moved
                    .iter()
                    .map(|&index| frame.entities[index as usize])
                    .collect(),
                exhaustive: first_frame,
            };
            let (records, dirty) = frame.renderer.profile_stage_records_changed(
                frame.setup.device,
                &frame.instances,
                &[],
                &frame.entities,
                &changes,
            );
            frame.records = records;
            frame.dirty = dirty;
        }
    }
    clock.elapsed()
}

/// The culler's source-buffer write: the dirty runs of this frame, as
/// `(start, count)` pairs. Returns the timer (which includes the full-frame
/// diff on the two pre-#185 paths) and the staged instance count.
// qual:allow(test_quality, untested) reason: "bench-only stage tap, driven by the criterion harness, never under libtest"
pub(super) fn fill_culling(frame: &mut Frame) -> (Duration, usize) {
    let clock = Instant::now();
    let ranges = match frame.path {
        BenchPath::Compact => profile::diff(&frame.prev, &frame.next),
        BenchPath::Records => {
            frame.next.clear();
            frame.next.extend_from_slice(&frame.records);
            profile::diff(&frame.prev, &frame.next)
        }
        BenchPath::Changed => profile::runs_from_indices(&frame.dirty),
    };
    let staged = frame.staged_records().len();
    frame
        .setup
        .culler
        .profile_write_dirty(frame.setup.queue, frame.staged_records(), &ranges);
    let elapsed = clock.elapsed();
    if frame.path != BenchPath::Changed {
        std::mem::swap(&mut frame.prev, &mut frame.next);
    }
    (elapsed, staged)
}

/// The uniform/args pre-write of the culler for this frame's count.
// qual:allow(test_quality, untested) reason: "bench-only stage tap, driven by the criterion harness, never under libtest"
pub(super) fn prewrite(frame: &mut Frame, staged: usize) -> Duration {
    let clock = Instant::now();
    frame.setup.culler.profile_prewrite(
        frame.setup.queue,
        &frame.setup.view_proj,
        INDEX_COUNT,
        staged as u32,
    );
    clock.elapsed()
}
