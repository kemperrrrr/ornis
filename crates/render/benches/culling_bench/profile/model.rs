//! Shared model of the stage profile: the scenario x path matrix, the
//! per-frame stage row, and the cell state every measurement shares.

use std::time::Duration;

use glam::Mat4;
use ornis_core::Entity;
use ornis_render::{Culler, InstanceData, PerObjectGpu, Renderer3D};

use super::super::probe::Probe;

/// Criterion samples per cell (the group minimum).
pub(super) const PROFILE_SAMPLE_SIZE: usize = 10;
/// Frames averaged into each table row, after two discarded warm-ups.
pub(super) const PROFILE_FRAMES: usize = 10;
/// Discarded frames before the measurement: the first aligns the fresh
/// renderer's previous-frame history, the second runs in the aligned
/// state, so measured frames start from a steady state (static cells
/// then fill nothing).
pub(super) const PROFILE_WARM_FRAMES: usize = 2;
/// Distance between the movers of the 1% scenario (one in a hundred).
pub(super) const MOVE_STRIDE: usize = 100;
/// Per-frame X drift of every mover; the sign alternates with the step.
pub(super) const DRIFT_STEP: f32 = 0.05;
/// Milliseconds per second — the table's cells are milliseconds.
const MS_PER_SECOND: f64 = 1000.0;

/// The issue's three scene behaviours.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Scenario {
    /// "сцена стоит".
    Static,
    /// "двигается 1%".
    MoveOnePercent,
    /// "двигается всё".
    MoveAll,
}

impl Scenario {
    /// The markdown label of this scenario.
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::MoveOnePercent => "moves_1pct",
            Self::MoveAll => "moves_all",
        }
    }
}

/// The two staging paths of the issue.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchPath {
    /// The culler path: the frame converts its own records.
    Compact,
    /// `compact_records`, fed by the renderer's conversion and upload.
    Records,
}

impl BenchPath {
    /// The markdown label of this path.
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Records => "compact_records",
        }
    }
}

/// One frame's stage timings; zero plus a cleared flag where unmeasured.
#[derive(Default)]
pub(super) struct Stages {
    /// CPU build: record conversion (renderer on the records path).
    pub(super) build: Duration,
    /// CPU diff (records path: incl. the staging hand-off copy).
    pub(super) diff: Duration,
    /// CPU fill: the dirty-run `queue.write_buffer`.
    pub(super) fill: Duration,
    /// CPU `upload_per_objects` (zero on the compact path).
    pub(super) upload: Duration,
    /// CPU pre-write: uniform plus the indirect-args reset.
    pub(super) prewrite: Duration,
    /// GPU culling compute pass, from the pass timestamps.
    pub(super) compute: Duration,
    /// GPU indirect draw, from the pass timestamps.
    pub(super) draw: Duration,
    /// The whole frame, submit and timestamp readback included.
    pub(super) wall: Duration,
    /// False without a stamp readback: GPU cells print "—".
    pub(super) gpu_measured: bool,
    /// False on the compact path: there is no renderer upload.
    pub(super) upload_measured: bool,
    /// Frames whose stamp readback succeeded (average divisor for GPU).
    pub(super) gpu_frames: u32,
    /// Frames with a renderer upload (average divisor for upload).
    pub(super) upload_frames: u32,
}

impl Stages {
    /// Fold one measured frame into this accumulating row. GPU and upload
    /// stages carry their own measured-frame counts: a failed readback
    /// contributes nothing, and the average divides by the frames that
    /// actually measured — a mid-run failure can no longer slip zeros
    /// into the mean.
    pub(super) fn accumulate(&mut self, frame: &Self) {
        self.build += frame.build;
        self.diff += frame.diff;
        self.fill += frame.fill;
        self.upload += frame.upload;
        self.prewrite += frame.prewrite;
        self.compute += frame.compute;
        self.draw += frame.draw;
        self.wall += frame.wall;
        self.gpu_measured |= frame.gpu_measured;
        self.upload_measured |= frame.upload_measured;
        self.gpu_frames += u32::from(frame.gpu_measured);
        self.upload_frames += u32::from(frame.upload_measured);
    }

    // qual:allow(test_quality, untested) reason: "bench-only formatting helper, driven by the criterion harness, never under libtest"
    /// The row average: the accumulated total over `frames`; the GPU and
    /// upload stages over their own measured-frame counts (flag false
    /// when nothing measured, so the cell prints "—").
    pub(super) fn averaged(self, frames: usize) -> Self {
        let frames = frames.max(1) as u32;
        let gpu_frames = self.gpu_frames.max(1);
        let upload_frames = self.upload_frames.max(1);
        Self {
            build: self.build / frames,
            diff: self.diff / frames,
            fill: self.fill / frames,
            upload: self.upload / upload_frames,
            prewrite: self.prewrite / frames,
            compute: self.compute / gpu_frames,
            draw: self.draw / gpu_frames,
            wall: self.wall / frames,
            gpu_measured: self.gpu_frames > 0,
            upload_measured: self.upload_frames > 0,
            gpu_frames: self.gpu_frames,
            upload_frames: self.upload_frames,
        }
    }

    /// The markdown row: scenario, size, path, then every stage cell
    /// (milliseconds; "—" where the stage was not measured).
    pub(super) fn row(&self, scenario: &str, size: usize, path: &str) -> String {
        format!(
            "| {scenario} | {size} | {path} | {} | {} | {} | {} | {} | {} | {} | {} |",
            ms(self.build),
            ms(self.diff),
            ms(self.fill),
            cell(self.upload, self.upload_measured),
            ms(self.prewrite),
            cell(self.compute, self.gpu_measured),
            cell(self.draw, self.gpu_measured),
            ms(self.wall),
        )
    }
}

/// One measured stage cell: milliseconds with three decimals.
fn ms(duration: Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * MS_PER_SECOND)
}

/// One possibly-unmeasured cell: ms, or "—" where nothing was measured.
fn cell(duration: Duration, measured: bool) -> String {
    if measured {
        ms(duration)
    } else {
        String::from("—")
    }
}

/// The shared GPU objects of the profile: built once, every cell reuses
/// them — one culler sized to the largest profiled scene, one probe.
/// The renderer is deliberately NOT shared: its `previous_instance_models`
/// history would leak from cell to cell and corrupt the records path's
/// diff, so each [`Frame`] builds its own (see [`Frame::new`]).
pub(super) struct Setup<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) queue: &'a wgpu::Queue,
    pub(super) surface_config: &'a wgpu::SurfaceConfiguration,
    pub(super) culler: &'a Culler,
    pub(super) probe: &'a Probe,
    pub(super) view_proj: Mat4,
}

/// One cell's frame state: the scene motion drive, the cell's own
/// renderer (fresh per cell — its previous-frame history must not leak
/// into the next cell), and the two staging histories the diff needs —
/// the bench owns them on both paths, exactly the runs [`Culler`]
/// staging would own internally.
pub(super) struct Frame<'a> {
    pub(super) setup: &'a Setup<'a>,
    pub(super) renderer: Renderer3D,
    pub(super) instances: Vec<InstanceData>,
    pub(super) entities: Vec<Entity>,
    /// Renderer records: uploaded, then handed to the staging diff.
    pub(super) records: Vec<PerObjectGpu>,
    pub(super) prev: Vec<PerObjectGpu>,
    pub(super) next: Vec<PerObjectGpu>,
    pub(super) scenario: Scenario,
    pub(super) path: BenchPath,
    pub(super) step: usize,
}

impl<'a> Frame<'a> {
    /// A fresh cell: a private copy of the base scene, a fresh renderer
    /// (clean previous-frame history) and empty histories — the warm-up
    /// frame fills them before the first measurement, so no row pays for
    /// a cold pipeline, a full first diff, or a previous cell's motion.
    pub(super) fn new(
        setup: &'a Setup<'a>,
        base: &[InstanceData],
        entities: &[Entity],
        scenario: Scenario,
        path: BenchPath,
    ) -> Self {
        Self {
            setup,
            renderer: Renderer3D::new(setup.device, setup.surface_config, 1),
            instances: base.to_vec(),
            entities: entities.to_vec(),
            records: Vec::new(),
            prev: Vec::new(),
            next: Vec::new(),
            scenario,
            path,
            step: 0,
        }
    }
}
