//! Opt-in debug statistics (`--stats` / `ORNIS_STATS=1`): a second OS
//! window beside the render window showing FPS, frame time (average,
//! recent worst, history graph) and per-stage / per-system time drained
//! from the schedule timers (`ornis_core::Engine::take_stage_timings`).
//!
//! Without the flag nothing here is constructed: no window, no clock
//! samples in the schedules, the frame loop is unchanged.

mod font;
mod panel;
mod window;

use std::time::{Duration, Instant};

use ornis_core::{ScheduleTimings, Stage};
use ornis_schedule::FrameHistory;

pub(crate) use window::StatsWindow;

/// Environment switch equivalent to `--stats`.
pub const STATS_ENV: &str = "ORNIS_STATS";
/// Command-line switch opening the stats window.
pub const STATS_ARG: &str = "--stats";
/// Frames kept for the average and the history graph.
const HISTORY_FRAMES: usize = 240;
/// Wall-time window of the "worst" frame readout.
const WORST_WINDOW: Duration = Duration::from_secs(2);
/// Stats window refresh period (the panel is redrawn at most this often).
const REFRESH: Duration = Duration::from_millis(100);
/// Rows of the per-system table (heaviest first).
const TOP_SYSTEMS: usize = 10;

/// Whether stats were requested: `--stats` among `args`, or `env` (the
/// value of [`STATS_ENV`]) set to `1`/`true`/`yes`/`on` (case-insensitive).
pub fn stats_requested<S: AsRef<str>>(args: &[S], env: Option<&str>) -> bool {
    let by_arg = args.iter().any(|arg| arg.as_ref() == STATS_ARG);
    let by_env = env.is_some_and(|value| {
        ["1", "true", "yes", "on"]
            .iter()
            .any(|on| value.trim().eq_ignore_ascii_case(on))
    });
    by_arg || by_env
}

/// One line of the timing table: a stage or a system with its mean and
/// worst per-frame time over the last refresh period.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TimingRow {
    /// `stage` or `stage/system`.
    pub label: String,
    /// Mean per-frame time over the period.
    pub avg: Duration,
    /// Worst single-frame time over the period.
    pub max: Duration,
}

/// Per-label running sums between two refreshes.
#[derive(Debug, Default)]
struct Accum {
    label: String,
    sum: Duration,
    max: Duration,
}

/// Frame-time and schedule-timing aggregation behind the stats window.
#[derive(Debug)]
pub(crate) struct StatsCollector {
    /// Wall interval between presented frames.
    pub frames: FrameHistory,
    /// CPU time of the runner's frame call (schedules + extraction).
    pub work: FrameHistory,
    /// Stage rows first, then the heaviest systems.
    pub rows: Vec<TimingRow>,
    pending: Vec<Accum>,
    pending_frames: u32,
    last_refresh: Option<Instant>,
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self {
            frames: FrameHistory::new(HISTORY_FRAMES),
            work: FrameHistory::new(HISTORY_FRAMES),
            rows: Vec::new(),
            pending: Vec::new(),
            pending_frames: 0,
            last_refresh: None,
        }
    }
}

impl StatsCollector {
    /// Records one frame: its wall interval, the runner work time and the
    /// stage timings drained from the engine for that frame.
    pub fn record(
        &mut self,
        interval: Duration,
        work: Duration,
        stages: &[(Stage, ScheduleTimings)],
    ) {
        self.frames.push(interval);
        self.work.push(work);
        self.pending_frames += 1;
        for (stage, timings) in stages {
            self.add(stage.name().to_owned(), timings.total);
            for (system, elapsed) in &timings.systems {
                self.add(format!("{}/{system}", stage.name()), *elapsed);
            }
        }
    }

    fn add(&mut self, label: String, elapsed: Duration) {
        let slot = match self.pending.iter().position(|a| a.label == label) {
            Some(i) => &mut self.pending[i],
            None => {
                self.pending.push(Accum {
                    label,
                    ..Accum::default()
                });
                let last = self.pending.len() - 1;
                &mut self.pending[last]
            }
        };
        slot.sum += elapsed;
        slot.max = slot.max.max(elapsed);
    }

    /// Folds the pending sums into [`Self::rows`] when a refresh is due
    /// at `now`; returns whether the panel should be redrawn.
    pub fn refresh(&mut self, now: Instant) -> bool {
        if self
            .last_refresh
            .is_some_and(|at| now.duration_since(at) < REFRESH)
        {
            return false;
        }
        self.last_refresh = Some(now);
        let frames = self.pending_frames.max(1);
        let (mut stages, mut systems): (Vec<TimingRow>, Vec<TimingRow>) = self
            .pending
            .drain(..)
            .map(|a| TimingRow {
                label: a.label,
                avg: a.sum / frames,
                max: a.max,
            })
            .partition(|row| !row.label.contains('/'));
        systems.sort_by_key(|row| std::cmp::Reverse(row.avg));
        systems.truncate(TOP_SYSTEMS);
        stages.append(&mut systems);
        self.rows = stages;
        self.pending_frames = 0;
        true
    }

    /// Worst frame interval over the recent [`WORST_WINDOW`].
    pub fn worst_frame(&self) -> Duration {
        self.frames.worst_within(WORST_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn flag_parsing_accepts_arg_and_env() {
        assert!(stats_requested(&["anim", "--stats"], None));
        assert!(!stats_requested(&["anim", "--frames", "5"], None));
        assert!(!stats_requested(&["--statsx"], None));
        for on in ["1", "true", "YES", " on "] {
            assert!(stats_requested::<&str>(&[], Some(on)), "{on}");
        }
        for off in ["0", "", "false", "no"] {
            assert!(!stats_requested::<&str>(&[], Some(off)), "{off}");
        }
    }

    #[test]
    fn collector_averages_stages_and_ranks_systems() {
        let mut c = StatsCollector::default();
        let stage = |total, a, b| {
            vec![(
                Stage::PostFrame,
                ScheduleTimings {
                    total,
                    systems: vec![("light", a), ("heavy", b)],
                },
            )]
        };
        let t0 = Instant::now();
        assert!(c.refresh(t0), "first refresh is immediate");
        c.record(ms(16), ms(10), &stage(ms(8), ms(1), ms(6)));
        c.record(ms(48), ms(30), &stage(ms(12), ms(3), ms(10)));
        assert!(!c.refresh(t0), "throttled");
        assert!(c.refresh(t0 + REFRESH));
        let labels: Vec<_> = c.rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(
            labels,
            ["post_frame", "post_frame/heavy", "post_frame/light"]
        );
        assert_eq!(c.rows[0].avg, ms(10));
        assert_eq!(c.rows[0].max, ms(12));
        assert_eq!(c.rows[1].avg, ms(8));
        assert_eq!(c.frames.average(), ms(32));
        assert_eq!(c.worst_frame(), ms(48));
        assert_eq!(c.work.average(), ms(20));
    }
}
