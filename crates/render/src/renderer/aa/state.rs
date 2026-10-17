//! The anti-aliasing state of the renderer: the active mode, the jitter
//! sequence of the temporal mode, the camera matrices of the last two
//! frames and the validity of the history buffer.
//!
//! Split from the pass module so each file stays within the production-line
//! gate.

use super::super::{read_lock, write_lock};
use crate::antialias::{AntiAliasMode, JitterSequence};
use glam::Mat4;

/// The camera matrices of the last two frames, as the temporal resolve
/// needs them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AaCameraFrames {
    /// View-projection of the previous frame (jittered).
    pub(crate) previous_view_proj: Mat4,
    /// View-projection of the current frame (jittered).
    pub(crate) current_view_proj: Mat4,
}

impl Default for AaCameraFrames {
    fn default() -> Self {
        Self {
            previous_view_proj: Mat4::IDENTITY,
            current_view_proj: Mat4::IDENTITY,
        }
    }
}

/// The anti-aliasing state of the renderer: the active mode, the jitter
/// sequence of the temporal mode, the camera matrices of the last two
/// frames and the validity of the history buffer.
pub(crate) struct AaState {
    /// The active mode (the discriminant of [`AntiAliasMode`]).
    mode: std::sync::atomic::AtomicU32,
    /// The jitter sequence of the temporal mode.
    jitter: std::sync::RwLock<JitterSequence>,
    /// The camera matrices of the last two frames.
    frames: std::sync::RwLock<AaCameraFrames>,
    /// Whether the history buffer holds a resolved frame.
    history_valid: std::sync::atomic::AtomicBool,
    /// Whether a temporal resolve ran: only then the projection jitters,
    /// so a frame without the resolve pass stays pixel-identical.
    temporal_armed: std::sync::atomic::AtomicBool,
}

impl AaState {
    /// A fresh state: no anti-aliasing beyond the hardware MSAA.
    pub(crate) fn new() -> Self {
        Self {
            mode: std::sync::atomic::AtomicU32::new(AntiAliasMode::Msaa as u32),
            jitter: std::sync::RwLock::new(JitterSequence::new()),
            frames: std::sync::RwLock::new(AaCameraFrames::default()),
            history_valid: std::sync::atomic::AtomicBool::new(false),
            temporal_armed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The active mode.
    pub(crate) fn mode(&self) -> AntiAliasMode {
        mode_from_u32(self.mode.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Switches the mode, resetting the temporal state when the mode
    /// changes: a stale history or a stale jitter sequence would leak into
    /// the new mode.
    pub(crate) fn set_mode(&self, mode: AntiAliasMode) {
        let previous = self
            .mode
            .swap(mode as u32, std::sync::atomic::Ordering::Relaxed);
        if previous != mode as u32 {
            *write_lock(&self.jitter) = JitterSequence::new();
            self.history_valid
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.temporal_armed
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The jitter offset of the next frame, advancing the sequence.
    pub(crate) fn advance_jitter(&self) -> glam::Vec2 {
        write_lock(&self.jitter).advance()
    }

    /// Records the camera matrices of a frame: the view-projection of the
    /// frame before becomes the one the next frame reprojects into.
    pub(crate) fn record_camera_frame(&self, view_proj: Mat4) {
        let mut frames = write_lock(&self.frames);
        frames.previous_view_proj = frames.current_view_proj;
        frames.current_view_proj = view_proj;
    }

    /// The camera matrices of the last two frames.
    pub(crate) fn camera_frames(&self) -> AaCameraFrames {
        *read_lock(&self.frames)
    }

    /// Whether the history buffer holds a resolved frame.
    pub(crate) fn history_valid(&self) -> bool {
        self.history_valid
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Marks the history buffer as holding a resolved frame.
    pub(crate) fn mark_history_valid(&self) {
        self.history_valid
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Marks the history buffer as empty: the next resolve reads the
    /// current frame only.
    pub(crate) fn invalidate_history(&self) {
        self.history_valid
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a temporal resolve ran: only then the projection jitters.
    pub(crate) fn temporal_armed(&self) -> bool {
        self.temporal_armed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Arms the temporal jitter: the resolve pass ran, so the next frame
    /// may jitter its projection.
    pub(crate) fn arm_temporal(&self) {
        self.temporal_armed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Decodes the mode discriminant of the renderer state.
fn mode_from_u32(value: u32) -> AntiAliasMode {
    match value {
        value if value == AntiAliasMode::Off as u32 => AntiAliasMode::Off,
        value if value == AntiAliasMode::Taa as u32 => AntiAliasMode::Taa,
        value if value == AntiAliasMode::Fxaa as u32 => AntiAliasMode::Fxaa,
        value if value == AntiAliasMode::Smaa as u32 => AntiAliasMode::Smaa,
        _ => AntiAliasMode::Msaa,
    }
}
