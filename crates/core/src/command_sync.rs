//! Command-based synchronization system for CPU↔GPU data transfer
//!
//! This module implements the "Instructions instead of Data" pattern (LEAK pattern from HVM2)
//! where CPU sends commands to GPU instead of copying data back and forth.
//!
//! The GPU-specific execution is in the render crate; core only handles CPU-side logic.

use crate::smart_store::SmartStore;
use std::any::TypeId;
use std::collections::HashMap;

/// Data residency state for a lane
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataResidency {
    /// Latest authoritative copy lives only on the CPU; a CPU-to-GPU
    /// upload is pending.
    #[default]
    CpuOnly,
    /// Latest authoritative copy lives only on the GPU; a GPU readback is
    /// pending before CPU code may consume the data.
    GpuOnly,
    /// Both copies are up to date; no transfer required.
    Both,
}

/// Tracks data residency across CPU and GPU
#[derive(Debug, Default)]
pub struct ResidencyTracker {
    residency: HashMap<TypeId, DataResidency>,
}

impl ResidencyTracker {
    /// Creates an empty tracker; every untracked type is implicitly
    /// `CpuOnly`.
    pub fn new() -> Self {
        Self {
            residency: HashMap::new(),
        }
    }

    /// Records that the CPU copy of `T` is the fresh one.
    pub fn mark_cpu<T: 'static + Send + Sync>(&mut self) {
        self.residency
            .insert(TypeId::of::<T>(), DataResidency::CpuOnly);
    }

    /// Records that the GPU copy of `T` is the fresh one.
    pub fn mark_gpu<T: 'static + Send + Sync>(&mut self) {
        self.residency
            .insert(TypeId::of::<T>(), DataResidency::GpuOnly);
    }

    /// Records that both copies of `T` are in sync (after a completed
    /// transfer).
    pub fn mark_both<T: 'static + Send + Sync>(&mut self) {
        self.residency
            .insert(TypeId::of::<T>(), DataResidency::Both);
    }

    /// Current residency of `T`; defaults to [`DataResidency::CpuOnly`]
    /// for types never marked.
    pub fn get<T: 'static + Send + Sync>(&self) -> DataResidency {
        self.residency
            .get(&TypeId::of::<T>())
            .copied()
            .unwrap_or(DataResidency::CpuOnly)
    }

    /// Number of tracked component types.
    pub fn len(&self) -> usize {
        self.residency.len()
    }

    /// Returns `true` if no type has been tracked yet.
    pub fn is_empty(&self) -> bool {
        self.residency.is_empty()
    }

    /// `true` when the GPU copy is stale and an upload of `T` is due.
    pub fn needs_cpu_to_gpu<T: 'static + Send + Sync>(&self) -> bool {
        matches!(self.get::<T>(), DataResidency::CpuOnly)
    }

    /// `true` when the CPU copy is stale and a readback of `T` is due.
    pub fn needs_gpu_to_cpu<T: 'static + Send + Sync>(&self) -> bool {
        matches!(self.get::<T>(), DataResidency::GpuOnly)
    }
}

/// Type of GPU command
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandType {
    /// Sync CPU data to GPU
    CpuToGpu,
    /// Sync GPU data to CPU
    GpuToCpu,
    /// Execute a compute shader
    Compute,
    /// Copy buffer
    Copy,
}

/// A GPU command to be executed on the GPU
#[derive(Debug)]
pub struct GpuCommand {
    /// What kind of transfer/dispatch to perform.
    pub command_type: CommandType,
    /// Which component lane the command applies to.
    pub component_type: TypeId,
}

impl GpuCommand {
    /// Creates a command targeting the lane of component type `T`.
    pub fn new<T: 'static + Send + Sync>(command_type: CommandType) -> Self {
        Self {
            command_type,
            component_type: TypeId::of::<T>(),
        }
    }
}

/// Queue of GPU commands
#[derive(Debug, Default)]
pub struct CommandQueue {
    commands: Vec<GpuCommand>,
    dirty_lanes: HashMap<TypeId, bool>,
}

impl CommandQueue {
    /// Creates an empty queue with no dirty lanes.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            dirty_lanes: HashMap::new(),
        }
    }

    /// Appends `command` (typed by `T` purely for call-site clarity).
    pub fn enqueue<T: 'static + Send + Sync>(&mut self, command: GpuCommand) {
        self.commands.push(command);
    }

    /// Flags the lane of `T` as having unsynced CPU-side changes.
    pub fn mark_dirty<T: 'static + Send + Sync>(&mut self) {
        self.dirty_lanes.insert(TypeId::of::<T>(), true);
    }

    /// `true` if the lane of `T` has changes not yet flushed to the GPU.
    pub fn is_dirty<T: 'static + Send + Sync>(&self) -> bool {
        self.dirty_lanes
            .get(&TypeId::of::<T>())
            .copied()
            .unwrap_or(false)
    }

    /// Clears the dirty flag for `T` (call after a successful upload).
    pub fn clear_dirty<T: 'static + Send + Sync>(&mut self) {
        self.dirty_lanes.insert(TypeId::of::<T>(), false);
    }

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns `true` if no commands are pending.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Takes all queued commands, leaving the queue empty; the render
    /// thread executes them and reports residency back.
    pub fn drain(&mut self) -> Vec<GpuCommand> {
        std::mem::take(&mut self.commands)
    }
}

/// Smart command-based sync system (CPU-side only)
pub struct CommandSync {
    queue: CommandQueue,
    residency: ResidencyTracker,
}

impl Default for CommandSync {
    fn default() -> Self {
        Self {
            queue: CommandQueue::new(),
            residency: ResidencyTracker::new(),
        }
    }
}

impl CommandSync {
    /// Creates an empty sync state with no commands or tracked lanes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Flags the lane of POD component `T` as dirty on the CPU side:
    /// marks it in the queue and records `CpuOnly` residency so the next
    /// flush issues an upload.
    pub fn mark_dirty<T: 'static + Send + Sync + bytemuck::Pod + bytemuck::Zeroable>(
        &mut self,
        _store: &SmartStore,
    ) {
        self.queue.mark_dirty::<T>();
        self.residency.mark_cpu::<T>();
    }

    /// Pending GPU commands accumulated since the last drain.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Residency state per component type.
    pub fn residency(&self) -> &ResidencyTracker {
        &self.residency
    }
}

/// Trait for components that can be synced via commands
/// Marker for component types eligible for command-based GPU sync:
/// plain-old-data so they can be copied into GPU buffers verbatim.
pub trait CommandSyncable: 'static + Send + Sync + bytemuck::Pod + bytemuck::Zeroable {}

impl<T: 'static + Send + Sync + bytemuck::Pod + bytemuck::Zeroable> CommandSyncable for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SmartStore;

    #[test]
    fn command_queue_enqueue_execute() {
        let queue = CommandQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn residency_tracker() {
        let mut tracker = ResidencyTracker::new();
        tracker.mark_cpu::<f32>();
        assert_eq!(tracker.get::<f32>(), DataResidency::CpuOnly);
        assert!(tracker.needs_cpu_to_gpu::<f32>());
        assert!(!tracker.needs_gpu_to_cpu::<f32>());

        tracker.mark_gpu::<f32>();
        assert_eq!(tracker.get::<f32>(), DataResidency::GpuOnly);
        assert!(!tracker.needs_cpu_to_gpu::<f32>());
        assert!(tracker.needs_gpu_to_cpu::<f32>());

        tracker.mark_both::<f32>();
        assert_eq!(tracker.get::<f32>(), DataResidency::Both);
        assert!(!tracker.needs_cpu_to_gpu::<f32>());
        assert!(!tracker.needs_gpu_to_cpu::<f32>());
    }

    #[test]
    fn command_sync_basic() {
        let mut sync = CommandSync::new();
        let store = SmartStore::new();

        sync.mark_dirty::<f32>(&store);
        assert!(sync.queue.is_dirty::<f32>());

        sync.queue.clear_dirty::<f32>();
        assert!(!sync.queue.is_dirty::<f32>());
    }

    #[test]
    fn command_queue_enqueue_drain() {
        let mut queue = CommandQueue::new();
        let cmd = GpuCommand::new::<f32>(CommandType::CpuToGpu);

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.enqueue::<f32>(cmd);
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        queue.mark_dirty::<f32>();
        assert!(queue.is_dirty::<f32>());
        queue.clear_dirty::<f32>();
        assert!(!queue.is_dirty::<f32>());

        let drained = queue.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].command_type, CommandType::CpuToGpu);
        assert!(queue.is_empty());
    }

    #[test]
    fn command_queue_dirty_per_type() {
        let mut queue = CommandQueue::new();

        queue.mark_dirty::<f32>();
        assert!(queue.is_dirty::<f32>());
        // Different type stays clean.
        assert!(!queue.is_dirty::<u64>());

        queue.clear_dirty::<f32>();
        assert!(!queue.is_dirty::<f32>());
    }

    #[test]
    fn command_sync_exposes_real_state() {
        let mut sync = CommandSync::new();
        let store = SmartStore::new();

        // Empty tracker before any mark: catches len()→1 / is_empty()→false
        // mutants on the tracker itself.
        assert_eq!(sync.residency().len(), 0);
        assert!(sync.residency().is_empty());

        // Accessors must return the live queue/tracker, not a default stub.
        sync.mark_dirty::<f32>(&store);
        assert!(sync.queue().is_dirty::<f32>());
        assert_eq!(sync.residency().get::<f32>(), DataResidency::CpuOnly);
        // A stub tracker would be empty; the live one holds the f32 entry.
        assert_eq!(sync.residency().len(), 1);
        assert!(!sync.residency().is_empty());
    }

    #[test]
    fn residency_mark_cpu_sets_residency() {
        let mut tracker = ResidencyTracker::new();
        assert_eq!(tracker.get::<u32>(), DataResidency::CpuOnly);

        tracker.mark_gpu::<u32>();
        assert_eq!(tracker.get::<u32>(), DataResidency::GpuOnly);
        assert!(tracker.needs_gpu_to_cpu::<u32>());

        tracker.mark_cpu::<u32>();
        assert_eq!(tracker.get::<u32>(), DataResidency::CpuOnly);
        assert!(tracker.needs_cpu_to_gpu::<u32>());
        assert!(!tracker.needs_gpu_to_cpu::<u32>());
    }
}
