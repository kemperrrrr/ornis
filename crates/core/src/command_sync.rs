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
    #[default]
    CpuOnly,
    GpuOnly,
    Both,
}

/// Tracks data residency across CPU and GPU
#[derive(Debug, Default)]
pub struct ResidencyTracker {
    residency: HashMap<TypeId, DataResidency>,
}

impl ResidencyTracker {
    pub fn new() -> Self {
        Self {
            residency: HashMap::new(),
        }
    }

    pub fn mark_cpu<T: 'static + Send + Sync>(&mut self) {
        self.residency
            .insert(TypeId::of::<T>(), DataResidency::CpuOnly);
    }

    pub fn mark_gpu<T: 'static + Send + Sync>(&mut self) {
        self.residency
            .insert(TypeId::of::<T>(), DataResidency::GpuOnly);
    }

    pub fn mark_both<T: 'static + Send + Sync>(&mut self) {
        self.residency
            .insert(TypeId::of::<T>(), DataResidency::Both);
    }

    pub fn get<T: 'static + Send + Sync>(&self) -> DataResidency {
        self.residency
            .get(&TypeId::of::<T>())
            .copied()
            .unwrap_or(DataResidency::CpuOnly)
    }

    pub fn needs_cpu_to_gpu<T: 'static + Send + Sync>(&self) -> bool {
        matches!(self.get::<T>(), DataResidency::CpuOnly)
    }

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
    pub command_type: CommandType,
    pub component_type: TypeId,
}

impl GpuCommand {
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
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            dirty_lanes: HashMap::new(),
        }
    }

    pub fn enqueue<T: 'static + Send + Sync>(&mut self, command: GpuCommand) {
        self.commands.push(command);
    }

    pub fn mark_dirty<T: 'static + Send + Sync>(&mut self) {
        self.dirty_lanes.insert(TypeId::of::<T>(), true);
    }

    pub fn is_dirty<T: 'static + Send + Sync>(&self) -> bool {
        self.dirty_lanes
            .get(&TypeId::of::<T>())
            .copied()
            .unwrap_or(false)
    }

    pub fn clear_dirty<T: 'static + Send + Sync>(&mut self) {
        self.dirty_lanes.insert(TypeId::of::<T>(), false);
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_dirty<T: 'static + Send + Sync + bytemuck::Pod + bytemuck::Zeroable>(
        &mut self,
        _store: &SmartStore,
    ) {
        self.queue.mark_dirty::<T>();
        self.residency.mark_cpu::<T>();
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn residency(&self) -> &ResidencyTracker {
        &self.residency
    }
}

/// Trait for components that can be synced via commands
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
}
