//! GPU execution backend for ornis, built on `wgpu`.
//!
//! The crate turns lane decisions made by [`ornis_core`] into concrete work:
//! [`CommandSync`](command_sync::CommandSync) records GPU compute dispatches
//! and CPU closures and flushes them together, [`PipelineRouter`] and
//! [`choose_platform`] map component lanes onto a CPU/GPU [`Platform`], and
//! [`SmartBuffer`] keeps data resident on both sides with dirty-flag tracking
//! so transfers happen only when needed. Compute pipelines are memoized by
//! [`PsoCache`], while [`AutoProfiler`] calibrates the GPU/CPU crossover
//! threshold on the local hardware. [`leak`] provides self-contained shader
//! generation and dispatch for LEAK-style kernels.

#![warn(missing_docs)]

/// Buffer creation helpers bridging [`ornis_core::ComponentStore`] and `wgpu`.
pub mod buffer;
/// Mixed CPU/GPU command recording with a single flush point.
pub mod command_sync;
/// `wgpu` instance/adapter/device/queue setup.
pub mod context;
/// Execution-target policy: CPU, GPU, or an element-count threshold.
pub mod dispatcher;
/// Zero-cost executors and dispatch entry points for lanes.
pub mod execute;
/// LEAK-style WGSL shader generation and one-shot dispatch.
pub mod leak;
/// Hardware calibration of the GPU/CPU crossover point.
pub mod profiler;
/// In-memory and on-disk caching of compiled compute pipelines.
pub mod pso_cache;
/// Type-level bridge from lane tags to execution platforms.
pub mod router;
/// Dual-resident buffer with dirty-flag synchronization.
pub mod smart_buffer;

pub use buffer::{create_buffer_from_slice, create_buffer_from_store};
pub use context::WgpuContext;
pub use dispatcher::{DispatchConfig, ExecutionTarget, Platform, choose_platform};
pub use execute::{CpuExecutor, ExecuteLane, GpuExecutor, dispatch_lane, dispatch_with};
pub use leak::{LeakDispatch, leak_wgsl, leak_wgsl_typed};
pub use profiler::{AutoProfiler, ProfilerConfig};
pub use pso_cache::PsoCache;
pub use router::PipelineRouter;
pub use smart_buffer::{ResidencyFlags, SmartBuffer};
