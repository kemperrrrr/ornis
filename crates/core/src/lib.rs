mod page_table;
mod component_store;
mod entity;
mod smart_store;
mod prefetch;
mod cold_store;
pub mod pipeline;
#[cfg(feature = "lock-free")]
mod lock_free_store;
pub mod physics;

pub use entity::{Entity, EntityAllocator};
pub use component_store::{ComponentStore, ZipIter, ChunkedIterMut};
pub use page_table::{PageTable, PAGE_SIZE};
pub use smart_store::{SmartStore, Pack};
pub use cold_store::ColdComponentStore;
pub use prefetch::prefetch_read;
pub use pipeline::{AutoPipeline, LaneTarget, PipelineConfig, TargetDiscriminant, Route, lane_target_of, GpuLane, CpuLane, HybridLane, pipeline_enter, pipeline_exit};
pub use physics::*;
