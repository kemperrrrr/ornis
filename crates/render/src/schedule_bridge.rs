//! E1 (S5e, 2026-09-07): frame passes as ordinary `Schedule` systems.
//!
//! [`try_project_schedule`] mirrors a [`SystemSet`] declaration into a
//! `ornis_core::Schedule`: every enabled pass becomes a [`PassSystem`]
//! declaration twin — same name, accesses projected from `ResourceId`s to
//! the `FrameResource` `TypeId`s through the registry. The projected
//! levels are pinned bitwise-equal to `FrameLayout::levels()` by
//! `scheduler_parity` (the anti-drift canon); pass bodies still record
//! through a borrowed encoder at the execution site
//! ([`crate::frame_exec::RenderFrame3D::render_schedule`]) — moving the
//! recording itself into the systems via a frame resource is E2
//! (`FrameCommandBuffers`), not this step.

use crate::system::SystemSet;
use crate::transient_pool::{PassId, PassNode, ResourceId};
use ornis_core::{Schedule, System, SystemAccess};
use std::any::TypeId;
use std::fmt;

/// System twin of one declared pass: the declaration (name + accesses)
/// projected into the core scheduler. `run` is a no-op by design — E1
/// executes passes through the borrowed-encoder dispatch; this type only
/// drives leveling (and, with it, the single-engine invariant of
/// `scheduler_parity`). E2 replaces the no-op with recording through
/// frame resources.
pub struct PassSystem {
    name: &'static str,
    access: SystemAccess,
}

impl System for PassSystem {
    fn name(&self) -> &'static str {
        self.name
    }

    fn access(&self) -> SystemAccess {
        self.access.clone()
    }

    fn run(&self, _resources: &ornis_core::Resources) {}
}

/// Why a pass cannot be projected into a [`PassSystem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionError {
    /// The pass accesses a resource declared without a typed identity
    /// (`create_resource` instead of `register_resource::<R>`); the core
    /// scheduler keys accesses by `TypeId` and has nothing to mirror.
    UntypedResource {
        /// Name of the offending pass.
        pass: &'static str,
        /// The resource without a `FrameResource` type.
        resource: ResourceId,
    },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UntypedResource { pass, resource } => write!(
                f,
                "pass '{pass}' accesses untyped resource {resource:?} \
                 (register it with register_resource::<R> to project)"
            ),
        }
    }
}

impl std::error::Error for ProjectionError {}

/// Projects the registry's passes into a core [`Schedule`].
///
/// Registration order is preserved and disabled passes are skipped, so
/// the projected system index equals the layout pass index — the invariant
/// `render_schedule` relies on and `scheduler_parity` pins. Explicit
/// `order_before` edges are mirrored by name; an edge touching a disabled
/// pass is dropped, exactly like `TransientPool`'s `layout_levels` does.
///
/// # Errors
/// Returns [`ProjectionError::UntypedResource`] for a pass whose accessed
/// resource has no typed registry identity.
pub fn try_project_schedule(set: &SystemSet) -> Result<Schedule, ProjectionError> {
    let mut schedule = Schedule::new();
    for (index, node) in set.pass_nodes().iter().enumerate() {
        if node.enabled {
            schedule.add_system(pass_system(set, PassId(index as u32), node)?);
        }
    }
    mirror_ordering_edges(set, &mut schedule);
    Ok(schedule)
}

/// Builds the declaration twin of one enabled pass.
fn pass_system(
    set: &SystemSet,
    id: PassId,
    node: &PassNode,
) -> Result<PassSystem, ProjectionError> {
    let name = set.pass_name(id);
    let mut access = SystemAccess::new();
    for &resource in &node.reads {
        push_typed_access(&mut access.reads, set, name, resource)?;
    }
    for &(resource, _) in &node.writes {
        push_typed_access(&mut access.writes, set, name, resource)?;
    }
    Ok(PassSystem { name, access })
}

/// Resolves one `ResourceId` access to its `FrameResource` `TypeId`.
fn push_typed_access(
    accesses: &mut Vec<TypeId>,
    set: &SystemSet,
    pass: &'static str,
    resource: ResourceId,
) -> Result<(), ProjectionError> {
    match set.resource_type(resource) {
        Some(type_id) => {
            accesses.push(type_id);
            Ok(())
        }
        None => Err(ProjectionError::UntypedResource { pass, resource }),
    }
}

/// Mirrors `order_before` edges onto the projected schedule (by pass name;
/// edges touching disabled passes are dropped, mirroring `layout_levels`).
fn mirror_ordering_edges(set: &SystemSet, schedule: &mut Schedule) {
    for &(before, after) in set.ordering_edges() {
        let names = match (edge_name(set, before), edge_name(set, after)) {
            (Some(before_name), Some(after_name)) => Some((before_name, after_name)),
            _ => None,
        };
        if let Some((before_name, after_name)) = names {
            let _ = schedule.try_order_before(before_name, after_name);
        }
    }
}

/// Name of a pass if it is enabled (i.e. present in the projection).
fn edge_name(set: &SystemSet, id: PassId) -> Option<&'static str> {
    set.pass_nodes()
        .get(id.0 as usize)
        .filter(|node| node.enabled)
        .map(|_| set.pass_name(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_exec::{RenderFrame3D, Technique};

    /// Full production matrix: every Technique × bloom wiring projects and
    /// levels match the frame layout bitwise (the E1 gate; the mirrored
    /// imperative topologies live in `scheduler_parity`).
    #[test]
    fn production_plans_project_with_matching_levels() {
        for technique in [Technique::Hybrid, Technique::Deferred, Technique::Forward] {
            for bloom in [false, true] {
                let mut plan = RenderFrame3D::new_with(
                    wgpu::TextureFormat::Rgba8Unorm,
                    (32, 32),
                    technique,
                    bloom,
                );
                let schedule =
                    try_project_schedule(plan.systems()).expect("production resources are typed");
                assert_eq!(
                    schedule.levels(),
                    plan.systems_mut().build().levels(),
                    "technique {technique:?} bloom {bloom}: adapter levels != layout levels"
                );
            }
        }
    }
}
