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
//! (`FrameCommandBuffers`), not this step. [`PassSystem::run`] only
//! appends the pass name to [`PassOrderLog`]; GPU work stays on the
//! borrowed-encoder dispatch.

use crate::system::SystemSet;
use crate::transient_pool::{PassId, PassNode, ResourceId};
use ornis_core::{Schedule, System, SystemAccess};
use std::any::TypeId;
use std::fmt;
use std::sync::Mutex;

/// Order log for projected pass systems: [`PassSystem::run`] appends its
/// pass name here when the resource is installed, giving tests a
/// sequential-`Schedule::run` observable of the projected order without
/// touching GPU state.
#[derive(Debug, Default)]
pub struct PassOrderLog(pub Mutex<Vec<&'static str>>);

/// System twin of one declared pass: the declaration (name + accesses)
/// projected into the core scheduler. `run` appends the pass name to
/// [`PassOrderLog`] when installed and is a silent no-op otherwise — E1
/// executes passes through the borrowed-encoder dispatch; this type only
/// drives leveling (and, with it, the single-engine invariant of
/// `scheduler_parity`) plus the order observable. E2 replaces the log
/// append with recording through frame resources.
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

    fn run(&self, resources: &ornis_core::Resources) {
        let Some(log) = resources.get::<PassOrderLog>() else {
            return;
        };
        log.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(self.name);
    }
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
    access = access.reads::<PassOrderLog>();
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
    use crate::flags::Bloom;
    use crate::frame_exec::{RenderFrame3D, Technique};
    use crate::system::{FrameResource, ResourceKind};
    use crate::transient_pool::{SizePolicy, TextureSpec};
    use ornis_core::Resources;

    /// Full production matrix: every Technique × bloom wiring projects and
    /// levels match the frame layout bitwise (the E1 gate; the mirrored
    /// imperative topologies live in `scheduler_parity`).
    #[test]
    fn production_plans_project_with_matching_levels() {
        for technique in [Technique::Hybrid, Technique::Deferred, Technique::Forward] {
            for bloom in [Bloom::Off, Bloom::On] {
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
                    "technique {technique:?} bloom {bloom:?}: adapter levels != layout levels"
                );
            }
        }
    }

    macro_rules! test_resources {
        ($($r:ident => $name:literal),+ $(,)?) => {
            $(
                struct $r;
                impl FrameResource for $r {
                    const NAME: &'static str = $name;
                    fn kind() -> ResourceKind {
                        ResourceKind::FrameOwned
                    }
                    fn spec(_: wgpu::TextureFormat) -> TextureSpec {
                        TextureSpec {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            samples: 1,
                            size: SizePolicy::Fixed { width: 4, height: 4 },
                        }
                    }
                }
            )+
        };
    }

    test_resources!(T0 => "t0", T1 => "t1", T2 => "t2");

    fn sequential_log(schedule: &Schedule) -> Vec<&'static str> {
        let mut resources = Resources::new();
        resources.insert(PassOrderLog::default());
        schedule.run(&resources);
        resources
            .get::<PassOrderLog>()
            .expect("log installed")
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Sequential `Schedule::run` on production Hybrid follows the flat
    /// DAG order `FrameExecutor::execute` iterates (on production wirings
    /// both coincide with registration order).
    #[test]
    fn sequential_run_matches_flat_dag_order_on_production_hybrid() {
        let mut plan = RenderFrame3D::new_with(
            wgpu::TextureFormat::Rgba8Unorm,
            (32, 32),
            Technique::Hybrid,
            Bloom::Off,
        );
        let mut schedule =
            try_project_schedule(plan.systems()).expect("production resources are typed");
        schedule.set_parallel(false);
        let log = sequential_log(&schedule);
        let layout = plan.systems_mut().build();
        let flat: Vec<String> = layout
            .levels()
            .iter()
            .flatten()
            .map(|&i| layout.passes[i].name.clone())
            .collect();
        let log_strings: Vec<String> = log.iter().map(|name| (*name).to_owned()).collect();
        assert_eq!(
            log_strings, flat,
            "sequential Schedule::run != flat DAG order of FrameExecutor::execute"
        );
    }

    /// Diamond: `p2` is independent of the `p0 → p1` chain, so it shares
    /// level 0 and the projected levels are `[[0, 2], [1]]`.
    #[test]
    fn diamond_shares_level_for_independent_pass() {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut set = SystemSet::new();
        set.set_surface_size((32, 32));
        let a = set.register_resource::<T0>(format);
        let b = set.register_resource::<T1>(format);
        let c = set.register_resource::<T2>(format);
        set.add_pass("p0").write(a);
        set.add_pass("p1").read(a).write(b);
        set.add_pass("p2").write(c);
        let schedule = try_project_schedule(&set).expect("typed registration projects");
        assert_eq!(schedule.levels(), vec![vec![0, 2], vec![1]]);
        assert_eq!(
            schedule.levels(),
            set.build().levels(),
            "projected levels != layout levels on the diamond"
        );
    }

    /// The shared [`PassOrderLog`] read is conflict-free: independent
    /// writers still share one level.
    #[test]
    fn shared_log_read_does_not_split_levels() {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut set = SystemSet::new();
        set.set_surface_size((32, 32));
        let a = set.register_resource::<T0>(format);
        let b = set.register_resource::<T1>(format);
        set.add_pass("p0").write(a);
        set.add_pass("p1").write(b);
        let schedule = try_project_schedule(&set).expect("typed registration projects");
        assert_eq!(schedule.levels(), vec![vec![0, 1]]);
    }

    /// Without an installed [`PassOrderLog`] the twins are silent no-ops.
    #[test]
    fn run_without_log_is_silent_noop() {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut set = SystemSet::new();
        set.set_surface_size((32, 32));
        let a = set.register_resource::<T0>(format);
        set.add_pass("p0").write(a);
        let mut schedule = try_project_schedule(&set).expect("typed registration projects");
        schedule.set_parallel(false);
        let resources = Resources::new();
        schedule.run(&resources);
    }
}
