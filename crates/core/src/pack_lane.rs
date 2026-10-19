//! Field-lane contract for [`Pack`](crate::Pack) bundles: the per-field
//! [`PackLane`] wrappers that `#[derive(Pack)]` generates, and the
//! [`PackMembership`] entity-set narrowing that `#[smart_system]` runs use.

use crate::component_store::ComponentStore;
use crate::entity::Entity;

/// One field lane of a [`Pack`] bundle: the generated wrapper type that
/// stores a single field for each entity owning it.
///
/// `#[derive(Pack)]` implements this for every wrapper. `#[smart_system]`
/// gathers a field through it, runs the body on the copy, and scatters the
/// value back, so the same code path serves dense and sparse fields.
pub trait PackLane: Sized + Send + Sync + 'static {
    /// Field type as declared on the bundle (`Option<T>` for sparse fields).
    type Field: Clone;

    /// `true` for `Option<T>` fields. Only entities that own a value hold an
    /// entry, and a missing entry reads as `None` instead of excluding the
    /// entity.
    const SPARSE: bool;

    /// Reads the field at dense `index`. Returns `None` for a dense field
    /// whose entity is missing from the lane, so the caller skips it. Sparse
    /// fields always return `Some`, with `None` as the value when absent.
    fn gather(lane: &ComponentStore<Self>, index: Option<usize>) -> Option<Self::Field>;

    /// Writes `value` back for `entity`. Dense fields overwrite `index`
    /// (inserting when it is `None`). Sparse fields overwrite or insert a
    /// `Some` value and remove the entry for `None`.
    fn scatter(
        lane: &mut ComponentStore<Self>,
        entity: Entity,
        index: Option<usize>,
        value: Self::Field,
    );

    /// Reads the field at dense `index` of a lane that is known to hold an
    /// entry for every entity of the run, so no entity lookup or skip check
    /// is needed. Sparse fields return the stored value as `Some`.
    fn gather_at(lane: &ComponentStore<Self>, index: usize) -> Self::Field;

    /// Writes `value` at dense `index` under the same precondition as
    /// [`gather_at`](Self::gather_at). Sparse `None` removes the entry at
    /// `index` through the lane's own entity list.
    fn scatter_at(lane: &mut ComponentStore<Self>, index: usize, value: Self::Field);
}

/// Dense (non-`Option`) field lane: the stored cell is [`PackLane::Field`].
///
/// Sparse wrappers do not implement this. Aligned `#[smart_system]` runs
/// zip the columns through it, so that path is only type-checked when every
/// own lane of the system is dense.
pub trait DensePackLane: PackLane {
    /// Shared view of the stored field.
    fn inner(&self) -> &Self::Field;

    /// Mutable view of the stored field.
    fn inner_mut(&mut self) -> &mut Self::Field;
}

/// Entity set of one `#[smart_system]` run: the entities that own every
/// required (non-`Option`) lane the body touches.
///
/// Membership comes from lane entity lists only, so no field values are
/// read. Entities are yielded in ascending entity-id order, the order
/// `Pack::pack_entities` has always used.
#[derive(Default)]
pub struct PackMembership {
    entities: Option<Vec<Entity>>,
}

impl PackMembership {
    /// Empty membership: no lane has been narrowed yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the set to entities present in `lane`. Sparse lanes do not
    /// restrict membership. The first required lane seeds the set, and later
    /// ones filter it.
    pub fn narrow<L: PackLane>(&mut self, lane: &ComponentStore<L>) {
        if L::SPARSE {
            return;
        }
        match &mut self.entities {
            Some(set) => set.retain(|entity| lane.contains(*entity)),
            None => self.entities = Some(lane.iter_entities().collect()),
        }
    }

    /// Consumes the set. Empty when no required lane was narrowed; the
    /// `#[smart_system]` expansion rejects such bodies at compile time.
    pub fn into_entities(self) -> Vec<Entity> {
        self.entities.unwrap_or_default()
    }
}
