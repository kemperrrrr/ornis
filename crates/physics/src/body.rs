//! Rigid-body data model: handles, body types and mass properties.
//!
//! [`RigidBody`] bundles pose, velocities, material coefficients and the
//! collision [`Shape`]; [`BodyType`] decides which solver terms apply. The
//! solver reads only inverse quantities, so `mass` and `inv_mass` must stay
//! consistent when mutated directly. [`BodyHandle`] is a plain vector index:
//! removal swaps the last body into the removed slot. Handles survive
//! solver migrations, but callers must remap the moved body after removal.

use glam::{Quat, Vec3};

use crate::invariants::{FrictionFrame, FrictionFrameError, Mass, MassKind, PositiveF32, UnitVec3};
use crate::shape::{Pose, Shape};

// Default material coefficients for constructor helpers. Named so the
// shape-specific recipes (`new_box` vs `new_sphere` vs hull rolling) stay
// readable and cannot drift apart when one call site is retuned.

/// Default restitution for boxes / cones / hulls / meshes.
const RESTITUTION_DEFAULT: f32 = 0.3;
/// Default slide friction for boxes / cones / hulls / meshes.
const FRICTION_DEFAULT: f32 = 0.5;
/// Sphere recipe: bouncier, slightly less sticky than the box default.
const RESTITUTION_SPHERE: f32 = 0.5;
const FRICTION_SPHERE: f32 = 0.3;
/// Capsule / cylinder recipe: mid rubber.
const RESTITUTION_CAPSULE: f32 = 0.4;
const FRICTION_CAPSULE: f32 = 0.4;
/// Heightfield terrain recipe: grip over bounce.
const FRICTION_TERRAIN: f32 = 0.6;
/// MuJoCo-style rolling/torsion torque caps (metres) for hulls and meshes.
const ROLLING_FRICTION_DEBRIS: f32 = 0.2;
const TORSION_FRICTION_DEBRIS: f32 = 0.05;
/// Fallback half-extents when trimesh soup construction fails.
const FALLBACK_BOX_HALF: f32 = 0.5;

/// Stable index of a body inside its owning [`SequentialImpulseEngine`](crate::engine::SequentialImpulseEngine).
///
/// Solver/routing migrations preserve handles. Removal swaps the final body
/// into the removed slot; only that surviving body's handle changes. A removed
/// slot can be reused, so these are not generational entity identifiers.
///
/// Newtype over `u32` (not a `usize` alias) so body handles never mix with
/// joint handles, particle indices or entity ids at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyHandle(u32);

impl BodyHandle {
    /// Wraps a raw `u32` body index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` body index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Body index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for BodyHandle {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for BodyHandle {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<BodyHandle> for u32 {
    fn from(h: BodyHandle) -> Self {
        h.0
    }
}

impl From<BodyHandle> for usize {
    fn from(h: BodyHandle) -> Self {
        h.0 as usize
    }
}

/// Local body index inside an [`AvbdEngine`](crate::avbd::AvbdEngine).
///
/// Same `u32` representation as [`BodyHandle`], but a different handle
/// space: the AVBD engine's dense table, not the [`Engine`](crate::Engine)
/// global registry. Convert explicitly at the split-registry boundary
/// (`split.rs`); the conversion is a lossless `u32` reinterpretation, so
/// simulation behavior is bit-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalAvbdBody(u32);

impl LocalAvbdBody {
    /// Wraps a raw `u32` local body index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` local body index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Local body index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LocalAvbdBody {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for LocalAvbdBody {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<LocalAvbdBody> for u32 {
    fn from(h: LocalAvbdBody) -> Self {
        h.0
    }
}

impl From<LocalAvbdBody> for usize {
    fn from(h: LocalAvbdBody) -> Self {
        h.0 as usize
    }
}

impl From<LocalAvbdBody> for BodyHandle {
    fn from(h: LocalAvbdBody) -> Self {
        Self::from_raw(h.0)
    }
}

impl From<BodyHandle> for LocalAvbdBody {
    fn from(h: BodyHandle) -> Self {
        Self::from_raw(h.as_u32())
    }
}

/// Local body index inside a [`SequentialImpulseEngine`](crate::engine::SequentialImpulseEngine).
///
/// Same contract as [`LocalAvbdBody`]: the SI engine's dense table, not
/// the global registry. Explicit conversions only, at the split boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalSiBody(u32);

impl LocalSiBody {
    /// Wraps a raw `u32` local body index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` local body index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Local body index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LocalSiBody {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for LocalSiBody {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<LocalSiBody> for u32 {
    fn from(h: LocalSiBody) -> Self {
        h.0
    }
}

impl From<LocalSiBody> for usize {
    fn from(h: LocalSiBody) -> Self {
        h.0 as usize
    }
}

impl From<LocalSiBody> for BodyHandle {
    fn from(h: LocalSiBody) -> Self {
        Self::from_raw(h.0)
    }
}

impl From<BodyHandle> for LocalSiBody {
    fn from(h: BodyHandle) -> Self {
        Self::from_raw(h.as_u32())
    }
}

/// How a body participates in simulation.
///
/// Determines which solver terms apply: static bodies have zero inverse mass
/// and never integrate, kinematic bodies integrate velocity but push dynamic
/// bodies with infinite effective mass, dynamic bodies respond to contacts
/// and gravity in full.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyType {
    /// Never moves (level geometry); treated as having infinite mass.
    Static,
    /// Fully simulated under forces, gravity, contacts and joints.
    Dynamic,
    /// Moved by setting [`RigidBody::velocity`] directly; collides with and
    /// pushes dynamic bodies but is unaffected by them.
    ///
    /// Driver contract: the engine never integrates kinematic bodies, the
    /// driver sets positions. A per-step position change (teleport) above
    /// the travel gate (half the thinnest shape feature, same gate as the
    /// linear CCD) acts as the implied motion everywhere for that step —
    /// broadphase sweep, speculative margin, kinematic sweep and contact
    /// responses — with the driver velocity fields restored at step end.
    /// Below the gate the fields rule and the teleport settles
    /// positionally, imparting no momentum (Box2D parity for nudges).
    Kinematic,
}

/// A single rigid body: pose, motion state, material properties and shape.
///
/// Mass properties are derived from [`Shape::inertia`] at construction
/// through the single [`Mass`](crate::invariants::Mass) site
/// ([`Mass::from_kind`](crate::invariants::Mass::from_kind)); mutate them
/// only via [`RigidBody::set_mass_kind`], [`RigidBody::set_mass`],
/// [`RigidBody::make_static`] or [`RigidBody::make_dynamic`] so `mass`,
/// `inv_mass`, `inertia` and `body_type` move together — the solver reads
/// only the inverse quantities. Direct field writes bypass the invariant.
#[derive(Debug, Clone)]
pub struct RigidBody {
    /// World-space position of the body's center of mass.
    pub position: Vec3,
    /// Rotation of the body, stored as a unit quaternion.
    pub orientation: Quat,
    /// Linear velocity of the center of mass (m/s).
    pub velocity: Vec3,
    /// Angular velocity in world space (rad/s), about the center of mass.
    pub angular_velocity: Vec3,
    /// Total mass (kg). Zero mass means static/infinite-mass behavior.
    /// Part of the mass triple — see the [`RigidBody`] docs; prefer
    /// [`RigidBody::set_mass`] over direct writes.
    pub mass: f32,
    /// Cached `1 / mass` (0 for statics) — what the solver actually uses.
    /// Part of the mass triple — see the [`RigidBody`] docs.
    pub inv_mass: f32,
    /// Diagonal (body-frame) inertia tensor. Part of the mass triple —
    /// see the [`RigidBody`] docs.
    pub inertia: Vec3,
    /// Accumulated external torque (N·m), consumed and cleared each step.
    pub torque: Vec3,
    /// Coefficient of restitution in `[0, 1]`: how much normal velocity
    /// survives a bounce (0 = dead stop, 1 = perfectly elastic).
    pub restitution: f32,
    /// Coulomb friction coefficient ≥ 0 used by the contact solver.
    pub friction: f32,
    /// Transverse friction coefficient ≥ 0 (ODE `mu2` parity): Coulomb
    /// coefficient along the second tangent axis when
    /// [`RigidBody::friction_dir`] fixes the first one. Initialized from
    /// `friction` at construction — mutating `friction` later does NOT
    /// re-mirror it, set both explicitly for anisotropic surfaces.
    /// Ignored while `friction_dir` is `None` (isotropic contact).
    pub friction_transverse: f32,
    /// First friction direction in body-local frame (ODE `fdir1` parity):
    /// fixes the manifold tangent basis `t1` (projected onto the plane
    /// ⊥ the contact normal, orthogonalized at use). Body A wins if both
    /// sides set one; degenerate projections fall back to the default
    /// normal-derived basis. `None` (default) = isotropic friction.
    pub friction_dir: Option<Vec3>,
    /// Rolling resistance in meters (MuJoCo `rolling` parity): per-contact
    /// torque cap = `rolling_friction × normal impulse`, opposing relative
    /// rotation about the tangent axes. Zero (default) disables it —
    /// existing scenes are bit-identical.
    pub rolling_friction: f32,
    /// Torsional (spin) friction in meters (MuJoCo `torsional` parity):
    /// torque cap = `torsion_friction × normal impulse`, opposing relative
    /// spin about the contact normal. Zero (default) disables it.
    pub torsion_friction: f32,
    /// Collision primitive; also drives the derived inertia tensor.
    pub shape: Shape,
    /// Bit identifying the collision layer this body belongs to.
    ///
    /// A zero layer is valid and makes the body ineligible for all pairs;
    /// ordinary layers are represented by one or more set bits.
    pub collision_layer: u32,
    /// Bit mask of layers this body is allowed to collide with.
    ///
    /// A pair collides only when both bodies' masks include the other
    /// body's layer. The default is all layers, preserving pre-filter
    /// behavior.
    pub collision_mask: u32,
    /// When true, the body reports overlap transitions but never applies
    /// contact or CCD impulses. Triggers still obey collision filters.
    /// Prefer [`Self::role`] / [`Self::set_role`] with [`crate::flags::BodyRole`].
    pub is_trigger: bool,
    /// Impact speed (m/s) at which the [`crate::Engine`] fracture pass
    /// splits this body, compared against [`ContactEventKind::Hit`]
    /// approach speeds after each step. `INFINITY` (default) never
    /// fractures — existing scenes are bit-identical. Only dynamic boxes
    /// split (along the longest axis, halves inherit everything incl.
    /// this threshold); other shapes and static bodies ignore it.
    /// Values below the Hit emission floor (1 m/s) behave as 1 m/s: no
    /// Hit event exists to trigger on.
    pub fracture_impact_speed: f32,
    /// Simulation role; derived from mass at construction, settable after.
    pub body_type: BodyType,
    /// Bullet (full-CCD) upgrade in the Rapier sense (`ccd.ccd_enabled`):
    /// a dynamic body that always sweeps the nonlinear (rotating) TOI path,
    /// bypassing the travel gate. Non-bullet bodies keep the legacy policy
    /// (linear sweep plus the gated angular sweep). Default `false`, so
    /// existing scenes are bit-identical.
    pub ccd_enabled: bool,
    /// Contact-force report threshold (N) for this body (Rapier
    /// `ContactForceEventThreshold` parity): the pair force (total normal
    /// impulse over the last substep divided by the substep length) at or
    /// above which a [`crate::trigger::ContactForceEvent`] is emitted.
    /// `INFINITY` (default) disables reporting for this side — existing
    /// scenes are bit-identical and the force pass early-outs with zero
    /// per-pair work. Set a finite value (opt-in, like Rapier's
    /// `ActiveEvents::CONTACT_FORCE_EVENTS`) to observe load-bearing
    /// contacts, impacts and presses involving this body; use
    /// [`Self::set_contact_force_threshold`] so negative/non-finite input
    /// folds to the documented states instead of poisoning comparisons.
    pub contact_force_threshold: f32,
}

impl RigidBody {
    // Internal constructor: derives derived quantities from mass and shape.
    fn build(position: Vec3, mass: f32, restitution: f32, friction: f32, shape: Shape) -> Self {
        let props = Mass::from_kind(MassKind::from_f32(mass), &shape);
        Self {
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: props.mass(),
            inv_mass: props.inv_mass(),
            inertia: props.inertia(),
            torque: Vec3::ZERO,
            restitution,
            friction,
            friction_transverse: friction,
            friction_dir: None,
            rolling_friction: 0.0,
            torsion_friction: 0.0,
            shape,
            collision_layer: 1,
            collision_mask: u32::MAX,
            is_trigger: false,
            ccd_enabled: false,
            contact_force_threshold: f32::INFINITY,
            fracture_impact_speed: f32::INFINITY,
            body_type: if mass > 0.0 {
                BodyType::Dynamic
            } else {
                BodyType::Static
            },
        }
    }

    /// Consistent mass triple backing this body.
    pub fn mass_props(&self) -> Mass {
        Mass::from_kind(MassKind::from_f32(self.mass), &self.shape)
    }

    /// Current mass classification.
    pub fn mass_kind(&self) -> MassKind {
        MassKind::from_f32(self.mass)
    }

    /// Replace the mass model in one consistent write (`mass`, `inv_mass`,
    /// `inertia` and `body_type` move together).
    pub fn set_mass_kind(&mut self, kind: MassKind) {
        let props = Mass::from_kind(kind, &self.shape);
        self.mass = props.mass();
        self.inv_mass = props.inv_mass();
        self.inertia = props.inertia();
        self.body_type = if props.is_fixed() {
            BodyType::Static
        } else {
            BodyType::Dynamic
        };
    }

    /// Freeze for sleep: zero inverse mass/inertia without touching `mass`.
    pub(crate) fn sleep_staticify(&mut self) {
        self.velocity = Vec3::ZERO;
        self.angular_velocity = Vec3::ZERO;
        self.inv_mass = 0.0;
        self.inertia = Vec3::ZERO;
    }

    /// Undo [`RigidBody::sleep_staticify`]: restore `inv_mass`/`inertia`
    /// from `mass` + `shape`.
    pub(crate) fn wake_restore(&mut self) {
        if self.body_type == BodyType::Dynamic {
            let props = Mass::from_kind(MassKind::from_f32(self.mass), &self.shape);
            self.inv_mass = props.inv_mass();
            self.inertia = props.inertia();
        }
    }

    /// Restore a possibly zeroed sleep triple (migration path).
    pub(crate) fn restore_sleep_triple(&mut self) {
        if self.body_type == BodyType::Dynamic && self.inv_mass <= 0.0 {
            self.wake_restore();
        }
    }

    /// Replace the mass with a positive value in one consistent write
    /// (`mass`, `inv_mass`, `inertia` and `body_type` move together;
    /// the body becomes dynamic).
    pub fn set_mass(&mut self, mass: PositiveF32) {
        self.set_mass_kind(MassKind::Free(mass));
    }

    /// Make this body static (infinite mass) in one consistent write.
    pub fn make_static(&mut self) {
        self.set_mass_kind(MassKind::Fixed);
    }

    /// Make this body dynamic with a positive mass.
    pub fn make_dynamic(&mut self, mass: PositiveF32) {
        self.set_mass_kind(MassKind::Free(mass));
    }

    /// Typed view of [`RigidBody::friction_dir`]: `None` is isotropic,
    /// `Some` must hold a normalizable axis.
    pub fn friction_frame(&self) -> Result<FrictionFrame, FrictionFrameError> {
        FrictionFrame::from_option(self.friction_dir)
    }

    /// Checked setter for the anisotropy axis: rejects zero/non-finite
    /// input instead of silently falling back at solve time.
    pub fn set_friction_axis(&mut self, dir: Vec3) -> Result<(), FrictionFrameError> {
        match UnitVec3::normalize_checked(dir) {
            Some(u) => {
                self.friction_dir = Some(u.get());
                Ok(())
            }
            None => Err(FrictionFrameError),
        }
    }

    /// Clear the anisotropy axis (isotropic friction).
    pub fn clear_friction_axis(&mut self) {
        self.friction_dir = None;
    }

    /// Sphere body with default material ([`RESTITUTION_SPHERE`] /
    /// [`FRICTION_SPHERE`]). Mass > 0 yields a dynamic body; mass 0 a static
    /// one.
    pub fn new_sphere(position: Vec3, radius: f32, mass: f32) -> Self {
        Self::build(
            position,
            mass,
            RESTITUTION_SPHERE,
            FRICTION_SPHERE,
            Shape::Sphere { radius },
        )
    }

    /// Typed sphere entry point: radius as [`ornis_core::units::Meters`],
    /// mass as [`ornis_core::units::Kilograms`]. Returns `None` unless the
    /// radius is finite and `> 0` and the mass is a valid model input
    /// (finite, `>= 0`).
    pub fn try_sphere_units(
        position: Vec3,
        radius: ornis_core::units::Meters,
        mass: ornis_core::units::Kilograms,
    ) -> Option<Self> {
        if !(radius.get().is_finite() && radius.get() > 0.0) || !mass.is_valid() {
            return None;
        }
        Some(Self::new_sphere(position, radius.get(), mass.get()))
    }

    /// Typed capsule entry point (`None` unless both extents are finite and
    /// `> 0` and the mass is valid; same defaults as [`RigidBody::new_capsule`]).
    pub fn try_capsule_units(
        position: Vec3,
        radius: ornis_core::units::Meters,
        half_height: ornis_core::units::Meters,
        mass: ornis_core::units::Kilograms,
    ) -> Option<Self> {
        if !(radius.get().is_finite() && radius.get() > 0.0)
            || !(half_height.get().is_finite() && half_height.get() > 0.0)
            || !mass.is_valid()
        {
            return None;
        }
        Some(Self::new_capsule(
            position,
            radius.get(),
            half_height.get(),
            mass.get(),
        ))
    }

    /// Total mass in kilograms (`0` = static).
    pub fn mass_units(&self) -> ornis_core::units::Kilograms {
        ornis_core::units::Kilograms::new(self.mass)
    }

    /// Raw mass getter (kept for solver-adjacent code; prefer
    /// [`RigidBody::mass_units`] plus [`RigidBody::set_mass_kind`]).
    pub fn mass_raw(&self) -> f32 {
        self.mass
    }

    /// Linear velocity in m/s (same storage as [`RigidBody::velocity`];
    /// the name pins the unit at the type level).
    pub fn linear_velocity_mps(&self) -> Vec3 {
        self.velocity
    }

    /// Sets the linear velocity in m/s.
    pub fn set_linear_velocity_mps(&mut self, velocity: Vec3) {
        self.velocity = velocity;
    }

    /// Angular velocity in rad/s (same storage as
    /// [`RigidBody::angular_velocity`).
    pub fn angular_velocity_rad_s(&self) -> Vec3 {
        self.angular_velocity
    }

    /// Accumulated torque in N·m (same storage as [`RigidBody::torque`).
    pub fn torque_n_m(&self) -> Vec3 {
        self.torque
    }

    /// Axis-aligned box body (half-extents per axis), default material
    /// ([`RESTITUTION_DEFAULT`] / [`FRICTION_DEFAULT`]).
    pub fn new_box(position: Vec3, half_extents: Vec3, mass: f32) -> Self {
        Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            Shape::Box { half_extents },
        )
    }

    /// Capsule body aligned to the local +Y axis (`half_height` is the
    /// cylinder half-length excluding the caps), capsule material
    /// ([`RESTITUTION_CAPSULE`] / [`FRICTION_CAPSULE`]).
    pub fn new_capsule(position: Vec3, radius: f32, half_height: f32, mass: f32) -> Self {
        Self::build(
            position,
            mass,
            RESTITUTION_CAPSULE,
            FRICTION_CAPSULE,
            Shape::Capsule {
                radius,
                half_height,
            },
        )
    }

    /// Flat-capped cylinder body aligned to the local +Y axis, capsule
    /// material ([`RESTITUTION_CAPSULE`] / [`FRICTION_CAPSULE`]).
    pub fn new_cylinder(position: Vec3, radius: f32, half_height: f32, mass: f32) -> Self {
        Self::build(
            position,
            mass,
            RESTITUTION_CAPSULE,
            FRICTION_CAPSULE,
            Shape::Cylinder {
                radius,
                half_height,
            },
        )
    }

    /// Solid cone body (apex `+half_height` on local +Y, base at
    /// `-half_height`), default material ([`RESTITUTION_DEFAULT`] /
    /// [`FRICTION_DEFAULT`]).
    pub fn new_cone(position: Vec3, radius: f32, half_height: f32, mass: f32) -> Self {
        Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            Shape::Cone {
                radius,
                half_height,
            },
        )
    }

    /// Convex-hull body from local vertices (faces triangulated at
    /// construction), restitution 0.3, friction 0.5. Rolling/torsion
    /// damping default to 0.2/0.05 (MuJoCo-style torque caps in meters):
    /// hulls are debris — without rolling resistance a tetra rocks on its
    /// vertices/edges indefinitely (measured perch at 0.52 with μr=0.1,
    /// face settle at 0.408 with μr=0.2). Slide friction is untouched.
    ///
    /// Legacy infallible wrapper over [`RigidBody::try_new_convex_hull`]:
    /// non-finite vertices fall back to an empty hull so existing scenes
    /// are bit-identical. Deprecated — do not use in new code, kept only
    /// for compat; new code should use the `try_` variant.
    ///
    /// # Errors
    ///
    /// This wrapper never fails (see [`RigidBody::try_new_convex_hull`]
    /// for the fallible canonical path and its errors).
    pub fn new_convex_hull(position: Vec3, vertices: Vec<Vec3>, mass: f32) -> Self {
        Self::try_new_convex_hull(position, vertices, mass).unwrap_or_else(|_| {
            Self::build(
                position,
                mass,
                RESTITUTION_DEFAULT,
                FRICTION_DEFAULT,
                Shape::ConvexHull(crate::shape::ConvexHull {
                    vertices: Vec::new(),
                    faces: Vec::new(),
                }),
            )
        })
    }

    /// Checked convex-hull body: non-finite vertices are a typed error.
    ///
    /// # Errors
    ///
    /// [`crate::errors::MeshError::NonFiniteVertex`] on non-finite input.
    pub fn try_new_convex_hull(
        position: Vec3,
        vertices: Vec<Vec3>,
        mass: f32,
    ) -> Result<Self, crate::errors::MeshError> {
        let mut body = Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            Shape::ConvexHull(crate::shape::ConvexHull::from_vertices(vertices)?),
        );
        body.rolling_friction = ROLLING_FRICTION_DEBRIS;
        body.torsion_friction = TORSION_FRICTION_DEBRIS;
        Ok(body)
    }

    /// Heightfield terrain body (static use intended: pass mass 0).
    /// Restitution 0.3, friction 0.6.
    ///
    /// Legacy infallible wrapper over [`RigidBody::try_new_heightfield`]:
    /// an invalid grid description is kept verbatim (queries degrade to
    /// separation/`0.0`) so existing scenes are bit-identical. Deprecated —
    /// do not use in new code, kept only for compat; new code should use
    /// the `try_` variant.
    ///
    /// # Errors
    ///
    /// This wrapper never fails (see [`RigidBody::try_new_heightfield`]
    /// for the fallible canonical path and its errors).
    pub fn new_heightfield(
        position: Vec3,
        heights: Vec<f32>,
        rows: usize,
        cols: usize,
        cell: f32,
        mass: f32,
    ) -> Self {
        Self::try_new_heightfield(position, heights.clone(), rows, cols, cell, mass).unwrap_or_else(
            |_| {
                Self::build(
                    position,
                    mass,
                    RESTITUTION_DEFAULT,
                    FRICTION_TERRAIN,
                    Shape::Heightfield(crate::shape::Heightfield {
                        heights,
                        rows,
                        cols,
                        cell,
                    }),
                )
            },
        )
    }

    /// Checked heightfield body: validates `len == rows * cols`, grid
    /// extents and finite samples up front instead of degrading silently
    /// in queries.
    pub fn try_new_heightfield(
        position: Vec3,
        heights: Vec<f32>,
        rows: usize,
        cols: usize,
        cell: f32,
        mass: f32,
    ) -> Result<Self, crate::invariants::HeightfieldError> {
        let hf = crate::shape::Heightfield::new(heights, rows, cols, cell)?;
        Ok(Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_TERRAIN,
            Shape::Heightfield(hf),
        ))
    }

    /// Triangle-mesh collider body from a vertex soup and triangle
    /// indices (concave meshes welcome; static use intended: pass mass
    /// 0). Restitution 0.3, friction 0.5, rolling/torsion damping 0.2/0.05
    /// like hull debris. Inertia is exact (Mirtich) for closed, outwardly
    /// wound soup, bounding-box fallback otherwise.
    ///
    /// Legacy infallible wrapper over [`RigidBody::try_new_trimesh`]:
    /// dangling indices fall back to an empty mesh instead of panicking,
    /// so existing scenes are bit-identical on valid input. Deprecated —
    /// do not use in new code, kept only for compat; new code should use
    /// the `try_` variant.
    ///
    /// # Errors
    ///
    /// This wrapper never fails (see [`RigidBody::try_new_trimesh`]
    /// for the fallible canonical path and its errors).
    pub fn new_trimesh(
        position: Vec3,
        vertices: &[Vec3],
        triangles: &[crate::shape::Triangle],
        mass: f32,
    ) -> Self {
        Self::try_new_trimesh(position, vertices, triangles, mass).unwrap_or_else(|_| {
            match crate::shape::TriMesh::from_triangles(&[], &[]) {
                Ok(empty) => {
                    let mut body = Self::build(
                        position,
                        mass,
                        RESTITUTION_DEFAULT,
                        FRICTION_DEFAULT,
                        Shape::TriMesh(empty),
                    );
                    body.rolling_friction = ROLLING_FRICTION_DEBRIS;
                    body.torsion_friction = TORSION_FRICTION_DEBRIS;
                    body
                }
                Err(_) => Self::new_box(position, Vec3::splat(FALLBACK_BOX_HALF), mass),
            }
        })
    }

    /// Checked triangle-mesh body: dangling indices and non-finite
    /// vertices are a typed error instead of a panic.
    ///
    /// # Errors
    ///
    /// [`crate::errors::MeshError`] from [`crate::shape::TriMesh::from_triangles`].
    pub fn try_new_trimesh(
        position: Vec3,
        vertices: &[Vec3],
        triangles: &[crate::shape::Triangle],
        mass: f32,
    ) -> Result<Self, crate::errors::MeshError> {
        let mut body = Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            Shape::TriMesh(crate::shape::TriMesh::from_triangles(vertices, triangles)?),
        );
        body.rolling_friction = ROLLING_FRICTION_DEBRIS;
        body.torsion_friction = TORSION_FRICTION_DEBRIS;
        Ok(body)
    }

    /// Checked triangle-mesh body from a flat index list.
    ///
    /// Chunks `indices` into [`crate::shape::Triangle`] via
    /// [`crate::shape::TriMesh::from_indexed`].
    ///
    /// # Errors
    ///
    /// [`crate::errors::MeshError`] from [`crate::shape::TriMesh::from_indexed`].
    pub fn try_new_trimesh_indexed(
        position: Vec3,
        vertices: &[Vec3],
        indices: &[u32],
        mass: f32,
    ) -> Result<Self, crate::errors::MeshError> {
        let mut body = Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            Shape::TriMesh(crate::shape::TriMesh::from_indexed(vertices, indices)?),
        );
        body.rolling_friction = ROLLING_FRICTION_DEBRIS;
        body.torsion_friction = TORSION_FRICTION_DEBRIS;
        Ok(body)
    }

    /// Checked compound body from placed child shapes: each child owns a
    /// [`Pose`] in the compound frame (see [`Shape::try_compound`]).
    /// Default material ([`RESTITUTION_DEFAULT`] / [`FRICTION_DEFAULT`]);
    /// mass splits evenly across children with parallel-axis terms.
    ///
    /// # Errors
    ///
    /// [`crate::errors::ShapeError::EmptyCompound`] when `shapes` is empty.
    pub fn try_new_compound(
        position: Vec3,
        shapes: Vec<(Shape, Pose)>,
        mass: f32,
    ) -> Result<Self, crate::errors::ShapeError> {
        Ok(Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            Shape::try_compound(shapes)?,
        ))
    }

    /// Checked rounded body: `inner` dilated by `border_radius` (see
    /// [`Shape::try_round`]). Default material ([`RESTITUTION_DEFAULT`] /
    /// [`FRICTION_DEFAULT`]).
    ///
    /// # Errors
    ///
    /// [`crate::errors::ShapeError::BadBorderRadius`] unless the radius is
    /// finite and `> 0`.
    pub fn try_new_round(
        position: Vec3,
        inner: Shape,
        border_radius: f32,
        mass: f32,
    ) -> Result<Self, crate::errors::ShapeError> {
        let shape = Shape::try_round(inner, border_radius).ok_or(
            crate::errors::ShapeError::BadBorderRadius {
                radius: border_radius,
            },
        )?;
        Ok(Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_DEFAULT,
            shape,
        ))
    }

    /// Checked static half-space body: the infinite plane through
    /// `position` with outward `normal` (see [`Shape::HalfSpace`]).
    /// Restitution 0.3, friction 0.6 (floors want grip, like terrain).
    /// Static-only: an infinite plane has no finite inertia, so dynamic
    /// mass is a typed error instead of a silently grounded body.
    ///
    /// # Errors
    ///
    /// [`crate::errors::ShapeError::BadNormal`] on zero/non-finite input,
    /// [`crate::errors::ShapeError::DynamicHalfSpace`] on dynamic mass.
    pub fn try_new_halfspace(
        position: Vec3,
        normal: Vec3,
        mass: f32,
    ) -> Result<Self, crate::errors::ShapeError> {
        if !MassKind::from_f32(mass).is_fixed() {
            return Err(crate::errors::ShapeError::DynamicHalfSpace { mass });
        }
        let shape = Shape::try_halfspace(normal).ok_or(crate::errors::ShapeError::BadNormal)?;
        Ok(Self::build(
            position,
            mass,
            RESTITUTION_DEFAULT,
            FRICTION_TERRAIN,
            shape,
        ))
    }

    /// Builder-style collision-layer and mask configuration.
    ///
    /// A pair is eligible only when both directions agree: `self`'s mask
    /// contains `other`'s layer and vice versa. This keeps filtering
    /// symmetric even when a body uses a restrictive mask.
    pub fn with_collision_filter(mut self, layer: u32, mask: u32) -> Self {
        self.set_collision_filter(layer, mask);
        self
    }

    /// Changes the collision layer and mask in place.
    pub fn set_collision_filter(&mut self, layer: u32, mask: u32) {
        self.collision_layer = layer;
        self.collision_mask = mask;
    }

    /// Returns whether this body and `other` pass their mutual layer masks.
    pub fn can_collide_with(&self, other: &Self) -> bool {
        self.collision_mask & other.collision_layer != 0
            && other.collision_mask & self.collision_layer != 0
    }

    /// Builder-style trigger configuration.
    ///
    /// A trigger remains in broadphase overlap queries, but its contacts are
    /// reported as [`crate::TriggerEvent`] values instead of being solved.
    pub fn with_trigger(mut self, is_trigger: bool) -> Self {
        self.set_trigger(is_trigger);
        self
    }

    /// Builder-style collision-role configuration (typed replacement for
    /// [`Self::with_trigger`]).
    pub fn with_role(mut self, role: crate::flags::BodyRole) -> Self {
        self.set_role(role);
        self
    }

    /// Enables or disables overlap-event behavior in place.
    pub fn set_trigger(&mut self, is_trigger: bool) {
        self.set_role(crate::flags::BodyRole::from(is_trigger));
    }

    /// Sets the collision role in place (typed replacement for
    /// [`Self::set_trigger`]).
    pub fn set_role(&mut self, role: crate::flags::BodyRole) {
        self.is_trigger = role.is_trigger();
    }

    /// Current collision role as a [`crate::flags::BodyRole`].
    pub fn role(&self) -> crate::flags::BodyRole {
        crate::flags::BodyRole::from(self.is_trigger)
    }

    /// Builder-style bullet (full-CCD) configuration.
    ///
    /// A bullet always sweeps the nonlinear (rotating) TOI path,
    /// bypassing the travel gate; non-bullet bodies keep the legacy
    /// policy (linear sweep plus the gated angular sweep).
    pub fn with_ccd_enabled(mut self, ccd_enabled: bool) -> Self {
        self.set_ccd_enabled(ccd_enabled);
        self
    }

    /// Enables or disables the bullet (full-CCD) upgrade in place.
    pub fn set_ccd_enabled(&mut self, ccd_enabled: bool) {
        self.ccd_enabled = ccd_enabled;
    }

    /// Sets the contact-force report threshold (N) for this body (Rapier
    /// `ContactForceEventThreshold` parity): finite values opt in
    /// (negatives clamp to `0.0` — every touching pair reports),
    /// non-finite values opt out (`INFINITY`, the default).
    pub fn set_contact_force_threshold(&mut self, threshold: f32) {
        self.contact_force_threshold = if threshold.is_finite() {
            threshold.max(0.0)
        } else {
            f32::INFINITY
        };
    }

    /// Builder-style variant of [`Self::set_contact_force_threshold`].
    pub fn with_contact_force_threshold(mut self, threshold: f32) -> Self {
        self.set_contact_force_threshold(threshold);
        self
    }

    /// Disables contact-force reporting for this body (restores the
    /// `INFINITY` default).
    pub fn clear_contact_force_threshold(&mut self) {
        self.contact_force_threshold = f32::INFINITY;
    }

    /// `true` when this body opts into
    /// [`crate::trigger::ContactForceEvent`] reporting (finite threshold).
    pub fn contact_force_events_enabled(&self) -> bool {
        self.contact_force_threshold.is_finite()
    }

    /// `true` when this body carries the bullet (full-CCD) upgrade and
    /// takes part in dynamics (Rapier `is_bullet`: dynamic plus
    /// `ccd_enabled`).
    pub fn is_bullet(&self) -> bool {
        self.ccd_enabled && self.body_type == BodyType::Dynamic
    }

    /// Builder-style variant of [`RigidBody::set_orientation`].
    pub fn with_orientation(mut self, orientation: Quat) -> Self {
        self.orientation = orientation;
        self
    }

    /// Overwrite the rotation; must remain a unit quaternion.
    pub fn set_orientation(&mut self, orientation: Quat) {
        self.orientation = orientation;
    }

    /// Directly set the world-space angular velocity (rad/s).
    pub fn set_angular_velocity(&mut self, w: Vec3) {
        self.angular_velocity = w;
    }

    /// Typed alias of [`RigidBody::set_angular_velocity`]: the vector
    /// carries rad/s per component, pinned in the name (raw storage stays
    /// `Vec3` so the solver layout is untouched).
    pub fn set_angular_velocity_rad_s(&mut self, w_rad_s: Vec3) {
        self.set_angular_velocity(w_rad_s);
    }

    /// Apply a torque (N·m) to the body; takes effect on the next step.
    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque += torque;
    }

    /// Typed alias of [`RigidBody::apply_torque`]: the vector carries N·m
    /// per component, pinned in the name.
    pub fn apply_torque_n_m(&mut self, torque_n_m: Vec3) {
        self.apply_torque(torque_n_m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for a mutation-testing zombie (`cargo mutants`, night gate,
    /// 2026-08-24): `replace RigidBody::set_orientation with ()` survived —
    /// no test asserted that the setter actually mutates `orientation`.
    #[test]
    fn set_orientation_mutates_the_body() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.orientation, Quat::IDENTITY);
        let target = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        body.set_orientation(target);
        assert_eq!(body.orientation, target);
    }

    /// `with_orientation` is the builder counterpart of `set_orientation`;
    /// same zombie risk, covered separately since it consumes/returns `self`
    /// rather than mutating in place.
    #[test]
    fn with_orientation_sets_the_body_on_construction() {
        let target = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_orientation(target);
        assert_eq!(body.orientation, target);
    }

    /// `apply_torque` accumulates (`+=`); a mutant flipping it to `-=` or
    /// dropping the call entirely must be caught. Two calls in the same
    /// direction must sum, not overwrite or cancel.
    #[test]
    fn apply_torque_accumulates() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.torque, Vec3::ZERO);
        body.apply_torque(Vec3::new(1.0, 0.0, 0.0));
        body.apply_torque(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(body.torque, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn collision_filter_defaults_to_all_layers() {
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.collision_layer, 1);
        assert_eq!(body.collision_mask, u32::MAX);
        assert!(body.can_collide_with(&body));
    }

    #[test]
    fn collision_filter_requires_mutual_mask_match() {
        let a = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_collision_filter(0b0001, 0b0010);
        let b = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_collision_filter(0b0010, 0b0001);
        assert!(a.can_collide_with(&b));

        let blocked =
            RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_collision_filter(0b0100, 0b0001);
        assert!(!a.can_collide_with(&blocked));
        assert!(!blocked.can_collide_with(&a));
    }

    #[test]
    fn collision_filter_setter_updates_layer_and_mask() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        body.set_collision_filter(0b1000, 0b0100);
        assert_eq!(body.collision_layer, 0b1000);
        assert_eq!(body.collision_mask, 0b0100);
    }

    #[test]
    fn trigger_builder_and_setter_update_sensor_state() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0).with_trigger(true);
        assert!(body.is_trigger);
        body.set_trigger(false);
        assert!(!body.is_trigger);
    }

    /// `set_angular_velocity` is a straight setter with the same zombie
    /// shape as `set_orientation` — assert it actually takes effect.
    #[test]
    fn set_angular_velocity_mutates_the_body() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0);
        assert_eq!(body.angular_velocity, Vec3::ZERO);
        let w = Vec3::new(0.0, 1.0, 2.0);
        body.set_angular_velocity(w);
        assert_eq!(body.angular_velocity, w);
    }

    /// `build`'s `inv_mass` derivation (`1.0 / mass`, guarded for statics):
    /// a mutant flipping `/` to `*` or dropping the `mass > 0.0` guard must
    /// be caught on both the dynamic and static branches.
    #[test]
    fn build_derives_inverse_mass_for_dynamic_bodies() {
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0);
        assert_eq!(body.mass, 2.0);
        assert!((body.inv_mass - 0.5).abs() < 1e-6, "inv_mass = 1/mass");
        assert_eq!(body.body_type, BodyType::Dynamic);
    }

    /// Static bodies (mass == 0) must have `inv_mass == 0`, not `1/0`
    /// (infinity) or a silently wrong non-zero value.
    #[test]
    fn build_static_body_has_zero_inverse_mass() {
        let body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 0.0);
        assert_eq!(body.inv_mass, 0.0);
        assert_eq!(body.body_type, BodyType::Static);
    }

    /// `set_mass`/`make_static`/`make_dynamic` move the whole triple
    /// (`mass`, `inv_mass`, `inertia`, `body_type`) in one write — no
    /// manual resync.
    #[test]
    fn set_mass_moves_the_whole_triple() {
        let mut body = RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0);
        body.set_mass(PositiveF32::try_new(4.0).expect("valid mass"));
        assert_eq!(body.mass, 4.0);
        assert!((body.inv_mass - 0.25).abs() < 1e-6, "inv_mass = 1/mass");
        assert_eq!(body.body_type, BodyType::Dynamic);
        assert!(body.inertia.x > 0.0, "dynamic inertia is shape-derived");
        body.make_static();
        assert_eq!(body.mass, 0.0);
        assert_eq!(body.inv_mass, 0.0);
        assert_eq!(body.body_type, BodyType::Static);
        body.make_dynamic(PositiveF32::try_new(1.0).expect("valid mass"));
        assert_eq!(body.body_type, BodyType::Dynamic);
        assert!((body.inv_mass - 1.0).abs() < 1e-6);
    }

    #[test]
    fn halfspace_body_is_static_and_rejects_dynamics() {
        use crate::errors::ShapeError;
        // Dynamic mass is a typed error, never a silent grounding.
        assert_eq!(
            RigidBody::try_new_halfspace(Vec3::ZERO, Vec3::Y, 1.0).expect_err("dynamic refused"),
            ShapeError::DynamicHalfSpace { mass: 1.0 }
        );
        assert_eq!(
            RigidBody::try_new_halfspace(Vec3::ZERO, Vec3::ZERO, 0.0)
                .expect_err("zero normal refused"),
            ShapeError::BadNormal
        );
        let floor =
            RigidBody::try_new_halfspace(Vec3::ZERO, Vec3::Y, 0.0).expect("static plane builds");
        assert_eq!(floor.body_type, BodyType::Static);
        assert_eq!(floor.inv_mass, 0.0);
        assert!(matches!(floor.shape, Shape::HalfSpace { .. }));
    }

    #[test]
    fn compound_and_round_bodies_reject_degenerates() {
        use crate::errors::ShapeError;
        assert_eq!(
            RigidBody::try_new_compound(Vec3::ZERO, vec![], 1.0).expect_err("empty refused"),
            ShapeError::EmptyCompound
        );
        assert_eq!(
            RigidBody::try_new_round(Vec3::ZERO, Shape::Sphere { radius: 1.0 }, 0.0, 1.0)
                .expect_err("zero radius refused"),
            ShapeError::BadBorderRadius { radius: 0.0 }
        );
        let body = RigidBody::try_new_compound(
            Vec3::ZERO,
            vec![(
                Shape::Box {
                    half_extents: Vec3::ONE,
                },
                Pose::IDENTITY,
            )],
            2.0,
        )
        .expect("single-box compound builds");
        assert_eq!(body.body_type, BodyType::Dynamic);
        assert_eq!(
            body.inertia,
            Shape::Box {
                half_extents: Vec3::ONE
            }
            .inertia(2.0)
        );
    }
}
