//! Deformable bodies (PLAN B2/D1): particles and distance constraints.
//!
//! [`Particle`] is a 3-DOF point mass — deliberately not a [`crate::body::RigidBody`]:
//! hauling a quaternion and an inertia tensor per cloth vertex is wasteful
//! and meaningless. [`SoftBody`] bundles particles with a distance-constraint
//! topology (structural/shear/bend groups, each with its own compliance).
//! The solver is XPBD on the shared [`crate::xpbd`] kernel: constraints are
//! equalities solved with `Δλ = (−C − α̃λ)/(w + α̃)`, `α̃ = α/h²`, one
//! Lagrange multiplier per scalar row, reset every substep (Small-Steps
//! regime — see [`crate::xpbd`]).
//!
//! Builders cover the first two D1 scenes: [`SoftBody::chain`] (rope/cable)
//! and [`SoftBody::cloth_grid`] (draping sheet). Volume preservation,
//! deformable↔rigid coupling and render-mesh upload are later D1 steps and
//! explicitly absent here.

use glam::Vec3;

/// Stable index of a soft body inside [`crate::xpbd::XpbdEngine`].
///
/// Dense like [`crate::body::BodyHandle`]: removal swaps the last body into
/// the freed slot, so only the moved body's handle changes.
///
/// Newtype over `u32` so soft-body handles never mix with rigid-body or
/// joint handles at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoftHandle(u32);

impl SoftHandle {
    /// Wraps a raw `u32` soft-body index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` soft-body index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Soft-body index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for SoftHandle {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for SoftHandle {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<SoftHandle> for u32 {
    fn from(h: SoftHandle) -> Self {
        h.0
    }
}

impl From<SoftHandle> for usize {
    fn from(h: SoftHandle) -> Self {
        h.0 as usize
    }
}

/// Index of one particle inside its owning [`SoftBody`].
///
/// Newtype over `u32` so particle indices never mix with body handles or
/// raw triangle soup indices at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParticleIdx(u32);

impl ParticleIdx {
    /// Wraps a raw `u32` particle index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` particle index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Particle index as `usize` for slice lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for ParticleIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for ParticleIdx {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<ParticleIdx> for u32 {
    fn from(h: ParticleIdx) -> Self {
        h.0
    }
}

impl From<ParticleIdx> for usize {
    fn from(h: ParticleIdx) -> Self {
        h.0 as usize
    }
}

/// Constraint group inside a deformable: structural/shear/bend rows share
/// one formula and differ only in topology and compliance (stiff stretch,
/// soft bend — the cloth recipe from the XPBD paper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeformKind {
    /// Primary topology (chain links, cloth warp/weft).
    Structural,
    /// Diagonals resisting in-plane shear.
    Shear,
    /// Skip-one rows resisting bending (soft).
    Bend,
}

/// One 3-DOF point mass of a [`SoftBody`].
///
/// `inv_mass == 0` pins the particle (static anchor): the solver skips it
/// and it never integrates. `prev_position` is the substep-start pose used
/// for the BDF1-style velocity update.
#[derive(Debug, Clone)]
pub struct Particle {
    /// World-space position (solved state).
    pub position: Vec3,
    /// Position at the substep start (velocity baseline).
    pub prev_position: Vec3,
    /// Linear velocity (m/s).
    pub velocity: Vec3,
    /// Cached `1 / mass` (0 = pinned) — what the solver actually uses.
    pub inv_mass: f32,
}

impl Particle {
    /// Free particle of `mass` at `position` (`mass <= 0` pins it).
    ///
    /// Legacy infallible wrapper over [`Particle::from_kind`]: non-positive
    /// input pins the particle instead of failing, so existing builders are
    /// bit-identical; new code should use [`Particle::try_new`] (checked
    /// free particle) or [`Particle::pinned`] for anchors. Deprecated —
    /// do not use in new code, kept only for compat.
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self::from_kind(position, crate::invariants::MassKind::from_f32(mass))
    }

    /// Checked free particle at `position`: `Some` only when `mass` is
    /// finite and `> 0`; `None` (instead of a silent pin) otherwise.
    pub fn try_new(position: Vec3, mass: f32) -> Option<Self> {
        let kind = crate::invariants::MassKind::try_free(mass)?;
        Some(Self::from_kind(position, kind))
    }

    /// Pinned anchor particle at `position`.
    pub fn pinned(position: Vec3) -> Self {
        Self::from_kind(position, crate::invariants::MassKind::Fixed)
    }

    /// Free particle with a statically checked positive mass.
    pub fn free(position: Vec3, mass: crate::invariants::PositiveF32) -> Self {
        Self::from_kind(position, crate::invariants::MassKind::Free(mass))
    }

    /// Particle from an explicit [`crate::invariants::MassKind`].
    pub fn from_kind(position: Vec3, kind: crate::invariants::MassKind) -> Self {
        Self {
            position,
            prev_position: position,
            velocity: Vec3::ZERO,
            inv_mass: kind.inv_mass_value(),
        }
    }

    /// Current mass classification.
    pub fn mass_kind(&self) -> crate::invariants::MassKind {
        if self.inv_mass <= 0.0 {
            crate::invariants::MassKind::Fixed
        } else {
            match crate::invariants::PositiveF32::try_new(1.0 / self.inv_mass) {
                Some(m) => crate::invariants::MassKind::Free(m),
                None => crate::invariants::MassKind::Fixed,
            }
        }
    }

    /// Pin this particle in place (zero inverse mass).
    pub fn pin(&mut self) {
        self.inv_mass = 0.0;
        self.velocity = Vec3::ZERO;
    }

    /// Whether the solver treats this particle as an immovable anchor.
    pub fn is_pinned(&self) -> bool {
        self.inv_mass <= 0.0
    }
}

/// One scalar distance constraint between two particles of the same body.
#[derive(Debug, Clone)]
pub struct DeformConstraint {
    /// First particle index.
    pub a: ParticleIdx,
    /// Second particle index.
    pub b: ParticleIdx,
    /// Rest length (m).
    pub rest: f32,
    /// Compliance `α` (inverse stiffness in m/N, 0 = inextensible).
    pub compliance: f32,
    /// Topology group (structural/shear/bend).
    pub kind: DeformKind,
    /// Accumulated multiplier, live only within one substep's iterations.
    pub lambda: f32,
}

/// Which particles of a [`SoftBody::cloth_grid`] start pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClothPin {
    /// Nothing pinned (free fall).
    None,
    /// The whole top row (`row == 0`) pinned — curtain scenario.
    TopRow,
    /// Only the two top corners pinned — drape scenario.
    Corners,
}

/// A deformable body: particles plus a distance-constraint topology.
///
/// Constraints reference particle indices of this body only (no cross-body
/// rows in D1.1); invalid indices are rejected by the builders, and the
/// solver skips degenerate rows defensively. `triangles` is an optional
/// closed, consistently-wound surface driving the global volume constraint
/// (D1.3, XPBD balloon model — zero triangles disables it).
#[derive(Debug, Clone)]
pub struct SoftBody {
    /// Particles in index order.
    pub particles: Vec<Particle>,
    /// Distance rows over the particles.
    pub constraints: Vec<DeformConstraint>,
    /// Closed surface triangles (outward-wound) for the volume row.
    pub triangles: Vec<[ParticleIdx; 3]>,
    /// Render-only surface topology (D1.5): triangle indices into
    /// `particles`, wound CCW from outside (same convention as the
    /// asset-side `Custom` mesh soup). Unlike `triangles` this may describe
    /// an OPEN sheet (cloth) — it never drives physics, only the per-frame
    /// mesh upload. Bodies without a sheet (chains) leave it empty.
    pub surface: Vec<[ParticleIdx; 3]>,
    /// Rest volume (m³) captured at build time.
    pub volume_rest: f32,
    /// Volume compliance `α` (0 = incompressible).
    pub volume_compliance: f32,
    /// Accumulated volume multiplier, live within one substep only.
    pub volume_lambda: f32,
    /// Velocity damping rate (1/s, 0 = undamped): velocities retain
    /// `exp(−damping·h)` per substep — a timestep-independent exponential
    /// decay, unlike a per-substep fraction (which would make the terminal
    /// velocity depend on the substep count). XPBD projection is nearly
    /// energy-preserving, so undamped bodies jiggle around equilibrium
    /// instead of settling — production solvers (Vellum included) all
    /// carry this control. Negative values are clamped to 0 on use;
    /// builders default to 0.
    pub damping: f32,
    /// Particle radius (m) for deformable↔rigid coupling (D1.4): each
    /// particle collides as a sphere of this radius via `shape_distance`.
    /// Zero disables the coupling for the whole body.
    pub contact_radius: f32,
    /// Breakage threshold as a stretch ratio `dist / rest` (0 = unbreakable,
    /// the default). When [`SoftBody::apply_breakage`] runs, structural rows
    /// stretched beyond this ratio are removed. E.g. `2.0` tears rows pulled
    /// past twice their rest length. Shear/bend rows never tear in D1.
    pub tear_strain: f32,
}

impl SoftBody {
    /// Bare body without a volume surface (volume row disabled).
    fn raw(particles: Vec<Particle>, constraints: Vec<DeformConstraint>) -> Self {
        Self {
            particles,
            constraints,
            triangles: Vec::new(),
            surface: Vec::new(),
            volume_rest: 0.0,
            volume_compliance: 0.0,
            volume_lambda: 0.0,
            damping: 0.0,
            contact_radius: 0.0,
            tear_strain: 0.0,
        }
    }

    /// Rope/cable: `count` particles from `origin` along `dir` (normalized
    /// internally; a zero/non-finite `dir` falls back to `-Y`) spaced
    /// `spacing` apart, each of `mass`, linked by structural rows of
    /// `compliance`. The first particle is pinned.
    pub fn chain(
        origin: Vec3,
        dir: Vec3,
        count: usize,
        spacing: f32,
        mass: f32,
        compliance: f32,
    ) -> Self {
        let dir = dir.normalize_or(Vec3::NEG_Y);
        let mut particles = Vec::with_capacity(count);
        for i in 0..count {
            let pos = origin + dir * (i as f32 * spacing);
            let mut p = Particle::try_new(pos, mass).unwrap_or_else(|| Particle::pinned(pos));
            if i == 0 {
                p.pin();
            }
            particles.push(p);
        }
        let mut constraints = Vec::with_capacity(count.saturating_sub(1));
        for i in 0..count.saturating_sub(1) {
            constraints.push(DeformConstraint {
                a: ParticleIdx::from(i),
                b: ParticleIdx::from(i + 1),
                rest: spacing,
                compliance,
                kind: DeformKind::Structural,
                lambda: 0.0,
            });
        }
        let mut body = Self::raw(particles, constraints);
        body.contact_radius = spacing * 0.2;
        body
    }

    /// Typed rope entry point: spacing as [`ornis_core::units::Meters`],
    /// per-particle mass as [`ornis_core::units::Kilograms`]. Returns `None`
    /// unless the spacing and mass are positive and finite and the
    /// compliance is finite and `>= 0`.
    pub fn try_chain_units(
        origin: Vec3,
        dir: Vec3,
        count: usize,
        spacing: ornis_core::units::Meters,
        mass: ornis_core::units::Kilograms,
        compliance: f32,
    ) -> Option<Self> {
        let (s, m) = (spacing.get(), mass.get());
        if !(s.is_finite() && s > 0.0)
            || !(m.is_finite() && m > 0.0)
            || !compliance.is_finite()
            || compliance < 0.0
        {
            return None;
        }
        Some(Self::chain(origin, dir, count, s, m, compliance))
    }

    /// Spacing-derived contact radius in meters.
    pub fn contact_radius_units(&self) -> ornis_core::units::Meters {
        ornis_core::units::Meters::new(self.contact_radius)
    }

    /// Cloth sheet in the local XY plane: `cols × rows` particles from
    /// `origin` (top-left corner), spaced `spacing` apart, each of `mass`.
    /// Structural rows link grid neighbours, shear rows link diagonals and
    /// bend rows skip one vertex — each group with its own compliance.
    /// `pin` selects the initially pinned set.
    #[allow(clippy::too_many_arguments)]
    pub fn cloth_grid(
        origin: Vec3,
        cols: usize,
        rows: usize,
        spacing: f32,
        mass: f32,
        structural: f32,
        shear: f32,
        bend: f32,
        pin: ClothPin,
    ) -> Self {
        let at = |c: usize, r: usize| ParticleIdx::from(r * cols + c);
        let pinned = |c: usize, r: usize| match pin {
            ClothPin::None => false,
            ClothPin::TopRow => r == 0,
            ClothPin::Corners => r == 0 && (c == 0 || c + 1 == cols),
        };
        let mut particles = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                let pos = origin + Vec3::new(c as f32 * spacing, -(r as f32) * spacing, 0.0);
                let mut p =
                    Particle::try_new(pos, mass).unwrap_or_else(|| Particle::pinned(pos));
                if pinned(c, r) {
                    p.pin();
                }
                particles.push(p);
            }
        }
        let mut constraints = Vec::new();
        let mut surface = Vec::new();
        let mut link =
            |a: ParticleIdx, b: ParticleIdx, rest: f32, compliance: f32, kind: DeformKind| {
                if a.index() < particles.len() && b.index() < particles.len() && a != b {
                    constraints.push(DeformConstraint {
                        a,
                        b,
                        rest,
                        compliance,
                        kind,
                        lambda: 0.0,
                    });
                }
            };
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    link(
                        at(c, r),
                        at(c + 1, r),
                        spacing,
                        structural,
                        DeformKind::Structural,
                    );
                }
                if r + 1 < rows {
                    link(
                        at(c, r),
                        at(c, r + 1),
                        spacing,
                        structural,
                        DeformKind::Structural,
                    );
                }
                if c + 1 < cols && r + 1 < rows {
                    let d = spacing * std::f32::consts::SQRT_2;
                    link(at(c, r), at(c + 1, r + 1), d, shear, DeformKind::Shear);
                    link(at(c + 1, r), at(c, r + 1), d, shear, DeformKind::Shear);
                    // Render sheet: two triangles per cell, CCW from +z.
                    surface.push([at(c, r), at(c, r + 1), at(c + 1, r + 1)]);
                    surface.push([at(c, r), at(c + 1, r + 1), at(c + 1, r)]);
                }
                if c + 2 < cols {
                    link(
                        at(c, r),
                        at(c + 2, r),
                        2.0 * spacing,
                        bend,
                        DeformKind::Bend,
                    );
                }
                if r + 2 < rows {
                    link(
                        at(c, r),
                        at(c, r + 2),
                        2.0 * spacing,
                        bend,
                        DeformKind::Bend,
                    );
                }
            }
        }
        let mut body = Self::raw(particles, constraints);
        body.contact_radius = spacing * 0.2;
        body.surface = surface;
        body
    }

    /// Typed cloth entry point (`None` unless spacing/mass are positive
    /// finite and all compliances are finite and `>= 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn try_cloth_grid_units(
        origin: Vec3,
        cols: usize,
        rows: usize,
        spacing: ornis_core::units::Meters,
        mass: ornis_core::units::Kilograms,
        structural: f32,
        shear: f32,
        bend: f32,
        pin: ClothPin,
    ) -> Option<Self> {
        let (s, m) = (spacing.get(), mass.get());
        if !(s.is_finite() && s > 0.0) || !(m.is_finite() && m > 0.0) {
            return None;
        }
        for c in [structural, shear, bend] {
            if !c.is_finite() || c < 0.0 {
                return None;
            }
        }
        Some(Self::cloth_grid(
            origin, cols, rows, s, m, structural, shear, bend, pin,
        ))
    }

    /// Soft cube of edge `size` at `origin` (minimum corner): 8 particles,
    /// 12 structural edges of `edge_compliance`, and a closed 12-triangle
    /// surface driving the global volume row of `volume_compliance`
    /// (0 = incompressible). Triangle winding is fixed up programmatically
    /// (each face must point away from the centroid), so the builder is
    /// correct by construction rather than by hand-checked order.
    pub fn soft_cube(
        origin: Vec3,
        size: f32,
        mass: f32,
        edge_compliance: f32,
        volume_compliance: f32,
    ) -> Self {
        let corner = |x: u32, y: u32, z: u32| {
            origin + Vec3::new(x as f32 * size, y as f32 * size, z as f32 * size)
        };
        let mut particles = Vec::with_capacity(8);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let pos = corner(x, y, z);
                    particles.push(
                        Particle::try_new(pos, mass).unwrap_or_else(|| Particle::pinned(pos)),
                    );
                }
            }
        }
        // Index = x + 2*y + 4*z.
        let mut constraints = Vec::with_capacity(12);
        for a in 0..8usize {
            for b in (a + 1)..8 {
                let diff = (a ^ b) as u32;
                // Exactly one coordinate differs: cube edge.
                if diff == 1 || diff == 2 || diff == 4 {
                    constraints.push(DeformConstraint {
                        a: ParticleIdx::from(a),
                        b: ParticleIdx::from(b),
                        rest: size,
                        compliance: edge_compliance,
                        kind: DeformKind::Structural,
                        lambda: 0.0,
                    });
                }
            }
        }
        // Six quad faces as corner loops; triangulated + outward-fixed below.
        let quads: [[ParticleIdx; 4]; 6] = [
            [
                ParticleIdx::from_raw(1),
                ParticleIdx::from_raw(3),
                ParticleIdx::from_raw(7),
                ParticleIdx::from_raw(5),
            ],
            [
                ParticleIdx::from_raw(0),
                ParticleIdx::from_raw(4),
                ParticleIdx::from_raw(6),
                ParticleIdx::from_raw(2),
            ],
            [
                ParticleIdx::from_raw(2),
                ParticleIdx::from_raw(6),
                ParticleIdx::from_raw(7),
                ParticleIdx::from_raw(3),
            ],
            [
                ParticleIdx::from_raw(0),
                ParticleIdx::from_raw(1),
                ParticleIdx::from_raw(5),
                ParticleIdx::from_raw(4),
            ],
            [
                ParticleIdx::from_raw(4),
                ParticleIdx::from_raw(5),
                ParticleIdx::from_raw(7),
                ParticleIdx::from_raw(6),
            ],
            [
                ParticleIdx::from_raw(0),
                ParticleIdx::from_raw(2),
                ParticleIdx::from_raw(3),
                ParticleIdx::from_raw(1),
            ],
        ];
        let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let center = positions.iter().sum::<Vec3>() / positions.len() as f32;
        let mut triangles: Vec<[ParticleIdx; 3]> = Vec::with_capacity(12);
        for [a, b, c, d] in quads {
            for (x, mut y, mut z) in [(a, b, c), (a, c, d)] {
                let n = (positions[y.index()] - positions[x.index()])
                    .cross(positions[z.index()] - positions[x.index()]);
                let face_center =
                    (positions[x.index()] + positions[y.index()] + positions[z.index()]) / 3.0;
                if n.dot(face_center - center) < 0.0 {
                    std::mem::swap(&mut y, &mut z);
                }
                triangles.push([x, y, z]);
            }
        }
        let volume_rest = mesh_volume(&positions, &triangles).abs();
        let mut body = Self::raw(particles, constraints);
        body.triangles = triangles.clone();
        body.surface = triangles;
        body.volume_rest = volume_rest;
        body.volume_compliance = volume_compliance;
        body.contact_radius = size * 0.1;
        body
    }

    /// Typed soft-cube entry point (`None` unless the edge size and mass
    /// are positive finite and both compliances are finite and `>= 0`).
    pub fn try_soft_cube_units(
        origin: Vec3,
        size: ornis_core::units::Meters,
        mass: ornis_core::units::Kilograms,
        edge_compliance: f32,
        volume_compliance: f32,
    ) -> Option<Self> {
        let (s, m) = (size.get(), mass.get());
        if !(s.is_finite() && s > 0.0) || !(m.is_finite() && m > 0.0) {
            return None;
        }
        for c in [edge_compliance, volume_compliance] {
            if !c.is_finite() || c < 0.0 {
                return None;
            }
        }
        Some(Self::soft_cube(
            origin,
            s,
            m,
            edge_compliance,
            volume_compliance,
        ))
    }

    /// Number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Number of distance rows.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Semi-implicit Euler prediction for one substep of size `h` under
    /// `gravity`: pinned particles only refresh their baseline.
    pub fn integrate(&mut self, h: f32, gravity: Vec3) {
        for p in &mut self.particles {
            p.prev_position = p.position;
            if p.inv_mass <= 0.0 {
                p.velocity = Vec3::ZERO;
                continue;
            }
            p.velocity += h * gravity;
            p.position += h * p.velocity;
        }
    }

    /// Gauss–Seidel sweep over the distance rows (equalities): each row
    /// evaluates `C = |pa − pb| − rest` at the live positions and applies
    /// the XPBD update. Multipliers start at 0 every substep (reset by
    /// [`SoftBody::begin_substep`], not here — iterations within one
    /// substep accumulate).
    pub fn solve_constraints(&mut self, h: f32) {
        for c in &mut self.constraints {
            let (pa, pb) = match pair(&self.particles, c.a, c.b) {
                Some(pair) => pair,
                None => continue,
            };
            let delta = pa.position - pb.position;
            let dist = delta.length();
            if dist < 1e-9 {
                continue;
            }
            let n = delta / dist;
            let w = pa.inv_mass + pb.inv_mass;
            if w <= 0.0 {
                continue;
            }
            let alpha_tilde = c.compliance / (h * h);
            let dlambda = crate::xpbd::delta_lambda(dist - c.rest, c.lambda, w, alpha_tilde);
            c.lambda += dlambda;
            let (pa, pb) = match pair_mut(&mut self.particles, c.a, c.b) {
                Some(pair) => pair,
                None => continue,
            };
            if dlambda != 0.0 {
                pa.position += n * (dlambda * pa.inv_mass);
                pb.position -= n * (dlambda * pb.inv_mass);
            }
        }
    }

    /// Reset per-substep multiplier state (Small-Steps: every substep is a
    /// fresh XPBD solve starting from `λ = 0`).
    pub fn begin_substep(&mut self) {
        for c in &mut self.constraints {
            c.lambda = 0.0;
        }
        self.volume_lambda = 0.0;
    }

    /// BDF1-style velocity update from the solved positions, with the
    /// body's exponential velocity damping applied.
    pub fn update_velocities(&mut self, h: f32) {
        let retain = (-self.damping.max(0.0) * h).exp();
        for p in &mut self.particles {
            if p.inv_mass <= 0.0 {
                p.velocity = Vec3::ZERO;
                continue;
            }
            p.velocity = (p.position - p.prev_position) / h * retain;
        }
    }

    /// Tears overstretched structural rows (separate pass, D1 leftover #2).
    ///
    /// Removes [`DeformKind::Structural`] rows whose live stretch ratio
    /// `dist / rest` exceeds [`SoftBody::tear_strain`], then drops every
    /// [`SoftBody::surface`] triangle containing both endpoints of any torn
    /// row (the sheet gets a hole). Does nothing when `tear_strain <= 0`
    /// (unbreakable). Shear/bend rows never tear in D1, `triangles` (the
    /// physics volume surface) is left intact, and particles — pinned or
    /// not — are never added, removed, or unpinned.
    ///
    /// This is holes-not-splits, without particle duplication: the two sides
    /// of a tear keep sharing the same particles, so a tear opens as missing
    /// triangles rather than two clean lips. The upgrade path (duplicating
    /// particles along the tear for clean cuts) is out of scope.
    ///
    /// Call this once per step, outside the solver iterations — never inside
    /// the [`SoftBody::solve_constraints`] sweep, which would fight the XPBD
    /// projection within one substep.
    pub fn apply_breakage(&mut self) {
        if self.tear_strain <= 0.0 {
            return;
        }
        let threshold = self.tear_strain;
        let particles = &self.particles;
        let mut torn: Vec<(ParticleIdx, ParticleIdx)> = Vec::new();
        self.constraints.retain(|c| {
            if c.kind != DeformKind::Structural {
                return true;
            }
            if c.rest <= 1e-9 {
                return true;
            }
            let (pa, pb) = match pair(particles, c.a, c.b) {
                Some(pair) => pair,
                None => return true,
            };
            let dist = (pa.position - pb.position).length();
            if dist / c.rest > threshold {
                torn.push((c.a, c.b));
                false
            } else {
                true
            }
        });
        if torn.is_empty() {
            return;
        }
        self.surface
            .retain(|tri| !torn.iter().any(|(a, b)| tri.contains(a) && tri.contains(b)));
    }

    /// Global volume row (XPBD balloon model, Macklin et al. 2016 §6.5):
    /// `C = (V − V0)/V0` over the closed surface, solved as one equality.
    /// Skipped without triangles or with a degenerate rest volume.
    pub fn solve_volume(&mut self, h: f32) {
        if self.triangles.is_empty() || self.volume_rest <= 1e-12 {
            return;
        }
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        let volume = mesh_volume(&positions, &self.triangles);
        let c = (volume - self.volume_rest) / self.volume_rest;
        // Per-particle gradients, accumulated over adjacent triangles:
        // triangle (a,b,c) contributes (pb×pc)/6, (pc×pa)/6, (pa×pb)/6.
        let mut grads = vec![Vec3::ZERO; self.particles.len()];
        let mut valid = true;
        for [a, b, c] in &self.triangles {
            if a.index() >= self.particles.len()
                || b.index() >= self.particles.len()
                || c.index() >= self.particles.len()
            {
                valid = false;
                break;
            }
            let (pa, pb, pc) = (
                positions[a.index()],
                positions[b.index()],
                positions[c.index()],
            );
            grads[a.index()] += pb.cross(pc) / (6.0 * self.volume_rest);
            grads[b.index()] += pc.cross(pa) / (6.0 * self.volume_rest);
            grads[c.index()] += pa.cross(pb) / (6.0 * self.volume_rest);
        }
        if !valid {
            return;
        }
        let mut w = 0.0;
        for (p, g) in self.particles.iter().zip(&grads) {
            w += p.inv_mass * g.length_squared();
        }
        if w <= 0.0 {
            return;
        }
        let alpha_tilde = self.volume_compliance / (h * h);
        let dlambda = crate::xpbd::delta_lambda(c, self.volume_lambda, w, alpha_tilde);
        self.volume_lambda += dlambda;
        if dlambda != 0.0 {
            for (p, g) in self.particles.iter_mut().zip(&grads) {
                p.position += *g * (dlambda * p.inv_mass);
            }
        }
    }

    /// Live volume of the closed surface (m³, signed by winding).
    pub fn volume(&self) -> f32 {
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        mesh_volume(&positions, &self.triangles)
    }

    /// World-space particle positions in index order (render upload
    /// source): pair with [`SoftBody::surface`] indices to build the mesh
    /// soup. Allocates — the bridge calls it once per frame per body.
    pub fn positions_snapshot(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }
}

/// Signed volume of a closed triangle surface (divergence theorem):
/// `V = Σ (pa × pb)·pc / 6`. Positive for outward-wound meshes.
fn mesh_volume(positions: &[Vec3], triangles: &[[ParticleIdx; 3]]) -> f32 {
    let mut volume = 0.0;
    for [a, b, c] in triangles {
        if let (Some(pa), Some(pb), Some(pc)) = (
            positions.get(a.index()),
            positions.get(b.index()),
            positions.get(c.index()),
        ) {
            volume += pa.cross(*pb).dot(*pc) / 6.0;
        }
    }
    volume
}

/// Shared read of two distinct particles (`None` for bad indices).
fn pair(particles: &[Particle], a: ParticleIdx, b: ParticleIdx) -> Option<(&Particle, &Particle)> {
    let (a, b) = (a.index(), b.index());
    if a == b || a >= particles.len() || b >= particles.len() {
        return None;
    }
    Some((&particles[a], &particles[b]))
}

/// Exclusive mutable borrow of two distinct particles.
fn pair_mut(
    particles: &mut [Particle],
    a: ParticleIdx,
    b: ParticleIdx,
) -> Option<(&mut Particle, &mut Particle)> {
    let (a, b) = (a.index(), b.index());
    if a == b || a >= particles.len() || b >= particles.len() {
        return None;
    }
    if a < b {
        let (lo, hi) = particles.split_at_mut(b);
        Some((&mut lo[a], &mut hi[0]))
    } else {
        let (lo, hi) = particles.split_at_mut(a);
        Some((&mut hi[0], &mut lo[b]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checked particle construction: `try_new` admits only positive
    /// finite masses, while the legacy `new` pins on bad input.
    #[test]
    fn try_new_rejects_non_positive_mass() {
        assert!(Particle::try_new(Vec3::ZERO, 1.0).is_some());
        assert!(Particle::try_new(Vec3::ZERO, 0.0).is_none());
        assert!(Particle::try_new(Vec3::ZERO, -1.0).is_none());
        assert!(Particle::try_new(Vec3::ZERO, f32::NAN).is_none());
        assert!(Particle::new(Vec3::ZERO, 0.0).is_pinned());
    }

    /// Builder topology: chain links neighbours, pins the first particle.
    #[test]
    fn chain_builder_links_and_pins() {
        let body = SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 6, 0.5, 1.0, 0.0);
        assert_eq!(body.particle_count(), 6);
        assert_eq!(body.constraint_count(), 5);
        assert!(body.particles[0].is_pinned());
        assert!(
            body.constraints
                .iter()
                .all(|c| c.kind == DeformKind::Structural)
        );
        assert!(body.particles[1..].iter().all(|p| !p.is_pinned()));
    }

    /// Builder topology: grid counts per group + top-row pinning.
    #[test]
    fn cloth_grid_builder_counts_and_pins() {
        let (cols, rows) = (5, 4);
        let body = SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            0.25,
            1.0,
            0.0,
            0.0,
            1e-4,
            ClothPin::TopRow,
        );
        let structural = 2 * cols * rows - cols - rows;
        let shear = 2 * (cols - 1) * (rows - 1);
        let bend = (cols - 2) * rows + cols * (rows - 2);
        assert_eq!(body.particle_count(), cols * rows);
        assert_eq!(body.constraint_count(), structural + shear + bend);
        assert_eq!(
            body.particles.iter().filter(|p| p.is_pinned()).count(),
            cols,
            "whole top row pinned"
        );
    }

    /// Degenerate rows (self-links, bad indices, coincident particles) are
    /// skipped, never panicking or producing NaN.
    #[test]
    fn degenerate_rows_are_skipped() {
        let mut body = SoftBody::raw(
            vec![
                Particle::new(Vec3::ZERO, 1.0),
                Particle::new(Vec3::ZERO, 1.0),
            ],
            vec![
                DeformConstraint {
                    a: ParticleIdx::from_raw(0),
                    b: ParticleIdx::from_raw(0),
                    rest: 1.0,
                    compliance: 0.0,
                    kind: DeformKind::Structural,
                    lambda: 0.0,
                },
                DeformConstraint {
                    a: ParticleIdx::from_raw(0),
                    b: ParticleIdx::from_raw(7),
                    rest: 1.0,
                    compliance: 0.0,
                    kind: DeformKind::Structural,
                    lambda: 0.0,
                },
                DeformConstraint {
                    a: ParticleIdx::from_raw(0),
                    b: ParticleIdx::from_raw(1),
                    rest: 1.0,
                    compliance: 0.0,
                    kind: DeformKind::Structural,
                    lambda: 0.0,
                },
            ],
        );
        body.integrate(1.0 / 1200.0, Vec3::new(0.0, -9.81, 0.0));
        body.solve_constraints(1.0 / 1200.0);
        body.update_velocities(1.0 / 1200.0);
        for p in &body.particles {
            assert!(p.position.is_finite() && p.velocity.is_finite());
        }
    }

    /// Render surface (D1.5): cloth carries two CCW triangles per cell with
    /// valid indices, the cube mirrors its volume surface, chains are empty.
    #[test]
    fn surface_topology_matches_builders() {
        let (cols, rows) = (5, 4);
        let cloth = SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            0.25,
            1.0,
            0.0,
            0.0,
            0.0,
            ClothPin::None,
        );
        assert_eq!(cloth.surface.len(), 2 * (cols - 1) * (rows - 1));
        assert!(
            cloth
                .surface
                .iter()
                .flatten()
                .all(|i| i.index() < cloth.particle_count()),
            "cloth surface indices valid"
        );
        // CCW from +z: every triangle normal must point +z on the flat grid.
        for [a, b, c] in &cloth.surface {
            let (pa, pb, pc) = (
                cloth.particles[a.index()].position,
                cloth.particles[b.index()].position,
                cloth.particles[c.index()].position,
            );
            assert!((pb - pa).cross(pc - pa).z > 0.0, "cloth winding +z");
        }

        let cube = SoftBody::soft_cube(Vec3::ZERO, 1.0, 1.0, 0.0, 0.0);
        assert_eq!(cube.surface, cube.triangles);
        assert_eq!(cube.positions_snapshot().len(), 8);

        let chain = SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 4, 0.5, 1.0, 0.0);
        assert!(chain.surface.is_empty(), "chains have no sheet");
    }

    /// Overloaded strip tears structural rows and opens a surface hole.
    #[test]
    fn breakage_tears_overstretched_structural_and_opens_hole() {
        let (cols, rows) = (4, 4);
        let mut body = SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            ClothPin::TopRow,
        );
        body.tear_strain = 1.5;
        let constraints_before = body.constraint_count();
        let surface_before = body.surface.len();
        // Overload: drag the bottom half down — vertical structural rows
        // crossing the middle stretch ~6x rest (shear/bend cross too, but
        // only structural may tear in D1).
        for r in 2..rows {
            for c in 0..cols {
                body.particles[r * cols + c].position.y -= 5.0;
            }
        }
        body.apply_breakage();
        // Exactly the `cols` vertical rows between row 1 and row 2 tear.
        assert_eq!(body.constraint_count(), constraints_before - cols);
        // Both triangles of every middle-strip cell reference a torn edge.
        assert_eq!(body.surface.len(), surface_before - 2 * (cols - 1));
        for c in 0..cols {
            let (a, b) = (ParticleIdx::from(c + cols), ParticleIdx::from(c + 2 * cols));
            assert!(
                !body
                    .constraints
                    .iter()
                    .any(|row| row.kind == DeformKind::Structural
                        && ((row.a == a && row.b == b) || (row.a == b && row.b == a))),
                "middle vertical structural row {c} is gone"
            );
        }
    }

    /// `tear_strain == 0` means unbreakable (also the builder default).
    #[test]
    fn breakage_disabled_by_default_zero_threshold() {
        let mut body =
            SoftBody::cloth_grid(Vec3::ZERO, 4, 4, 1.0, 1.0, 0.0, 0.0, 0.0, ClothPin::TopRow);
        assert_eq!(body.tear_strain, 0.0, "raw() defaults to unbreakable");
        for p in body.particles.iter_mut().skip(8) {
            p.position.y -= 50.0;
        }
        let constraints_before = body.constraint_count();
        let surface_before = body.surface.len();
        body.apply_breakage();
        assert_eq!(body.constraint_count(), constraints_before);
        assert_eq!(body.surface.len(), surface_before);
    }

    /// Breakage only removes structural rows: shear/bend rows survive even
    /// far past the threshold, and pins/particles are untouched.
    #[test]
    fn breakage_preserves_pins_and_non_structural() {
        let (cols, rows) = (4, 4);
        let mut body = SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            ClothPin::TopRow,
        );
        body.tear_strain = 1.1;
        let structural_before = body
            .constraints
            .iter()
            .filter(|c| c.kind == DeformKind::Structural)
            .count();
        let shear_before = body
            .constraints
            .iter()
            .filter(|c| c.kind == DeformKind::Shear)
            .count();
        let bend_before = body
            .constraints
            .iter()
            .filter(|c| c.kind == DeformKind::Bend)
            .count();
        let pinned_before = body.particles.iter().filter(|p| p.is_pinned()).count();
        let particles_before = body.particle_count();
        // Stretch everything crossing the middle (structural + shear + bend).
        for r in 2..rows {
            for c in 0..cols {
                body.particles[r * cols + c].position.y -= 5.0;
            }
        }
        body.apply_breakage();
        assert!(structural_before > 0 && shear_before > 0 && bend_before > 0);
        assert_eq!(
            body.constraints
                .iter()
                .filter(|c| c.kind == DeformKind::Shear)
                .count(),
            shear_before,
            "shear rows never tear in D1"
        );
        assert_eq!(
            body.constraints
                .iter()
                .filter(|c| c.kind == DeformKind::Bend)
                .count(),
            bend_before,
            "bend rows never tear in D1"
        );
        assert!(
            body.constraints
                .iter()
                .filter(|c| c.kind == DeformKind::Structural)
                .count()
                < structural_before,
            "some structural rows did tear"
        );
        assert_eq!(body.particle_count(), particles_before);
        assert_eq!(
            body.particles.iter().filter(|p| p.is_pinned()).count(),
            pinned_before,
            "pins untouched"
        );
        assert_eq!(pinned_before, cols, "top row stays pinned");
    }
}
