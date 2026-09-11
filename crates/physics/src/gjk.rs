//! GJK/EPA distance queries between convex shapes.
//!
//! Fallback narrow phase for pairs involving cylinder, cone and convex
//! hull: the analytic pairwise routines in `distance` cover
//! sphere/box/capsule only. GJK gives the separation (distance +
//! witnesses), EPA the penetration (normal + depth + witnesses) — together
//! they answer the [`GjkDistance`] query the generic contact builder needs.
//! Heightfields never enter here (they are not convex; they collide
//! column-wise in `distance`).
//!
//! Determinism: fixed iteration caps (64 + 32), index-ordered simplex
//! reduction, exact predicates for skips (never tolerances on the hot
//! path), no hashing, no RNG. Degenerate cases (coincident witnesses,
//! exhausted budget) return best-effort values instead of panicking.

use glam::{Quat, Vec3};

use crate::distance::ShapeRef;
use crate::shape::Shape;

/// Separation/depth query result: surface distance (negative =
/// penetration), the contact normal (first shape toward the second), and
/// one witness point on each shape. The normal always comes from the
/// resolving query (GJK witness axis when separated, EPA face normal when
/// overlapping) — never rebuilt from the witnesses downstream, because
/// degenerate blends point sideways while the plane is right.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GjkDistance {
    /// Surface-to-surface distance (negative when overlapping).
    pub dist: f32,
    /// Contact normal: from the first shape toward the second.
    pub normal: Vec3,
    /// Witness on the first shape.
    pub point_a: Vec3,
    /// Witness on the second shape.
    pub point_b: Vec3,
}

/// Support point of a convex shape in world direction `dir` (farthest
/// point along `dir`). Closed forms per shape; hulls scan vertices in
/// index order (first maximum wins — deterministic).
fn support(shape: &Shape, pos: Vec3, rot: Quat, dir: Vec3) -> Vec3 {
    match shape {
        Shape::Sphere { radius } => pos + dir.normalize_or(Vec3::X) * *radius,
        Shape::Box { half_extents } => {
            let local = rot.conjugate() * dir;
            let s = Vec3::new(
                if local.x >= 0.0 { 1.0 } else { -1.0 },
                if local.y >= 0.0 { 1.0 } else { -1.0 },
                if local.z >= 0.0 { 1.0 } else { -1.0 },
            );
            pos + rot * (*half_extents * s)
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let axis = rot * Vec3::Y;
            let tip = if dir.dot(axis) >= 0.0 { axis } else { -axis } * *half_height;
            pos + tip + dir.normalize_or(Vec3::X) * *radius
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let axis = rot * Vec3::Y;
            let along = dir.dot(axis);
            let cap = axis
                * (if along >= 0.0 {
                    *half_height
                } else {
                    -*half_height
                });
            let radial = dir - axis * along;
            let rim = if radial.length_squared() > 1e-12 {
                radial.normalize() * *radius
            } else {
                Vec3::ZERO
            };
            pos + cap + rim
        }
        Shape::Cone {
            radius,
            half_height,
        } => {
            let axis = rot * Vec3::Y;
            let apex = pos + axis * *half_height;
            let base = pos - axis * *half_height;
            let along = dir.dot(axis);
            let radial = dir - axis * along;
            let rim = if radial.length_squared() > 1e-12 {
                base + radial.normalize() * *radius
            } else {
                base
            };
            if dir.dot(apex) >= dir.dot(rim) {
                apex
            } else {
                rim
            }
        }
        Shape::ConvexHull(hull) => {
            let mut best = pos;
            let mut best_d = f32::MIN;
            for v in &hull.vertices {
                let w = pos + rot * *v;
                let d = w.dot(dir);
                if d > best_d {
                    best_d = d;
                    best = w;
                }
            }
            best
        }
        Shape::Heightfield(_) => {
            // Unreachable by construction: heightfield pairs dispatch to
            // the column loop before GJK. A wrong dispatch must fail loudly
            // in tests, never silently collide.
            debug_assert!(false, "heightfield has no support function");
            pos
        }
    }
}

/// One Minkowski-difference vertex with its two witness preimages.
#[derive(Clone, Copy)]
struct SVertex {
    /// `sa - sb`.
    v: Vec3,
    /// Support point on A.
    sa: Vec3,
    /// Support point on B.
    sb: Vec3,
}

/// Minkowski support along `dir` with witnesses.
fn minkowski(a: ShapeRef, b: ShapeRef, dir: Vec3) -> SVertex {
    let sa = support(a.shape, a.pos, a.rot, dir);
    let sb = support(b.shape, b.pos, b.rot, -dir);
    SVertex { v: sa - sb, sa, sb }
}

/// Closest point on segment [p, q] to the origin with barycentric weights
/// (wp, wq) for (p, q).
fn closest_segment(p: Vec3, q: Vec3) -> (Vec3, f32, f32) {
    let pq = q - p;
    let len_sq = pq.length_squared();
    if len_sq < 1e-18 {
        return (p, 1.0, 0.0);
    }
    let t = (-p.dot(pq) / len_sq).clamp(0.0, 1.0);
    (p + pq * t, 1.0 - t, t)
}

/// Closest point on triangle (a, b, c) to the origin with barycentric
/// weights (Ericson, RTCD §5.1.5 — Voronoi regions, exact predicates).
pub(crate) fn closest_triangle(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, f32, f32, f32) {
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 1.0, 0.0, 0.0);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 0.0, 1.0, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, 1.0 - v, v, 0.0);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 0.0, 0.0, 1.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, 1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, 0.0, 1.0 - w, w);
    }
    // Inside face region: barycentric via (va, vb, vc).
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, 1.0 - v - w, v, w)
}

/// GJK distance: separation between two convex shapes with witnesses.
/// Returns `None` when the origin is enclosed (penetration — hand over to
/// EPA) or on a degenerate simplex. Fixed 64-iteration cap: curved
/// closest features (rim circles) zigzag before settling.
fn gjk_separated(a: ShapeRef, b: ShapeRef) -> Option<GjkDistance> {
    const MAX_ITERS: usize = 64;
    // Initial direction: center delta, exact fallback.
    let mut dir = a.pos - b.pos;
    if dir.length_squared() < 1e-18 {
        dir = Vec3::X;
    }
    let mut simplex = [minkowski(a, b, dir); 4];
    let mut count = 1usize;
    // Barycentric weights aligned with `simplex[..count]`; the witnesses
    // are always the weight blend of the stored preimages.
    let mut weights = [1.0f32, 0.0, 0.0, 0.0];
    let mut closest = simplex[0].v;
    if closest.length_squared() < 1e-18 {
        // Centroid difference already spans the origin: touching.
        return Some(GjkDistance {
            dist: 0.0,
            normal: (b.pos - a.pos).normalize_or(Vec3::Y),
            point_a: simplex[0].sa,
            point_b: simplex[0].sb,
        });
    }
    for _ in 0..MAX_ITERS {
        dir = -closest;
        let w = minkowski(a, b, dir);
        // No progress past the current closest point: separated.
        if w.v.dot(dir) - closest.dot(dir) <= 1e-6 * closest.length_squared().max(1.0) {
            let (mut pa, mut pb) = (Vec3::ZERO, Vec3::ZERO);
            for i in 0..count {
                pa += simplex[i].sa * weights[i];
                pb += simplex[i].sb * weights[i];
            }
            let delta = pa - pb;
            // Stall at ~zero distance is the same straddling-segment trap
            // as above (interior weight blend, not a contact): route to
            // EPA instead of reporting a zero-depth contact that sinks.
            if delta.length() <= 1e-6 {
                return None;
            }
            return Some(GjkDistance {
                dist: delta.length(),
                normal: delta.normalize_or((b.pos - a.pos).normalize_or(Vec3::Y)),
                point_a: pa,
                point_b: pb,
            });
        }
        // Insert and reduce.
        simplex[count] = w;
        count += 1;
        let (next, reduced, wts) = reduce_simplex(&simplex[..count]);
        // Compress the simplex + weights in place, preserving order.
        let mut next_count = 0;
        for (i, &keep) in reduced.iter().enumerate() {
            if keep {
                simplex[next_count] = simplex[i];
                weights[next_count] = wts[i];
                next_count += 1;
            }
        }
        count = next_count;
        // Origin within 1e-6 of the simplex: either touching (gap < 1e-6)
        // or penetrating with the simplex straddling the interior (a
        // segment through the origin is interior to the Minkowski
        // difference, not a contact — the classic false-touching trap).
        // Both route to EPA, which resolves touching as dist ~ 0 and
        // penetration as negative. Never report dist ~ 0 from here: a
        // zero-depth contact disables positional correction and sinks.
        if count == 0 || next.length_squared() <= 1e-12 {
            return None; // Enclosed (or touching): EPA owns it.
        }
        closest = next;
    }
    // Budget exhausted: best-effort from the current weights.
    let (mut pa, mut pb) = (Vec3::ZERO, Vec3::ZERO);
    for i in 0..count {
        pa += simplex[i].sa * weights[i];
        pb += simplex[i].sb * weights[i];
    }
    let delta = pa - pb;
    Some(GjkDistance {
        dist: delta.length(),
        normal: delta.normalize_or((b.pos - a.pos).normalize_or(Vec3::Y)),
        point_a: pa,
        point_b: pb,
    })
}

/// Reduce a 2–4 point simplex to the sub-simplex closest to the origin.
/// Returns the closest point, a keep-mask, and barycentric weights aligned
/// with the input order. A 4-point simplex enclosing the origin reduces to
/// nothing (empty mask = penetration signal).
fn reduce_simplex(s: &[SVertex]) -> (Vec3, [bool; 4], [f32; 4]) {
    match s.len() {
        1 => (s[0].v, [true, false, false, false], [1.0, 0.0, 0.0, 0.0]),
        2 => {
            let (p, wa, wb) = closest_segment(s[0].v, s[1].v);
            let keep = [wa > 0.0, wb > 0.0, false, false];
            (p, keep, [wa, wb, 0.0, 0.0])
        }
        3 => {
            let (p, wa, wb, wc) = closest_triangle(s[0].v, s[1].v, s[2].v);
            let keep = [wa > 0.0, wb > 0.0, wc > 0.0, false];
            (p, keep, [wa, wb, wc, 0.0])
        }
        _ => reduce_tetra(s),
    }
}

/// Tetrahedron reduction: the closest of the four faces (each tested with
/// outward orientation) wins; an origin strictly inside all four means
/// penetration (empty mask). Barycentrics map back to tet vertices.
fn reduce_tetra(s: &[SVertex]) -> (Vec3, [bool; 4], [f32; 4]) {
    // Degenerate tetra (flat: repeated or coplanar supports): volume ~ 0
    // makes the inside test a coin flip and reports false enclosure for
    // separated shapes. Fall back to the best face without enclosing.
    let e1 = s[1].v - s[0].v;
    let e2 = s[2].v - s[0].v;
    let e3 = s[3].v - s[0].v;
    let vol = e1.dot(e2.cross(e3)).abs();
    let scale = e1.length() * e2.length() * e3.length();
    if vol <= 1e-9 * scale.max(1e-18) {
        let mut best: Option<(Vec3, f32, [f32; 4])> = None;
        for (a, b, c) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
            let (p, wa, wb, wc) = closest_triangle(s[a].v, s[b].v, s[c].v);
            let gap = p.length();
            if best.is_none_or(|(_, g, _)| gap < g) {
                let mut w = [0.0f32; 4];
                w[a] = wa;
                w[b] = wb;
                w[c] = wc;
                best = Some((p, gap, w));
            }
        }
        let (p, _, w) = best.unwrap_or((Vec3::ZERO, 0.0, [0.0; 4]));
        let keep = [w[0] > 0.0, w[1] > 0.0, w[2] > 0.0, w[3] > 0.0];
        return (p, keep, w);
    }
    // Faces with outward winding checked both ways; the split below keeps
    // the face whose plane separates the origin with the smallest gap.
    const FACES: [(usize, usize, usize, usize); 4] =
        [(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)];
    let mut best: Option<(Vec3, f32, [f32; 4])> = None;
    for (a, b, c, _apex) in FACES {
        let (p, wa, wb, wc) = closest_triangle(s[a].v, s[b].v, s[c].v);
        let gap = p.length();
        let better = best.is_none_or(|(_, g, _)| gap < g);
        if better {
            let mut w = [0.0f32; 4];
            w[a] = wa;
            w[b] = wb;
            w[c] = wc;
            best = Some((p, gap, w));
        }
    }
    let (p, _, w) = best.unwrap_or((Vec3::ZERO, 0.0, [0.0; 4]));
    // Inside test: origin strictly behind every face plane means enclosed.
    // Reuse the gap: if the closest face still contains the origin in its
    // Voronoi interior AND all four face distances are ~0, the tet holds
    // the origin. Cheaper exact check: all four signed face distances <= 0.
    let mut inside = true;
    for (a, b, c, apex) in FACES {
        let n = (s[b].v - s[a].v).cross(s[c].v - s[a].v);
        // Orient outward (away from the apex vertex).
        let n = if n.dot(s[apex].v - s[a].v) > 0.0 {
            -n
        } else {
            n
        };
        // Origin beyond the outward face plane (n·(0 - a) > 0) means
        // strictly outside this face: not enclosed.
        if n.dot(-s[a].v) > 0.0 {
            inside = false;
            break;
        }
    }
    if inside {
        return (Vec3::ZERO, [false; 4], [0.0; 4]);
    }
    let keep = [w[0] > 0.0, w[1] > 0.0, w[2] > 0.0, w[3] > 0.0];
    (p, keep, w)
}

/// EPA penetration query from an enclosing tetrahedron: expands the
/// polytope toward the closest face until the support converges. Returns
/// (depth, normal toward B, witness_a, witness_b). Fixed 32-iteration cap;
/// degenerates return a zero-depth contact instead of panicking.
fn epa(a: ShapeRef, b: ShapeRef, tet: [SVertex; 4]) -> (f32, Vec3, Vec3, Vec3) {
    const MAX_ITERS: usize = 32;
    const EPS: f32 = 1e-6;
    // Polytope vertices (Minkowski) with witness preimages.
    let mut verts: Vec<SVertex> = tet.into_iter().collect();
    // Faces as index triples, outward-oriented.
    let mut faces: Vec<[usize; 3]> = vec![[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]];
    // Orient outward: apex check per face via the fourth vertex.
    for f in faces.iter_mut() {
        let n = (verts[f[1]].v - verts[f[0]].v).cross(verts[f[2]].v - verts[f[0]].v);
        // Any vertex not on the face works as the interior reference.
        let apex = (0..verts.len()).find(|i| *i != f[0] && *i != f[1] && *i != f[2]);
        if let Some(ai) = apex
            && n.dot(verts[ai].v - verts[f[0]].v) > 0.0
        {
            f.swap(1, 2);
        }
    }
    let mut best = (0.0f32, Vec3::X, a.pos, b.pos);
    for _ in 0..MAX_ITERS {
        // Closest face to the origin.
        let mut bi = 0usize;
        let mut bdist = f32::MAX;
        let mut bnormal = Vec3::X;
        for (i, f) in faces.iter().enumerate() {
            let n = (verts[f[1]].v - verts[f[0]].v).cross(verts[f[2]].v - verts[f[0]].v);
            let len = n.length();
            if len < 1e-18 {
                continue;
            }
            let n = n / len;
            let d = n.dot(verts[f[0]].v);
            if d < bdist {
                bdist = d;
                bnormal = n;
                bi = i;
            }
        }
        let w = minkowski(a, b, bnormal);
        // Converged: support cannot push past the face.
        if w.v.dot(bnormal) - bdist <= EPS * bdist.max(1.0) {
            // Witnesses from the face barycentrics.
            let f = faces[bi];
            let (closest, wa, wb, wc) =
                closest_triangle(verts[f[0]].v, verts[f[1]].v, verts[f[2]].v);
            let _ = closest;
            let pa = verts[f[0]].sa * wa + verts[f[1]].sa * wb + verts[f[2]].sa * wc;
            let pb = verts[f[0]].sb * wa + verts[f[1]].sb * wb + verts[f[2]].sb * wc;
            // Normal toward B in world: separating direction of the face.
            return (bdist.max(0.0), bnormal, pa, pb);
        }
        // Visible set: faces whose outward plane puts the new vertex
        // strictly outside.
        let mut visible = vec![false; faces.len()];
        for (i, f) in faces.iter().enumerate() {
            let n = (verts[f[1]].v - verts[f[0]].v).cross(verts[f[2]].v - verts[f[0]].v);
            let len = n.length();
            if len < 1e-18 {
                continue;
            }
            if (n / len).dot(w.v - verts[f[0]].v) > 1e-9 {
                visible[i] = true;
            }
        }
        // Horizon: directed edges of visible faces with no visible neighbor
        // across them (edge counting over the visible set, winding kept).
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (i, f) in faces.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            for e in 0..3 {
                edges.push((f[e], f[(e + 1) % 3]));
            }
        }
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for &e in &edges {
            let twin = edges.iter().filter(|&&o| o.0 == e.1 && o.1 == e.0).count();
            if twin == 0 {
                horizon.push(e);
            }
        }
        // Remove the visible set, add the vertex, reseal the fan.
        let mut kept: Vec<[usize; 3]> = Vec::with_capacity(faces.len() + horizon.len());
        for (i, f) in faces.drain(..).enumerate() {
            if !visible[i] {
                kept.push(f);
            }
        }
        faces = kept;
        let wi = verts.len();
        verts.push(w);
        for (e0, e1) in horizon {
            faces.push([e0, e1, wi]);
        }
        // Re-orient every face away from the polytope interior (centroid
        // side) — cheap safety net for winding mistakes above.
        let centroid = verts.iter().fold(Vec3::ZERO, |s, v| s + v.v) / verts.len() as f32;
        for f in faces.iter_mut() {
            let n = (verts[f[1]].v - verts[f[0]].v).cross(verts[f[2]].v - verts[f[0]].v);
            if n.dot(centroid - verts[f[0]].v) > 0.0 {
                f.swap(1, 2);
            }
        }
        best = (bdist.max(0.0), bnormal, a.pos, b.pos);
        if verts.len() > 128 {
            break; // Degenerate growth guard: report the best face so far.
        }
    }
    best
}

/// Full convex query: GJK separation, EPA penetration. Both shapes must be
/// convex (never a heightfield — the dispatcher guarantees it).
pub(crate) fn convex_distance(a: ShapeRef, b: ShapeRef) -> GjkDistance {
    if let Some(sep) = gjk_separated(a, b) {
        return sep;
    }
    // Penetration: rebuild an enclosing tetrahedron for EPA by sampling
    // support vertices around the axes (deterministic, index-free).
    let dirs = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    let mut tet = [minkowski(a, b, Vec3::X); 4];
    let mut found = 0;
    for d in dirs {
        let w = minkowski(a, b, d);
        // Keep the extreme vertex per direction (dedupe by value).
        if (0..found).all(|i| (tet[i].v - w.v).length_squared() > 1e-12) && found < 4 {
            tet[found] = w;
            found += 1;
        }
        if found == 4 {
            break;
        }
    }
    if found < 4 {
        // Degenerate (coincident shapes): zero-depth contact at the centers.
        return GjkDistance {
            dist: 0.0,
            normal: (b.pos - a.pos).normalize_or(Vec3::Y),
            point_a: a.pos,
            point_b: b.pos,
        };
    }
    let (depth, normal, raw_a, raw_b) = epa(a, b, tet);
    // EPA reports the right plane; its preimage blend is best-effort
    // (collapses onto box corners against sliding supports). Keep the raw
    // witnesses — the distance dispatcher re-seats them by closest-point
    // projection under this normal.
    GjkDistance {
        dist: -depth,
        normal,
        point_a: raw_a,
        point_b: raw_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{ConvexHull, Heightfield};

    const EPS: f32 = 1e-4;

    fn hull_cube() -> Shape {
        let mut verts = Vec::new();
        for &x in &[-1.0f32, 1.0] {
            for &y in &[-1.0f32, 1.0] {
                for &z in &[-1.0f32, 1.0] {
                    verts.push(Vec3::new(x, y, z));
                }
            }
        }
        Shape::ConvexHull(ConvexHull::from_vertices(verts))
    }

    #[test]
    fn gjk_sphere_vs_cylinder_separation() {
        let sphere = Shape::Sphere { radius: 1.0 };
        let cyl = Shape::Cylinder {
            radius: 1.0,
            half_height: 1.0,
        };
        let d = convex_distance(
            ShapeRef {
                shape: &sphere,
                pos: Vec3::new(4.0, 0.0, 0.0),
                rot: Quat::IDENTITY,
            },
            ShapeRef {
                shape: &cyl,
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
        );
        // Sphere surface at x=3, cylinder rim at x=1: gap 2.
        assert!((d.dist - 2.0).abs() < EPS, "gap, got {}", d.dist);
        // Witness tolerance 1e-3, not EPS: the wall point (1,0,0) is a
        // 50/50 blend of the two rim-circle supports, and GJK weight
        // settling has a slow linear tail. The residual is purely
        // tangential (both witnesses shift together), so the contact
        // normal stays exact — zero physics impact.
        assert!(
            (d.point_a - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-3,
            "pa={} pb={}",
            d.point_a,
            d.point_b
        );
        assert!(
            (d.point_b - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3,
            "pb={}",
            d.point_b
        );
    }

    /// Regression: a tetrahedron vertex buried 0.4 deep in a box must
    /// report penetration (negative distance), not a stalled separation.
    /// The tetra reduction's inside test once had its sign inverted, so
    /// GJK never saw enclosure: the loop kept discarding the improving
    /// vertex and returned a false gap from the iteration budget.
    #[test]
    fn gjk_tetra_vertex_buried_reports_penetration() {
        let tetra = Shape::ConvexHull(ConvexHull::from_vertices(vec![
            Vec3::new(1.0, 0.0, -1.0 / 2.0f32.sqrt()),
            Vec3::new(-1.0, 0.0, -1.0 / 2.0f32.sqrt()),
            Vec3::new(0.0, 1.0, 1.0 / 2.0f32.sqrt()),
            Vec3::new(0.0, -1.0, 1.0 / 2.0f32.sqrt()),
        ]));
        let floor = Shape::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        };
        let d = convex_distance(
            ShapeRef {
                shape: &floor,
                pos: Vec3::new(0.0, -0.5, 0.0),
                rot: Quat::IDENTITY,
            },
            ShapeRef {
                shape: &tetra,
                pos: Vec3::new(0.0, 0.6, 0.0),
                rot: Quat::IDENTITY,
            },
        );
        // Vertex (0,-0.4,0.707) is 0.4 below the floor top: exact depth.
        assert!(
            (d.dist + 0.4).abs() < 1e-3,
            "dist={} n={}",
            d.dist,
            d.normal
        );
        // Contact normal runs floor toward tetra (up).
        assert!(d.normal.dot(Vec3::Y) > 0.999, "n={}", d.normal);
    }

    #[test]
    fn gjk_cube_hull_matches_box_oracle() {
        // A hull of the unit-cube corners must agree with the analytic box
        // distance to within GJK tolerance.
        let hull = hull_cube();
        let other = Shape::Sphere { radius: 0.5 };
        let d = convex_distance(
            ShapeRef {
                shape: &hull,
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            ShapeRef {
                shape: &other,
                pos: Vec3::new(3.0, 0.0, 0.0),
                rot: Quat::IDENTITY,
            },
        );
        // Cube face at x=1, sphere surface at 2.5: gap 1.5.
        assert!((d.dist - 1.5).abs() < 1e-3, "gap, got {}", d.dist);
    }

    #[test]
    fn epa_reports_box_box_penetration() {
        let a = Shape::Box {
            half_extents: Vec3::splat(1.0),
        };
        let b = Shape::Box {
            half_extents: Vec3::splat(1.0),
        };
        let d = convex_distance(
            ShapeRef {
                shape: &a,
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            ShapeRef {
                shape: &b,
                pos: Vec3::new(1.5, 0.0, 0.0),
                rot: Quat::IDENTITY,
            },
        );
        // 0.5 penetration along x.
        assert!(d.dist < 0.0, "must report overlap, got {}", d.dist);
        assert!((d.dist + 0.5).abs() < 1e-3, "depth, got {}", d.dist);
    }

    #[test]
    fn support_cone_apex_vs_base() {
        let cone = Shape::Cone {
            radius: 1.0,
            half_height: 1.0,
        };
        // Straight up: apex.
        let s = support(&cone, Vec3::ZERO, Quat::IDENTITY, Vec3::Y);
        assert!((s - Vec3::new(0.0, 1.0, 0.0)).length() < EPS);
        // Straight down: base center.
        let s = support(&cone, Vec3::ZERO, Quat::IDENTITY, Vec3::NEG_Y);
        assert!((s - Vec3::new(0.0, -1.0, 0.0)).length() < EPS);
        // Sideways: base rim.
        let s = support(&cone, Vec3::ZERO, Quat::IDENTITY, Vec3::X);
        assert!((s - Vec3::new(1.0, -1.0, 0.0)).length() < EPS);
    }

    #[test]
    fn hull_triangulates_cube_faces() {
        // 8 cube corners, exact ±1 arithmetic: every quad yields all
        // C(4,3) = 4 coplanar triples, each in its 3 outward (even
        // permutation) orderings — 4 * 3 * 6 = 72 oriented faces, all
        // outward. Overlapping coplanar triangles share one normal, which
        // is harmless for SAT/GJK (support never reads faces).
        let Shape::ConvexHull(hull) = hull_cube() else {
            panic!("hull");
        };
        assert_eq!(hull.faces.len(), 72);
        for f in &hull.faces {
            let (a, b, c) = (
                hull.vertices[f[0] as usize],
                hull.vertices[f[1] as usize],
                hull.vertices[f[2] as usize],
            );
            let n = (b - a).cross(c - a).normalize();
            let center = (a + b + c) / 3.0;
            assert!(n.dot(center) > 0.0, "face must point outward: {f:?} n={n}");
        }
    }

    #[test]
    fn hull_over_cap_skips_faces_but_keeps_support() {
        let verts: Vec<Vec3> = (0..70).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        let hull = ConvexHull::from_vertices(verts);
        assert!(hull.faces.is_empty());
        assert_eq!(hull.vertices.len(), 70);
        let s = support(
            &Shape::ConvexHull(hull),
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::X,
        );
        assert!((s.x - 69.0).abs() < EPS);
    }

    #[test]
    fn heightfield_height_at_bilinear() {
        let hf = Heightfield {
            heights: vec![0.0, 0.0, 0.0, 2.0],
            rows: 2,
            cols: 2,
            cell: 1.0,
        };
        // Corners clamp to edge samples.
        assert!((hf.height_at(-10.0, -10.0) - 0.0).abs() < EPS);
        assert!((hf.height_at(10.0, 10.0) - 2.0).abs() < EPS);
        // Center of the single quad: mean of corners.
        assert!((hf.height_at(0.0, 0.0) - 0.5).abs() < EPS);
    }
}
