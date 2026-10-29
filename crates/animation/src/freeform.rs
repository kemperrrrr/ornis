//! Deterministic **freeform** blend weights ([`freeform2d_weights`]) —
//! the pure position-to-weight math behind
//! [`TreeNodeDef::Freeform2d`](crate::TreeNodeDef::Freeform2d): a
//! Delaunay triangulation of the arbitrary child points (f64
//! determinants, symbolic SoS tie-break of cocircular sets, `span`-
//! scaled tolerances) and the barycentric fold over the containing
//! triangle — allocation-free and deterministic for the hot path.

use crate::tree::{FREEFORM_MAX_POINTS, FREEFORM_MAX_TRIANGLES, FreeformCache, TreeChild2d};

/// The live `(node, weight)` slots of one [`freeform2d_weights_cached`] fold —
/// up to three triangle corners (the outside-hull clamp needs at most
/// two). The slots carry the child's **node index** (the freeform folds
/// consume nodes directly) and their weights sum to one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FreeformSlots {
    entries: [(usize, f32); FREEFORM_SLOTS],
    filled: usize,
}

impl FreeformSlots {
    /// The zero-slot fold — the caller reads rest.
    pub(crate) const EMPTY: Self = Self {
        entries: [(0, 0.0); FREEFORM_SLOTS],
        filled: 0,
    };

    /// The live `(node, weight)` slots.
    pub(crate) fn as_slice(&self) -> &[(usize, f32)] {
        &self.entries[..self.filled]
    }
}

/// Slots of a `freeform2d_weights` fold: one triangle's corners.
pub(crate) const FREEFORM_SLOTS: usize = 3;

/// Dimensionless slop of the freeform fold: barycentric weights and
/// edge projections may dip a few ulps below zero or the unit span.
const FREEFORM_EPS: f32 = 1e-9;

/// Degenerate-area tolerance of the freeform geometry, relative to the
/// squared span of the point set: triangles thinner than this are
/// collinear and do not cover the plane.
const FREEFORM_AREA_EPS: f64 = 1e-12;

/// Cocircularity tolerance of the incircle test, relative to span⁴: f32
/// coordinates round an ideal circle by ~6e-8·span⁴, so near-cocircular
/// sets (an octagon of `cos 45°` directions) must tie-break instead of
/// failing the strict Delaunay filter at every scale.
const FREEFORM_CIRCLE_EPS: f64 = 1e-6;

/// Twice the signed area of `a → b → c` (positive = counter-clockwise),
/// in f64 over the exact f32 inputs — exactly collinear points give
/// exactly zero.
fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f64 {
    let ((abx, aby), (acx, acy)) = (
        (
            f64::from(b[0]) - f64::from(a[0]),
            f64::from(b[1]) - f64::from(a[1]),
        ),
        (
            f64::from(c[0]) - f64::from(a[0]),
            f64::from(c[1]) - f64::from(a[1]),
        ),
    );
    abx * acy - aby * acx
}

/// Largest axis extent of `children` — the scale of the point set.
fn set_span(children: &[TreeChild2d]) -> f64 {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for child in children {
        for (axis, v) in child.position.iter().enumerate() {
            let v = f64::from(*v);
            lo[axis] = lo[axis].min(v);
            hi[axis] = hi[axis].max(v);
        }
    }
    (hi[0] - lo[0]).max(hi[1] - lo[1])
}

/// Relative geometric tolerances of the point set as
/// `(circle, area)` — `span⁴`-scaled cocircularity band and
/// `span²`-scaled degenerate area, where `span` is the largest axis
/// extent ([`set_span`]).
fn tolerances(children: &[TreeChild2d]) -> (f64, f64) {
    let span = set_span(children);
    (
        span * span * span * span * FREEFORM_CIRCLE_EPS,
        span * span * FREEFORM_AREA_EPS,
    )
}

/// Raw barycentric weights of `value` in triangle `a b c`, `None` when
/// the triangle is degenerate or the value sits outside (inclusive
/// edges, [`FREEFORM_EPS`] tolerance). The weights sum to one up to
/// float error; [`compact`] drops the non-positive ones and
/// renormalizes.
fn triangle_weights(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    value: [f32; 2],
    area_eps: f64,
) -> Option<[f32; 3]> {
    let area = orient(a, b, c);
    if area.abs() <= area_eps {
        return None;
    }
    let weights = [
        (orient(b, c, value) / area) as f32,
        (orient(c, a, value) / area) as f32,
        (orient(a, b, value) / area) as f32,
    ];
    weights
        .iter()
        .all(|weight| *weight >= -FREEFORM_EPS)
        .then_some(weights)
}

/// Which side of the circumcircle of `a b c` `value` sits on: positive
/// — strictly inside, negative — strictly outside, near zero —
/// cocircular within the f32 input noise (tie-broken by [`sos_side`]).
/// The f64 determinant over differences translated to `value` is exact
/// for the f32 inputs up to a few ulps.
fn circumcircle_side(a: [f32; 2], b: [f32; 2], c: [f32; 2], value: [f32; 2]) -> f64 {
    let ((ax, ay), (bx, by), (cx, cy)) = (
        (
            f64::from(a[0]) - f64::from(value[0]),
            f64::from(a[1]) - f64::from(value[1]),
        ),
        (
            f64::from(b[0]) - f64::from(value[0]),
            f64::from(b[1]) - f64::from(value[1]),
        ),
        (
            f64::from(c[0]) - f64::from(value[0]),
            f64::from(c[1]) - f64::from(value[1]),
        ),
    );
    let det = (ax * ax + ay * ay) * (bx * cy - by * cx)
        + (bx * bx + by * by) * (cx * ay - cy * ax)
        + (cx * cx + cy * cy) * (ax * by - ay * bx);
    // The translation to `value` turns the incircle test into this
    // determinant: positive = `value` inside for a CCW triangle.
    det * orient(a, b, c).signum()
}

/// Symbolic tie-break of a cocircular quadruple (simulation of
/// simplicity): perturbing the lifted points `z += ε^rank` in
/// lexicographic `(x, y)` rank turns the incircle sign into the sign of
/// the first non-degenerate cofactor in that order. The same rule
/// decides every triple of a cocircular set — the survivors form one
/// triangulation of the hull whichever order the children arrive in —
/// and the output matches [`circumcircle_side`]'s convention (positive
/// — the fourth point inside for a CCW triple).
fn sos_side(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2], eps: f64) -> f64 {
    let mut terms = [
        (a, orient(b, c, d)),
        (b, -orient(a, c, d)),
        (c, orient(a, b, d)),
        (d, -orient(a, b, c)),
    ];
    terms.sort_unstable_by(|x, y| x.0[0].total_cmp(&y.0[0]).then(x.0[1].total_cmp(&y.0[1])));
    for (_, cofactor) in terms {
        if cofactor.abs() > eps {
            return cofactor.signum() * orient(a, b, c).signum();
        }
    }
    -1.0
}

/// The Delaunay filter: `children[triple]` survives only when no other
/// point sits strictly inside its circumcircle. Cocircular fourth
/// points (inside the `span⁴`-scaled [`FREEFORM_CIRCLE_EPS`] band) are
/// tie-broken by [`sos_side`] — one diagonal per cocircular quadruple,
/// geometric rather than arrival-ordered, so the survivors partition
/// the hull and the weights stay continuous across their shared edges.
/// The scan is in index order — deterministic, no allocations.
fn passes_delaunay(children: &[TreeChild2d], triple: [usize; 3], eps: (f64, f64)) -> bool {
    let (a, b, c) = (
        children[triple[0]].position,
        children[triple[1]].position,
        children[triple[2]].position,
    );
    for (slot, child) in children.iter().enumerate() {
        if triple.contains(&slot) {
            continue;
        }
        let side = circumcircle_side(a, b, c, child.position);
        let side = if side.abs() <= eps.0 {
            sos_side(a, b, c, child.position, eps.1)
        } else {
            side
        };
        if side > 0.0 {
            return false;
        }
    }
    true
}

/// `children` indices in ascending (x, y) order — the fixed point order
/// of the freeform triangulation and hull. Insertion sort, no
/// allocations.
fn sorted_indices(children: &[TreeChild2d]) -> [usize; FREEFORM_MAX_POINTS] {
    let mut order = [0usize; FREEFORM_MAX_POINTS];
    for (slot, entry) in order.iter_mut().enumerate().take(children.len()) {
        *entry = slot;
    }
    for i in 1..children.len() {
        let mut j = i;
        while j > 0
            && (
                children[order[j - 1]].position[0],
                children[order[j - 1]].position[1],
            ) > (
                children[order[j]].position[0],
                children[order[j]].position[1],
            )
        {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    order
}

/// Convex hull of `children` in counter-clockwise index order (monotone
/// chain over [`sorted_indices`], ties broken by the fixed point order).
/// Writes the hull into `out` and returns its length; `children` must be
/// distinct finite points with at least two of them (the fold contract —
/// hull length is then at least two). Collinear input collapses to its
/// two extreme points.
fn hull_indices(children: &[TreeChild2d], out: &mut [usize; FREEFORM_MAX_POINTS]) -> usize {
    let order = sorted_indices(children);
    let mut len = 0usize;
    for pass in 0..2 {
        let lower = len;
        let count = children.len() - usize::from(pass == 1);
        for slot in 0..count {
            let point = order[if pass == 0 { slot } else { count - 1 - slot }];
            while len > lower.max(1)
                && orient(
                    children[out[len - 2]].position,
                    children[out[len - 1]].position,
                    children[point].position,
                ) <= 0.0
            {
                len -= 1;
            }
            out[len] = point;
            len += 1;
        }
    }
    // The last point duplicates the first one of the lower run.
    len.saturating_sub(1)
}

/// The hull edge nearest to `value` as `(edge, t)` — `hull[edge]` to
/// `hull[(edge + 1) % hull_len]`, clamped parametric position `t`.
fn nearest_hull_edge(
    children: &[TreeChild2d],
    hull: &[usize; FREEFORM_MAX_POINTS],
    hull_len: usize,
    value: [f32; 2],
) -> (usize, f32) {
    let mut best = (0usize, 0.0f32, f32::INFINITY);
    for edge in 0..hull_len {
        let (a, b) = (
            children[hull[edge]].position,
            children[hull[(edge + 1) % hull_len]].position,
        );
        let span = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
        let t = if span <= FREEFORM_EPS {
            0.0
        } else {
            (((value[0] - a[0]) * (b[0] - a[0]) + (value[1] - a[1]) * (b[1] - a[1])) / span)
                .clamp(0.0, 1.0)
        };
        let (px, py) = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
        let distance = (value[0] - px) * (value[0] - px) + (value[1] - py) * (value[1] - py);
        if distance < best.2 {
            best = (edge, t, distance);
        }
    }
    (best.0, best.1)
}

/// Slots of `value` clamped to the nearest hull segment of `children`
/// (linear weights along the segment); the trailing zero-weight slot is
/// dropped by [`compact`].
fn clamp_to_hull(children: &[TreeChild2d], value: [f32; 2]) -> [(usize, f32); FREEFORM_SLOTS] {
    let mut hull = [0usize; FREEFORM_MAX_POINTS];
    let hull_len = hull_indices(children, &mut hull);
    let (edge, t) = nearest_hull_edge(children, &hull, hull_len, value);
    [
        (hull[edge], 1.0 - t),
        (hull[(edge + 1) % hull_len.max(1)], t),
        (0, 0.0),
    ]
}

/// Drops the non-positive weights of `fold` and renormalizes the rest
/// (the live-count semantics of the weight tables).
fn normalize(fold: &mut FreeformSlots) {
    let sum: f32 = fold.as_slice().iter().map(|(_, weight)| *weight).sum();
    if sum > 0.0 {
        for entry in &mut fold.entries[..fold.filled] {
            entry.1 /= sum;
        }
    }
}

/// Copies `slots` into a [`FreeformSlots`] fold, dropping non-positive
/// weights and renormalizing the rest.
fn compact(slots: [(usize, f32); FREEFORM_SLOTS]) -> FreeformSlots {
    let mut fold = FreeformSlots::EMPTY;
    for &(node, weight) in &slots {
        if weight > 0.0 {
            fold.entries[fold.filled] = (node, weight);
            fold.filled += 1;
        }
    }
    normalize(&mut fold);
    fold
}

/// The surviving Delaunay triples of `children` — every triple with an
/// empty circumcircle, cocircular sets folded to one geometric diagonal
/// by the SoS tie-break. Built once per install; fixed arrays only, no
/// allocations. The scan order matches the fold's, so the cached fold
/// is bit-for-bit the on-the-fly one.
pub fn freeform_cache(children: &[TreeChild2d]) -> FreeformCache {
    let mut cache = FreeformCache::default();
    let eps = tolerances(children);
    for i in 0..children.len() {
        for j in (i + 1)..children.len() {
            for k in (j + 1)..children.len() {
                if passes_delaunay(children, [i, j, k], eps) && cache.count < FREEFORM_MAX_TRIANGLES
                {
                    cache.kept[cache.count] = [i, j, k];
                    cache.count += 1;
                }
            }
        }
    }
    cache
}

/// Weights of a 2-D **freeform** blend at `value` (x, y) over `children`
/// — arbitrary distinct points ([`TreeNodeDef::Freeform2d`]) — against
/// the node's install-time [`FreeformCache`].
///
/// Returns up to three live `(node, weight)` slots — the child's node
/// index and its weight, summing to one. The value inside a
/// triangulation triangle gets the corners' barycentric weights — the
/// triangulation is a partition of the hull (cocircular sets fold to
/// one geometric diagonal, see [`sos_side`]), so the weights are
/// continuous across shared edges; on the convex hull edge or a corner
/// the edge or corner owns the pose; outside the convex hull the value
/// clamps to its nearest hull segment (linear along it); a collinear set
/// degenerates to its hull segment; empty `children` yields
/// [`FreeformSlots::EMPTY`] (the caller reads rest). More than
/// [`FREEFORM_MAX_POINTS`] children yields the empty fold (the eval
/// layer reads it as bad skin first).
///
/// Deterministic and allocation-free — evaluated every frame per blend
/// node: a containment scan over the cached triangles, no incircle
/// tests.
pub(crate) fn freeform2d_weights_cached(
    cache: &FreeformCache,
    children: &[TreeChild2d],
    value: [f32; 2],
) -> FreeformSlots {
    match children.len() {
        0 => return FreeformSlots::EMPTY,
        1 => return compact([(children[0].node, 1.0), (0, 0.0), (0, 0.0)]),
        count if count > FREEFORM_MAX_POINTS => return FreeformSlots::EMPTY,
        _ => {}
    }
    let (_, area_eps) = tolerances(children);
    for triple in cache.triples() {
        if let Some(weights) = triangle_weights(
            children[triple[0]].position,
            children[triple[1]].position,
            children[triple[2]].position,
            value,
            area_eps,
        ) {
            return compact([
                (children[triple[0]].node, weights[0]),
                (children[triple[1]].node, weights[1]),
                (children[triple[2]].node, weights[2]),
            ]);
        }
    }
    compact(clamp_to_hull(children, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `count` freeform inputs at `positions` with sequential node ids.
    fn freeform(positions: [[f32; 2]; 3]) -> Vec<TreeChild2d> {
        positions
            .into_iter()
            .enumerate()
            .map(|(node, position)| TreeChild2d { position, node })
            .collect()
    }

    fn weights_of(children: &[TreeChild2d], value: [f32; 2]) -> Vec<(usize, f32)> {
        freeform2d_weights_cached(&freeform_cache(children), children, value)
            .as_slice()
            .to_vec()
    }

    /// Barycentric weights of `value` in the candidate triangle
    /// `children[triple]`, `None` when the triple is degenerate, fails
    /// the Delaunay filter (a fourth point strictly inside its
    /// circumcircle), or leaves the value outside.
    fn triangulated_at(
        children: &[TreeChild2d],
        triple: [usize; 3],
        value: [f32; 2],
        eps: (f64, f64),
    ) -> Option<[f32; 3]> {
        if !passes_delaunay(children, triple, eps) {
            return None;
        }
        triangle_weights(
            children[triple[0]].position,
            children[triple[1]].position,
            children[triple[2]].position,
            value,
            eps.1,
        )
    }

    /// Barycentric weights of `value` in triangle (0, 0) (2, 0) (0, 2).
    fn bary(value: [f32; 2]) -> Vec<(usize, f32)> {
        let (x, y) = (value[0], value[1]);
        let mut out = vec![(0, 1.0 - x / 2.0 - y / 2.0), (1, x / 2.0), (2, y / 2.0)];
        out.retain(|(_, weight)| *weight > 0.0);
        out
    }

    #[test]
    fn a_triangle_interpolates_barycentrically() {
        let children = freeform([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        for value in [[0.5, 0.5], [1.0, 0.25], [0.25, 1.0], [1.0, 1.0]] {
            let got = weights_of(&children, value);
            let want = bary(value);
            assert_eq!(got.len(), want.len(), "at {value:?}");
            for ((child, weight), (want_child, want_weight)) in got.iter().zip(want.iter()) {
                assert_eq!(child, want_child, "at {value:?}");
                assert!((weight - want_weight).abs() < 1e-4, "at {value:?}");
            }
        }
    }

    #[test]
    fn a_hull_edge_carries_the_whole_pose() {
        let children = freeform([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        // Midpoint of the hypotenuse (1, 1): the edge, not a corner.
        let got = weights_of(&children, [1.0, 1.0]);
        assert_eq!(
            got.iter().map(|(child, _)| *child).collect::<Vec<_>>(),
            vec![1, 2]
        );
        // Exactly on a corner: the corner owns it.
        let got = weights_of(&children, [2.0, 0.0]);
        assert_eq!(got, vec![(1, 1.0)]);
    }

    #[test]
    fn outside_the_hull_the_nearest_edge_carries_the_pose() {
        let children = freeform([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        // Below the bottom edge: clamps onto it at the nearest point.
        let got = weights_of(&children, [0.5, -3.0]);
        assert_eq!(got.len(), 2);
        let total: f32 = got.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(
            got.iter().all(|(child, _)| *child != 2),
            "the far corner owns nothing"
        );
        // Far past the corner along x: the corner owns it entirely.
        let got = weights_of(&children, [9.0, -0.5]);
        assert_eq!(got, vec![(1, 1.0)]);
    }

    #[test]
    fn four_points_blend_inside_their_triangles() {
        // A square: the query sits inside one of the two triangles of
        // the deterministic triangulation.
        let children = vec![
            TreeChild2d {
                position: [0.0, 0.0],
                node: 0,
            },
            TreeChild2d {
                position: [2.0, 0.0],
                node: 1,
            },
            TreeChild2d {
                position: [2.0, 2.0],
                node: 2,
            },
            TreeChild2d {
                position: [0.0, 2.0],
                node: 3,
            },
        ];
        let got = weights_of(&children, [1.75, 0.25]);
        assert!(
            got.iter().any(|(child, _)| *child == 1),
            "the nearby corner takes part"
        );
        let total: f32 = got.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-6);
        // Outside the square: clamps to the nearest edge.
        let got = weights_of(&children, [3.0, 1.0]);
        assert_eq!(
            got.iter().map(|(child, _)| *child).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn a_collinear_set_degenerates_to_its_segment() {
        let children = vec![
            TreeChild2d {
                position: [0.0, 0.0],
                node: 0,
            },
            TreeChild2d {
                position: [1.0, 0.0],
                node: 1,
            },
            TreeChild2d {
                position: [2.0, 0.0],
                node: 2,
            },
        ];
        let got = weights_of(&children, [0.5, 1.0]);
        assert_eq!(got, vec![(0, 0.75), (2, 0.25)]);
    }

    /// Sparse fold over `children` as a dense per-node weight vector.
    fn dense(children: &[TreeChild2d], value: [f32; 2]) -> Vec<f32> {
        let mut out = vec![0.0; children.len()];
        for (node, weight) in weights_of(children, value) {
            out[node] += weight;
        }
        out
    }

    fn l1(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }

    /// Five points in convex position plus one inside — deliberately
    /// off any lattice.
    fn irregular() -> Vec<TreeChild2d> {
        vec![
            TreeChild2d {
                position: [-1.0, 0.5],
                node: 0,
            },
            TreeChild2d {
                position: [0.5, -1.0],
                node: 1,
            },
            TreeChild2d {
                position: [2.5, 0.0],
                node: 2,
            },
            TreeChild2d {
                position: [1.5, 2.0],
                node: 3,
            },
            TreeChild2d {
                position: [-0.5, 1.8],
                node: 4,
            },
            TreeChild2d {
                position: [1.0, 0.7],
                node: 5,
            },
        ]
    }

    /// Whether `value` lies inside (or on) the hull of `children`.
    fn inside_hull(children: &[TreeChild2d], value: [f32; 2]) -> bool {
        let mut hull = [0usize; FREEFORM_MAX_POINTS];
        let hull_len = hull_indices(children, &mut hull);
        (0..hull_len).all(|edge| {
            orient(
                children[hull[edge]].position,
                children[hull[(edge + 1) % hull_len]].position,
                value,
            ) >= -tolerances(children).1
        })
    }

    /// Cocircular square: the fold keeps one diagonal (tie-break), so
    /// the weights are continuous across it — the reviewer repro pair
    /// on either side of the seam and a pair straddling the diagonal
    /// differ by O(ε), not O(1).
    #[test]
    fn cocircular_square_folds_continuously_across_one_diagonal() {
        let children = vec![
            TreeChild2d {
                position: [0.0, 0.0],
                node: 0,
            },
            TreeChild2d {
                position: [2.0, 0.0],
                node: 1,
            },
            TreeChild2d {
                position: [2.0, 2.0],
                node: 2,
            },
            TreeChild2d {
                position: [0.0, 2.0],
                node: 3,
            },
        ];
        let straddle = l1(
            &dense(&children, [0.5, 0.5 - 1e-3]),
            &dense(&children, [0.5, 0.5 + 1e-3]),
        );
        assert!(straddle < 5e-3, "jump across the diagonal: {straddle}");
        let repro = l1(
            &dense(&children, [0.5, 0.49]),
            &dense(&children, [0.49, 0.5]),
        );
        assert!(repro < 0.05, "repro pair diverges: {repro}");
    }

    /// Cocircular diamond (the speed × direction case): same
    /// continuity across the kept diagonal; the x = 0 seam the old
    /// overlapping fold produced is gone.
    #[test]
    fn cocircular_diamond_folds_continuously_across_one_diagonal() {
        let children = vec![
            TreeChild2d {
                position: [-1.0, 0.0],
                node: 0,
            },
            TreeChild2d {
                position: [0.0, -1.0],
                node: 1,
            },
            TreeChild2d {
                position: [1.0, 0.0],
                node: 2,
            },
            TreeChild2d {
                position: [0.0, 1.0],
                node: 3,
            },
        ];
        let straddle = l1(
            &dense(&children, [0.0, -1e-3]),
            &dense(&children, [0.0, 1e-3]),
        );
        assert!(straddle < 5e-3, "jump across the diagonal: {straddle}");
        let repro = l1(
            &dense(&children, [0.01, 0.2]),
            &dense(&children, [-0.01, 0.2]),
        );
        assert!(repro < 0.05, "repro pair diverges: {repro}");
    }

    /// Irregular 5+1-point set: interior folds are non-negative,
    /// sum to one, and reconstitute the query point (barycentric
    /// identity — the weights are the corners' barycentrics of the
    /// containing triangle).
    #[test]
    fn irregular_points_fold_barycentrically() {
        let children = irregular();
        for value in [[0.0, 0.5], [1.0, 0.5], [0.5, 0.2], [1.2, 1.2], [0.2, 1.2]] {
            let weights = weights_of(&children, value);
            assert!(weights.len() >= 2 && weights.len() <= 3, "at {value:?}");
            let sum: f32 = weights.iter().map(|(_, weight)| *weight).sum();
            assert!((sum - 1.0).abs() < 1e-4, "sum at {value:?}: {sum}");
            for (node, weight) in &weights {
                assert!((0.0..=1.0).contains(weight), "at {value:?}");
                assert!(inside_hull(&children, children[*node].position));
            }
            let mut rebuilt = [0.0f32; 2];
            for (node, weight) in &weights {
                let p = children[*node].position;
                rebuilt[0] += weight * p[0];
                rebuilt[1] += weight * p[1];
            }
            assert!(
                (rebuilt[0] - value[0]).abs() < 1e-4 && (rebuilt[1] - value[1]).abs() < 1e-4,
                "rebuild at {value:?}: {rebuilt:?}"
            );
        }
    }

    /// «Close values → close weights» on a grid inside the hull of the
    /// irregular set: a step of 1e-3 in either axis moves the weights
    /// by little; a discontinuity would show O(1) jumps.
    #[test]
    fn nearby_values_give_nearby_weights_on_a_grid() {
        let children = irregular();
        let mut x = -1.0;
        while x <= 2.5 {
            let mut y = -1.0;
            while y <= 2.0 {
                let value = [x, y];
                if inside_hull(&children, value) {
                    let base = dense(&children, value);
                    for step in [[1e-3, 0.0], [0.0, 1e-3]] {
                        let moved = dense(&children, [x + step[0], y + step[1]]);
                        let jump = l1(&base, &moved);
                        assert!(jump < 0.05, "jump at {value:?} step {step:?}: {jump}");
                    }
                }
                y += 0.25;
            }
            x += 0.25;
        }
    }

    /// `count` freeform inputs at `positions` with sequential node ids.
    fn freeform_n(positions: &[[f32; 2]]) -> Vec<TreeChild2d> {
        positions
            .iter()
            .enumerate()
            .map(|(node, position)| TreeChild2d {
                position: *position,
                node,
            })
            .collect()
    }

    /// The triples the Delaunay filter keeps — the query sits at each
    /// centroid, so the value never decides acceptance.
    fn survivors(children: &[TreeChild2d]) -> Vec<[usize; 3]> {
        let eps = tolerances(children);
        let mut kept = Vec::new();
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                for k in (j + 1)..children.len() {
                    let centroid = [
                        (children[i].position[0]
                            + children[j].position[0]
                            + children[k].position[0])
                            / 3.0,
                        (children[i].position[1]
                            + children[j].position[1]
                            + children[k].position[1])
                            / 3.0,
                    ];
                    if triangulated_at(children, [i, j, k], centroid, eps).is_some() {
                        kept.push([i, j, k]);
                    }
                }
            }
        }
        kept
    }

    /// The partition contract of the triangulation over `children`:
    /// the surviving triangles tile the hull exactly (area sum = hull
    /// area — a hole or an overlap both break the equality), every grid
    /// point strictly inside the hull folds inside a surviving triangle
    /// (never `clamp_to_hull`), and neighbouring grid points fold to
    /// nearby weights. `children` needs at least three non-collinear
    /// points.
    #[track_caller]
    fn assert_partition(children: &[TreeChild2d]) {
        let kept = survivors(children);
        let mut hull = [0usize; FREEFORM_MAX_POINTS];
        let hull_len = hull_indices(children, &mut hull);
        let mut hull_area = 0f64;
        for edge in 1..(hull_len - 1) {
            hull_area += orient(
                children[hull[0]].position,
                children[hull[edge]].position,
                children[hull[edge + 1]].position,
            );
        }
        let kept_area: f64 = kept
            .iter()
            .map(|&[i, j, k]| {
                orient(
                    children[i].position,
                    children[j].position,
                    children[k].position,
                )
                .abs()
            })
            .sum();
        assert!(
            (kept_area - hull_area).abs() <= 1e-6 * hull_area.max(1.0),
            "kept area {kept_area} vs hull area {hull_area} over {:?} — holes or overlaps",
            kept
        );
        let extent = set_span(children) as f32;
        let step = extent / 16.0;
        let delta = extent * 1e-3;
        let lo = bounds(children);
        for gx in 0..16 {
            for gy in 0..16 {
                let value = [
                    lo[0] + step * (gx as f32 + 0.37),
                    lo[1] + step * (gy as f32 + 0.29),
                ];
                if !inside_hull(children, value) {
                    continue;
                }
                assert!(
                    kept.iter().any(
                        |&t| triangulated_at(children, t, value, tolerances(children)).is_some()
                    ),
                    "hole at {value:?} — the fold would clamp inside the hull"
                );
                for step in [[delta, 0.0], [0.0, delta]] {
                    let jump = l1(
                        &dense(children, value),
                        &dense(children, [value[0] + step[0], value[1] + step[1]]),
                    );
                    assert!(jump < 0.05, "weight jump {jump} at {value:?} step {step:?}");
                }
            }
        }
    }

    /// Lowest corner of the `children` bounding box.
    fn bounds(children: &[TreeChild2d]) -> [f32; 2] {
        let mut lo = [f32::INFINITY; 2];
        for child in children {
            for (axis, v) in child.position.iter().enumerate() {
                lo[axis] = lo[axis].min(*v);
            }
        }
        lo
    }

    /// Every reordering of four children — the tie-break must not
    /// depend on the order the points arrive in.
    const PERMUTATIONS_4: [[usize; 4]; 24] = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];

    /// `n` directions (`cos`/`sin` of `2πk/n`, f32) at `radius` — the
    /// speed × direction blend spaces — optionally with the centre
    /// point; the f32 rounding keeps the rings only *nearly*
    /// cocircular.
    fn ngon(n: usize, radius: f32, centre: bool) -> Vec<TreeChild2d> {
        let mut points: Vec<[f32; 2]> = (0..n)
            .map(|k| {
                let angle = k as f32 * core::f32::consts::TAU / n as f32;
                [angle.cos() * radius, angle.sin() * radius]
            })
            .collect();
        if centre {
            points.push([0.0, 0.0]);
        }
        freeform_n(&points)
    }

    /// The reviewer's hole repro: the square in the child order
    /// `(0,0) (2,2) (2,0) (0,2)` — the kept diagonal is geometric, not
    /// index-based, so the survivors still tile the hull and no
    /// interior grid point falls into `clamp_to_hull`.
    #[test]
    fn permuted_square_partitions_the_hull() {
        assert_partition(&freeform_n(&[
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]));
    }

    /// The reviewer's second repro: the diamond in the child order
    /// `(1,0) (-1,0) (0,1) (0,-1)`.
    #[test]
    fn permuted_rhombus_partitions_the_hull() {
        assert_partition(&freeform_n(&[
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]));
    }

    /// Every one of the 24 orderings of the square tiles the hull —
    /// the tie-break reads coordinates, not arrival order.
    #[test]
    fn square_permutations_all_partition() {
        let base = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        for perm in PERMUTATIONS_4 {
            let ordered: Vec<[f32; 2]> = perm.iter().map(|&i| base[i]).collect();
            assert_partition(&freeform_n(&ordered));
        }
    }

    /// The eight-direction octagon at r = 1, 3 and 300 (a cm/s speed
    /// axis), centre or not: the `span⁴` band catches the f32 rounding
    /// at every scale, so the ring tiles instead of collapsing into
    /// `clamp_to_hull` or overlapping fans.
    #[test]
    fn octagons_partition_the_hull_at_every_scale() {
        for radius in [1.0f32, 3.0, 300.0] {
            for centre in [false, true] {
                assert_partition(&ngon(8, radius, centre));
            }
        }
    }

    #[test]
    fn one_and_two_points_own_their_span() {
        let one = vec![TreeChild2d {
            position: [1.0, 1.0],
            node: 0,
        }];
        assert_eq!(weights_of(&one, [9.0, -9.0]), vec![(0, 1.0)]);
        let two = vec![
            TreeChild2d {
                position: [0.0, 0.0],
                node: 0,
            },
            TreeChild2d {
                position: [4.0, 0.0],
                node: 1,
            },
        ];
        assert_eq!(weights_of(&two, [1.0, 0.0]), vec![(0, 0.75), (1, 0.25)]);
        assert_eq!(weights_of(&two, [-2.0, 0.0]), vec![(0, 1.0)]);
        assert_eq!(weights_of(&[], [0.0, 0.0]).len(), 0);
    }

    /// The on-the-fly fold of r-38 — the reference the cached hot path
    /// must match bit for bit (r-39 acceptance).
    fn freeform2d_weights_uncached(children: &[TreeChild2d], value: [f32; 2]) -> FreeformSlots {
        match children.len() {
            0 => return FreeformSlots::EMPTY,
            1 => return compact([(children[0].node, 1.0), (0, 0.0), (0, 0.0)]),
            count if count > FREEFORM_MAX_POINTS => return FreeformSlots::EMPTY,
            _ => {}
        }
        let eps = tolerances(children);
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                for k in (j + 1)..children.len() {
                    if let Some(weights) = triangulated_at(children, [i, j, k], value, eps) {
                        return compact([
                            (children[i].node, weights[0]),
                            (children[j].node, weights[1]),
                            (children[k].node, weights[2]),
                        ]);
                    }
                }
            }
        }
        compact(clamp_to_hull(children, value))
    }

    /// Probe values over the `children` bounding box plus four outside
    /// offsets — the comparison must cover the clamp path too.
    fn probes(children: &[TreeChild2d]) -> Vec<[f32; 2]> {
        let lo = bounds(children);
        let extent = set_span(children) as f32;
        let step = extent / 6.0;
        let mut out = Vec::new();
        for gx in 0..7 {
            for gy in 0..7 {
                out.push([lo[0] + step * gx as f32, lo[1] + step * gy as f32]);
            }
        }
        out.push([lo[0] - extent, lo[1] - extent]);
        out.push([lo[0] + 2.0 * extent, lo[1] + 0.5 * extent]);
        out.push([lo[0] + 0.5 * extent, lo[1] + 2.0 * extent]);
        out.push([lo[0] + 2.0 * extent, lo[1] + 2.0 * extent]);
        out
    }

    /// Deterministic xorshift noise in `[0, 1)` — no rand dependency.
    fn noise(seed: &mut u32) -> f32 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *seed = x;
        (x >> 8) as f32 / 16_777_216.0
    }

    /// The cached hot fold is bit-for-bit the on-the-fly fold on every
    /// reviewed set: same survivors, same scan order, same arithmetic
    /// (r-39 acceptance — «веса с кэшем совпадают со старым путём»).
    #[test]
    fn cached_weights_match_the_uncached_fold_bit_for_bit() {
        let base = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let mut sets: Vec<(String, Vec<TreeChild2d>)> = Vec::new();
        for perm in PERMUTATIONS_4 {
            sets.push((
                format!("square {perm:?}"),
                freeform_n(&perm.map(|i| base[i])),
            ));
        }
        sets.push((
            "square + centre".to_string(),
            freeform_n(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]]),
        ));
        sets.push((
            "rhombus".to_string(),
            freeform_n(&[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        ));
        for radius in [1.0f32, 3.0, 300.0, 1e-3] {
            for centre in [false, true] {
                sets.push((
                    format!("octagon {radius}/{centre}"),
                    ngon(8, radius, centre),
                ));
            }
        }
        for sides in [5usize, 6, 12] {
            sets.push((format!("{sides}-gon"), ngon(sides, 5.0, true)));
        }
        sets.push(("irregular".to_string(), irregular()));
        for (name, children) in &sets {
            let cache = freeform_cache(children);
            for value in probes(children) {
                assert_eq!(
                    freeform2d_weights_cached(&cache, children, value).as_slice(),
                    freeform2d_weights_uncached(children, value).as_slice(),
                    "{name} at {value:?}"
                );
            }
        }
    }

    /// The hot fold trusts the cache, not the points: forced to the
    /// other diagonal of the square, the fold keeps its triangles — no
    /// incircle pass runs per frame («триангуляция не пересчитывается»).
    #[test]
    fn the_cache_decides_without_retriangulating() {
        let children = freeform_n(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        let mut forced = FreeformCache::default();
        forced.kept[0] = [0, 1, 2];
        forced.kept[1] = [0, 2, 3];
        forced.count = 2;
        let value = [1.7, 0.2];
        let cached = freeform2d_weights_cached(&forced, &children, value);
        let support =
            |slots: &[(usize, f32)]| slots.iter().map(|(node, _)| *node).collect::<Vec<_>>();
        assert_eq!(
            support(cached.as_slice()),
            vec![0, 1, 2],
            "the forced cache's triangle decides"
        );
        let on_the_fly = freeform2d_weights_uncached(&children, value);
        assert_eq!(support(on_the_fly.as_slice()), vec![0, 1, 3]);
        assert_ne!(cached.as_slice(), on_the_fly.as_slice());
    }

    /// Near-cocircular rings with f32-scale jitter stay a partition —
    /// the `span⁴` band folds the whole ring consistently (Grok's
    /// property-test request): no holes, no overlaps, no weight jumps.
    #[test]
    fn jittered_circles_partition_cleanly() {
        let mut seed = 0x9E37_79B9u32;
        for round in 0..12 {
            for sides in [8usize, 12] {
                let mut points: Vec<[f32; 2]> = (0..sides)
                    .map(|k| {
                        let angle = k as f32 * core::f32::consts::TAU / sides as f32;
                        let wobble = 4e-7;
                        [
                            angle.cos() + (noise(&mut seed) - 0.5) * wobble,
                            angle.sin() + (noise(&mut seed) - 0.5) * wobble,
                        ]
                    })
                    .collect();
                if round % 2 == 0 {
                    points.push([noise(&mut seed) * 0.4 - 0.2, noise(&mut seed) * 0.4 - 0.2]);
                }
                assert_partition(&freeform_n(&points));
            }
        }
    }
}
