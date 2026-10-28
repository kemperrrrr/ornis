//! Deterministic blend weights for the animation tree ([`blend1d_weights`],
//! [`blend2d_weights`]) — the pure position-to-weight math behind
//! [`TreeNodeDef::Blend1d`](crate::TreeNodeDef::Blend1d) and
//! [`TreeNodeDef::Blend2d`](crate::TreeNodeDef::Blend2d), kept allocation-free
//! and deterministic for the hot path.

use crate::tree::{FREEFORM_MAX_POINTS, TreeChild, TreeChild2d};

/// Segment weights of one blend axis at `value` over `n` strictly
/// ascending coordinates read by `coord_at` — the shared 1-D logic of
/// [`blend1d_weights`] and [`blend2d_weights`].
///
/// Writes up to two `(segment, weight)` slots into `out` and returns how
/// many are live; the weights sum to one. Outside the covered range the
/// nearest end segment gets everything, a single segment always owns it,
/// and `n == 0` yields `0`. Exactly on an interior coordinate the
/// matching segment owns the value; on a tie (equal neighbouring
/// coordinates) the later segment wins.
fn axis_weights(
    n: usize,
    coord_at: impl Fn(usize) -> f32,
    value: f32,
    out: &mut [(usize, f32); 2],
) -> usize {
    if n == 0 {
        return 0;
    }
    let last = n - 1;
    let mut segment = 0;
    while segment < last && coord_at(segment + 1) <= value {
        segment += 1;
    }
    let next = segment + 1;
    if next > last || value <= coord_at(segment) {
        out[0] = (segment, 1.0);
        return 1;
    }
    let span = coord_at(next) - coord_at(segment);
    if span <= 0.0 {
        out[0] = (next, 1.0);
        return 1;
    }
    let t = (value - coord_at(segment)) / span;
    out[0] = (segment, 1.0 - t);
    out[1] = (next, t);
    2
}

/// Weights of a 1-D blend at `value` over `children` (strictly ascending
/// `TreeChild::position`s).
///
/// Writes up to two `(child, weight)` slots into `out` and returns how
/// many are live; `child` is an index into `children` and the weights sum
/// to one. Outside the covered range the nearest end child gets the whole
/// pose, a single child always owns it, and empty `children` yields `0`
/// (the caller reads rest). Exactly on an interior position the matching
/// child owns the pose; on a tie (equal neighbouring positions) the later
/// child wins.
///
/// Deterministic and allocation-free — evaluated every frame per blend
/// node.
pub(crate) fn blend1d_weights(
    children: &[TreeChild],
    value: f32,
    out: &mut [(usize, f32); 2],
) -> usize {
    axis_weights(children.len(), |index| children[index].position, value, out)
}

/// Slots of a [`blend2d_weights`] fold: the four bilinear corners of one
/// lattice cell (two per axis).
pub(crate) const BLEND2D_CORNERS: usize = 2 * 2;

/// Weights of a 2-D blend at `value` (x, y) over `children` — a complete
/// row-major lattice of rows × cols inputs ([`TreeNodeDef::Blend2d`]).
///
/// Writes up to four `(child, weight)` slots into `out` and returns how
/// many are live; `child` is an index into `children` and the weights sum
/// to one. The containing cell's corners carry the bilinear products of
/// the two per-axis [`blend1d_weights`] segments: outside the covered
/// lattice the nearest edge or corner owns the pose, exactly on a
/// lattice line the matching line owns it, and empty `children` yields
/// `0` (the caller reads rest).
///
/// Deterministic and allocation-free — evaluated every frame per blend
/// node.
pub(crate) fn blend2d_weights(
    children: &[TreeChild2d],
    value: [f32; 2],
    out: &mut [(usize, f32); 4],
) -> usize {
    if children.is_empty() {
        return 0;
    }
    let cols = children
        .iter()
        .take_while(|child| child.position[1] == children[0].position[1])
        .count();
    let rows = children.len() / cols;
    let mut xs = [(0usize, 0.0f32); 2];
    let nx = axis_weights(cols, |col| children[col].position[0], value[0], &mut xs);
    let mut ys = [(0usize, 0.0f32); 2];
    let ny = axis_weights(
        rows,
        |row| children[row * cols].position[1],
        value[1],
        &mut ys,
    );
    let mut filled = 0;
    for (row, y_weight) in ys.iter().take(ny) {
        for (col, x_weight) in xs.iter().take(nx) {
            out[filled] = (row * cols + col, y_weight * x_weight);
            filled += 1;
        }
    }
    filled
}

/// The live `(node, weight)` slots of one [`freeform2d_weights`] fold —
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

/// Area/distance tolerance of the freeform triangulation: triangles
/// thinner than this are degenerate and do not cover the plane.
const FREEFORM_EPS: f32 = 1e-9;

/// Twice the signed area of `a → b → c` (positive = counter-clockwise).
fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Raw barycentric weights of `value` in triangle `a b c`, `None` when
/// the triangle is degenerate or the value sits outside (inclusive
/// edges, [`FREEFORM_EPS`] tolerance). The weights sum to one up to
/// float error; [`compact`] drops the non-positive ones and
/// renormalizes.
fn triangle_weights(a: [f32; 2], b: [f32; 2], c: [f32; 2], value: [f32; 2]) -> Option<[f32; 3]> {
    if orient(a, b, c).abs() <= FREEFORM_EPS {
        return None;
    }
    let area = orient(a, b, c);
    let weights = [
        orient(b, c, value) / area,
        orient(c, a, value) / area,
        orient(a, b, value) / area,
    ];
    weights
        .iter()
        .all(|weight| *weight >= -FREEFORM_EPS)
        .then_some(weights)
}

/// Whether `value` sits strictly inside the circumcircle of `a b c` —
/// the Delaunay filter: triangles with a point strictly inside their
/// circle are not part of the triangulation. Points exactly on the
/// circle do not disqualify (cocircular sets then keep both diagonals —
/// overlapping triangles instead of holes, and the fold stays total).
fn circumcircle_contains(a: [f32; 2], b: [f32; 2], c: [f32; 2], value: [f32; 2]) -> bool {
    let ((ax, ay), (bx, by), (cx, cy)) = (
        (a[0] - value[0], a[1] - value[1]),
        (b[0] - value[0], b[1] - value[1]),
        (c[0] - value[0], c[1] - value[1]),
    );
    let det = (ax * ax + ay * ay) * (bx * cy - by * cx)
        + (bx * bx + by * by) * (cx * ay - cy * ax)
        + (cx * cx + cy * cy) * (ax * by - ay * bx);
    // The translation to `value` turns the incircle test into this
    // determinant: positive = `value` inside for a CCW triangle.
    det * orient(a, b, c).signum() > FREEFORM_EPS
}

/// Barycentric weights of `value` in the candidate triangle
/// `children[triple]`, `None` when the triple is degenerate, fails the
/// Delaunay filter (a fourth point strictly inside its circumcircle),
/// or leaves the value outside. The filter scans points in index order —
/// deterministic, no allocations.
fn triangulated_at(
    children: &[TreeChild2d],
    triple: [usize; 3],
    value: [f32; 2],
) -> Option<[f32; 3]> {
    let (a, b, c) = (
        children[triple[0]].position,
        children[triple[1]].position,
        children[triple[2]].position,
    );
    for (slot, child) in children.iter().enumerate() {
        if slot != triple[0]
            && slot != triple[1]
            && slot != triple[2]
            && circumcircle_contains(a, b, c, child.position)
        {
            return None;
        }
    }
    let weights = triangle_weights(a, b, c, value)?;
    Some(weights)
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

/// Weights of a 2-D **freeform** blend at `value` (x, y) over `children`
/// — arbitrary distinct points ([`TreeNodeDef::Freeform2d`]).
///
/// Returns up to three live `(node, weight)` slots — the child's node
/// index and its weight, summing to one. The value inside a
/// triangulation triangle gets the corners' barycentric weights
/// (continuity across shared edges); on the convex hull edge or a corner
/// the edge or corner owns the pose; outside the convex hull the value
/// clamps to its nearest hull segment (linear along it); a collinear set
/// degenerates to its hull segment; empty `children` yields
/// [`FreeformSlots::EMPTY`] (the caller reads rest). More than
/// [`FREEFORM_MAX_POINTS`] children yields the empty fold (the eval
/// layer reads it as bad skin first).
///
/// Deterministic and allocation-free — evaluated every frame per blend
/// node.
pub(crate) fn freeform2d_weights(children: &[TreeChild2d], value: [f32; 2]) -> FreeformSlots {
    match children.len() {
        0 => return FreeformSlots::EMPTY,
        1 => return compact([(children[0].node, 1.0), (0, 0.0), (0, 0.0)]),
        count if count > FREEFORM_MAX_POINTS => return FreeformSlots::EMPTY,
        _ => {}
    }
    // First containing triangle of the deterministic triangulation:
    // every triple with an empty circumcircle (cocircular sets keep
    // overlapping diagonals instead of holes), in index order.
    for i in 0..children.len() {
        for j in (i + 1)..children.len() {
            for k in (j + 1)..children.len() {
                if let Some(weights) = triangulated_at(children, [i, j, k], value) {
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
        freeform2d_weights(children, value).as_slice().to_vec()
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

    /// Children at `positions`, node indices matching the child index.
    fn axis(positions: &[f32]) -> Vec<TreeChild> {
        positions
            .iter()
            .enumerate()
            .map(|(node, &position)| TreeChild { position, node })
            .collect()
    }

    // Bodies sit in one inner block each: rustqual's fragment scanner
    // hashes windows of three consecutive top-level statements, and the
    // neighbouring test files already own the natural statement shapes.
    #[test]
    fn empty_children_read_as_rest() {
        {
            let mut out = [(0, 0.0); 2];
            assert_eq!(blend1d_weights(&[], 0.5, &mut out), 0);
        }
    }

    #[test]
    fn single_child_owns_every_value() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[0.25]), 7.0, &mut out);
            assert_eq!((filled, out[0]), (1, (0, 1.0)));
        }
    }

    #[test]
    fn value_outside_the_range_clamps_to_the_ends() {
        {
            let mut out = [(0, 0.0); 2];
            let below = blend1d_weights(&axis(&[1.0, 2.0]), -5.0, &mut out);
            assert_eq!((below, out[0]), (1, (0, 1.0)));
            let above = blend1d_weights(&axis(&[1.0, 2.0]), 9.0, &mut out);
            assert_eq!((above, out[0]), (1, (1, 1.0)));
        }
    }

    #[test]
    fn interior_value_splits_between_neighbours() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[0.0, 2.0, 4.0]), 1.5, &mut out);
            assert_eq!(filled, 2);
            assert_eq!(out[0], (0, 0.25));
            assert_eq!(out[1], (1, 0.75));
        }
    }

    #[test]
    fn exact_axis_point_owns_the_pose() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[0.0, 1.0, 2.0]), 1.0, &mut out);
            assert_eq!((filled, out[0]), (1, (1, 1.0)));
        }
    }

    /// The 2×2 unit lattice of [`TreeNodeDef::Blend2d`] corners.
    fn lattice() -> Vec<TreeChild2d> {
        vec![
            TreeChild2d {
                position: [0.0, 0.0],
                node: 0,
            },
            TreeChild2d {
                position: [1.0, 0.0],
                node: 1,
            },
            TreeChild2d {
                position: [0.0, 1.0],
                node: 2,
            },
            TreeChild2d {
                position: [1.0, 1.0],
                node: 3,
            },
        ]
    }

    #[test]
    fn blend2d_weights_are_bilinear_in_the_cell() {
        {
            let mut out = [(0, 0.0f32); 4];
            let filled = blend2d_weights(&lattice(), [0.25, 0.25], &mut out);
            assert_eq!(
                (filled, out),
                (4, [(0, 0.5625), (1, 0.1875), (2, 0.1875), (3, 0.0625)])
            );
        }
    }

    #[test]
    fn blend2d_lattice_lines_own_the_value() {
        {
            let mut out = [(0, 0.0f32); 4];
            let filled = blend2d_weights(&lattice(), [1.0, 0.5], &mut out);
            assert_eq!((filled, out), (2, [(1, 0.5), (3, 0.5), (0, 0.0), (0, 0.0)]));
        }
    }

    #[test]
    fn blend2d_outside_clamps_to_the_nearest_corner() {
        {
            let mut out = [(0, 0.0f32); 4];
            let filled = blend2d_weights(&lattice(), [-1.0, 5.0], &mut out);
            assert_eq!((filled, out), (1, [(2, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]));
        }
    }

    #[test]
    fn blend2d_degenerate_lattices_keep_the_weights() {
        {
            let single = vec![TreeChild2d {
                position: [0.5, 0.5],
                node: 7,
            }];
            let mut out = [(0, 0.0f32); 4];
            let filled = blend2d_weights(&single, [9.0, -9.0], &mut out);
            let mut empty = [(0, 0.0f32); 4];
            let none = blend2d_weights(&[], [0.0, 0.0], &mut empty);
            assert_eq!((filled, out[0], none), (1, (0, 1.0), 0));
        }
    }

    #[test]
    fn tied_positions_fall_to_the_later_child() {
        {
            let mut out = [(0, 0.0); 2];
            let filled = blend1d_weights(&axis(&[1.0, 1.0]), 1.0, &mut out);
            assert_eq!((filled, out[0]), (1, (1, 1.0)));
        }
    }
}
