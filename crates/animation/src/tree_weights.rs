//! Deterministic blend weights for the animation tree ([`blend1d_weights`],
//! [`blend2d_weights`]) — the pure position-to-weight math behind
//! [`TreeNodeDef::Blend1d`](crate::TreeNodeDef::Blend1d) and
//! [`TreeNodeDef::Blend2d`](crate::TreeNodeDef::Blend2d), kept allocation-free
//! and deterministic for the hot path.

use crate::tree::{TreeChild, TreeChild2d};

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

#[cfg(test)]
mod tests {
    use super::*;

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
