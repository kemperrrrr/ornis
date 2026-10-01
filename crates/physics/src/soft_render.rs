//! Tube mesh soup for rope/chain soft bodies (PLAN B2/D1 leftover #3).
//!
//! Chains ([`crate::soft::SoftBody::chain`]) carry particles but no render
//! `surface`, so the surface-upload path has nothing to draw. This module
//! builds a render-only tube along the particle polyline instead: one ring of
//! `sides` vertices per particle, consecutive rings stitched with side quads
//! (two triangles each). The soup is world-space — the bridge uploads it as a
//! `Custom` mesh with an identity transform, exactly like the cloth path.
//!
//! Frames propagate along the chain by parallel transport
//! (rotation-minimizing): each ring reuses the previous ring's normal
//! projected onto the plane perpendicular to the new tangent, so straight
//! sections never twist-pop. Zero-length segments (duplicated particles)
//! reuse the last valid tangent instead of normalizing a zero vector, so the
//! output never contains NaN for finite inputs. Tube ends are open (no caps
//! are emitted) — ropes vanish into anchors and pulleys rather than showing
//! a flat disc.

use glam::Vec3;

use crate::soft::Particle;

/// Fallback tangent when no chain segment has a valid direction (all
/// particles stacked): matches the default `dir` of
/// [`crate::soft::SoftBody::chain`], so a fully degenerate chain still forms
/// a finite tube along −Y instead of emitting NaN.
const FALLBACK_TANGENT: Vec3 = Vec3::NEG_Y;

/// Squared length below which a chain segment counts as degenerate
/// (zero-length) and reuses the last valid tangent.
const MIN_SEGMENT_LEN_SQ: f32 = crate::constants::DEGENERATE_LEN2;
/// Minimum ring sides for a usable tube cross-section (a triangle).
const MIN_TUBE_SIDES: u32 = 3;
/// Indices per side quad (two triangles).
const INDICES_PER_QUAD: usize = 6;

/// Triangle index list for a tube of `count` rings with `sides` vertices each.
///
/// Rings are laid out linearly (`ring * sides + side`); consecutive rings are
/// stitched with `sides` side quads (two triangles each, wound CCW from
/// outside to match the `Custom` soup convention). Both ends stay open — no
/// caps are emitted (see the module docs).
///
/// Returns exactly `(count - 1) * sides * 6` indices, or an empty vector when
/// no tube can be formed (`count < 2` or `sides < 3`).
pub fn tube_indices(count: usize, sides: u32) -> Vec<u32> {
    if count < 2 || sides < MIN_TUBE_SIDES {
        return Vec::new();
    }
    let sides_usize = sides as usize;
    let mut indices = Vec::with_capacity((count - 1) * sides_usize * INDICES_PER_QUAD);
    for ring in 0..count - 1 {
        let r0 = (ring * sides_usize) as u32;
        let r1 = ((ring + 1) * sides_usize) as u32;
        for side in 0..sides {
            let next = (side + 1) % sides;
            let a = r0 + side;
            let b = r0 + next;
            let c = r1 + next;
            let d = r1 + side;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    indices
}

/// World-space ring vertices for a tube of `radius` around the particle polyline.
///
/// Emits `particles.len() * sides` vertices (`sides` per ring, starting at the
/// transported normal and sweeping toward the binormal), each at distance
/// `radius` from its particle. Ring frames use parallel-transport propagation
/// along per-ring tangents (central differences inside, one-sided at the
/// ends), so straight sections share one stable frame with no twist popping.
/// Degenerate segments reuse the last valid tangent (leading runs use the
/// first valid one); every output is finite for finite inputs — never NaN.
///
/// Returns an empty vector when no tube can be formed (`particles.len() < 2`,
/// `sides < 3`, or a non-positive/non-finite `radius`).
pub fn tube_positions(particles: &[Particle], radius: f32, sides: u32) -> Vec<[f32; 3]> {
    let count = particles.len();
    if count < 2 || sides < MIN_TUBE_SIDES || !radius.is_finite() || radius <= 0.0 {
        return Vec::new();
    }
    let centers: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let tangents = chain_tangents(&centers);
    let sides_usize = sides as usize;
    let mut out = Vec::with_capacity(count * sides_usize);
    let mut normal = perpendicular(tangents[0]);
    for (i, center) in centers.iter().enumerate() {
        let tangent = tangents[i];
        if i > 0 {
            normal = transport(normal, tangent);
        }
        let binormal = tangent.cross(normal);
        for side in 0..sides {
            let theta = std::f32::consts::TAU * (side as f32) / (sides as f32);
            let (sin, cos) = theta.sin_cos();
            out.push((center + (normal * cos + binormal * sin) * radius).to_array());
        }
    }
    out
}

/// Per-ring unit tangents along the polyline: central differences inside,
/// one-sided differences at the ends. Degenerate (zero-length) segments reuse
/// the last valid tangent; leading degenerates (no past tangent yet) are
/// backfilled with the first valid tangent, or [`FALLBACK_TANGENT`] when the
/// whole chain is stacked. Always finite and unit-length for finite inputs.
fn chain_tangents(centers: &[Vec3]) -> Vec<Vec3> {
    let mut tangents: Vec<Option<Vec3>> = Vec::with_capacity(centers.len());
    let mut last_valid: Option<Vec3> = None;
    for (i, center) in centers.iter().enumerate() {
        let next = if i + 1 < centers.len() {
            centers[i + 1]
        } else {
            *center
        };
        let prev = if i > 0 { centers[i - 1] } else { *center };
        let delta = next - prev;
        if delta.length_squared() > MIN_SEGMENT_LEN_SQ {
            last_valid = Some(delta.normalize());
        }
        tangents.push(last_valid);
    }
    let fill = tangents.iter().find_map(|t| *t).unwrap_or(FALLBACK_TANGENT);
    tangents.into_iter().map(|t| t.unwrap_or(fill)).collect()
}

/// Parallel-transports `normal` onto the plane perpendicular to `tangent`
/// (rotation-minimizing frame propagation — no twist on straight sections).
/// A ~90° kink can collapse the projection onto the tangent; that case picks
/// a fresh perpendicular instead of normalizing a near-zero vector.
fn transport(normal: Vec3, tangent: Vec3) -> Vec3 {
    let projected = normal - tangent * normal.dot(tangent);
    if projected.length_squared() > MIN_SEGMENT_LEN_SQ {
        projected.normalize()
    } else {
        perpendicular(tangent)
    }
}

/// Any unit vector perpendicular to `tangent`: subtracts the tangent
/// component from the least-aligned coordinate axis (never near-parallel by
/// construction) and normalizes.
fn perpendicular(tangent: Vec3) -> Vec3 {
    let abs = tangent.abs();
    let reference = if abs.x <= abs.y && abs.x <= abs.z {
        Vec3::X
    } else if abs.y <= abs.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    (reference - tangent * tangent.dot(reference)).normalize_or(Vec3::X)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::SoftBody;

    fn straight_chain(count: usize) -> SoftBody {
        SoftBody::chain(Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y, count, 0.25, 1.0, 0.0)
    }

    #[test]
    fn tube_indices_cover_side_quads() {
        let count = 5;
        let sides = 6;
        let indices = tube_indices(count, sides);
        assert_eq!(indices.len(), (count - 1) * sides as usize * 6);
        assert!(
            indices.iter().all(|&i| i < (count * sides as usize) as u32),
            "every index references a ring vertex"
        );
        assert!(
            tube_indices(1, sides).is_empty(),
            "single ring has no sides"
        );
        assert!(tube_indices(0, sides).is_empty());
        assert!(tube_indices(count, 2).is_empty(), "a tube needs 3+ sides");
    }

    #[test]
    fn straight_tube_vertices_hold_radius_without_twist() {
        let body = straight_chain(6);
        let radius = 0.05;
        let sides = 8u32;
        let verts = tube_positions(&body.particles, radius, sides);
        assert_eq!(verts.len(), 6 * sides as usize);
        // All vertices sit at `radius` from the chain axis (here −Y through
        // the origin column), with no axial drift out of their ring plane.
        for (i, particle) in body.particles.iter().enumerate() {
            let center = particle.position;
            for s in 0..sides as usize {
                let v = Vec3::from_array(verts[i * sides as usize + s]);
                let radial = v - center;
                let dist = Vec3::new(radial.x, 0.0, radial.z).length();
                assert!((dist - radius).abs() < 1e-4, "ring {i} side {s}: {dist}");
                assert!(radial.y.abs() < 1e-4, "ring {i} side {s} stays planar");
            }
        }
        // No twist pop on the straight run: every ring repeats ring 0's frame.
        for i in 1..6 {
            for s in 0..sides as usize {
                let a = Vec3::from_array(verts[s]);
                let b = Vec3::from_array(verts[i * sides as usize + s]);
                let da = a - body.particles[0].position;
                let db = b - body.particles[i].position;
                assert!((da - db).length() < 1e-5, "ring {i} side {s} twisted");
            }
        }
    }

    #[test]
    fn degenerate_segment_reuses_tangent_without_nan() {
        let mut body = straight_chain(4);
        body.particles[2].position = body.particles[1].position;
        let verts = tube_positions(&body.particles, 0.05, 6);
        assert_eq!(verts.len(), 4 * 6);
        assert!(
            verts.iter().flatten().all(|v| v.is_finite()),
            "zero-length segment must not emit NaN"
        );
        // The duplicated ring still forms a full-radius circle (reused frame,
        // not a collapse onto the axis).
        let center = body.particles[2].position;
        for s in 0..6 {
            let v = Vec3::from_array(verts[2 * 6 + s]);
            let dist = (v - center).length();
            assert!(
                (dist - 0.05).abs() < 1e-4,
                "degenerate ring side {s}: {dist}"
            );
        }
    }
}
