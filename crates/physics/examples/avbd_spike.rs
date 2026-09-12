//! THROWAWAY SPIKE (M0, PLAN B2): minimal rigid-AVBD port for a head-to-head
//! against SI on the 4-box stack scene. NOT production code — delete after
//! the verdict in `spikes/001-avbd-stack/README.md`.
//!
//! Faithful port of the `Solver`/`Manifold` update rules from
//! savant117/avbd-demo3d (MIT, Giles SIGGRAPH'25), second attempt: the first
//! attempt guessed the dual rules (live gap, multiplicative x2 penalty ramp,
//! incremental lambda) and produced coupling instability. This version keeps
//! the official formulation:
//! - constraint value is a Taylor approximation around step start,
//!   `C = C0*(1-alpha) + J*dq` (Sec. 4), not the live gap;
//! - penalty ramp is ADDITIVE (`penalty += beta*|C|`, Eq. 16), not x2;
//! - dual ASSIGNS the clamped force (`lambda = F`), warmstarted per step via
//!   Eq. 19 (`lambda *= alpha*gamma`, penalty decayed with gamma).
//!
//! Conscious simplifications vs the official demo (all documented, none silent):
//! - boxes stay ~axis-aligned; rotation is a small-angle rotation vector,
//!   corner offsets rotate as `r + rot x r`, inertia stays body-frame diagonal;
//! - contacts are "4 bottom corners vs support height" (floor plane or the top
//!   face of the box below found by XZ overlap) — exact for aligned stacks,
//!   not a general narrowphase; no feature tracking, anchors always refreshed
//!   (the demo's `!stick` case);
//! - shared tangential penalty for both tangent rows (demo: per-axis float3);
//! - no restitution (SI scene uses 0.3; settle comparison stays valid);
//! - contacts couple both bodies (upper +n side, lower -n side), statics skip
//!   their own solve; sweep order is bottom-to-top (helps GS stacks).

use glam::Vec3;
use ornis_physics::{BodyHandle, BuiltinPhysicsEngine, PhysicsEngine, RigidBody};
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;
const ITERS: usize = 10;
const STEPS: usize = 300;
const GRAVITY: f32 = -9.81;
// Official defaults (Solver::defaultParams + solver.h).
const ALPHA: f32 = 0.99;
const GAMMA: f32 = 0.999;
const BETA: f32 = 10000.0;
const MARGIN: f32 = 0.01;
const PENALTY_INIT: f32 = 1.0;
const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1.0e10;
const GEN_MARGIN: f32 = 0.05;
const MU: f32 = 0.5;
const HALF: f32 = 0.5;

#[derive(Clone)]
struct Body {
    pos: Vec3,
    rot: Vec3,
    vel: Vec3,
    angvel: Vec3,
    inv_mass: f32,
    inv_inertia: Vec3,
    is_static: bool,
}

struct Contact {
    upper: usize,
    lower: usize, // body below, or usize::MAX for the floor plane
    corner: Vec3, // body-frame corner offset of the upper box
    support: f32, // support height under the corner (culling + debug)
    c0: f32,      // normal C0 at step start, gap + MARGIN (official: +margin)
    /// Lower-body material anchor at step start (official `rB` equivalent):
    /// MUST be fixed at generation — recomputing it from live positions
    /// folds the upper body's motion into the lower Jacobian and pumps
    /// energy during fast relative motion. Unused for the static floor.
    r_lo0: Vec3,
    penalty_n: f32,
    lambda_n: f32,
    penalty_t: f32,
    lambda_t1: f32,
    lambda_t2: f32,
}

fn rot_small(v: Vec3, r: Vec3) -> Vec3 {
    v + r.cross(v)
}

fn outer(a: Vec3, b: Vec3) -> [[f32; 3]; 3] {
    let av = [a.x, a.y, a.z];
    let bv = [b.x, b.y, b.z];
    [
        [av[0] * bv[0], av[0] * bv[1], av[0] * bv[2]],
        [av[1] * bv[0], av[1] * bv[1], av[1] * bv[2]],
        [av[2] * bv[0], av[2] * bv[1], av[2] * bv[2]],
    ]
}

/// Dense LDL (no pivoting) for a 6x6 SPD system. Returns None on breakdown.
fn solve_6x6(lhs: [[f32; 6]; 6], rhs: [f32; 6]) -> Option<[f32; 6]> {
    let mut l = [[0.0f32; 6]; 6];
    let mut d = [0.0f32; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut s = lhs[i][j];
            for k in 0..j {
                s -= l[i][k] * d[k] * l[j][k];
            }
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                d[i] = s;
                l[i][i] = 1.0;
            } else {
                l[i][j] = s / d[j];
            }
        }
    }
    let mut y = [0.0f32; 6];
    for i in 0..6 {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s;
    }
    let mut z = [0.0f32; 6];
    for i in 0..6 {
        z[i] = y[i] / d[i];
    }
    let mut x = [0.0f32; 6];
    for i in (0..6).rev() {
        let mut s = z[i];
        for k in (i + 1)..6 {
            s -= l[k][i] * x[k];
        }
        x[i] = s;
    }
    Some(x)
}

fn support_height(bodies: &[Body], upper: usize, x: f32, z: f32, corner_y: f32) -> (f32, usize) {
    // Nearest top face with XZ overlap, above OR below the corner: a corner
    // inside a box is a regular penetrated contact (negative gap), not a
    // reason to drop the face. Dropping above-corner faces (the old ceiling
    // filter) silently re-seated deep corners onto the floor and let boxes
    // fall through each other.
    let mut best_h = 0.0f32;
    let mut best_id = usize::MAX;
    let mut best_dist = f32::INFINITY;
    for (i, b) in bodies.iter().enumerate() {
        if i == upper {
            continue;
        }
        // Spike simplification: body 0 is the static floor slab whose contact
        // surface is the y=0 plane (its OBB top at -0.5 is NOT the surface,
        // and its XZ extent is the whole plane, not HALF).
        if i != 0
            && (x < b.pos.x - HALF
                || x > b.pos.x + HALF
                || z < b.pos.z - HALF
                || z > b.pos.z + HALF)
        {
            continue;
        }
        let top = if i == 0 { 0.0 } else { b.pos.y + HALF };
        let d = (corner_y - top).abs();
        if d < best_dist {
            best_dist = d;
            best_h = top;
            best_id = i;
        }
    }
    (best_h, best_id)
}

fn low_side(bodies: &[Body], c: &Contact) -> bool {
    c.lower != usize::MAX && !bodies[c.lower].is_static
}

/// Live corner anchors for a contact: upper corner world now/at step start,
/// lower anchor world now/at step start. The lower anchor is a MATERIAL point
/// (`r_lo0` fixed at generation, rotated live) — never recomputed from the
/// live gap (see `Contact::r_lo0`).
fn anchors(
    bodies: &[Body],
    _pos0: &[Vec3],
    rot0: &[Vec3],
    c: &Contact,
) -> (Vec3, Vec3, Vec3, Vec3) {
    let r_up = rot_small(c.corner, bodies[c.upper].rot);
    let r_up0 = rot_small(c.corner, rot0[c.upper]);
    if low_side(bodies, c) {
        let lo = c.lower;
        let r_lo = rot_small(c.r_lo0, bodies[lo].rot);
        let r_lo0 = rot_small(c.r_lo0, rot0[lo]);
        (r_up, r_up0, r_lo, r_lo0)
    } else {
        (r_up, r_up0, Vec3::ZERO, Vec3::ZERO)
    }
}

/// Taylor constraint value `C = C0*(1-alpha) + axis.(dUp - dLow)` (official
/// Sec. 4), with `C0 = c0` for the normal row and `0.0` for tangent rows
/// (anchors are step-start by construction here).
fn rel_c(
    bodies: &[Body],
    pos0: &[Vec3],
    rot0: &[Vec3],
    c: &Contact,
    axis: Vec3,
    c0: f32,
) -> (f32, Vec3, Vec3) {
    let (r_up, r_up0, r_lo, r_lo0) = anchors(bodies, pos0, rot0, c);
    let up = c.upper;
    let d_up = (bodies[up].pos + r_up) - (pos0[up] + r_up0)
        + axis_cross_dot(r_up, axis, bodies[up].rot - rot0[up]);
    let d_low = if low_side(bodies, c) {
        let lo = c.lower;
        (bodies[lo].pos + r_lo) - (pos0[lo] + r_lo0)
            + axis_cross_dot(r_lo, axis, bodies[lo].rot - rot0[lo])
    } else {
        Vec3::ZERO
    };
    // (r x axis).drot contributes axis.((drot x r)) = axis.d(corner) — the
    // rotational part of the corner displacement along the axis.
    (c0 * (1.0 - ALPHA) + axis.dot(d_up - d_low), r_up, r_lo)
}

fn axis_cross_dot(r: Vec3, axis: Vec3, drot: Vec3) -> Vec3 {
    // Returns the vector whose dot with `axis` is the rotational corner
    // displacement along `axis`: axis.((drot x r)) is a scalar, so this
    // contributes it as axis * that scalar.
    axis * (r.cross(axis).dot(drot))
}

/// Clamped contact force triple (official Manifold primal/dual core):
/// normal is push-only, tangent pair shares one friction-cone scale.
fn clamped_force(
    cn: f32,
    pen_n: f32,
    lam_n: f32,
    ct: [f32; 2],
    pen_t: f32,
    lam_t: [f32; 2],
) -> [f32; 3] {
    let fn_raw = pen_n * cn + lam_n;
    let fn_c = fn_raw.min(0.0);
    let ft_raw = [pen_t * ct[0] + lam_t[0], pen_t * ct[1] + lam_t[1]];
    let scale = (ft_raw[0] * ft_raw[0] + ft_raw[1] * ft_raw[1]).sqrt();
    let bound = fn_c.abs() * MU;
    let k = if scale > bound && scale > 0.0 {
        bound / scale
    } else {
        1.0
    };
    [fn_c, ft_raw[0] * k, ft_raw[1] * k]
}

fn avbd_step(bodies: &mut [Body], contacts: &mut Vec<Contact>) {
    // --- contact generation: 4 bottom corners of each dynamic box.
    // Contacts PERSIST across steps (matched by upper body + corner sign);
    // per-step Eq. 19 decay below is the warmstart, not a reset.
    let mut seen = vec![false; contacts.len()];
    for u in 1..bodies.len() {
        for &sx in &[-1.0f32, 1.0] {
            for &sz in &[-1.0f32, 1.0] {
                let corner = Vec3::new(sx * HALF, -HALF, sz * HALF);
                let world = bodies[u].pos + rot_small(corner, bodies[u].rot);
                let (h, lower) = support_height(bodies, u, world.x, world.z, world.y);
                let found = contacts.iter_mut().enumerate().find(|(_, c)| {
                    c.upper == u
                        && (c.corner.x - corner.x).abs() < 1e-6
                        && (c.corner.z - corner.z).abs() < 1e-6
                });
                match found {
                    Some((idx, c)) => {
                        if world.y < h + GEN_MARGIN {
                            seen[idx] = true;
                            c.support = h;
                            c.lower = lower;
                            c.c0 = world.y - h + MARGIN;
                            c.r_lo0 = if lower != usize::MAX {
                                world - bodies[lower].pos
                            } else {
                                Vec3::ZERO
                            };
                        }
                    }
                    None => {
                        if world.y < h + GEN_MARGIN {
                            seen.push(true);
                            contacts.push(Contact {
                                upper: u,
                                lower,
                                corner,
                                support: h,
                                c0: world.y - h + MARGIN,
                                r_lo0: if lower != usize::MAX {
                                    world - bodies[lower].pos
                                } else {
                                    Vec3::ZERO
                                },
                                penalty_n: PENALTY_INIT,
                                lambda_n: 0.0,
                                penalty_t: PENALTY_INIT,
                                lambda_t1: 0.0,
                                lambda_t2: 0.0,
                            });
                        }
                    }
                }
            }
        }
    }
    for idx in (0..seen.len()).rev() {
        if !seen[idx] {
            contacts.remove(idx);
        }
    }

    // --- Eq. 19 warmstart: decay duals + penalties once per step ---
    for c in contacts.iter_mut() {
        c.lambda_n *= ALPHA * GAMMA;
        c.lambda_t1 *= ALPHA * GAMMA;
        c.lambda_t2 *= ALPHA * GAMMA;
        c.penalty_n = c.penalty_n * GAMMA + 0.0;
        c.penalty_n = c.penalty_n.clamp(PENALTY_MIN, PENALTY_MAX);
        c.penalty_t = c.penalty_t.clamp(PENALTY_MIN, PENALTY_MAX);
    }

    // --- inertial + warmstart, save step-start pose ---
    let n = bodies.len();
    let mut inertial = vec![Vec3::ZERO; n];
    let mut inertial_rot = vec![Vec3::ZERO; n];
    let mut pos0 = vec![Vec3::ZERO; n];
    let mut rot0 = vec![Vec3::ZERO; n];
    for (i, b) in bodies.iter_mut().enumerate() {
        pos0[i] = b.pos;
        rot0[i] = b.rot;
        if b.is_static {
            inertial[i] = b.pos;
            inertial_rot[i] = b.rot;
            continue;
        }
        inertial[i] = b.pos + b.vel * DT + Vec3::new(0.0, GRAVITY * DT * DT, 0.0);
        inertial_rot[i] = b.rot + b.angvel * DT;
        // Warmstarted position (full gravity; the demo scales it by accelWeight).
        b.pos = inertial[i];
        b.rot = inertial_rot[i];
    }

    // --- main loop ---
    for _ in 0..ITERS {
        // Primal: GS sweep over dynamic bodies. Order is a real parameter:
        // the official demo sweeps its (LIFO) body list, i.e. effectively
        // top-to-bottom for stacked creation order; AVBD_SWP=1 restores
        // bottom-to-top. Default follows the official order.
        let top_down = std::env::var("AVBD_SWP").is_err();
        for k in 1..n {
            let i = if top_down { n - k } else { k };
            let m_dt2 = 1.0 / bodies[i].inv_mass / (DT * DT);
            let i_dt2 = Vec3::new(
                1.0 / bodies[i].inv_inertia.x,
                1.0 / bodies[i].inv_inertia.y,
                1.0 / bodies[i].inv_inertia.z,
            ) / (DT * DT);
            let mut lhs = [[0.0f32; 6]; 6];
            for (i, row) in lhs.iter_mut().enumerate().take(3) {
                row[i] = m_dt2;
            }
            lhs[3][3] = i_dt2.x;
            lhs[4][4] = i_dt2.y;
            lhs[5][5] = i_dt2.z;
            let mut rhs = [0.0f32; 6];
            let rl = m_dt2 * (bodies[i].pos - inertial[i]);
            rhs[0] = rl.x;
            rhs[1] = rl.y;
            rhs[2] = rl.z;
            let ra = Vec3::new(
                i_dt2.x * (bodies[i].rot - inertial_rot[i]).x,
                i_dt2.y * (bodies[i].rot - inertial_rot[i]).y,
                i_dt2.z * (bodies[i].rot - inertial_rot[i]).z,
            );
            rhs[3] = ra.x;
            rhs[4] = ra.y;
            rhs[5] = ra.z;

            for c in contacts.iter() {
                let is_up = c.upper == i;
                let is_low = low_side(bodies, c) && c.lower == i;
                if !is_up && !is_low {
                    continue;
                }
                let sign = if is_up { 1.0 } else { -1.0 };
                // Row values with joint friction-cone clamp (official core).
                let (cn, r_up, r_lo) = rel_c(bodies, &pos0, &rot0, c, Vec3::Y, c.c0);
                let (ct1, _, _) = rel_c(bodies, &pos0, &rot0, c, Vec3::X, 0.0);
                let (ct2, _, _) = rel_c(bodies, &pos0, &rot0, c, Vec3::Z, 0.0);
                let f = clamped_force(
                    cn,
                    c.penalty_n,
                    c.lambda_n,
                    [ct1, ct2],
                    c.penalty_t,
                    [c.lambda_t1, c.lambda_t2],
                );
                let rows = [
                    (Vec3::Y, c.penalty_n, f[0]),
                    (Vec3::X, c.penalty_t, f[1]),
                    (Vec3::Z, c.penalty_t, f[2]),
                ];
                let r_side = if is_up { r_up } else { r_lo };
                for (axis, pen, fv) in rows {
                    // Own-side gradients: the two sides differ only here.
                    let nn = sign * axis;
                    let t = r_side.cross(nn);
                    let o_nn = outer(nn, nn);
                    let o_tt = outer(t, t);
                    let o_nt = outer(nn, t);
                    for a in 0..3 {
                        for b2 in 0..3 {
                            lhs[a][b2] += pen * o_nn[a][b2];
                            lhs[3 + a][3 + b2] += pen * o_tt[a][b2];
                            lhs[a][3 + b2] += pen * o_nt[a][b2];
                            lhs[3 + a][b2] += pen * o_nt[b2][a];
                        }
                    }
                    rhs[0] += fv * nn.x;
                    rhs[1] += fv * nn.y;
                    rhs[2] += fv * nn.z;
                    rhs[3] += fv * t.x;
                    rhs[4] += fv * t.y;
                    rhs[5] += fv * t.z;
                }
            }

            let neg_rhs = [-rhs[0], -rhs[1], -rhs[2], -rhs[3], -rhs[4], -rhs[5]];
            if let Some(dx) = solve_6x6(lhs, neg_rhs) {
                bodies[i].pos += Vec3::new(dx[0], dx[1], dx[2]);
                bodies[i].rot += Vec3::new(dx[3], dx[4], dx[5]);
            }
        }

        // Dual update per contact (official updateDual core).
        for c in contacts.iter_mut() {
            let (cn, _, _) = rel_c(bodies, &pos0, &rot0, c, Vec3::Y, c.c0);
            let (ct1, _, _) = rel_c(bodies, &pos0, &rot0, c, Vec3::X, 0.0);
            let (ct2, _, _) = rel_c(bodies, &pos0, &rot0, c, Vec3::Z, 0.0);
            let f = clamped_force(
                cn,
                c.penalty_n,
                c.lambda_n,
                [ct1, ct2],
                c.penalty_t,
                [c.lambda_t1, c.lambda_t2],
            );
            c.lambda_n = f[0];
            c.lambda_t1 = f[1];
            c.lambda_t2 = f[2];
            // Eq. 16: additive penalty ramp inside the force bounds.
            if f[0] < 0.0 {
                c.penalty_n = (c.penalty_n + BETA * cn.abs()).min(PENALTY_MAX);
            }
            let t_scale = (f[1] * f[1] + f[2] * f[2]).sqrt() / (f[0].abs() * MU + 1e-12);
            if t_scale <= 1.0 {
                c.penalty_t = (c.penalty_t + BETA * (ct1.abs() + ct2.abs())).min(PENALTY_MAX);
            }
        }
    }

    // --- BDF1 velocities after the final iteration ---
    for i in 1..n {
        bodies[i].vel = (bodies[i].pos - pos0[i]) / DT;
        bodies[i].angvel = (bodies[i].rot - rot0[i]) / DT;
    }
}

fn build_avbd_scene() -> Vec<Body> {
    let mut bodies = vec![Body {
        pos: Vec3::new(0.0, -1.0, 0.0),
        rot: Vec3::ZERO,
        vel: Vec3::ZERO,
        angvel: Vec3::ZERO,
        inv_mass: 0.0,
        inv_inertia: Vec3::ZERO,
        is_static: true,
    }];
    // Unit-box inertia m=1: I = m/3*(hy^2+hz^2) = 1/6 per axis.
    let inv_i = Vec3::splat(6.0);
    for level in 0..4 {
        bodies.push(Body {
            // Drop transient (same 1.02 spacing as the SI scene): the stack
            // falls 2cm before engaging. Robustness probe, not the reference.
            pos: Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
            rot: Vec3::ZERO,
            vel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            inv_mass: 1.0,
            inv_inertia: inv_i,
            is_static: false,
        });
    }
    bodies
}

fn build_si_scene() -> (BuiltinPhysicsEngine, Vec<BodyHandle>) {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let mut handles = Vec::new();
    for level in 0..4 {
        handles.push(physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        )));
    }
    (physics, handles)
}

fn main() {
    // --- AVBD run ---
    let mut bodies = build_avbd_scene();
    let mut contacts: Vec<Contact> = Vec::new();
    let mut settle_step = STEPS;
    let t0 = Instant::now();
    for s in 0..STEPS {
        avbd_step(&mut bodies, &mut contacts);
        let max_v = bodies[1..]
            .iter()
            .map(|b| b.vel.length().max(b.angvel.length()))
            .fold(0.0f32, f32::max);
        if s > 10 && max_v < 0.08 && settle_step == STEPS {
            settle_step = s;
        }
    }
    let avbd_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("=== AVBD (spike, {ITERS} iters, dt={DT}) ===");
    println!("wall: {avbd_ms:.1} ms for {STEPS} steps, settled at step {settle_step}");
    let mut avbd_ok = true;
    for (k, b) in bodies[1..].iter().enumerate() {
        let expected_y = 0.5 + k as f32;
        let dy = (b.pos.y - expected_y).abs();
        let drift = b.pos.x.abs().max(b.pos.z.abs());
        let v = b.vel.length();
        let w = b.angvel.length();
        let ok = dy < 0.1 && drift < 0.15 && v < 0.08 && w < 0.08;
        avbd_ok &= ok;
        println!(
            "box{k}: y={:.4} (exp {expected_y:.2}, d={dy:.4}) drift={drift:.4} \
             v={v:.4} w={w:.4} rot={:.4} {}",
            b.pos.y,
            b.rot.length(),
            if ok { "OK" } else { "FAIL" },
        );
    }

    // --- SI run, same scene/gates ---
    let (mut physics, si_handles) = build_si_scene();
    let t0 = Instant::now();
    for _ in 0..STEPS {
        physics.step(DT);
    }
    let si_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("=== SI (BuiltinPhysicsEngine) ===");
    println!("wall: {si_ms:.1} ms for {STEPS} steps");
    let mut si_ok = true;
    for (k, h) in si_handles.iter().enumerate() {
        let b = physics.get_body(*h).unwrap();
        let expected_y = 0.5 + k as f32;
        let dy = (b.position.y - expected_y).abs();
        let drift = b.position.x.abs().max(b.position.z.abs());
        let v = b.velocity.length();
        let w = b.angular_velocity.length();
        let ok = dy < 0.1 && drift < 0.15 && v < 0.08 && w < 0.08;
        si_ok &= ok;
        println!(
            "box{k}: y={:.4} (exp {expected_y:.2}, d={dy:.4}) drift={drift:.4} \
             v={v:.4} w={w:.4} {}",
            b.position.y,
            if ok { "OK" } else { "FAIL" },
        );
    }

    println!(
        "HEAD-TO-HEAD: AVBD {} ({avbd_ms:.1} ms) vs SI {} ({si_ms:.1} ms)",
        if avbd_ok { "PASS" } else { "FAIL" },
        if si_ok { "PASS" } else { "FAIL" },
    );
}
