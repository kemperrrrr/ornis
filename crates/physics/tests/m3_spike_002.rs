//! THROWAWAY M3 SPIKE 002 — island ownership, no proxies.
//! Not production code: hardcodes everything, no quality gates apply.
//! Delete with `spikes/002-ownership-proxies/` after the verdict.
//!
//! Lesson from v1 (staggered dynamic proxies): two independent lagged
//! pairs chasing each other pump energy through the floor (bottom
//! tunneled to y=-360). Island routing removes cross pairs as a class:
//! a contact-connected component always steps in ONE solver.

use glam::Vec3;
use ornis_physics::{AvbdEngine, BodyHandle, BuiltinPhysicsEngine, PhysicsEngine, RigidBody};

const DT: f32 = 1.0 / 60.0;
const GRAV: Vec3 = Vec3::new(0.0, -9.81, 0.0);
const SLEEP_V: f32 = 0.2;
const WAKE_V: f32 = 0.5;
const SLEEP_STEPS: u32 = 30;
/// Proximity band for island linkage (spike-hardcoded for 1m boxes).
const LINK_DIST: f32 = 1.15;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Owner {
    Avbd,
    Builtin,
}

struct M3 {
    avbd: AvbdEngine,
    builtin: BuiltinPhysicsEngine,
    /// Global truth (0 floor static, 1..=3 dynamics). Fixed order.
    reg: Vec<RigidBody>,
    owner: Vec<Owner>,
    local: Vec<BodyHandle>,
    sleepy: Vec<u32>,
    rebuilds: u32,
}

fn floor() -> RigidBody {
    RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(10.0, 1.0, 10.0), 0.0)
}

fn dyn_box(p: Vec3) -> RigidBody {
    RigidBody::new_box(p, Vec3::splat(0.5), 1.0)
}

fn find(r: &mut [usize], x: usize) -> usize {
    if r[x] != x {
        r[x] = find(r, r[x]);
    }
    r[x]
}

impl M3 {
    fn new() -> Self {
        let reg = vec![
            floor(),
            dyn_box(Vec3::new(0.0, 0.5, 0.0)),
            dyn_box(Vec3::new(0.0, 1.6, 0.0)),
            dyn_box(Vec3::new(3.0, 4.0, 0.0)),
        ];
        let mut m = Self {
            avbd: AvbdEngine::new(GRAV),
            builtin: BuiltinPhysicsEngine::new(GRAV),
            reg,
            owner: vec![Owner::Avbd; 4],
            local: vec![0; 4],
            sleepy: vec![0; 4],
            rebuilds: 0,
        };
        m.rebuild();
        m
    }

    fn rebuild(&mut self) {
        self.avbd = AvbdEngine::new(GRAV);
        self.builtin = BuiltinPhysicsEngine::new(GRAV);
        self.rebuilds += 1;
        for (g, body) in self.reg.iter().enumerate() {
            // Zombie guard (M3 reconcile rule): a snapshot taken from a
            // sleeping engine carries inv_mass=0 AND zero inertia; a fresh
            // engine starts it awake, hence permanently unsolvable (and
            // zero inertia tumbles it corner-first through the floor).
            // Restore the mass model like wake_body does.
            let mut b = body.clone();
            if b.body_type == ornis_physics::BodyType::Dynamic && b.inv_mass <= 0.0 {
                b.inv_mass = 1.0 / b.mass;
                b.inertia = b.shape.inertia(b.mass);
            }
            if g == 0 {
                self.local[g] = self.avbd.add_body(b.clone());
                self.builtin.add_body(b.clone());
                continue;
            }
            let h = match self.owner[g] {
                Owner::Avbd => self.avbd.add_body(b.clone()),
                Owner::Builtin => self.builtin.add_body(b.clone()),
            };
            self.local[g] = h;
        }
    }

    fn pull(&mut self) {
        for g in 0..self.reg.len() {
            let b = match self.owner[g] {
                Owner::Avbd => self.avbd.get_body(self.local[g]).unwrap().clone(),
                // Floor truth lives in AVBD (static, identical anyway).
                Owner::Builtin if g == 0 => self.avbd.get_body(self.local[g]).unwrap().clone(),
                Owner::Builtin => self.builtin.get_body(self.local[g]).unwrap().clone(),
            };
            self.reg[g] = b;
        }
    }

    /// Contact islands over the registry (union-find on proximity).
    /// Returns owner decision per global: calm islands migrate to
    /// Builtin, awake ones to AVBD. True when anything changed.
    fn route(&mut self) -> bool {
        let n = self.reg.len();
        let mut root: Vec<usize> = (0..n).collect();
        for a in 1..n {
            for b in (a + 1)..n {
                if (self.reg[a].position - self.reg[b].position).length() < LINK_DIST {
                    let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                    root[ra] = rb;
                }
            }
        }
        let mut changed = false;
        for g in 1..n {
            let v = self.reg[g].velocity.length();
            if v > WAKE_V {
                self.sleepy[g] = 0;
            } else if v < SLEEP_V {
                self.sleepy[g] += 1;
            } else {
                self.sleepy[g] = 0;
            }
        }
        // Island calm <=> every member sleepy long enough.
        for g in 1..n {
            let r = find(&mut root, g);
            let mut calm = true;
            for h in 1..n {
                if find(&mut root, h) == r && self.sleepy[h] < SLEEP_STEPS {
                    calm = false;
                    break;
                }
            }
            // Wake is per-body immediate; sleep is per-island unanimous.
            let want = if self.reg[g].velocity.length() > WAKE_V {
                Owner::Avbd
            } else if calm {
                Owner::Builtin
            } else {
                Owner::Avbd
            };
            if want != self.owner[g] {
                self.owner[g] = want;
                changed = true;
            }
        }
        changed
    }

    fn step(&mut self) {
        self.avbd.step(DT);
        self.builtin.step(DT);
        self.pull();
        if self.route() {
            self.rebuild();
        }
    }
}

#[test]
fn spike_island_stack_settles_and_migrates() {
    let mut m = M3::new();
    for i in 0..600 {
        if i == 300 {
            // Deterministic kick on the migrant (global 3).
            m.reg[3].velocity = Vec3::new(0.0, 6.0, 0.0);
            let (o, l) = (m.owner[3], m.local[3]);
            let b = match o {
                Owner::Avbd => m.avbd.get_body_mut(l).unwrap(),
                Owner::Builtin => m.builtin.get_body_mut(l).unwrap(),
            };
            b.velocity = Vec3::new(0.0, 6.0, 0.0);
            // Clean-launch probe: identity orientation, no spin. If the
            // re-entry lands now, the tunnel came from a tipped state
            // (corner-manifold weakness, out of M3 scope).
            b.orientation = glam::Quat::IDENTITY;
            b.angular_velocity = Vec3::ZERO;
            m.reg[3].orientation = glam::Quat::IDENTITY;
            m.reg[3].angular_velocity = Vec3::ZERO;
        }
        m.step();
    }
    let ys: Vec<f32> = m.reg.iter().map(|b| b.position.y).collect();
    assert!(
        (ys[1] - 0.5).abs() < 0.12 && (ys[2] - 1.5).abs() < 0.18,
        "island stack must rest: {ys:?}"
    );
    assert!(
        (ys[3] - 0.5).abs() < 0.12
            && (m.reg[3].position.x - 3.0).abs() < 0.2
            && m.reg[3].velocity.length() < 0.6,
        "migrant must land back on the floor after the kick: {:?} {:?}",
        m.reg[3].position,
        m.reg[3].velocity
    );
    assert!(
        (2..=10).contains(&m.rebuilds),
        "hysteresis must bound migrations: {} rebuilds",
        m.rebuilds
    );
}

#[test]
fn spike_island_deterministic() {
    let run = || {
        let mut m = M3::new();
        for i in 0..600 {
            if i == 300 {
                m.reg[3].velocity = Vec3::new(0.0, 6.0, 0.0);
                let (o, l) = (m.owner[3], m.local[3]);
                let b = match o {
                    Owner::Avbd => m.avbd.get_body_mut(l).unwrap(),
                    Owner::Builtin => m.builtin.get_body_mut(l).unwrap(),
                };
                b.velocity = Vec3::new(0.0, 6.0, 0.0);
            }
            m.step();
        }
        m.reg
            .iter()
            .map(|b| (b.position, b.velocity))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run(), "island routing must be rerun-identical");
}

/// Isolation control: the same 6 m/s launch from floor rest in a SINGLE
/// AVBD engine. If this tunnels too, the miss is engine behavior, not
/// the M3 harness.
#[test]
fn spike_single_launch_catches() {
    let mut e = AvbdEngine::new(GRAV);
    e.add_body(floor());
    let h = e.add_body(dyn_box(Vec3::new(3.0, 0.5, 0.0)));
    e.get_body_mut(h).unwrap().velocity = Vec3::new(0.0, 6.0, 0.0);
    for _ in 0..300 {
        e.step(DT);
    }
    let b = e.get_body(h).unwrap();
    assert!(
        b.position.y > 0.0,
        "single-solver launch must not tunnel: {:?} {:?}",
        b.position,
        b.velocity
    );
}
