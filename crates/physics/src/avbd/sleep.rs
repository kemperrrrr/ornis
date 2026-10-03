//! Residual-aware sleep for articulated AVBD bodies. Low speed alone does
//! not mean a hard constraint has reached its assembly pose.

use super::*;

const JOINT_REST_ERROR: f32 = 0.005;

impl AvbdEngine {
    fn joint_members(&self, j: &AvbdJoint) -> [usize; 4] {
        if j.kind == AvbdJointKind::Gear
            && let (Some(a), Some(b)) = (self.joints.get(j.gb[0]), self.joints.get(j.gb[1]))
        {
            return [a.a, a.b, b.a, b.b];
        }
        [j.a, j.b, j.a, j.b]
    }

    fn joint_driven(&self, j: &AvbdJoint) -> bool {
        j.mot.is_some_and(|[speed, cap]| speed != 0.0 && cap > 0.0)
            || j.servo.is_some_and(|m| m.keeps_awake())
            || j.spec.spring_motor().is_some_and(|m| m.keeps_awake())
            || self.joint_members(j).into_iter().any(|h| {
                let b = &self.bodies[h];
                b.body_type != BodyType::Dynamic
                    && (b.position != self.prev_pos[h] || b.orientation != self.prev_rot[h])
            })
    }

    // qual:allow(iosp) reason: per-kind residual dispatcher — anchor delta precompute feeds one match; splitting per kind would scatter the assembly-error definition across six fragments.
    fn joint_rest_error(&self, j: &AvbdJoint) -> f32 {
        let (a, b) = (&self.bodies[j.a], &self.bodies[j.b]);
        let delta = (b.position + b.orientation * j.lb) - (a.position + a.orientation * j.la);
        let axis = a.orientation * j.ax_a;
        match j.kind {
            AvbdJointKind::Ball => delta.length(),
            AvbdJointKind::Revolute => delta.length().max((axis - b.orientation * j.ax_b).length()),
            AvbdJointKind::Fixed => (delta - a.orientation * j.dref)
                .length()
                .max(quat_diff_vec(a.orientation.conjugate() * b.orientation, j.q_ref).length()),
            AvbdJointKind::Distance => (delta.length() - j.ref_val).abs(),
            // Slack is rest (zero residual); a stretched rope reports
            // the overshoot.
            AvbdJointKind::Rope => (delta.length() - j.ref_val).max(0.0),
            // Spring rest comes from the spec motor when present.
            AvbdJointKind::Spring => {
                let rest = j
                    .spec
                    .spring_motor()
                    .map(|m| m.target_position)
                    .unwrap_or(j.ref_val);
                (delta.length() - rest).abs()
            }
            AvbdJointKind::Prismatic | AvbdJointKind::Wheel => {
                let perpendicular = delta - axis * delta.dot(axis);
                let mut error = perpendicular.length();
                if let Some([lo, hi]) = j.lim {
                    let value = delta.dot(axis) - j.ref_val;
                    error = error.max((lo - value).max(value - hi));
                }
                if j.kind == AvbdJointKind::Prismatic {
                    error = error.max((axis - b.orientation * j.ax_b).length());
                } else {
                    error = error.max((a.orientation * j.bx_a - b.orientation * j.bx_b).length());
                    if j.susp[0] <= 0.0 {
                        error = error.max((delta.dot(axis) - j.ref_val).abs());
                    }
                }
                error
            }
            AvbdJointKind::Gear => self
                .gear_sides(j)
                .map(|(a, b)| (a.coord + j.gratio * b.coord - j.ref_val).abs())
                .unwrap_or(0.0),
            AvbdJointKind::SixDof => self.sixdof_rest_error(j, delta),
        }
    }

    // qual:allow(iosp) reason: six-axis residual fold — loop over lin/ang axis configs plus quat_diff_vec; a pure-logic split would duplicate the config match per axis.
    fn sixdof_rest_error(&self, j: &AvbdJoint, delta: Vec3) -> f32 {
        let (a, b) = (&self.bodies[j.a], &self.bodies[j.b]);
        let linear = a.orientation.conjugate() * delta - j.dref;
        let angular = quat_diff_vec(a.orientation.conjugate() * b.orientation, j.q_ref);
        let mut error = 0.0f32;
        for i in 0..3 {
            for (value, config) in [(linear[i], j.six_lin[i]), (angular[i], j.six_ang[i])] {
                let residual = match config {
                    AxisConfig::Free => 0.0,
                    AxisConfig::Locked => value.abs(),
                    AxisConfig::Limited { min, max } => (min - value).max(value - max).max(0.0),
                };
                error = error.max(residual);
            }
        }
        error
    }

    /// Wake changed inputs and propagate wake through complete joint components.
    // qual:allow(iosp) reason: wake orchestrator — one joint walk both decides and applies wakes; separating decision from application would double the traversal.
    pub(super) fn wake_joint_motion(&mut self) {
        let mut awake: Vec<bool> = self
            .bodies
            .iter()
            .enumerate()
            .map(|(h, b)| {
                b.body_type == BodyType::Dynamic
                    && (!self.asleep[h]
                        || b.velocity != Vec3::ZERO
                        || b.angular_velocity != Vec3::ZERO
                        || b.torque != Vec3::ZERO
                        || b.position != self.prev_pos[h]
                        || b.orientation != self.prev_rot[h])
            })
            .collect();
        for j in &self.joints {
            if self.joint_driven(j) || self.joint_rest_error(j) > JOINT_REST_ERROR {
                for h in self.joint_members(j) {
                    if self.bodies[h].body_type == BodyType::Dynamic {
                        awake[h] = true;
                    }
                }
            }
        }
        self.propagate_joint_flags(&mut awake, true);
        for (h, wake) in awake.into_iter().enumerate() {
            if wake && self.asleep[h] {
                self.wake_body(h);
            }
        }
    }

    // qual:allow(iosp) reason: sleep-readiness orchestrator — same joint-component walk as wake_joint_motion with inverted flag folding; splitting would duplicate the propagation contract.
    pub(super) fn joint_sleep_ready(&self) -> Vec<bool> {
        let mut ready = vec![true; self.bodies.len()];
        for j in &self.joints {
            if self.joint_driven(j) || self.joint_rest_error(j) > JOINT_REST_ERROR {
                for h in self.joint_members(j) {
                    ready[h] = false;
                }
            }
        }
        self.propagate_joint_flags(&mut ready, false);
        ready
    }

    /// One awake/unsatisfied member keeps its entire joint component awake.
    pub(super) fn propagate_joint_flags(&self, flags: &mut [bool], value: bool) {
        loop {
            let mut changed = false;
            for j in &self.joints {
                let members = self.joint_members(j);
                if members
                    .iter()
                    .any(|&h| self.bodies[h].body_type == BodyType::Dynamic && flags[h] == value)
                {
                    for h in members {
                        if self.bodies[h].body_type == BodyType::Dynamic && flags[h] != value {
                            flags[h] = value;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}
