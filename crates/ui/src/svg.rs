//! Minimal SVG path (`d` attribute) parser producing a `kurbo::BezPath`.
//!
//! Supports the common commands used by UI icons: `M m L l H h V v C c S s
//! Q q T t Z z`. Elliptical arcs (`A a`) are approximated with a straight line
//! to the endpoint (good enough for simple icons; no curved arcs yet).

use vello::peniko::kurbo::{BezPath, Point};

/// Parses an SVG path `d` string into a `BezPath`. Returns `None` if the string
/// is empty or contains no drawable commands.
pub fn parse_svg_path(d: &str) -> Option<BezPath> {
    let mut path = BezPath::new();
    let bytes = d.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();

    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut start_x = 0.0f64;
    let mut start_y = 0.0f64;
    // Last control point for smooth curves (S/T).
    let mut last_ctrl: Option<(f64, f64)> = None;
    let mut cmd = b' ';

    let mut nums = Vec::<f64>::new();

    while i < n {
        let c = bytes[i];
        if c.is_ascii_alphabetic() {
            if !nums.is_empty() {
                emit(&mut path, cmd, &nums, &mut cx, &mut cy, &mut start_x, &mut start_y, &mut last_ctrl);
                nums.clear();
            }
            cmd = c;
            i += 1;
        } else if c.is_ascii_digit() || c == b'.' || c == b'-' || c == b'+' || c == b'e' || c == b'E' {
            let num_start = i;
            let mut j = i;
            let mut seen_dot = false;
            let mut seen_exp = false;
            while j < n {
                let k = bytes[j];
                if k.is_ascii_digit() {
                    j += 1;
                } else if k == b'.' && !seen_dot && !seen_exp {
                    seen_dot = true;
                    j += 1;
                } else if (k == b'e' || k == b'E') && !seen_exp {
                    seen_exp = true;
                    j += 1;
                } else if (k == b'-' || k == b'+')
                    && (j == num_start || bytes[j - 1] == b'e' || bytes[j - 1] == b'E')
                {
                    j += 1;
                } else {
                    break;
                }
            }
            if let Ok(v) = d[num_start..j].parse::<f64>() {
                nums.push(v);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if cmd != b' ' {
        emit(&mut path, cmd, &nums, &mut cx, &mut cy, &mut start_x, &mut start_y, &mut last_ctrl);
    }

    if path.elements().is_empty() {
        None
    } else {
        Some(path)
    }
}

/// Signed angle (radians) from vector `u` to vector `v`.
fn angle_between(u: (f64, f64), v: (f64, f64)) -> f64 {
    let dot = u.0 * v.0 + u.1 * v.1;
    let len_u = (u.0 * u.0 + u.1 * u.1).sqrt();
    let len_v = (v.0 * v.0 + v.1 * v.1).sqrt();
    let mut a = if len_u * len_v > 0.0 {
        (dot / (len_u * len_v)).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };
    if u.0 * v.1 - u.1 * v.0 < 0.0 {
        a = -a;
    }
    a
}

/// Converts an SVG elliptical-arc command into a sequence of cubic-bézier
/// control-point triplets (c1, c2, end) expressed in user space, suitable for
/// `BezPath::curve_to`. This is the standard endpoint-parameterization
/// algorithm (SVG spec, F.6) with arcs split into <= 90° bézier segments.
fn arc_to_beziers(
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
    phi_deg: f64,
    large_arc: bool,
    sweep: bool,
    ex: f64,
    ey: f64,
) -> Vec<Point> {
    let phi = phi_deg.to_radians();
    let cos_p = phi.cos();
    let sin_p = phi.sin();
    let dx = (cx - ex) / 2.0;
    let dy = (cy - ey) / 2.0;
    let x1p = cos_p * dx + sin_p * dy;
    let y1p = -sin_p * dx + cos_p * dy;
    let mut rx = rx.abs();
    let mut ry = ry.abs();
    let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }
    let rx2 = rx * rx;
    let ry2 = ry * ry;
    let x1p2 = x1p * x1p;
    let y1p2 = y1p * y1p;
    let sign = if large_arc == sweep { -1.0 } else { 1.0 };
    let num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2;
    let den = rx2 * y1p2 + ry2 * x1p2;
    let coef = if den <= 0.0 {
        0.0
    } else {
        (num / den).max(0.0).sqrt() * sign
    };
    let cxp = coef * (rx * y1p / ry);
    let cyp = coef * (-ry * x1p / rx);
    let cxp_user = cos_p * cxp - sin_p * cyp + (cx + ex) / 2.0;
    let cyp_user = sin_p * cxp + cos_p * cyp + (cy + ey) / 2.0;

    let theta1 = angle_between(
        (1.0, 0.0),
        ((x1p - cxp) / rx, (y1p - cyp) / ry),
    );
    let mut dtheta = angle_between(
        ((x1p - cxp) / rx, (y1p - cyp) / ry),
        ((-x1p - cxp) / rx, (-y1p - cyp) / ry),
    );
    if !sweep && dtheta > 0.0 {
        dtheta -= 2.0 * std::f64::consts::PI;
    }
    if sweep && dtheta < 0.0 {
        dtheta += 2.0 * std::f64::consts::PI;
    }

    let n_segs = ((dtheta.abs() / std::f64::consts::FRAC_PI_2).ceil()).max(1.0) as i32;
    let seg = dtheta / n_segs as f64;
    let to_user = |px: f64, py: f64| -> Point {
        Point::new(cos_p * px - sin_p * py + cxp_user, sin_p * px + cos_p * py + cyp_user)
    };

    let mut out = Vec::new();
    let mut theta = theta1;
    for _ in 0..n_segs {
        let t1 = theta;
        let t2 = theta + seg;
        let f = (4.0 / 3.0) * (seg / 4.0).tan();
        let q1 = to_user(rx * t1.cos() - f * ry * t1.sin(), ry * t1.sin() + f * rx * t1.cos());
        let q2 = to_user(rx * t2.cos() + f * ry * t2.sin(), ry * t2.sin() - f * rx * t2.cos());
        let end = to_user(rx * t2.cos(), ry * t2.sin());
        out.push(q1);
        out.push(q2);
        out.push(end);
        theta = t2;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn emit(
    path: &mut BezPath,
    cmd: u8,
    nums: &[f64],
    cx: &mut f64,
    cy: &mut f64,
    start_x: &mut f64,
    start_y: &mut f64,
    last_ctrl: &mut Option<(f64, f64)>,
) {
    let rel = cmd.is_ascii_lowercase();
    let abs = |v: f64, c: f64| if rel { c + v } else { v };
    let mut idx = 0;
    let mut k = 0;
    let n = nums.len();

    macro_rules! next {
        () => {
            if idx < n {
                let v = nums[idx];
                idx += 1;
                Some(v)
            } else {
                None
            }
        };
    }

    if cmd.to_ascii_uppercase() == b'Z' {
        path.close_path();
        *cx = *start_x;
        *cy = *start_y;
        *last_ctrl = None;
        return;
    }

    while idx < n {
        match cmd.to_ascii_uppercase() {
            b'M' => {
                if let (Some(x), Some(y)) = (next!(), next!()) {
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    if k == 0 {
                        path.move_to((ax, ay));
                        *start_x = ax;
                        *start_y = ay;
                    } else {
                        path.line_to((ax, ay));
                    }
                    *cx = ax;
                    *cy = ay;
                    *last_ctrl = None;
                    k += 1;
                } else {
                    break;
                }
            }
            b'L' => {
                if let (Some(x), Some(y)) = (next!(), next!()) {
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    path.line_to((ax, ay));
                    *cx = ax;
                    *cy = ay;
                    *last_ctrl = None;
                } else {
                    break;
                }
            }
            b'H' => {
                if let Some(x) = next!() {
                    let ax = abs(x, *cx);
                    path.line_to((ax, *cy));
                    *cx = ax;
                    *last_ctrl = None;
                } else {
                    break;
                }
            }
            b'V' => {
                if let Some(y) = next!() {
                    let ay = abs(y, *cy);
                    path.line_to((*cx, ay));
                    *cy = ay;
                    *last_ctrl = None;
                } else {
                    break;
                }
            }
            b'C' => {
                if let (Some(x1), Some(y1), Some(x2), Some(y2), Some(x), Some(y)) =
                    (next!(), next!(), next!(), next!(), next!(), next!())
                {
                    let (a1x, a1y) = (abs(x1, *cx), abs(y1, *cy));
                    let (a2x, a2y) = (abs(x2, *cx), abs(y2, *cy));
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    path.curve_to((a1x, a1y), (a2x, a2y), (ax, ay));
                    *last_ctrl = Some((a2x, a2y));
                    *cx = ax;
                    *cy = ay;
                } else {
                    break;
                }
            }
            b'S' => {
                if let (Some(x2), Some(y2), Some(x), Some(y)) = (next!(), next!(), next!(), next!()) {
                    let (a2x, a2y) = (abs(x2, *cx), abs(y2, *cy));
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    let (c1x, c1y) = match *last_ctrl {
                        Some((px, py)) => (2.0 * *cx - px, 2.0 * *cy - py),
                        None => (*cx, *cy),
                    };
                    path.curve_to((c1x, c1y), (a2x, a2y), (ax, ay));
                    *last_ctrl = Some((a2x, a2y));
                    *cx = ax;
                    *cy = ay;
                } else {
                    break;
                }
            }
            b'Q' => {
                if let (Some(x1), Some(y1), Some(x), Some(y)) = (next!(), next!(), next!(), next!()) {
                    let (a1x, a1y) = (abs(x1, *cx), abs(y1, *cy));
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    path.quad_to((a1x, a1y), (ax, ay));
                    *last_ctrl = Some((a1x, a1y));
                    *cx = ax;
                    *cy = ay;
                } else {
                    break;
                }
            }
            b'T' => {
                if let (Some(x), Some(y)) = (next!(), next!()) {
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    let (c1x, c1y) = match *last_ctrl {
                        Some((px, py)) => (2.0 * *cx - px, 2.0 * *cy - py),
                        None => (*cx, *cy),
                    };
                    path.quad_to((c1x, c1y), (ax, ay));
                    *last_ctrl = Some((c1x, c1y));
                    *cx = ax;
                    *cy = ay;
                } else {
                    break;
                }
            }
            b'A' => {
                if nums.len() - idx >= 7 {
                    let rx = nums[idx];
                    let ry = nums[idx + 1];
                    let rot = nums[idx + 2];
                    let large = nums[idx + 3] != 0.0;
                    let sweep = nums[idx + 4] != 0.0;
                    let x = nums[idx + 5];
                    let y = nums[idx + 6];
                    let (ax, ay) = (abs(x, *cx), abs(y, *cy));
                    if rx <= 0.0 || ry <= 0.0 {
                        path.line_to((ax, ay));
                    } else {
                        let bez = arc_to_beziers(*cx, *cy, rx, ry, rot, large, sweep, ax, ay);
                        let mut it = bez.into_iter();
                        while let (Some(c1), Some(c2), Some(end)) =
                            (it.next(), it.next(), it.next())
                        {
                            path.curve_to(c1, c2, end);
                        }
                    }
                    *cx = ax;
                    *cy = ay;
                    *last_ctrl = None;
                    idx += 7;
                } else {
                    break;
                }
            }
            b'Z' => {
                path.close_path();
                *cx = *start_x;
                *cy = *start_y;
                *last_ctrl = None;
                break;
            }
            _ => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_svg_path;

    #[test]
    fn parses_move_line_close() {
        let p = parse_svg_path("M0 0 L10 0 L10 10 Z").unwrap();
        assert_eq!(p.elements().len(), 4);
    }

    #[test]
    fn parses_curve_relative() {
        let p = parse_svg_path("m10 10 c5 0 5 5 0 5 z").unwrap();
        assert!(!p.elements().is_empty());
    }

    #[test]
    fn empty_is_none() {
        assert!(parse_svg_path("").is_none());
        assert!(parse_svg_path("   ").is_none());
    }

    #[test]
    fn arc_produces_curves() {
        // A circle made of two arcs should yield cubic beziers, not straight
        // lines (otherwise round icons render as diamonds).
        let p = parse_svg_path("M10 0 A10 10 0 1 1 10 0.01 Z").unwrap();
        let has_curve = p
            .elements()
            .iter()
            .any(|e| matches!(e, vello::peniko::kurbo::PathEl::CurveTo(..)));
        assert!(has_curve, "arc command must produce curve segments");
    }
}
