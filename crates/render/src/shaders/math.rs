use ornis_macros::kernel;

#[kernel]
fn luminance(c: glam::Vec3) -> f32 {
    c.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722))
}

#[kernel]
fn aces_tonemap(color: glam::Vec3) -> glam::Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    color * (a * color + b) / (color * (c * color + d) + e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_black() {
        let result = luminance::eval(glam::Vec3::ZERO);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn luminance_white() {
        let result = luminance::eval(glam::Vec3::ONE);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn aces_tonemap_red() {
        let result = aces_tonemap::eval(glam::Vec3::new(1.0, 0.0, 0.0));
        assert!((result.x - 0.8038).abs() < 0.001);
        assert!((result.y).abs() < 0.001);
        assert!((result.z).abs() < 0.001);
    }

    #[test]
    fn aces_tonemap_black() {
        let result = aces_tonemap::eval(glam::Vec3::ZERO);
        assert!((result.x).abs() < 0.001);
        assert!((result.y).abs() < 0.001);
        assert!((result.z).abs() < 0.001);
    }

    #[test]
    fn aces_tonemap_wgsl_generates() {
        let source = aces_tonemap::wgsl_source();
        assert!(source.contains("fn aces_tonemap"));
        assert!(source.contains("let a = 2.51;"));
        assert!(source.contains("color * (a * color + b)"));
    }

    #[test]
    fn wgsl_and_rust_match() {
        let source = luminance::wgsl_source();
        assert!(source.starts_with("fn luminance"));
        assert!(source.contains("-> f32"));
    }
}
