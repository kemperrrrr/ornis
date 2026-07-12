use ornis_macros::kernel;
use glam::Vec3Swizzles;

pub const PI: f32 = std::f32::consts::PI;
pub const INV_PI: f32 = 1.0 / PI;
pub const EPS: f32 = 1e-6;

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

#[kernel]
fn fresnel0_from_ior(ior: f32) -> f32 {
    let f = (ior - 1.0) / (ior + 1.0);
    f * f
}

#[kernel]
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).powf(5.0)
}

#[kernel]
fn fresnel_schlick_vec(cos_theta: f32, f0: glam::Vec3) -> glam::Vec3 {
    f0 + (glam::Vec3::splat(1.0) - f0) * (1.0 - cos_theta).powf(5.0)
}

#[kernel]
fn fresnel_f82_tint(cos_theta: f32, f0: glam::Vec3, f82_tint: glam::Vec3) -> glam::Vec3 {
    let mu_bar = 1.0 / 7.0;
    let schlick_at_mu_bar = f0 + (glam::Vec3::splat(1.0) - f0) * (1.0_f32 - mu_bar).powf(5.0);
    let f82 = f82_tint * schlick_at_mu_bar;
    let numerator = cos_theta * (1.0_f32 - cos_theta).powf(6.0);
    let denominator = mu_bar * (1.0_f32 - mu_bar).powf(6.0);
    let scale = numerator / denominator;
    let f_schlick = f0 + (glam::Vec3::splat(1.0) - f0) * (1.0 - cos_theta).powf(5.0);
    let f82_correction = f_schlick - glam::Vec3::splat(scale) * (schlick_at_mu_bar - f82);
    f82_correction.max(glam::Vec3::splat(0.0))
}

#[kernel]
fn ggx_ndf(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = NoH * NoH * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

#[kernel]
fn ggx_ndf_aniso(_NoH: f32, H: glam::Vec3, T: glam::Vec3, B: glam::Vec3, alpha_u: f32, alpha_v: f32) -> f32 {
    let Hu = H.dot(T);
    let Hv = H.dot(B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let denom = 1.0 + (Hu * Hu) / a2u + (Hv * Hv) / a2v;
    1.0 / (PI * a2u * a2v * denom * denom)
}

#[kernel]
fn openpbr_anisotropy(roughness: f32, anisotropy: f32) -> glam::Vec2 {
    let r2 = roughness * roughness;
    let aniso_inv = 1.0 - anisotropy;
    let aniso_inv_sq = aniso_inv * aniso_inv;
    let denom = aniso_inv_sq + 1.0;
    let fraction = 2.0 / denom;
    let sqrt_frac = fraction.sqrt();
    let alpha_u = r2 * sqrt_frac;
    let alpha_v = aniso_inv * alpha_u;
    glam::Vec2::new(alpha_u, alpha_v)
}

#[kernel]
fn smith_ggx_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggxv = NoV * (NoL * NoL * (1.0 - a2) + a2).max(EPS).sqrt();
    let ggxl = NoL * (NoV * NoV * (1.0 - a2) + a2).max(EPS).sqrt();
    0.5 / (ggxv + ggxl).max(EPS)
}

#[kernel]
fn smith_ggx_aniso(NoV: f32, NoL: f32, V: glam::Vec3, L: glam::Vec3, T: glam::Vec3, B: glam::Vec3, alpha_u: f32, alpha_v: f32) -> f32 {
    let Vu = V.dot(T);
    let Vv = V.dot(B);
    let Lu = L.dot(T);
    let Lv = L.dot(B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let ggxv = NoV * (Lu * Lu * a2u + Lv * Lv * a2v + NoL * NoL).max(EPS).sqrt();
    let ggxl = NoL * (Vu * Vu * a2u + Vv * Vv * a2v + NoV * NoV).max(EPS).sqrt();
    0.5 / (ggxv + ggxl).max(EPS)
}

#[kernel]
fn oren_nayar_brdf(NoV: f32, NoL: f32, cos_phi: f32, alpha: f32) -> f32 {
    let sigma = alpha.max(EPS);
    let sigma2 = sigma * sigma;
    let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.57);
    let B = 0.45 * sigma2 / (sigma2 + 0.09);
    let theta_v = NoV.max(0.0).acos();
    let theta_l = NoL.max(0.0).acos();
    let alpha_max = theta_v.max(theta_l);
    let beta_min = theta_v.min(theta_l);
    let tan_beta = beta_min.tan();
    (A + B * cos_phi * alpha_max.sin() * tan_beta) * INV_PI
}

#[kernel]
fn sheen_brdf(NoV: f32, NoL: f32, NoH: f32, VoH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let D = alpha / (PI * (NoH * NoH * (alpha - 1.0) + 1.0).powf(2.0));
    let G = 1.0 / (1.0 + alpha * (1.0 / NoV + 1.0 / NoL - 2.0));
    let F = VoH;
    D * G * F / (4.0 * NoV * NoL).max(EPS)
}

#[kernel]
fn transmission_color_to_extinction(transmission_color: glam::Vec3, transmission_depth: f32) -> glam::Vec3 {
    if transmission_depth <= 0.0 {
        return glam::Vec3::splat(0.0);
    }
    let c = transmission_color.max(glam::Vec3::splat(1e-6));
    -c.ln() / transmission_depth
}

#[kernel]
fn subsurface_brdf(NoV: f32, NoL: f32, distance: f32, radius: glam::Vec3, anisotropy: f32) -> glam::Vec3 {
    let sigma_tr = 3.0_f32.sqrt() / radius;
    let profile = (-distance * sigma_tr).exp() / distance.max(EPS);
    let phase = 1.0 + anisotropy * distance.cos();
    profile * phase * INV_PI
}

#[kernel]
fn srgb_to_linear(c: glam::Vec3) -> glam::Vec3 {
    let cutoff = 0.04045;
    let low = c / 12.92;
    let high = ((c + 0.055) / 1.055).powf(2.4);
    glam::Vec3::new(
        if c.x <= cutoff { low.x } else { high.x },
        if c.y <= cutoff { low.y } else { high.y },
        if c.z <= cutoff { low.z } else { high.z },
    )
}

#[kernel]
fn coat_darkening(
    coat_ior: f32,
    coat_weight: f32,
    coat_darkening: f32,
    base_metalness: f32,
    base_color: glam::Vec3,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: glam::Vec3,
) -> glam::Vec3 {
    let coat_f0 = ((coat_ior - 1.0) / (coat_ior + 1.0)).powf(2.0);
    let one_minus_coat_f0 = 1.0 - coat_f0;
    let coat_ior_sq = coat_ior * coat_ior;
    let Kcoat = 1.0 - one_minus_coat_f0 / coat_ior_sq;

    let Emetal = base_color * base_weight * specular_weight;
    let Edielectric = subsurface_color.lerp(base_color, subsurface_weight);
    let Ebase = Emetal.lerp(Edielectric, base_metalness);

    let Ebase_Kcoat = Ebase * Kcoat;
    let one_minus_Kcoat = 1.0 - Kcoat;
    let one_minus_Ebase_Kcoat = glam::Vec3::splat(1.0) - Ebase_Kcoat;
    let base_darkening = glam::Vec3::splat(one_minus_Kcoat) / one_minus_Ebase_Kcoat.max(glam::Vec3::splat(1e-6));

    let mix_factor = coat_weight * coat_darkening;
    glam::Vec3::splat(1.0).lerp(base_darkening, mix_factor)
}

#[kernel]
fn thin_film_modulation(cos_theta: f32, film_ior: f32, thickness_nm: f32, ior_outside: f32) -> glam::Vec3 {
    let sin_theta_film = ior_outside * (1.0 - cos_theta * cos_theta).max(0.0).sqrt() / film_ior;
    let cos_theta_film = (1.0 - sin_theta_film * sin_theta_film).max(0.0).sqrt();
    let lambda = glam::Vec3::new(650.0, 550.0, 450.0);
    let phase = 4.0 * PI * film_ior * thickness_nm * cos_theta_film / lambda;
    let r0 = ((film_ior - ior_outside) / (film_ior + ior_outside)).powf(2.0);
    let modulation = glam::Vec3::splat(1.0) + glam::Vec3::splat(2.0 * r0) * glam::Vec3::new(phase.x.cos(), phase.y.cos(), phase.z.cos()) / glam::Vec3::splat(1.0 - r0 * r0);
    modulation
}

#[kernel]
fn octahedral_encode(n: glam::Vec3) -> glam::Vec2 {
    let p = n.xy() / (n.x.abs() + n.y.abs() + n.z.abs());
    let q = glam::Vec2::new(p.x, p.y);
    if n.z < 0.0 {
        glam::Vec2::new(1.0 - q.y.abs(), 1.0 - q.x.abs()) * glam::Vec2::new(
            if q.x >= 0.0 { 1.0 } else { -1.0 },
            if q.y >= 0.0 { 1.0 } else { -1.0 },
        )
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_black() {
        assert_eq!(luminance::eval(glam::Vec3::ZERO), 0.0);
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
    }

    #[test]
    fn aces_tonemap_black() {
        let result = aces_tonemap::eval(glam::Vec3::ZERO);
        assert!((result.x).abs() < 0.001);
    }

    #[test]
    fn wgsl_generates() {
        assert!(luminance::wgsl_source().contains("fn luminance"));
        assert!(aces_tonemap::wgsl_source().contains("fn aces_tonemap"));
    }

    #[test]
    fn fresnel0_from_ior_glass() {
        let result = fresnel0_from_ior::eval(1.5);
        assert!((result - 0.04).abs() < 0.01);
    }

    #[test]
    fn ggx_ndf_peak() {
        let result = ggx_ndf::eval(1.0, 0.1);
        assert!(result > 0.0);
    }

    #[test]
    fn smith_ggx_zero_to_one() {
        let result = smith_ggx_correlated::eval(1.0, 1.0, 0.5);
        assert!(result > 0.0 && result <= 1.0);
    }

    #[test]
    fn srgb_to_linear_black() {
        let result = srgb_to_linear::eval(glam::Vec3::ZERO);
        assert_eq!(result, glam::Vec3::ZERO);
    }

    #[test]
    fn srgb_to_linear_white() {
        let result = srgb_to_linear::eval(glam::Vec3::ONE);
        assert!((result.x - 1.0).abs() < 0.001);
    }

    #[test]
    fn transmission_color_to_extinction_degenerate() {
        let result = transmission_color_to_extinction::eval(glam::Vec3::splat(0.5), 0.0);
        assert_eq!(result, glam::Vec3::ZERO);
    }

    #[test]
    fn thin_film_modulation_produces_finite() {
        let result = thin_film_modulation::eval(0.5, 1.5, 300.0, 1.0);
        assert!(result.x.is_finite());
        assert!(result.y.is_finite());
        assert!(result.z.is_finite());
    }

    #[test]
    fn octahedral_encode_roundtrip() {
        let n = glam::Vec3::new(0.577, 0.577, 0.577).normalize();
        let enc = octahedral_encode::eval(n);
        let dec = crate::shader::octahedral_decode_rust(enc);
        for i in 0..3 {
            assert!((n[i] - dec[i]).abs() < 0.01, "mismatch at {i}: {n} vs {dec}");
        }
    }

    #[test]
    fn all_wgsl_sources_compile() {
        let funcs: [&str; 19] = [
            luminance::wgsl_source(),
            aces_tonemap::wgsl_source(),
            fresnel0_from_ior::wgsl_source(),
            fresnel_schlick::wgsl_source(),
            fresnel_schlick_vec::wgsl_source(),
            fresnel_f82_tint::wgsl_source(),
            ggx_ndf::wgsl_source(),
            ggx_ndf_aniso::wgsl_source(),
            openpbr_anisotropy::wgsl_source(),
            smith_ggx_correlated::wgsl_source(),
            smith_ggx_aniso::wgsl_source(),
            oren_nayar_brdf::wgsl_source(),
            sheen_brdf::wgsl_source(),
            transmission_color_to_extinction::wgsl_source(),
            subsurface_brdf::wgsl_source(),
            srgb_to_linear::wgsl_source(),
            coat_darkening::wgsl_source(),
            thin_film_modulation::wgsl_source(),
            octahedral_encode::wgsl_source(),
        ];
        for src in funcs.iter() {
            assert!(src.starts_with("fn "), "bad source: {src}");
            assert!(src.contains("->"), "missing return type: {src}");
        }
    }
}
