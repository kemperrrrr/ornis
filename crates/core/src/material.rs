//! OpenPBR material definition.
//!
//! The material is a flat, GPU-friendly blob of 20 `vec4` slots, but its
//! setter surface is split across parameter-group structs (`BaseGroup`,
//! `SpecularGroup`, …) so each concern owns a small, focused set of
//! builders instead of one type carrying dozens of methods. The nested
//! `#[repr(C)]` groups preserve the exact flat memory layout expected by
//! the WGSL uniform/storage buffers.

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BaseGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpecularGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransmissionGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
    pub scatter: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SubsurfaceGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
    pub radius_scale_gb: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FuzzGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoatGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
    pub ior: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ThinFilmGroup {
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmissionGroup {
    pub params: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeometryGroup {
    pub params: [f32; 4],
    pub params2: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OpenPBRMaterial {
    pub base: BaseGroup,
    pub specular: SpecularGroup,

    pub transmission: TransmissionGroup,

    pub subsurface: SubsurfaceGroup,

    pub fuzz: FuzzGroup,

    pub coat: CoatGroup,

    pub thin_film: ThinFilmGroup,

    pub emission: EmissionGroup,

    pub geometry: GeometryGroup,
}

impl BaseGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn diffuse_roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }
    pub fn metalness(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.clamp(0.0, 1.0);
        self
    }

    pub fn color(&mut self, r: f32, g: f32, b: f32, a: f32) -> &mut Self {
        self.color = [r, g, b, a];
        self
    }
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl SpecularGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }
    pub fn ior(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(1.0);
        self
    }
    pub fn anisotropy(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.clamp(0.0, 1.0);
        self
    }

    pub fn edge_tint(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    pub fn edge_tint_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl TransmissionGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn depth(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.max(0.0);
        self
    }
    pub fn dispersion_scale(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(0.0);
        self
    }
    pub fn dispersion_abbe(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.max(0.0);
        self
    }

    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }

    pub fn scatter_color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.scatter = [r, g, b, 0.0];
        self
    }
    pub fn scatter_anisotropy(&mut self, v: f32) -> &mut Self {
        self.scatter[3] = v.clamp(-1.0, 1.0);
        self
    }
}

impl SubsurfaceGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn radius(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.max(0.0);
        self
    }
    pub fn radius_scale_r(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(0.0);
        self
    }
    pub fn scatter_anisotropy(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.clamp(-1.0, 1.0);
        self
    }

    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 1.0];
        self
    }
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }

    pub fn radius_scale_g(&mut self, v: f32) -> &mut Self {
        self.radius_scale_gb[0] = v.max(0.0);
        self
    }
    pub fn radius_scale_b(&mut self, v: f32) -> &mut Self {
        self.radius_scale_gb[1] = v.max(0.0);
        self
    }
}

impl FuzzGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }

    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl CoatGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn roughness(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.clamp(0.0, 1.0);
        self
    }
    pub fn anisotropy(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.clamp(0.0, 1.0);
        self
    }
    pub fn darkening(&mut self, v: f32) -> &mut Self {
        self.params[3] = v.clamp(0.0, 1.0);
        self
    }

    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 0.0];
        self
    }
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }

    pub fn ior(&mut self, v: f32) -> &mut Self {
        self.ior[0] = v.max(1.0);
        self
    }
}

impl ThinFilmGroup {
    pub fn weight(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn thickness_um(&mut self, v: f32) -> &mut Self {
        self.params[1] = v.max(0.0);
        self
    }
    pub fn ior(&mut self, v: f32) -> &mut Self {
        self.params[2] = v.max(1.0);
        self
    }
}

impl EmissionGroup {
    pub fn luminance(&mut self, nits: f32) -> &mut Self {
        self.params[0] = nits.max(0.0);
        self
    }
    pub fn color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color = [r, g, b, 1.0];
        self
    }
    pub fn color_rgb(&mut self, rgb: [f32; 3]) -> &mut Self {
        self.color[0] = rgb[0];
        self.color[1] = rgb[1];
        self.color[2] = rgb[2];
        self
    }
}

impl GeometryGroup {
    pub fn opacity(&mut self, v: f32) -> &mut Self {
        self.params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn thin_walled(&mut self, v: bool) -> &mut Self {
        self.params[1] = if v { 1.0 } else { 0.0 };
        self
    }
}

impl OpenPBRMaterial {
    pub const VEC4_COUNT: usize = 20;
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn pbr() -> Self {
        Self {
            base: BaseGroup {
                params: [1.0, 0.0, 0.0, 0.0],
                color: [0.8, 0.8, 0.8, 1.0],
            },
            specular: SpecularGroup {
                params: [1.0, 0.3, 1.5, 0.0],
                color: [1.0, 1.0, 1.0, 0.0],
            },
            transmission: TransmissionGroup {
                params: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 0.0],
                scatter: [0.0, 0.0, 0.0, 0.0],
            },
            subsurface: SubsurfaceGroup {
                params: [0.0, 0.0, 1.0, 0.0],
                color: [0.8, 0.8, 0.8, 1.0],
                radius_scale_gb: [1.0, 1.0, 0.0, 0.0],
            },
            fuzz: FuzzGroup {
                params: [0.0, 0.5, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 0.0],
            },
            coat: CoatGroup {
                params: [0.0, 0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 0.0],
                ior: [1.5, 0.0, 0.0, 0.0],
            },
            thin_film: ThinFilmGroup {
                params: [0.0, 0.0, 1.33, 0.0],
            },
            emission: EmissionGroup {
                params: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            geometry: GeometryGroup {
                params: [1.0, 0.0, 0.0, 0.0],
                params2: [0.0, 0.0, 0.0, 0.0],
            },
        }
    }

    pub fn metal() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(1.0);
        m.base.color_rgb([0.9, 0.9, 0.9]);
        m.specular.weight(1.0);
        m.specular.roughness(0.1);
        m
    }

    pub fn dielectric() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.5);
        m
    }

    pub fn glass() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(1.0);
        m.specular.roughness(0.0);
        m.specular.ior(1.5);
        m.transmission.weight(1.0);
        m.transmission.depth(1.0);
        m.transmission.color_rgb([1.0, 1.0, 1.0]);
        m.geometry.thin_walled(true);
        m
    }

    pub fn subsurface() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.3);
        m.subsurface.weight(1.0);
        m.subsurface.radius(0.1);
        m.subsurface.color_rgb([0.8, 0.2, 0.1]);
        m
    }

    pub fn coat() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.5);
        m.coat.weight(1.0);
        m.coat.roughness(0.1);
        m.coat.ior(1.5);
        m
    }

    pub fn fuzz() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.weight(1.0);
        m.specular.roughness(0.3);
        m.specular.ior(1.5);
        m.fuzz.weight(1.0);
        m.fuzz.roughness(0.3);
        m.fuzz.color_rgb([0.8, 0.8, 0.8]);
        m
    }

    pub fn thin_film() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(1.0);
        m.specular.roughness(0.0);
        m.specular.ior(1.5);
        m.thin_film.weight(1.0);
        m.thin_film.thickness_um(300.0);
        m.thin_film.ior(1.33);
        m
    }

    pub fn emission() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(0.0);
        m.emission.luminance(1000.0);
        m.emission.color_rgb([1.0, 0.8, 0.6]);
        m
    }

    pub fn transmission() -> Self {
        let mut m = Self::pbr();
        m.base.metalness(0.0);
        m.base.color_rgb([0.0, 0.0, 0.0]);
        m.specular.weight(1.0);
        m.specular.roughness(0.0);
        m.specular.ior(1.5);
        m.transmission.weight(1.0);
        m.transmission.depth(2.0);
        m.transmission.color_rgb([1.0, 1.0, 1.0]);
        m
    }
}

impl Default for OpenPBRMaterial {
    fn default() -> Self {
        Self::pbr()
    }
}

pub const OPENPBR_MATERIAL_VEC4_COUNT: usize = 20;
pub const OPENPBR_MATERIAL_SIZE: usize = std::mem::size_of::<OpenPBRMaterial>();

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn test_openpbr_material_size() {
        assert_eq!(OPENPBR_MATERIAL_SIZE, 320);
        assert_eq!(OPENPBR_MATERIAL_VEC4_COUNT, 20);
    }

    #[test]
    fn test_openpbr_material_alignment() {
        let mat = OpenPBRMaterial::default();
        let bytes = bytemuck::bytes_of(&mat);
        assert_eq!(bytes.len(), OPENPBR_MATERIAL_SIZE);
        assert_eq!(bytes.len() % 16, 0);
    }

    #[test]
    fn test_default_material_is_neutral_dielectric() {
        let mat = OpenPBRMaterial::default();
        assert_eq!(mat.base.params[0], 1.0);
        assert_eq!(mat.base.color[0], 0.8);
        assert_eq!(mat.base.color[1], 0.8);
        assert_eq!(mat.base.color[2], 0.8);
        assert_eq!(mat.specular.params[1], 0.3);
        assert_eq!(mat.specular.params[2], 1.5);
    }

    #[test]
    fn test_metal_builder() {
        let mat = OpenPBRMaterial::metal();
        assert_eq!(mat.base.params[2], 1.0);
        assert_eq!(mat.specular.params[0], 1.0);
        assert_eq!(mat.specular.params[1], 0.1);
    }

    #[test]
    fn test_dielectric_builder() {
        let mat = OpenPBRMaterial::dielectric();
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.specular.params[0], 1.0);
        assert_eq!(mat.specular.params[1], 0.3);
        assert_eq!(mat.specular.params[2], 1.5);
    }

    #[test]
    fn test_glass_builder() {
        let mat = OpenPBRMaterial::glass();
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.transmission.params[0], 1.0);
        assert_eq!(mat.geometry.params[1], 1.0);
    }

    #[test]
    fn test_subsurface_builder() {
        let mat = OpenPBRMaterial::subsurface();
        assert_eq!(mat.subsurface.params[0], 1.0);
        assert_eq!(mat.subsurface.params[1], 0.1);
        assert_eq!(mat.subsurface.color[0], 0.8);
        assert_eq!(mat.subsurface.color[1], 0.2);
        assert_eq!(mat.subsurface.color[2], 0.1);
    }

    #[test]
    fn test_coat_builder() {
        let mat = OpenPBRMaterial::coat();
        assert_eq!(mat.coat.params[0], 1.0);
        assert_eq!(mat.coat.params[1], 0.1);
        assert_eq!(mat.coat.ior[0], 1.5);
    }

    #[test]
    fn test_fuzz_builder() {
        let mat = OpenPBRMaterial::fuzz();
        assert_eq!(mat.fuzz.params[0], 1.0);
        assert_eq!(mat.fuzz.params[1], 0.3);
    }

    #[test]
    fn test_thin_film_builder() {
        let mat = OpenPBRMaterial::thin_film();
        assert_eq!(mat.thin_film.params[0], 1.0);
        assert_eq!(mat.thin_film.params[1], 300.0);
        assert_eq!(mat.thin_film.params[2], 1.33);
    }

    #[test]
    fn test_emission_builder() {
        let mat = OpenPBRMaterial::emission();
        assert_eq!(mat.emission.params[0], 1000.0);
        assert_eq!(mat.emission.color[0], 1.0);
        assert_eq!(mat.emission.color[1], 0.8);
        assert_eq!(mat.emission.color[2], 0.6);
    }

    #[test]
    fn test_transmission_builder() {
        let mat = OpenPBRMaterial::transmission();
        assert_eq!(mat.transmission.params[0], 1.0);
        assert_eq!(mat.transmission.params[1], 2.0);
        assert_eq!(mat.transmission.color[0], 1.0);
    }

    #[test]
    fn test_parameter_clamping() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.weight(1.5);
        mat.base.metalness(-0.5);
        mat.specular.roughness(2.0);
        mat.specular.ior(0.5);
        assert_eq!(mat.base.params[0], 1.0);
        assert_eq!(mat.base.params[2], 0.0);
        assert_eq!(mat.specular.params[1], 1.0);
        assert_eq!(mat.specular.params[2], 1.0);
    }

    #[test]
    fn test_opacity_and_thin_walled() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.geometry.opacity(0.5);
        mat.geometry.thin_walled(true);
        assert_eq!(mat.geometry.params[0], 0.5);
        assert_eq!(mat.geometry.params[1], 1.0);
    }

    #[test]
    fn test_coat_darkening_parameter() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.coat.weight(1.0);
        mat.coat.roughness(0.1);
        mat.coat.anisotropy(0.3);
        mat.coat.darkening(0.5);
        assert_eq!(mat.coat.params[0], 1.0);
        assert_eq!(mat.coat.params[1], 0.1);
        assert_eq!(mat.coat.params[2], 0.3);
        assert_eq!(mat.coat.params[3], 0.5);
    }

    #[test]
    fn test_anisotropy_parameters() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.specular.anisotropy(0.5);
        mat.coat.anisotropy(0.8);
        assert_eq!(mat.specular.params[3], 0.5);
        assert_eq!(mat.coat.params[2], 0.8);
    }

    #[test]
    fn test_transmission_dispersion() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.transmission.weight(1.0);
        mat.transmission.dispersion_scale(0.5);
        mat.transmission.dispersion_abbe(50.0);
        assert_eq!(mat.transmission.params[0], 1.0);
        assert_eq!(mat.transmission.params[2], 0.5);
        assert_eq!(mat.transmission.params[3], 50.0);
    }

    #[test]
    fn test_subsurface_radius_scales() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.subsurface.weight(1.0);
        mat.subsurface.radius(0.5);
        mat.subsurface.radius_scale_r(1.0);
        mat.subsurface.radius_scale_g(0.5);
        mat.subsurface.radius_scale_b(0.25);
        assert_eq!(mat.subsurface.params[0], 1.0);
        assert_eq!(mat.subsurface.params[1], 0.5);
        assert_eq!(mat.subsurface.params[2], 1.0);
        assert_eq!(mat.subsurface.radius_scale_gb[0], 0.5);
        assert_eq!(mat.subsurface.radius_scale_gb[1], 0.25);
    }

    #[test]
    fn test_fuzz_parameters() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.fuzz.weight(0.8);
        mat.fuzz.roughness(0.4);
        mat.fuzz.color_rgb([0.9, 0.9, 0.9]);
        assert_eq!(mat.fuzz.params[0], 0.8);
        assert_eq!(mat.fuzz.params[1], 0.4);
        assert_eq!(mat.fuzz.color[0], 0.9);
    }

    #[test]
    fn test_pod_zeroable() {
        let _mat = OpenPBRMaterial::default();
        let _zeroed = OpenPBRMaterial::zeroed();
        let _pod = unsafe {
            std::mem::transmute::<[u8; OPENPBR_MATERIAL_SIZE], OpenPBRMaterial>(
                [0u8; OPENPBR_MATERIAL_SIZE],
            )
        };
    }

    #[test]
    fn scalar_builders_write_distinct_values() {
        // Each scalar builder must write its argument into the right slot.
        // Values are chosen to differ from the pbr() defaults so a mutant
        // that returns Default::default() instead of `self` fails.
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.weight(0.5);
        mat.base.diffuse_roughness(0.4);
        mat.transmission.scatter_anisotropy(-0.5);
        mat.subsurface.scatter_anisotropy(-0.25);
        assert_eq!(mat.base.params[0], 0.5);
        assert_eq!(mat.base.params[1], 0.4);
        assert_eq!(mat.transmission.scatter[3], -0.5);
        assert_eq!(mat.subsurface.params[3], -0.25);
    }

    #[test]
    fn color_builders_write_distinct_values() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.color(0.1, 0.2, 0.3, 0.9);
        mat.specular.edge_tint(0.4, 0.5, 0.6);
        mat.specular.edge_tint_rgb([0.7, 0.8, 0.9]);
        mat.transmission.color(0.11, 0.12, 0.13);
        mat.transmission.scatter_color(0.21, 0.22, 0.23);
        mat.subsurface.color(0.31, 0.32, 0.33);
        mat.fuzz.color(0.41, 0.42, 0.43);
        mat.coat.color(0.51, 0.52, 0.53);
        mat.coat.color_rgb([0.61, 0.62, 0.63]);
        mat.emission.color(0.71, 0.72, 0.73);
        assert_eq!(mat.base.color, [0.1, 0.2, 0.3, 0.9]);
        assert_eq!(mat.specular.color, [0.7, 0.8, 0.9, 0.0]);
        assert_eq!(mat.transmission.color, [0.11, 0.12, 0.13, 0.0]);
        assert_eq!(&mat.transmission.scatter[..3], &[0.21, 0.22, 0.23]);
        assert_eq!(mat.subsurface.color, [0.31, 0.32, 0.33, 1.0]);
        assert_eq!(mat.fuzz.color, [0.41, 0.42, 0.43, 0.0]);
        assert_eq!(mat.coat.color, [0.61, 0.62, 0.63, 0.0]);
        assert_eq!(mat.emission.color, [0.71, 0.72, 0.73, 1.0]);
    }

    #[test]
    fn builders_do_not_disturb_other_fields() {
        let mut mat = OpenPBRMaterial::pbr();
        mat.base.weight(0.5);
        mat.specular.edge_tint(0.1, 0.2, 0.3);
        // Fields untouched by the chain keep their pbr() defaults.
        assert_eq!(mat.base.params[1], 0.0);
        assert_eq!(mat.base.color, [0.8, 0.8, 0.8, 1.0]);
        assert_eq!(mat.transmission.color, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn dielectric_preset_writes_expected_fields() {
        // `dielectric()` is a convenience preset; a mutant that collapses
        // it to Default::default() must be caught by checking the fields
        // it is supposed to set.
        let mat = OpenPBRMaterial::dielectric();
        assert_eq!(mat.base.params[2], 0.0); // base metalness
        assert_eq!(&mat.base.color[..3], &[0.8, 0.8, 0.8]);
        assert_eq!(mat.specular.params[0], 1.0); // specular weight
        assert_eq!(mat.specular.params[1], 0.3); // specular roughness
    }
}
