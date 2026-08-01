#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OpenPBRMaterial {
    pub base_params: [f32; 4],
    pub base_color: [f32; 4],

    pub specular_params: [f32; 4],
    pub specular_color: [f32; 4],

    pub transmission_params: [f32; 4],
    pub transmission_color: [f32; 4],
    pub transmission_scatter: [f32; 4],

    pub subsurface_params: [f32; 4],
    pub subsurface_color: [f32; 4],
    pub subsurface_radius_scale_gb: [f32; 4],

    pub fuzz_params: [f32; 4],
    pub fuzz_color: [f32; 4],

    pub coat_params: [f32; 4],
    pub coat_color: [f32; 4],
    pub coat_ior: [f32; 4],

    pub thin_film_params: [f32; 4],

    pub emission_params: [f32; 4],
    pub emission_color: [f32; 4],

    pub geometry_params: [f32; 4],
    pub geometry_params2: [f32; 4],
}

impl OpenPBRMaterial {
    pub const VEC4_COUNT: usize = 20;
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn pbr() -> Self {
        Self {
            base_params: [1.0, 0.0, 0.0, 0.0],
            base_color: [0.8, 0.8, 0.8, 1.0],

            specular_params: [1.0, 0.3, 1.5, 0.0],
            specular_color: [1.0, 1.0, 1.0, 0.0],

            transmission_params: [0.0, 0.0, 0.0, 0.0],
            transmission_color: [1.0, 1.0, 1.0, 0.0],
            transmission_scatter: [0.0, 0.0, 0.0, 0.0],

            subsurface_params: [0.0, 0.0, 1.0, 0.0],
            subsurface_color: [0.8, 0.8, 0.8, 1.0],
            subsurface_radius_scale_gb: [1.0, 1.0, 0.0, 0.0],

            fuzz_params: [0.0, 0.5, 0.0, 0.0],
            fuzz_color: [1.0, 1.0, 1.0, 0.0],

            coat_params: [0.0, 0.0, 0.0, 1.0],
            coat_color: [1.0, 1.0, 1.0, 0.0],
            coat_ior: [1.5, 0.0, 0.0, 0.0],

            thin_film_params: [0.0, 0.0, 1.33, 0.0],

            emission_params: [0.0, 0.0, 0.0, 0.0],
            emission_color: [1.0, 1.0, 1.0, 1.0],

            geometry_params: [1.0, 0.0, 0.0, 0.0],
            geometry_params2: [0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn base_weight(mut self, v: f32) -> Self {
        self.base_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn base_diffuse_roughness(mut self, v: f32) -> Self {
        self.base_params[1] = v.clamp(0.0, 1.0);
        self
    }
    pub fn base_metalness(mut self, v: f32) -> Self {
        self.base_params[2] = v.clamp(0.0, 1.0);
        self
    }

    pub fn base_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.base_color = [r, g, b, a];
        self
    }
    pub fn base_color_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.base_color[0] = rgb[0];
        self.base_color[1] = rgb[1];
        self.base_color[2] = rgb[2];
        self
    }

    pub fn specular_weight(mut self, v: f32) -> Self {
        self.specular_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn specular_roughness(mut self, v: f32) -> Self {
        self.specular_params[1] = v.clamp(0.0, 1.0);
        self
    }
    pub fn specular_ior(mut self, v: f32) -> Self {
        self.specular_params[2] = v.max(1.0);
        self
    }
    pub fn specular_anisotropy(mut self, v: f32) -> Self {
        self.specular_params[3] = v.clamp(0.0, 1.0);
        self
    }

    pub fn specular_edge_tint(mut self, r: f32, g: f32, b: f32) -> Self {
        self.specular_color = [r, g, b, 0.0];
        self
    }
    pub fn specular_edge_tint_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.specular_color[0] = rgb[0];
        self.specular_color[1] = rgb[1];
        self.specular_color[2] = rgb[2];
        self
    }

    pub fn transmission_weight(mut self, v: f32) -> Self {
        self.transmission_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn transmission_depth(mut self, v: f32) -> Self {
        self.transmission_params[1] = v.max(0.0);
        self
    }
    pub fn transmission_dispersion_scale(mut self, v: f32) -> Self {
        self.transmission_params[2] = v.max(0.0);
        self
    }
    pub fn transmission_dispersion_abbe(mut self, v: f32) -> Self {
        self.transmission_params[3] = v.max(0.0);
        self
    }

    pub fn transmission_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.transmission_color = [r, g, b, 0.0];
        self
    }
    pub fn transmission_color_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.transmission_color[0] = rgb[0];
        self.transmission_color[1] = rgb[1];
        self.transmission_color[2] = rgb[2];
        self
    }

    pub fn transmission_scatter_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.transmission_scatter = [r, g, b, 0.0];
        self
    }
    pub fn transmission_scatter_anisotropy(mut self, v: f32) -> Self {
        self.transmission_scatter[3] = v.clamp(-1.0, 1.0);
        self
    }

    pub fn subsurface_weight(mut self, v: f32) -> Self {
        self.subsurface_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn subsurface_radius(mut self, v: f32) -> Self {
        self.subsurface_params[1] = v.max(0.0);
        self
    }
    pub fn subsurface_radius_scale_r(mut self, v: f32) -> Self {
        self.subsurface_params[2] = v.max(0.0);
        self
    }
    pub fn subsurface_scatter_anisotropy(mut self, v: f32) -> Self {
        self.subsurface_params[3] = v.clamp(-1.0, 1.0);
        self
    }

    pub fn subsurface_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.subsurface_color = [r, g, b, 1.0];
        self
    }
    pub fn subsurface_color_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.subsurface_color[0] = rgb[0];
        self.subsurface_color[1] = rgb[1];
        self.subsurface_color[2] = rgb[2];
        self
    }

    pub fn subsurface_radius_scale_g(mut self, v: f32) -> Self {
        self.subsurface_radius_scale_gb[0] = v.max(0.0);
        self
    }
    pub fn subsurface_radius_scale_b(mut self, v: f32) -> Self {
        self.subsurface_radius_scale_gb[1] = v.max(0.0);
        self
    }

    pub fn fuzz_weight(mut self, v: f32) -> Self {
        self.fuzz_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn fuzz_roughness(mut self, v: f32) -> Self {
        self.fuzz_params[1] = v.clamp(0.0, 1.0);
        self
    }

    pub fn fuzz_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.fuzz_color = [r, g, b, 0.0];
        self
    }
    pub fn fuzz_color_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.fuzz_color[0] = rgb[0];
        self.fuzz_color[1] = rgb[1];
        self.fuzz_color[2] = rgb[2];
        self
    }

    pub fn coat_weight(mut self, v: f32) -> Self {
        self.coat_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn coat_roughness(mut self, v: f32) -> Self {
        self.coat_params[1] = v.clamp(0.0, 1.0);
        self
    }
    pub fn coat_anisotropy(mut self, v: f32) -> Self {
        self.coat_params[2] = v.clamp(0.0, 1.0);
        self
    }
    pub fn coat_darkening(mut self, v: f32) -> Self {
        self.coat_params[3] = v.clamp(0.0, 1.0);
        self
    }

    pub fn coat_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.coat_color = [r, g, b, 0.0];
        self
    }
    pub fn coat_color_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.coat_color[0] = rgb[0];
        self.coat_color[1] = rgb[1];
        self.coat_color[2] = rgb[2];
        self
    }

    pub fn coat_ior(mut self, v: f32) -> Self {
        self.coat_ior[0] = v.max(1.0);
        self
    }

    pub fn thin_film_weight(mut self, v: f32) -> Self {
        self.thin_film_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn thin_film_thickness_um(mut self, v: f32) -> Self {
        self.thin_film_params[1] = v.max(0.0);
        self
    }
    pub fn thin_film_ior(mut self, v: f32) -> Self {
        self.thin_film_params[2] = v.max(1.0);
        self
    }

    pub fn emission_luminance(mut self, nits: f32) -> Self {
        self.emission_params[0] = nits.max(0.0);
        self
    }
    pub fn emission_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.emission_color = [r, g, b, 1.0];
        self
    }
    pub fn emission_color_rgb(mut self, rgb: [f32; 3]) -> Self {
        self.emission_color[0] = rgb[0];
        self.emission_color[1] = rgb[1];
        self.emission_color[2] = rgb[2];
        self
    }

    pub fn opacity(mut self, v: f32) -> Self {
        self.geometry_params[0] = v.clamp(0.0, 1.0);
        self
    }
    pub fn thin_walled(mut self, v: bool) -> Self {
        self.geometry_params[1] = if v { 1.0 } else { 0.0 };
        self
    }

    pub fn metal() -> Self {
        Self::pbr()
            .base_metalness(1.0)
            .base_color_rgb([0.9, 0.9, 0.9])
            .specular_weight(1.0)
            .specular_roughness(0.1)
    }

    pub fn dielectric() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.8, 0.8, 0.8])
            .specular_weight(1.0)
            .specular_roughness(0.3)
            .specular_ior(1.5)
    }

    pub fn glass() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.0, 0.0, 0.0])
            .specular_weight(1.0)
            .specular_roughness(0.0)
            .specular_ior(1.5)
            .transmission_weight(1.0)
            .transmission_depth(1.0)
            .transmission_color_rgb([1.0, 1.0, 1.0])
            .thin_walled(true)
    }

    pub fn subsurface() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.8, 0.8, 0.8])
            .specular_weight(1.0)
            .specular_roughness(0.3)
            .specular_ior(1.3)
            .subsurface_weight(1.0)
            .subsurface_radius(0.1)
            .subsurface_color_rgb([0.8, 0.2, 0.1])
    }

    pub fn coat() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.8, 0.8, 0.8])
            .specular_weight(1.0)
            .specular_roughness(0.3)
            .specular_ior(1.5)
            .coat_weight(1.0)
            .coat_roughness(0.1)
            .coat_ior(1.5)
    }

    pub fn fuzz() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.8, 0.8, 0.8])
            .specular_weight(1.0)
            .specular_roughness(0.3)
            .specular_ior(1.5)
            .fuzz_weight(1.0)
            .fuzz_roughness(0.3)
            .fuzz_color_rgb([0.8, 0.8, 0.8])
    }

    pub fn thin_film() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.0, 0.0, 0.0])
            .specular_weight(1.0)
            .specular_roughness(0.0)
            .specular_ior(1.5)
            .thin_film_weight(1.0)
            .thin_film_thickness_um(300.0)
            .thin_film_ior(1.33)
    }

    pub fn emission() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.0, 0.0, 0.0])
            .specular_weight(0.0)
            .emission_luminance(1000.0)
            .emission_color_rgb([1.0, 0.8, 0.6])
    }

    pub fn transmission() -> Self {
        Self::pbr()
            .base_metalness(0.0)
            .base_color_rgb([0.0, 0.0, 0.0])
            .specular_weight(1.0)
            .specular_roughness(0.0)
            .specular_ior(1.5)
            .transmission_weight(1.0)
            .transmission_depth(2.0)
            .transmission_color_rgb([1.0, 1.0, 1.0])
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
        assert_eq!(mat.base_params[0], 1.0);
        assert_eq!(mat.base_color[0], 0.8);
        assert_eq!(mat.base_color[1], 0.8);
        assert_eq!(mat.base_color[2], 0.8);
        assert_eq!(mat.specular_params[1], 0.3);
        assert_eq!(mat.specular_params[2], 1.5);
    }

    #[test]
    fn test_metal_builder() {
        let mat = OpenPBRMaterial::metal();
        assert_eq!(mat.base_params[2], 1.0);
        assert_eq!(mat.specular_params[0], 1.0);
        assert_eq!(mat.specular_params[1], 0.1);
    }

    #[test]
    fn test_dielectric_builder() {
        let mat = OpenPBRMaterial::dielectric();
        assert_eq!(mat.base_params[2], 0.0);
        assert_eq!(mat.specular_params[0], 1.0);
        assert_eq!(mat.specular_params[1], 0.3);
        assert_eq!(mat.specular_params[2], 1.5);
    }

    #[test]
    fn test_glass_builder() {
        let mat = OpenPBRMaterial::glass();
        assert_eq!(mat.base_params[2], 0.0);
        assert_eq!(mat.transmission_params[0], 1.0);
        assert_eq!(mat.geometry_params[1], 1.0);
    }

    #[test]
    fn test_subsurface_builder() {
        let mat = OpenPBRMaterial::subsurface();
        assert_eq!(mat.subsurface_params[0], 1.0);
        assert_eq!(mat.subsurface_params[1], 0.1);
        assert_eq!(mat.subsurface_color[0], 0.8);
        assert_eq!(mat.subsurface_color[1], 0.2);
        assert_eq!(mat.subsurface_color[2], 0.1);
    }

    #[test]
    fn test_coat_builder() {
        let mat = OpenPBRMaterial::coat();
        assert_eq!(mat.coat_params[0], 1.0);
        assert_eq!(mat.coat_params[1], 0.1);
        assert_eq!(mat.coat_ior[0], 1.5);
    }

    #[test]
    fn test_fuzz_builder() {
        let mat = OpenPBRMaterial::fuzz();
        assert_eq!(mat.fuzz_params[0], 1.0);
        assert_eq!(mat.fuzz_params[1], 0.3);
    }

    #[test]
    fn test_thin_film_builder() {
        let mat = OpenPBRMaterial::thin_film();
        assert_eq!(mat.thin_film_params[0], 1.0);
        assert_eq!(mat.thin_film_params[1], 300.0);
        assert_eq!(mat.thin_film_params[2], 1.33);
    }

    #[test]
    fn test_emission_builder() {
        let mat = OpenPBRMaterial::emission();
        assert_eq!(mat.emission_params[0], 1000.0);
        assert_eq!(mat.emission_color[0], 1.0);
        assert_eq!(mat.emission_color[1], 0.8);
        assert_eq!(mat.emission_color[2], 0.6);
    }

    #[test]
    fn test_transmission_builder() {
        let mat = OpenPBRMaterial::transmission();
        assert_eq!(mat.transmission_params[0], 1.0);
        assert_eq!(mat.transmission_params[1], 2.0);
        assert_eq!(mat.transmission_color[0], 1.0);
    }

    #[test]
    fn test_parameter_clamping() {
        let mat = OpenPBRMaterial::pbr()
            .base_weight(1.5)
            .base_metalness(-0.5)
            .specular_roughness(2.0)
            .specular_ior(0.5);
        assert_eq!(mat.base_params[0], 1.0);
        assert_eq!(mat.base_params[2], 0.0);
        assert_eq!(mat.specular_params[1], 1.0);
        assert_eq!(mat.specular_params[2], 1.0);
    }

    #[test]
    fn test_opacity_and_thin_walled() {
        let mat = OpenPBRMaterial::pbr().opacity(0.5).thin_walled(true);
        assert_eq!(mat.geometry_params[0], 0.5);
        assert_eq!(mat.geometry_params[1], 1.0);
    }

    #[test]
    fn test_coat_darkening_parameter() {
        let mat = OpenPBRMaterial::pbr()
            .coat_weight(1.0)
            .coat_roughness(0.1)
            .coat_anisotropy(0.3)
            .coat_darkening(0.5);
        assert_eq!(mat.coat_params[0], 1.0);
        assert_eq!(mat.coat_params[1], 0.1);
        assert_eq!(mat.coat_params[2], 0.3);
        assert_eq!(mat.coat_params[3], 0.5);
    }

    #[test]
    fn test_anisotropy_parameters() {
        let mat = OpenPBRMaterial::pbr()
            .specular_anisotropy(0.5)
            .coat_anisotropy(0.8);
        assert_eq!(mat.specular_params[3], 0.5);
        assert_eq!(mat.coat_params[2], 0.8);
    }

    #[test]
    fn test_transmission_dispersion() {
        let mat = OpenPBRMaterial::pbr()
            .transmission_weight(1.0)
            .transmission_dispersion_scale(0.5)
            .transmission_dispersion_abbe(50.0);
        assert_eq!(mat.transmission_params[0], 1.0);
        assert_eq!(mat.transmission_params[2], 0.5);
        assert_eq!(mat.transmission_params[3], 50.0);
    }

    #[test]
    fn test_subsurface_radius_scales() {
        let mat = OpenPBRMaterial::pbr()
            .subsurface_weight(1.0)
            .subsurface_radius(0.5)
            .subsurface_radius_scale_r(1.0)
            .subsurface_radius_scale_g(0.5)
            .subsurface_radius_scale_b(0.25);
        assert_eq!(mat.subsurface_params[0], 1.0);
        assert_eq!(mat.subsurface_params[1], 0.5);
        assert_eq!(mat.subsurface_params[2], 1.0);
        assert_eq!(mat.subsurface_radius_scale_gb[0], 0.5);
        assert_eq!(mat.subsurface_radius_scale_gb[1], 0.25);
    }

    #[test]
    fn test_fuzz_parameters() {
        let mat = OpenPBRMaterial::pbr()
            .fuzz_weight(0.8)
            .fuzz_roughness(0.4)
            .fuzz_color_rgb([0.9, 0.9, 0.9]);
        assert_eq!(mat.fuzz_params[0], 0.8);
        assert_eq!(mat.fuzz_params[1], 0.4);
        assert_eq!(mat.fuzz_color[0], 0.9);
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
}
