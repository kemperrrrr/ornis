#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub base_color: [f32; 4],
    pub emission_color: [f32; 4],
    pub pbr_params: [f32; 4],
    pub ior_params: [f32; 4],
    pub subsurface_color: [f32; 4],
    pub coat_color: [f32; 4],
    pub sheen_color: [f32; 4],
}

impl Material {
    pub fn pbr(base_color: [f32; 4], metalness: f32, roughness: f32) -> Self {
        Self {
            base_color,
            emission_color: [0.0; 4],
            pbr_params: [metalness, roughness, 1.0, 0.0],
            ior_params: [1.5, 0.0, 0.0, 0.0],
            subsurface_color: [0.0; 4],
            coat_color: [0.0; 4],
            sheen_color: [0.0; 4],
        }
    }

    pub fn metalness(&self) -> f32 { self.pbr_params[0] }
    pub fn roughness(&self) -> f32 { self.pbr_params[1] }
    pub fn specular_weight(&self) -> f32 { self.pbr_params[2] }
    pub fn coat_weight(&self) -> f32 { self.pbr_params[3] }

    pub fn specular_ior(&self) -> f32 { self.ior_params[0] }
    pub fn subsurface_weight(&self) -> f32 { self.ior_params[1] }
    pub fn sheen_weight(&self) -> f32 { self.ior_params[2] }
    pub fn coat_roughness(&self) -> f32 { self.ior_params[3] }

    pub fn set_metalness(&mut self, v: f32) { self.pbr_params[0] = v; }
    pub fn set_roughness(&mut self, v: f32) { self.pbr_params[1] = v; }
    pub fn set_specular_weight(&mut self, v: f32) { self.pbr_params[2] = v; }
    pub fn set_specular_ior(&mut self, v: f32) { self.ior_params[0] = v; }
    pub fn set_coat_weight(&mut self, v: f32) { self.pbr_params[3] = v; }

    pub fn set_subsurface(&mut self, weight: f32, color: [f32; 3]) {
        self.ior_params[1] = weight;
        self.subsurface_color = [color[0], color[1], color[2], 0.0];
    }

    pub fn set_sheen(&mut self, weight: f32, color: [f32; 3]) {
        self.ior_params[2] = weight;
        self.sheen_color = [color[0], color[1], color[2], 0.0];
    }

    pub fn set_coat(&mut self, weight: f32, color: [f32; 3], roughness: f32) {
        self.pbr_params[3] = weight;
        self.ior_params[3] = roughness;
        self.coat_color = [color[0], color[1], color[2], 0.0];
    }

    pub fn set_emission(&mut self, color: [f32; 3], intensity: f32) {
        self.emission_color = [color[0], color[1], color[2], intensity];
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::pbr([0.8, 0.8, 0.8, 1.0], 0.0, 0.5)
    }
}
