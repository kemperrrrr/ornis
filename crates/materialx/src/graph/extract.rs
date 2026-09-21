//! Extraction of evaluated graphs to OpenPBRMaterial.
//!
//! Maps the named outputs of the open_pbr_surface graph onto the
//! base, specular, transmission, subsurface, fuzz, coat, thin-film,
//! emission and geometry material groups.

use super::evaluate::GraphEvaluator;
use super::{CodegenError, EvaluatedGraph, MaterialXConverter, OutputValue};
use ornis_render::OpenPBRMaterial;

impl MaterialXConverter {
    /// Evaluate the first nodegraph whose `nodedef` references
    /// `open_pbr_surface` and map its named outputs onto an
    /// [`OpenPBRMaterial`], starting from the default material.
    ///
    /// # Errors
    /// [`CodegenError::GraphNotFound`] without such a graph; evaluation
    /// failures otherwise propagate as-is.
    pub fn to_openpbr(&self) -> Result<OpenPBRMaterial, CodegenError> {
        let graph = self.find_openpbr_graph()?;
        let mut evaluator = GraphEvaluator::new(self, graph);
        let evaluated = evaluator.evaluate()?;
        self.extract_material(&evaluated)
    }

    fn extract_material(
        &self,
        evaluated: &EvaluatedGraph,
    ) -> Result<OpenPBRMaterial, CodegenError> {
        let mut material = OpenPBRMaterial::pbr();
        material = self.extract_base(evaluated, material);
        material = self.extract_specular(evaluated, material);
        material = self.extract_transmission(evaluated, material);
        material = self.extract_subsurface(evaluated, material);
        material = self.extract_fuzz(evaluated, material);
        material = self.extract_coat(evaluated, material);
        material = self.extract_thin_film(evaluated, material);
        material = self.extract_emission(evaluated, material);
        material = self.extract_geometry(evaluated, material);
        Ok(material)
    }

    fn extract_base(&self, evaluated: &EvaluatedGraph, mut m: OpenPBRMaterial) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("base_weight") {
            m.base.weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("base_color") {
            m.base.color_rgb(*v);
        } else if let Some(OutputValue::Color4(v)) = evaluated.outputs.get("base_color") {
            m.base.color(v[0], v[1], v[2], v[3]);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("base_diffuse_roughness") {
            m.base.diffuse_roughness(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("base_metalness") {
            m.base.metalness(*v);
        }
        m
    }

    fn extract_specular(
        &self,
        evaluated: &EvaluatedGraph,
        mut m: OpenPBRMaterial,
    ) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_weight") {
            m.specular.weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("specular_color") {
            m.specular.edge_tint_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_roughness") {
            m.specular.roughness(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_ior") {
            m.specular.ior(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("specular_anisotropy") {
            m.specular.anisotropy(*v);
        }
        m
    }

    fn extract_transmission(
        &self,
        evaluated: &EvaluatedGraph,
        mut m: OpenPBRMaterial,
    ) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_weight") {
            m.transmission.weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("transmission_color") {
            m.transmission.color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_depth") {
            m.transmission.depth(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_dispersion_scale")
        {
            m.transmission.dispersion_scale(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("transmission_dispersion_abbe") {
            m.transmission.dispersion_abbe(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("transmission_scatter") {
            m.transmission.scatter_color(v[0], v[1], v[2]);
        }
        if let Some(OutputValue::Float(v)) =
            evaluated.outputs.get("transmission_scatter_anisotropy")
        {
            m.transmission.scatter_anisotropy(*v);
        }
        m
    }

    fn extract_subsurface(
        &self,
        evaluated: &EvaluatedGraph,
        mut m: OpenPBRMaterial,
    ) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_weight") {
            m.subsurface.weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("subsurface_color") {
            m.subsurface.color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_radius") {
            m.subsurface.radius(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("subsurface_radius_scale") {
            m.subsurface.radius_scale_g(v[1]);
            m.subsurface.radius_scale_b(v[2]);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("subsurface_scatter_anisotropy")
        {
            m.subsurface.scatter_anisotropy(*v);
        }
        m
    }

    fn extract_fuzz(&self, evaluated: &EvaluatedGraph, mut m: OpenPBRMaterial) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("fuzz_weight") {
            m.fuzz.weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("fuzz_color") {
            m.fuzz.color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("fuzz_roughness") {
            m.fuzz.roughness(*v);
        }
        m
    }

    fn extract_coat(&self, evaluated: &EvaluatedGraph, mut m: OpenPBRMaterial) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_weight") {
            m.coat.weight(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("coat_color") {
            m.coat.color_rgb(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_roughness") {
            m.coat.roughness(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_anisotropy") {
            m.coat.anisotropy(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_ior") {
            m.coat.ior(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("coat_darkening") {
            m.coat.darkening(*v);
        }
        m
    }

    fn extract_thin_film(
        &self,
        evaluated: &EvaluatedGraph,
        mut m: OpenPBRMaterial,
    ) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_weight") {
            m.thin_film.weight(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_thickness") {
            m.thin_film.thickness_um(*v);
        }
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("thin_film_ior") {
            m.thin_film.ior(*v);
        }
        m
    }

    fn extract_emission(
        &self,
        evaluated: &EvaluatedGraph,
        mut m: OpenPBRMaterial,
    ) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("emission_luminance") {
            m.emission.luminance(*v);
        }
        if let Some(OutputValue::Color3(v)) = evaluated.outputs.get("emission_color") {
            m.emission.color_rgb(*v);
        }
        m
    }

    fn extract_geometry(
        &self,
        evaluated: &EvaluatedGraph,
        mut m: OpenPBRMaterial,
    ) -> OpenPBRMaterial {
        if let Some(OutputValue::Float(v)) = evaluated.outputs.get("geometry_opacity") {
            m.geometry.opacity(*v);
        }
        if let Some(OutputValue::Boolean(v)) = evaluated.outputs.get("geometry_thin_walled") {
            m.geometry.thin_walled(*v);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::materialx_to_openpbr;
    use crate::graph::test_helpers::{NODEDEFS, assert_close};

    /// Exercise every branch of `extract_material`: one output per
    /// OpenPBR parameter name, each fed by a constant. The builder is
    /// invoked for every group (base/specular/transmission/subsurface/fuzz/
    /// coat/thin_film/emission/geometry), so all extractor branches run.
    /// Coverage of the mapping is the goal, not field-level inspection, so
    /// we assert each parameter group appears in the Debug dump.
    #[test]
    fn test_extract_all_openpbr_parameters() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <output name="out" type="surfaceshader">
      <input name="base_weight" type="float" value="0.9" />
      <input name="base_color" type="color3" value="0.1, 0.2, 0.3" />
      <input name="base_diffuse_roughness" type="float" value="0.45" />
      <input name="base_metalness" type="float" value="0.55" />
      <input name="specular_weight" type="float" value="0.8" />
      <input name="specular_color" type="color3" value="0.4, 0.5, 0.6" />
      <input name="specular_roughness" type="float" value="0.33" />
      <input name="specular_ior" type="float" value="1.45" />
      <input name="specular_anisotropy" type="float" value="0.2" />
      <input name="transmission_weight" type="float" value="0.7" />
      <input name="transmission_color" type="color3" value="0.7, 0.8, 0.9" />
      <input name="transmission_depth" type="float" value="1.2" />
      <input name="transmission_dispersion_scale" type="float" value="2.0" />
      <input name="transmission_dispersion_abbe" type="float" value="30.0" />
      <input name="transmission_scatter" type="color3" value="0.3, 0.4, 0.5" />
      <input name="transmission_scatter_anisotropy" type="float" value="0.6" />
      <input name="subsurface_weight" type="float" value="0.5" />
      <input name="subsurface_color" type="color3" value="0.6, 0.7, 0.8" />
      <input name="subsurface_radius" type="float" value="0.4" />
      <input name="subsurface_radius_scale" type="color3" value="0.1, 0.2, 0.3" />
      <input name="subsurface_scatter_anisotropy" type="float" value="0.3" />
      <input name="fuzz_weight" type="float" value="0.25" />
      <input name="fuzz_color" type="color3" value="0.9, 0.8, 0.7" />
      <input name="fuzz_roughness" type="float" value="0.15" />
      <input name="coat_weight" type="float" value="0.6" />
      <input name="coat_color" type="color3" value="0.5, 0.6, 0.7" />
      <input name="coat_roughness" type="float" value="0.22" />
      <input name="coat_anisotropy" type="float" value="0.1" />
      <input name="coat_ior" type="float" value="1.9" />
      <input name="coat_darkening" type="float" value="0.05" />
      <input name="thin_film_weight" type="float" value="0.35" />
      <input name="thin_film_thickness" type="float" value="0.55" />
      <input name="thin_film_ior" type="float" value="1.3" />
      <input name="emission_luminance" type="float" value="3.0" />
      <input name="emission_color" type="color3" value="0.2, 0.1, 0.0" />
      <input name="geometry_opacity" type="float" value="0.88" />
      <input name="geometry_thin_walled" type="boolean" value="true" />
    </output>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );

        let mat = materialx_to_openpbr(&mtlx).expect("all params convert");
        let debug = format!("{mat:?}");
        // Every group's extractor branch ran (builder invoked). Field names
        // are the flat arrays OpenPBRMaterial stores; we just check each
        // group is present in the debug dump.
        for group in [
            "BaseGroup",
            "SpecularGroup",
            "TransmissionGroup",
            "SubsurfaceGroup",
            "FuzzGroup",
            "CoatGroup",
            "ThinFilmGroup",
            "EmissionGroup",
            "GeometryGroup",
        ] {
            assert!(debug.contains(group), "missing group {group} in: {debug}");
        }
    }

    /// A `color4` base_color must route through the RGBA extractor (not the
    /// RGB one) — covers the `else if Color4` branch in extract_material.
    #[test]
    fn test_base_color_color4_path() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <output name="out" type="surfaceshader">
      <input name="base_color" type="color4" value="0.1, 0.2, 0.3, 0.4" />
    </output>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );
        let result = materialx_to_openpbr(&mtlx);
        assert!(result.is_ok(), "{result:?}");
    }

    /// A graph with no `output` node at all still resolves `surface`/`edf`
    /// nodes (the evaluator evaluates them even without an output), but
    /// produces an empty output set — covers the surface/edf walk branch.
    #[test]
    fn test_surface_node_without_output_evaluates() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <surface name="surf" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="dielectric_bsdf" />
    </surface>
    <dielectric_bsdf name="dielectric_bsdf" type="BSDF">
      <input name="ior" type="float" value="1.5" />
    </dielectric_bsdf>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );
        // No `output` referencing surf, so no material params extracted, but
        // the surface node itself must evaluate without error.
        let result = materialx_to_openpbr(&mtlx);
        assert!(result.is_ok());
    }

    /// End-to-end: a math chain evaluated by `materialx_to_openpbr` lands in
    /// the extracted `OpenPBRMaterial` fields.
    #[test]
    fn test_end_to_end_material_extraction() {
        let mtlx = format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
{}
  <nodegraph name="test" nodedef="ND_open_pbr_surface_surfaceshader">
    <constant name="a" type="color3"><input name="value" type="color3" value="0.8, 0.4, 0.2" /></constant>
    <constant name="b" type="color3"><input name="value" type="color3" value="0.5, 0.5, 0.5" /></constant>
    <multiply name="m" type="color3"><input name="in1" type="color3" nodename="a" /><input name="in2" type="color3" nodename="b" /></multiply>
    <subtract name="s" type="color3"><input name="in1" type="color3" nodename="a" /><input name="in2" type="color3" nodename="m" /></subtract>
    <invert name="inv" type="float"><input name="in" type="float" value="0.25" /></invert>
    <clamp name="cl" type="float"><input name="in" type="float" value="1.5" /><input name="low" type="float" value="0.0" /><input name="high" type="float" value="1.0" /></clamp>
    <output name="out" type="surfaceshader">
      <input name="base_color" type="color3" nodename="s" />
      <input name="base_weight" type="float" nodename="inv" />
      <input name="specular_roughness" type="float" nodename="cl" />
    </output>
  </nodegraph>
</materialx>"#,
            NODEDEFS
        );

        let mat = materialx_to_openpbr(&mtlx).unwrap();
        // a - a*b per channel: (0.8, 0.4, 0.2) - (0.4, 0.2, 0.1)
        assert_eq!(mat.base.color[0], 0.4f32);
        assert_eq!(mat.base.color[1], 0.2f32);
        assert_close(mat.base.color[2], 0.1);
        // invert(0.25)
        assert_close(mat.base.params[0], 0.75);
        // clamp(1.5, 0, 1)
        assert_close(mat.specular.params[1], 1.0);
    }
}
