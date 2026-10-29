//! Straight-line IR to a naga module: one vertex entry and one fragment entry.

use naga::Function;
use naga::FunctionArgument;
use naga::Scalar;
use naga::ScalarKind;
use naga::ShaderStage;
use naga::Span;
use naga::Type;
use naga::TypeInner;
use naga::VectorSize;

use crate::cfg;
use crate::cx::Cx;
use crate::cx::F32_WIDTH;
use crate::cx::FS_PARAMS;
use crate::cx::FS_RESULTS;
use crate::cx::Stage;
use crate::cx::VS_PARAMS;
use crate::cx::VS_RESULTS;
use crate::cx::location;
use crate::error::TranslateError;
use crate::flow;
use crate::ir::Func;
use crate::ir::Kind;
use crate::ir::Module;
use crate::ir::Sig;
use crate::ret;
use ornis_gpu_abi::symbols;

/// Lower `module` to naga IR. Group and binding come from `i32.const` operands.
pub fn lower(module: &Module) -> Result<naga::Module, TranslateError> {
    let mut out = naga::Module::default();
    let scalar = scalar_ty(&mut out, ScalarKind::Float);
    let uint = scalar_ty(&mut out, ScalarKind::Uint);
    let sint = scalar_ty(&mut out, ScalarKind::Sint);
    let bool_ty = bool_ty(&mut out);
    let vec2 = vector_ty(&mut out, VectorSize::Bi);
    let vec3 = vector_ty(&mut out, VectorSize::Tri);
    let vec4 = vector_ty(&mut out, VectorSize::Quad);
    let mut cx = Cx {
        module: &mut out,
        scalar,
        uint,
        sint,
        bool_ty,
        vec2,
        vec3,
        vec4,
        uniforms: Vec::new(),
        storages: Vec::new(),
        textures: Vec::new(),
        samplers: Vec::new(),
        struct_tys: Vec::new(),
    };
    let vertex = stage(symbols::VS_MAIN, VS_PARAMS, VS_RESULTS, true);
    let fragment = stage(symbols::FS_MAIN, FS_PARAMS, FS_RESULTS, false);
    let vs = shader(&mut cx, module, &vertex)?;
    let fs = shader(&mut cx, module, &fragment)?;
    cx.module
        .entry_points
        .push(entry(symbols::VS_MAIN, ShaderStage::Vertex, vs));
    cx.module
        .entry_points
        .push(entry(symbols::FS_MAIN, ShaderStage::Fragment, fs));
    Ok(out)
}

fn stage(name: &'static str, params: u32, results: u32, vertex: bool) -> Stage {
    Stage {
        name,
        params,
        results,
        vertex,
    }
}

fn scalar_ty(module: &mut naga::Module, kind: ScalarKind) -> naga::Handle<Type> {
    let inner = TypeInner::Scalar(Scalar {
        kind,
        width: F32_WIDTH,
    });
    module
        .types
        .insert(Type { name: None, inner }, Span::default())
}

fn bool_ty(module: &mut naga::Module) -> naga::Handle<Type> {
    let inner = TypeInner::Scalar(Scalar::BOOL);
    module
        .types
        .insert(Type { name: None, inner }, Span::default())
}

fn vector_ty(module: &mut naga::Module, size: VectorSize) -> naga::Handle<Type> {
    let inner = TypeInner::Vector {
        size,
        scalar: Scalar {
            kind: ScalarKind::Float,
            width: F32_WIDTH,
        },
    };
    module
        .types
        .insert(Type { name: None, inner }, Span::default())
}

fn shader(cx: &mut Cx<'_>, module: &Module, stage: &Stage) -> Result<Function, TranslateError> {
    let func = find(module, stage.name)?;
    if !signature_ok(stage, &func.sig) {
        return Err(TranslateError::Unsupported);
    }
    cfg::check(func, &module.imports, &module.structs)?;
    let mut body = Function {
        name: Some(stage.name.to_string()),
        arguments: args(cx, &func.sig),
        ..Function::default()
    };
    let stack = flow::exec(cx, module, func, &mut body)?;
    let value = if stage.vertex {
        ret::vertex(cx, &mut body, &stack)?
    } else {
        ret::fragment(cx, &mut body, &stack)?
    };
    ret::finish(&mut body, value);
    Ok(body)
}

fn find<'a>(module: &'a Module, name: &'static str) -> Result<&'a Func, TranslateError> {
    let mut index = 0usize;
    while index < module.funcs.len() {
        if module.funcs[index].export == name {
            return Ok(&module.funcs[index]);
        }
        index += 1;
    }
    Err(TranslateError::MissingEntry(name))
}

fn entry(name: &str, stage: ShaderStage, function: Function) -> naga::EntryPoint {
    naga::EntryPoint {
        name: name.to_string(),
        stage,
        early_depth_test: None,
        workgroup_size: [0, 0, 0],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}

fn signature_ok(stage: &Stage, sig: &Sig) -> bool {
    if sig.results != stage.results {
        return false;
    }
    if stage.vertex {
        return sig.params == stage.params && f32_params(sig);
    }
    true
}

fn f32_params(sig: &Sig) -> bool {
    let mut index = 0usize;
    while index < sig.params as usize {
        if sig.param_kind[index] != Kind::F32 {
            return false;
        }
        index += 1;
    }
    true
}

fn args(cx: &Cx<'_>, sig: &Sig) -> Vec<FunctionArgument> {
    let mut out = Vec::new();
    let mut index = 0u32;
    while index < sig.params {
        let kind = sig.param_kind[index as usize];
        let ty = if kind == Kind::I32 {
            cx.uint
        } else {
            cx.scalar
        };
        out.push(FunctionArgument {
            name: None,
            ty,
            binding: Some(arg_location(index, kind)),
        });
        index += 1;
    }
    out
}

fn arg_location(index: u32, kind: Kind) -> naga::Binding {
    if kind == Kind::F32 {
        return location(index);
    }
    // Integer fragment inputs have no default interpolation. Flat is required.
    naga::Binding::Location {
        location: index,
        interpolation: Some(naga::Interpolation::Flat),
        sampling: None,
        blend_src: None,
        per_primitive: false,
    }
}
