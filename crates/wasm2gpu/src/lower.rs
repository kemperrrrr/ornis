//! Toy IR to a naga module: vertex and fragment entries, plus `cs_main`
//! when that export exists.

use naga::BuiltIn;
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
use crate::env::Code;
use crate::env::ExecIn;
use crate::error::TranslateError;
use crate::flow;
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
        specs: Vec::new(),
        active: Vec::new(),
    };
    let vertex = stage(symbols::VS_MAIN, VS_PARAMS, VS_RESULTS, true);
    let fragment = stage(symbols::FS_MAIN, FS_PARAMS, FS_RESULTS, false);
    let vs = shader(&mut cx, module, &vertex, None)?;
    let fs = shader(&mut cx, module, &fragment, None)?;
    cx.module
        .entry_points
        .push(entry(symbols::VS_MAIN, ShaderStage::Vertex, [0, 0, 0], vs));
    cx.module.entry_points.push(entry(
        symbols::FS_MAIN,
        ShaderStage::Fragment,
        [0, 0, 0],
        fs,
    ));
    if let Some(index) = find_index(module, symbols::CS_MAIN) {
        let compute = stage(symbols::CS_MAIN, 1, 0, false);
        let cs = shader(&mut cx, module, &compute, Some(index))?;
        cx.module.entry_points.push(entry(
            symbols::CS_MAIN,
            ShaderStage::Compute,
            symbols::CS_WORKGROUP,
            cs,
        ));
    }
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

fn shader(
    cx: &mut Cx<'_>,
    module: &Module,
    stage: &Stage,
    compute: Option<u32>,
) -> Result<Function, TranslateError> {
    let index = match compute {
        Some(index) => index,
        None => find_index(module, stage.name).ok_or(TranslateError::MissingEntry(stage.name))?,
    };
    let sig = module
        .funcs
        .get(index as usize)
        .ok_or(TranslateError::Unsupported)?
        .sig;
    if !signature_ok(stage, &sig, compute.is_some()) {
        return Err(TranslateError::Unsupported);
    }
    cfg::check(
        &module.funcs[index as usize],
        &module.imports,
        &module.structs,
        &module.funcs,
    )?;
    let id_param = compute.map(|_| 0);
    let arguments = if compute.is_some() {
        vec![invocation_arg(cx)]
    } else {
        args(cx, &sig)
    };
    let mut body = Function {
        name: Some(stage.name.to_string()),
        arguments,
        ..Function::default()
    };
    cx.active.push(index);
    let code = Code {
        imports: &module.imports,
        structs: &module.structs,
        funcs: &module.funcs,
    };
    let stack = flow::exec(
        cx,
        &code,
        &mut body,
        &ExecIn {
            func_index: index,
            baked: Vec::new(),
            id_param,
        },
    );
    cx.active.pop();
    let stack = stack?;
    if compute.is_some() {
        if !stack.is_empty() {
            return Err(TranslateError::Unsupported);
        }
        return Ok(body);
    }
    let value = if stage.vertex {
        ret::vertex(cx, &mut body, &stack)?
    } else {
        ret::fragment(cx, &mut body, &stack)?
    };
    ret::finish(&mut body, value);
    Ok(body)
}

fn find_index(module: &Module, name: &str) -> Option<u32> {
    let mut index = 0usize;
    while index < module.funcs.len() {
        if module.funcs[index].export == name {
            return Some(index as u32);
        }
        index += 1;
    }
    None
}

fn invocation_arg(cx: &mut Cx<'_>) -> FunctionArgument {
    let inner = TypeInner::Vector {
        size: VectorSize::Tri,
        scalar: Scalar {
            kind: ScalarKind::Uint,
            width: F32_WIDTH,
        },
    };
    let ty = cx
        .module
        .types
        .insert(Type { name: None, inner }, Span::default());
    FunctionArgument {
        name: Some("global_invocation_id".to_string()),
        ty,
        binding: Some(naga::Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
    }
}

fn entry(
    name: &str,
    stage: ShaderStage,
    workgroup_size: [u32; 3],
    function: Function,
) -> naga::EntryPoint {
    naga::EntryPoint {
        name: name.to_string(),
        stage,
        early_depth_test: None,
        workgroup_size,
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}

fn signature_ok(stage: &Stage, sig: &Sig, compute: bool) -> bool {
    if compute {
        return sig.results == 0 && sig.params == 1 && sig.param_kind[0] == Kind::I32;
    }
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
