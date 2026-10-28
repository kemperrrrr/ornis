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

use crate::cx::Cx;
use crate::cx::F32_WIDTH;
use crate::cx::FS_PARAMS;
use crate::cx::FS_RESULTS;
use crate::cx::Stage;
use crate::cx::VS_PARAMS;
use crate::cx::VS_RESULTS;
use crate::cx::location;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Module;
use crate::ops;
use crate::ret;
use ornis_gpu_abi::symbols;

/// Lower `module` to naga IR. Group and binding come from `i32.const` operands.
pub fn lower(module: &Module) -> Result<naga::Module, TranslateError> {
    let mut out = naga::Module::default();
    let scalar = scalar_ty(&mut out);
    let vec2 = vector_ty(&mut out, VectorSize::Bi);
    let vec4 = vector_ty(&mut out, VectorSize::Quad);
    let mut cx = Cx {
        module: &mut out,
        scalar,
        vec2,
        vec4,
        uniforms: Vec::new(),
        textures: Vec::new(),
        samplers: Vec::new(),
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

fn scalar_ty(module: &mut naga::Module) -> naga::Handle<Type> {
    let inner = TypeInner::Scalar(Scalar {
        kind: ScalarKind::Float,
        width: F32_WIDTH,
    });
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
    if func.sig.params != stage.params || func.sig.results != stage.results {
        return Err(TranslateError::Unsupported);
    }
    let mut body = Function {
        name: Some(stage.name.to_string()),
        arguments: args(cx.scalar, stage.params),
        ..Function::default()
    };
    let stack = run(cx, module, func, &mut body)?;
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

fn args(ty: naga::Handle<Type>, count: u32) -> Vec<FunctionArgument> {
    let mut out = Vec::new();
    let mut index = 0u32;
    while index < count {
        out.push(FunctionArgument {
            name: None,
            ty,
            binding: Some(location(index)),
        });
        index += 1;
    }
    out
}

fn run(
    cx: &mut Cx<'_>,
    module: &Module,
    func: &Func,
    body: &mut Function,
) -> Result<Vec<crate::cx::Val>, TranslateError> {
    let mut stack = Vec::new();
    let mut index = 0usize;
    while index < func.ops.len() {
        ops::step(cx, &module.imports, body, &mut stack, func.ops[index])?;
        index += 1;
    }
    Ok(stack)
}
