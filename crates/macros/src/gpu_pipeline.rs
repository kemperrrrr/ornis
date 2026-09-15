//! `#[gpu_pipeline]` — translate a Rust function to a WGSL compute shader.
//!
//! Two modes:
//!
//! 1. **Legacy (no arguments):** the function's parameters become read-only
//!    storage buffers (`array<T>`) and its return value is written into an
//!    output storage buffer; the body is the tail expression. See
//!    [`crate::wgsl::wgsl_source_from_fn`].
//!
//! 2. **Full shader (with arguments):** the function body *is* the compute
//!    entry point body, written in the kernel DSL (see [`crate::wgsl`]).
//!    Bindings, workgroup size and built-ins are declared in the attribute:
//!
//! ```ignore
//! #[gpu_pipeline(
//!     workgroup_size = 4,
//!     storage(body_buf: [BodyState; 64], read_write),
//!     storage(batch_buf: [ContactBatch; 64], read_write),
//!     uniform(params: [u32; 4]),
//!     builtin(gid: workgroup_id, lid: local_invocation_id),
//!     helpers(solve_ldl, mat3_ops),
//! )]
//! fn solver() {
//!     // kernel DSL body; builtins and bindings are in scope as identifiers
//!     if gid.x >= params.x { return; }
//!     body_buf[gid.x] = body_buf[gid.x] + batch_buf[gid.x].acc[lid.x];
//! }
//! ```
//!
//! `helpers(a, b, ...)` stitches the named helper modules (emitted by
//! `#[wgsl_fn]` / `#[kernel]`, each exposing `wgsl_source()`) ahead of
//! the entry body, in order — the compute-path analog of helper reuse
//! on the stage path. Paths resolve at the annotated item's site. With
//! helpers the generated `wgsl_source()` returns an owned `String`
//! (without them it stays a `&'static str` literal, as before).
//!
//! `storage(name: Type, access)` declares `@group(0) @binding(i) var<storage>
//! name: array<Elem>` (bindings are numbered in declaration order). A Rust
//! array type `[Elem; N]` contributes its element type; `N` is documentation
//! only (WGSL storage arrays are runtime-sized). A scalar array `[f32; N]`
//! with N = 2..=4 contributes `vecN<f32>` as the element (WGSL vector);
//! longer scalar arrays contribute `array<f32>`. `uniform(name: Type)`
//! declares a uniform buffer: `[f32; N]` maps to `vecN<f32>` for N = 2..=4,
//! any other type is passed through. `builtin(name: semantic, ...)` declares
//! `main` parameters as `@builtin(semantic) name: vec3<u32>`.
//!
//! The generated module exposes `wgsl_source()`, `pipeline_label()` and
//! `create_shader_module(device)`.

use ornis_shader_lang::{
    ShaderType,
    ir::{IrFieldAttr, IrGlobal, IrItem, IrStructField, IrTexture, IrType},
    writer::{print_builtin_param, print_item},
};
use proc_macro::TokenStream;
use proc_macro2::{Delimiter, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{ItemFn, LitInt, parse_macro_input, spanned::Spanned};

/// Parsed `#[gpu_pipeline(...)]` options (`syn::Path` has no `Debug`, so
/// neither does this — format fields individually when debugging).
struct ShaderConfig {
    workgroup_size: u32,
    items: Vec<IrItem>,
    builtins: Vec<(String, String)>,
    vertex_entry: Option<String>,
    fragment_entry: Option<String>,
    /// Helper modules (from `#[wgsl_fn]` / `#[kernel]`) whose
    /// `wgsl_source()` is stitched ahead of the entry body, in order.
    helpers: Vec<syn::Path>,
}

fn is_punct(tt: &TokenTree, ch: char) -> bool {
    matches!(tt, TokenTree::Punct(p) if p.as_char() == ch)
}

fn parse_u32(tt: &TokenTree) -> syn::Result<u32> {
    let lit = match tt {
        TokenTree::Literal(l) => l,
        other => {
            return Err(syn::Error::new(other.span(), "expected an integer literal"));
        }
    };
    let int: LitInt = syn::parse2(TokenStream2::from(TokenTree::from(lit.clone())))?;
    int.base10_parse()
}

/// Map a Rust type token (ident, `[T; N]` or nested scalar array) to a
/// structural [`IrType`]: scalar arrays of length 2..=4 classify to WGSL
/// vectors (a layout rule, not a spelling); longer ones are
/// runtime-sized arrays (the length documents buffer capacity).
fn parse_binding_base_ty(tt: &TokenTree) -> syn::Result<IrType> {
    match tt {
        TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket => {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            let elem_tt = inner
                .first()
                .ok_or_else(|| syn::Error::new(tt.span(), "expected a type inside `[..]`"))?;
            let len = inner
                .get(2)
                .ok_or_else(|| syn::Error::new(tt.span(), "expected `[Type; N]`"))
                .and_then(parse_u32)? as usize;
            match elem_tt {
                TokenTree::Ident(i) => {
                    let name = i.to_string();
                    if (2..=4).contains(&len)
                        && let Some(vec) = short_vec_ty(&name, len)
                    {
                        return Ok(vec);
                    }
                    Ok(IrType::RuntimeArray(Box::new(binding_leaf_ty(&name))))
                }
                TokenTree::Group(g2) if g2.delimiter() == Delimiter::Bracket => {
                    // Nested scalar array: [[f32; 3]; 64] → array<vec3<f32>>.
                    let inner_ty = parse_binding_base_ty(&TokenTree::Group(g2.clone()))?;
                    Ok(IrType::RuntimeArray(Box::new(inner_ty)))
                }
                other => Err(syn::Error::new(
                    other.span(),
                    "unsupported binding element type",
                )),
            }
        }
        TokenTree::Ident(i) => Ok(binding_leaf_ty(&i.to_string())),
        other => Err(syn::Error::new(
            other.span(),
            "expected a type: an identifier or `[Type; N]`",
        )),
    }
}

/// One scalar leaf: registry scalars/vectors/matrices, `bool`, or an
/// opaque struct mirror. Lexical classification — spellings stay in the
/// writer.
fn binding_leaf_ty(name: &str) -> IrType {
    if name == "bool" {
        return IrType::Bool;
    }
    match ShaderType::from_rust(name) {
        Some(t) => IrType::Scalar(t),
        None => IrType::Custom(name.to_string()),
    }
}

/// `[f32; 3]`-style fixed arrays that fit a WGSL vector.
fn short_vec_ty(name: &str, len: usize) -> Option<IrType> {
    let variant = match (name, len) {
        ("f32", 2) => ShaderType::Vec2,
        ("f32", 3) => ShaderType::Vec3,
        ("f32", 4) => ShaderType::Vec4,
        ("u32", 2) => ShaderType::UVec2,
        ("u32", 3) => ShaderType::UVec3,
        ("u32", 4) => ShaderType::UVec4,
        ("i32", 2) => ShaderType::IVec2,
        ("i32", 3) => ShaderType::IVec3,
        ("i32", 4) => ShaderType::IVec4,
        ("bool", 2) => ShaderType::BVec2,
        ("bool", 3) => ShaderType::BVec3,
        ("bool", 4) => ShaderType::BVec4,
        _ => return None,
    };
    Some(IrType::Scalar(variant))
}

/// Parse `storage(...)` / `uniform(...)` groups into a structural global
/// item. `binding_index` is the auto-assigned `@binding` number.
fn parse_binding_group(kind: &str, ts: TokenStream2, binding_index: usize) -> syn::Result<IrItem> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let name = match toks.first() {
        Some(TokenTree::Ident(i)) => i.to_string(),
        other => {
            return Err(syn::Error::new(
                other.map_or_else(proc_macro2::Span::call_site, |t| t.span()),
                "expected a binding name",
            ));
        }
    };
    if toks.len() < 3 || !is_punct(&toks[1], ':') {
        return Err(syn::Error::new(
            toks.first()
                .map_or_else(proc_macro2::Span::call_site, |t| t.span()),
            format!("expected `{kind}(name: Type, ...)`"),
        ));
    }
    let base = parse_binding_base_ty(&toks[2])?;
    let binding = binding_index as u32;
    match kind {
        "storage" => {
            let read_write = parse_access_rw(kind, &toks)?;
            // The declaration wraps one runtime array itself; a base that
            // already is one (long/scalar capacities) unwraps a level.
            let elem = match base {
                IrType::RuntimeArray(inner) => *inner,
                other => other,
            };
            Ok(IrItem::Global(IrGlobal::Storage {
                group: 0,
                binding,
                name,
                elem,
                read_only: !read_write,
            }))
        }
        "uniform" => {
            parse_access_rw(kind, &toks)?;
            Ok(IrItem::Global(IrGlobal::Uniform {
                group: 0,
                binding,
                name,
                ty: base,
            }))
        }
        _ => unreachable!("only storage/uniform are valid binding kinds"),
    }
}

/// Parse the optional `, read` / `, read_write` tail (storage only):
/// true for `read_write`.
fn parse_access_rw(kind: &str, toks: &[TokenTree]) -> syn::Result<bool> {
    if toks.len() <= 3 {
        return Ok(false);
    }
    if !is_punct(&toks[3], ',') {
        return Err(syn::Error::new(
            toks[3].span(),
            "expected `,` before the access mode",
        ));
    }
    let Some(acc_tt) = toks.get(4) else {
        return Ok(false);
    };
    let acc = match acc_tt {
        TokenTree::Ident(i) => i.to_string(),
        other => {
            return Err(syn::Error::new(
                other.span(),
                "expected access mode: `read` or `read_write`",
            ));
        }
    };
    if kind == "uniform" {
        return Err(syn::Error::new(
            acc_tt.span(),
            "uniform buffers do not take an access mode",
        ));
    }
    if acc != "read" && acc != "read_write" {
        return Err(syn::Error::new(
            acc_tt.span(),
            "expected access mode: `read` or `read_write`",
        ));
    }
    Ok(acc == "read_write")
}

/// Parse one `name: semantic` pair into a `(name, semantic)` entry-point
/// parameter (spelled by the writer: every invocation builtin is a
/// `vec3<u32>`).
fn parse_builtin_pair(seg: &[TokenTree]) -> syn::Result<(String, String)> {
    let name = match seg.first() {
        Some(TokenTree::Ident(i)) => i.to_string(),
        other => {
            return Err(syn::Error::new(
                other.map_or_else(proc_macro2::Span::call_site, |t| t.span()),
                "expected a builtin parameter name",
            ));
        }
    };
    let semantic = match seg.get(2) {
        Some(TokenTree::Ident(i)) => i.to_string(),
        other => {
            return Err(syn::Error::new(
                other.map_or_else(proc_macro2::Span::call_site, |t| t.span()),
                "expected a WGSL builtin semantic",
            ));
        }
    };
    Ok((name, semantic))
}

/// Parse `builtin(name: semantic, ...)` into entry-point parameters.
fn parse_builtin_group(ts: TokenStream2) -> syn::Result<Vec<(String, String)>> {
    let mut params = Vec::new();
    let mut seg: Vec<TokenTree> = Vec::new();
    for tt in ts {
        if is_punct(&tt, ',') {
            if !seg.is_empty() {
                params.push(parse_builtin_pair(&seg)?);
                seg.clear();
            }
        } else {
            seg.push(tt);
        }
    }
    if !seg.is_empty() {
        params.push(parse_builtin_pair(&seg)?);
    }
    Ok(params)
}

fn parse_config(args: TokenStream2) -> syn::Result<Option<ShaderConfig>> {
    if args.is_empty() {
        return Ok(None);
    }

    let mut config = ShaderConfig {
        workgroup_size: 64,
        items: Vec::new(),
        builtins: Vec::new(),
        vertex_entry: None,
        fragment_entry: None,
        helpers: Vec::new(),
    };
    let mut binding_index = 0usize;
    for item in split_top_level(args) {
        if item.is_empty() {
            continue; // trailing comma
        }
        apply_option(&item, &mut config, &mut binding_index)?;
    }
    Ok(Some(config))
}

/// Split the top level on commas, respecting nested groups.
fn split_top_level(args: TokenStream2) -> Vec<Vec<TokenTree>> {
    let mut items: Vec<Vec<TokenTree>> = Vec::new();
    let mut cur: Vec<TokenTree> = Vec::new();
    for tt in args {
        if is_punct(&tt, ',') {
            items.push(std::mem::take(&mut cur));
        } else {
            cur.push(tt);
        }
    }
    if !cur.is_empty() {
        items.push(cur);
    }
    items
}

/// Apply one `key(...)` / `key = value` option to the shader config.
fn apply_option(
    item: &[TokenTree],
    config: &mut ShaderConfig,
    binding_index: &mut usize,
) -> syn::Result<()> {
    let first = item[0].clone();
    let TokenTree::Ident(kw) = &first else {
        return Err(syn::Error::new(
            first.span(),
            "expected an option: `workgroup_size`, `storage`, `uniform`, `texture`, `sampler`, `vertex`, `fragment`, `builtin` or `helpers`",
        ));
    };
    match kw.to_string().as_str() {
        "workgroup_size" => parse_workgroup_size(item, config),
        "storage" | "uniform" => {
            let global = parse_binding_option(kw.to_string(), item, *binding_index)?;
            *binding_index += 1;
            config.items.push(global);
            Ok(())
        }
        "texture" | "sampler" => {
            let global = parse_texture_sampler_option(kw.to_string(), item, *binding_index)?;
            *binding_index += 1;
            config.items.push(global);
            Ok(())
        }
        "vertex" => {
            let entry = parse_stage_entry("vertex", item)?;
            config.vertex_entry = Some(entry);
            Ok(())
        }
        "fragment" => {
            let entry = parse_stage_entry("fragment", item)?;
            config.fragment_entry = Some(entry);
            Ok(())
        }
        "builtin" => {
            let params = builtin_params(item)?;
            config.builtins.extend(params);
            Ok(())
        }
        "helpers" => {
            let paths = helper_paths(item)?;
            config.helpers.extend(paths);
            Ok(())
        }
        other => Err(syn::Error::new(
            first.span(),
            format!(
                "unknown gpu_pipeline option `{other}`; expected `workgroup_size`, `storage`, `uniform`, `texture`, `sampler`, `vertex`, `fragment`, `builtin` or `helpers`"
            ),
        )),
    }
}

/// `workgroup_size = <integer>`
fn parse_workgroup_size(item: &[TokenTree], config: &mut ShaderConfig) -> syn::Result<()> {
    if item.len() < 3 || !is_punct(&item[1], '=') {
        return Err(syn::Error::new(
            item[0].span(),
            "expected `workgroup_size = <integer>`",
        ));
    }
    config.workgroup_size = parse_u32(&item[2])?;
    Ok(())
}

/// The parenthesized group of a `storage(..)` / `uniform(..)` /
/// `builtin(..)` option.
fn option_group(usage: &str, item: &[TokenTree]) -> syn::Result<TokenStream2> {
    let group_tt = item
        .get(1)
        .ok_or_else(|| syn::Error::new(item[0].span(), format!("expected `{usage}`")))?;
    let TokenTree::Group(group) = group_tt else {
        return Err(syn::Error::new(
            group_tt.span(),
            format!("expected `{usage}`"),
        ));
    };
    Ok(group.stream())
}

/// `storage(name: Type[, access])` / `uniform(name: Type)`
fn parse_binding_option(
    kw: String,
    item: &[TokenTree],
    binding_index: usize,
) -> syn::Result<IrItem> {
    let usage = format!("{}(name: Type, ...)", kw);
    let group = option_group(&usage, item)?;
    parse_binding_group(&kw, group, binding_index)
}

/// `builtin(name: semantic, ...)`
fn builtin_params(item: &[TokenTree]) -> syn::Result<Vec<(String, String)>> {
    let group = option_group("builtin(name: semantic, ...)", item)?;
    parse_builtin_group(group)
}

/// `helpers(mod_a, mod_b, ...)` — modules exposing `wgsl_source()`
/// (emitted by `#[wgsl_fn]` / `#[kernel]`), stitched ahead of the entry
/// body in order. Paths resolve at the annotated item's site.
fn helper_paths(item: &[TokenTree]) -> syn::Result<Vec<syn::Path>> {
    let group = option_group("helpers(mod_a, mod_b, ...)", item)?;
    let mut paths = Vec::new();
    for seg in split_top_level(group) {
        if seg.is_empty() {
            continue; // trailing comma
        }
        let path: syn::Path = syn::parse2(TokenStream2::from_iter(seg)).map_err(|_| {
            syn::Error::new(
                item[0].span(),
                "helpers(...) expects module paths: `helpers(ldl_solve, mat_ops)`",
            )
        })?;
        if path.segments.is_empty() {
            return Err(syn::Error::new(
                item[0].span(),
                "helpers(...) expects module paths: `helpers(ldl_solve, mat_ops)`",
            ));
        }
        paths.push(path);
    }
    if paths.is_empty() {
        return Err(syn::Error::new(
            item[0].span(),
            "helpers(...) needs at least one module path",
        ));
    }
    Ok(paths)
}

/// `texture(name: texture_2d<f32>)` / `sampler(name: sampler)`
fn parse_texture_sampler_option(
    kw: String,
    item: &[TokenTree],
    binding_index: usize,
) -> syn::Result<IrItem> {
    let usage = format!("{}(name: Type)", kw);
    let group = option_group(&usage, item)?;
    parse_texture_sampler_group(&kw, group, binding_index)
}

/// `vertex(entry)` / `fragment(entry)` — entry point names for render pipelines.
fn parse_stage_entry(kind: &str, item: &[TokenTree]) -> syn::Result<String> {
    let usage = format!("{kind}(entry)");
    let group = option_group(&usage, item)?;
    let toks: Vec<TokenTree> = group.into_iter().collect();
    let ident = toks
        .first()
        .ok_or_else(|| syn::Error::new(item[0].span(), format!("expected `{usage}`")))?;
    match ident {
        TokenTree::Ident(i) => Ok(i.to_string()),
        other => Err(syn::Error::new(other.span(), format!("expected `{usage}`"))),
    }
}

fn parse_texture_sampler_group(
    kind: &str,
    ts: TokenStream2,
    binding_index: usize,
) -> syn::Result<IrItem> {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let name = match toks.first() {
        Some(TokenTree::Ident(i)) => i.to_string(),
        other => {
            return Err(syn::Error::new(
                other.map_or_else(proc_macro2::Span::call_site, |t| t.span()),
                "expected a binding name",
            ));
        }
    };
    if toks.len() < 3 || !is_punct(&toks[1], ':') {
        return Err(syn::Error::new(
            toks.first()
                .map_or_else(proc_macro2::Span::call_site, |t| t.span()),
            format!("expected `{kind}(name: Type)`"),
        ));
    }
    // Structural type: tokens up to the next top-level comma parse as one
    // `syn::Type`, classified into the closed texture world below. No
    // space-normalization paste — unknown shapes are loud here, not at naga.
    let type_toks: Vec<TokenTree> = toks[2..]
        .iter()
        .take_while(|tt| !is_punct(tt, ','))
        .cloned()
        .collect();
    let ty: syn::Type = syn::parse2(TokenStream2::from_iter(type_toks)).map_err(|_| {
        syn::Error::new(
            toks[2].span(),
            "expected a texture/sampler type like `texture_2d<f32>` or `sampler`",
        )
    })?;
    let binding = binding_index as u32;
    let global = match texture_shape(&ty)? {
        TextureShape::Texture(kind) => IrGlobal::Texture {
            group: 0,
            binding,
            name,
            kind,
        },
        TextureShape::Sampler(comparison) => IrGlobal::Sampler {
            group: 0,
            binding,
            name,
            comparison,
        },
    };
    Ok(IrItem::Global(global))
}

/// Classified texture/sampler type (the `kind` prefix is already known
/// to be `texture`/`sampler` — this pins the spelling set).
enum TextureShape {
    Texture(IrTexture),
    Sampler(bool),
}

fn texture_shape(ty: &syn::Type) -> syn::Result<TextureShape> {
    use syn::{GenericArgument, PathArguments};
    let unsupported = || {
        syn::Error::new(
            ty.span(),
            "unsupported texture/sampler type: `texture_2d<f32|u32|i32>`, \
             `texture_depth_2d[_array]`, `texture_depth_cube_array`, \
             `sampler`, `sampler_comparison`",
        )
    };
    let syn::Type::Path(tp) = ty else {
        return Err(unsupported());
    };
    let Some(seg) = tp.path.segments.last() else {
        return Err(unsupported());
    };
    let scalar_arg = |seg: &syn::PathSegment| -> Option<String> {
        let PathArguments::AngleBracketed(args) = &seg.arguments else {
            return None;
        };
        let mut iter = args.args.iter();
        let arg = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        let GenericArgument::Type(syn::Type::Path(p)) = arg else {
            return None;
        };
        let name = p.path.segments.last()?.ident.to_string();
        Some(name)
    };
    match seg.ident.to_string().as_str() {
        "texture_2d" => {
            let kind = match scalar_arg(seg).as_deref() {
                Some("f32") => IrTexture::Tex2dF32,
                Some("u32") => IrTexture::Tex2dU32,
                Some("i32") => IrTexture::Tex2dI32,
                _ => return Err(unsupported()),
            };
            Ok(TextureShape::Texture(kind))
        }
        "texture_depth_2d" if seg.arguments.is_none() => {
            Ok(TextureShape::Texture(IrTexture::Depth2d))
        }
        "texture_depth_2d_array" if seg.arguments.is_none() => {
            Ok(TextureShape::Texture(IrTexture::Depth2dArray))
        }
        "texture_depth_cube_array" if seg.arguments.is_none() => {
            Ok(TextureShape::Texture(IrTexture::DepthCubeArray))
        }
        "sampler" if seg.arguments.is_none() => Ok(TextureShape::Sampler(false)),
        "sampler_comparison" if seg.arguments.is_none() => Ok(TextureShape::Sampler(true)),
        _ => Err(unsupported()),
    }
}

/// Module framing: entry signatures stay string assembly (like every
/// `wgsl.rs` entry), but every line fed into them — items, bodies —
/// arrives IR-printed. Pure for unit-level naga validation.
fn assemble_render_module(
    items_wgsl: &str,
    v_entry: &str,
    f_entry: &str,
    vertex_params: &str,
    vs_body: &str,
    fs_body: &str,
) -> String {
    format!(
        "{items_wgsl}\n@vertex\nfn {v_entry}({vertex_params}) -> VertexOutput {{\n{vs_body}\n}}\n@fragment\nfn {f_entry}(in_: VertexOutput) -> @location(0) vec4<f32> {{\n{fs_body}\n}}\n"
    )
}

/// Compute entry framing (same boundary as above).
fn assemble_compute_module(
    items_wgsl: &str,
    workgroup_size: u32,
    builtins_wgsl: &str,
    body_wgsl: &str,
) -> String {
    format!(
        "{items_wgsl}\n@compute @workgroup_size({workgroup_size})\nfn main({builtins_wgsl}) {{\n{body_wgsl}\n}}\n"
    )
}

/// Full-screen quad boilerplate as IR items (shared by bloom/composite
/// passes): QUAD/UVS constants plus the `VertexOutput` mirror. Built
/// from the same nodes as user code — the writer spells them, nothing
/// is pasted as text.
fn quad_items() -> Vec<IrItem> {
    use ornis_shader_lang::ir::{IrCallee, IrExpr, IrUnOp};
    fn lit(x: &str) -> IrExpr {
        IrExpr::Float(x.to_string())
    }
    fn neg(x: &str) -> IrExpr {
        IrExpr::Unary {
            op: IrUnOp::Neg,
            expr: Box::new(lit(x)),
        }
    }
    fn vec4(x: IrExpr, y: IrExpr, z: IrExpr, w: IrExpr) -> IrExpr {
        IrExpr::Call {
            target: IrCallee::Constructor {
                ty: ShaderType::Vec4,
            },
            args: vec![x, y, z, w],
        }
    }
    fn vec2(x: IrExpr, y: IrExpr) -> IrExpr {
        IrExpr::Call {
            target: IrCallee::Constructor {
                ty: ShaderType::Vec2,
            },
            args: vec![x, y],
        }
    }
    let quad_ty = IrType::Array {
        elem: Box::new(IrType::Scalar(ShaderType::Vec4)),
        len: 4,
    };
    let uvs_ty = IrType::Array {
        elem: Box::new(IrType::Scalar(ShaderType::Vec2)),
        len: 4,
    };
    vec![
        IrItem::Global(IrGlobal::Private {
            name: "QUAD".to_string(),
            ty: quad_ty,
            init: IrExpr::Array {
                elem: IrType::Scalar(ShaderType::Vec4),
                items: vec![
                    vec4(neg("1.0"), neg("1.0"), lit("0.0"), lit("1.0")),
                    vec4(lit("1.0"), neg("1.0"), lit("0.0"), lit("1.0")),
                    vec4(neg("1.0"), lit("1.0"), lit("0.0"), lit("1.0")),
                    vec4(lit("1.0"), lit("1.0"), lit("0.0"), lit("1.0")),
                ],
            },
        }),
        IrItem::Global(IrGlobal::Private {
            name: "UVS".to_string(),
            ty: uvs_ty,
            init: IrExpr::Array {
                elem: IrType::Scalar(ShaderType::Vec2),
                items: vec![
                    vec2(lit("0.0"), lit("1.0")),
                    vec2(lit("1.0"), lit("1.0")),
                    vec2(lit("0.0"), lit("0.0")),
                    vec2(lit("1.0"), lit("0.0")),
                ],
            },
        }),
        IrItem::Struct {
            name: "VertexOutput".to_string(),
            fields: vec![
                IrStructField {
                    attr: Some(IrFieldAttr::Builtin("position".to_string())),
                    name: "position".to_string(),
                    ty: IrType::Scalar(ShaderType::Vec4),
                },
                IrStructField {
                    attr: Some(IrFieldAttr::Location(0)),
                    name: "uv".to_string(),
                    ty: IrType::Scalar(ShaderType::Vec2),
                },
            ],
        },
    ]
}

pub fn gpu_pipeline(args: TokenStream, input: TokenStream) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);
    let fn_name = &func.sig.ident;

    let config = match parse_config(args.into()) {
        Ok(c) => c,
        Err(e) => return e.to_compile_error().into(),
    };

    let Some(config) = config else {
        return legacy_gpu_pipeline(&func);
    };

    // ── Full-shader mode: the function body is the compute entry body ──────
    if !func.sig.inputs.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "gpu_pipeline with bindings: the function must take no parameters \
             (built-ins are declared via `builtin(...)`)",
        )
        .to_compile_error()
        .into();
    }
    if let syn::ReturnType::Type(_, ty) = &func.sig.output {
        return syn::Error::new(
            ty.span(),
            "gpu_pipeline with bindings: the function must not return a value",
        )
        .to_compile_error()
        .into();
    }

    let body_wgsl = crate::wgsl::wgsl_main_body(&func);
    let mut items = config.items;
    let builtins_wgsl = config
        .builtins
        .iter()
        .map(|(name, semantic)| print_builtin_param(name, semantic))
        .collect::<Vec<_>>()
        .join(", ");
    // Render-pipeline mode: vertex/fragment entry points replace the compute entry.
    // Generates @vertex/@fragment shims with full-screen quad boilerplate
    // (QUAD/UVS + VertexOutput) so `texture`/`sampler`/`uniform`/`vertex`/`fragment`
    // are already validated and produce a usable bloom/PBR post-process.
    let wgsl_source = if config.vertex_entry.is_some() || config.fragment_entry.is_some() {
        let v_entry = config
            .vertex_entry
            .clone()
            .unwrap_or_else(|| "vs".to_string());
        let f_entry = config
            .fragment_entry
            .clone()
            .unwrap_or_else(|| "fs".to_string());
        let vertex_params = if builtins_wgsl.is_empty() {
            print_builtin_param("idx", "vertex_index")
        } else {
            builtins_wgsl.clone()
        };
        // Full-screen quad constants shared by bloom/composite passes,
        // built from the same IR nodes as user code — no raw WGSL text.
        items.extend(quad_items());
        let items_wgsl = items.iter().map(print_item).collect::<Vec<_>>().join("\n");
        let vs_body = crate::wgsl::wgsl_quad_vs_body();
        assemble_render_module(
            &items_wgsl,
            &v_entry,
            &f_entry,
            &vertex_params,
            &vs_body,
            &body_wgsl,
        )
    } else {
        let items_wgsl = items.iter().map(print_item).collect::<Vec<_>>().join("\n");
        assemble_compute_module(
            &items_wgsl,
            config.workgroup_size,
            &builtins_wgsl,
            &body_wgsl,
        )
    };
    let wgsl_lit = proc_macro2::Literal::string(&wgsl_source);

    // Helper inclusion (the compute-path analog of `#[wgsl_fn]` reuse on
    // the stage path): `helpers(a, b)` stitches `a::wgsl_source()` ahead
    // of the entry body at *runtime* — the macro cannot evaluate another
    // item's expansion, so it emits the concatenation instead. Declaration
    // before use, always valid WGSL. Without helpers the source stays a
    // `&'static str` literal (existing callers unaffected); with helpers
    // it becomes an owned `String`.
    let helper_sources: Vec<TokenStream2> = config
        .helpers
        .iter()
        .map(|path| quote!(#path::wgsl_source()))
        .collect();
    let wgsl_source_fn = if helper_sources.is_empty() {
        quote! {
            #[allow(dead_code)]
            pub fn wgsl_source() -> &'static str {
                #wgsl_lit
            }
        }
    } else {
        quote! {
            #[allow(dead_code)]
            pub fn wgsl_source() -> String {
                [#(#helper_sources),*].join("\n") + "\n" + #wgsl_lit
            }
        }
    };
    // `Cow::Borrowed` needs a `&str` place; the owned stitch has none,
    // so the two spellings diverge here (and only here).
    let create_module_fn = if helper_sources.is_empty() {
        quote! {
            #[allow(dead_code)]
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(pipeline_label()),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_source())),
                })
            }
        }
    } else {
        quote! {
            #[allow(dead_code)]
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(pipeline_label()),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl_source())),
                })
            }
        }
    };

    let expanded = quote! {
        pub mod #fn_name {
            #[allow(dead_code)]
            pub fn pipeline_label() -> &'static str {
                stringify!(#fn_name)
            }

            #wgsl_source_fn

            #create_module_fn
        }
    };

    TokenStream::from(expanded)
}

/// Legacy mode: parameters → storage arrays, tail expression → output.
fn legacy_gpu_pipeline(func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let wgsl = crate::wgsl::wgsl_source_from_fn(func);

    let expanded = quote! {
        pub mod #fn_name {
            #[allow(dead_code)]
            pub fn pipeline_label() -> &'static str {
                stringify!(#fn_name)
            }

            #[allow(dead_code)]
            pub fn wgsl_source() -> &'static str {
                #wgsl
            }

            #[allow(dead_code)]
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(pipeline_label()),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_source())),
                })
            }

            #[allow(dead_code)]
            pub fn create_pipeline(
                device: &wgpu::Device,
            ) -> wgpu::ComputePipeline {
                let shader = create_shader_module(device);
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(pipeline_label()),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(pipeline_label()),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &str) -> ShaderConfig {
        let ts: TokenStream2 = args.parse().expect("test args must parse");
        parse_config(ts)
            .expect("test args must not error")
            .expect("non-empty args must yield a config")
    }

    #[test]
    fn helpers_option_parses_paths_in_order() {
        let cfg = parse("workgroup_size = 4, helpers(ldl_solve, crate::mat_ops)");
        let names: Vec<String> = cfg
            .helpers
            .iter()
            .map(|p| {
                p.segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::")
            })
            .collect();
        assert_eq!(names, ["ldl_solve", "crate::mat_ops"]);
    }

    #[test]
    fn helpers_empty_is_rejected() {
        let ts: TokenStream2 = "helpers()".parse().unwrap();
        assert!(parse_config(ts).is_err());
    }

    #[test]
    fn helpers_without_parens_is_rejected() {
        let ts: TokenStream2 = "helpers".parse().unwrap();
        assert!(parse_config(ts).is_err());
    }

    #[test]
    fn unknown_option_still_rejected() {
        let ts: TokenStream2 = "workgroup_size = 4, frobnicate(x)".parse().unwrap();
        match parse_config(ts) {
            Ok(_) => panic!("unknown option must be rejected"),
            Err(err) => assert!(err.to_string().contains("helpers")),
        }
    }

    fn printed_items(args: &str) -> Vec<String> {
        parse(args).items.iter().map(print_item).collect()
    }

    #[test]
    fn storage_uniform_builtin_print_identically() {
        let decls = printed_items(
            "storage(body_buf: [f32; 64], read_write), uniform(params: [u32; 4]), builtin(gid: workgroup_id)",
        );
        assert_eq!(
            decls,
            [
                "@group(0) @binding(0) var<storage, read_write> body_buf: array<f32>;",
                "@group(0) @binding(1) var<uniform> params: vec4<u32>;",
            ]
        );
        let cfg = parse("builtin(gid: workgroup_id, lid: local_invocation_id)");
        let params: Vec<String> = cfg
            .builtins
            .iter()
            .map(|(name, semantic)| print_builtin_param(name, semantic))
            .collect();
        assert_eq!(
            params,
            [
                "@builtin(workgroup_id) gid: vec3<u32>",
                "@builtin(local_invocation_id) lid: vec3<u32>",
            ]
        );
    }

    #[test]
    fn texture_sampler_print_identically() {
        let decls = printed_items(
            "texture(albedo: texture_2d<f32>), sampler(smp: sampler), texture(depth: texture_depth_2d_array)",
        );
        assert_eq!(
            decls,
            [
                "@group(0) @binding(0) var albedo: texture_2d<f32>;",
                "@group(0) @binding(1) var smp: sampler;",
                "@group(0) @binding(2) var depth: texture_depth_2d_array;",
            ]
        );
    }

    #[test]
    fn unknown_texture_shape_is_loud() {
        let ts: TokenStream2 = "texture(x: texture_3d<f32>)".parse().unwrap();
        assert!(parse_config(ts).is_err());
    }

    #[test]
    fn render_quad_assembly_validates_with_naga() {
        // The render-mode path has no in-repo callers, so this pins it
        // directly: IR-built QUAD/UVS/VertexOutput + IR-built vs body +
        // a trivial fs body must form a valid module.
        let items_wgsl = quad_items()
            .iter()
            .map(print_item)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(items_wgsl.contains("var<private> QUAD: array<vec4<f32>, 4>"));
        assert!(items_wgsl.contains("struct VertexOutput {"));
        let source = assemble_render_module(
            &items_wgsl,
            "vs",
            "fs",
            &print_builtin_param("idx", "vertex_index"),
            &crate::wgsl::wgsl_quad_vs_body(),
            "return vec4<f32>(0.0, 0.0, 0.0, 1.0);",
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("render assembly must parse: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("render assembly must validate: {e:?}"));
    }
}
