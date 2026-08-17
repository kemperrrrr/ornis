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
//! )]
//! fn solver() {
//!     // kernel DSL body; builtins and bindings are in scope as identifiers
//!     if gid.x >= params.x { return; }
//!     body_buf[gid.x] = body_buf[gid.x] + batch_buf[gid.x].acc[lid.x];
//! }
//! ```
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

use proc_macro::TokenStream;
use proc_macro2::{Delimiter, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{ItemFn, LitInt, parse_macro_input, spanned::Spanned};

#[derive(Debug)]
struct ShaderConfig {
    workgroup_size: u32,
    bindings: Vec<String>,
    builtins: Vec<String>,
}

fn is_punct(tt: &TokenTree, ch: char) -> bool {
    matches!(tt, TokenTree::Punct(p) if p.as_char() == ch)
}

fn parse_u32(tt: &TokenTree) -> syn::Result<u32> {
    let lit = match tt {
        TokenTree::Literal(l) => l,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "expected an integer literal",
            ));
        }
    };
    let int: LitInt = syn::parse2(TokenStream2::from(lit.clone()))?;
    int.base10_parse()
}

/// Map a Rust type token (ident, `[T; N]` or nested scalar array) to the
/// WGSL type text and the element type used for storage arrays.
fn parse_binding_type(tt: &TokenTree) -> syn::Result<(String, String)> {
    match tt {
        TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket => {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            let elem_tt = inner.first().ok_or_else(|| {
                syn::Error::new(tt.span(), "expected a type inside `[..]`")
            })?;
            let len = inner
                .get(2)
                .ok_or_else(|| syn::Error::new(tt.span(), "expected `[Type; N]`"))
                .and_then(parse_u32)?;
            let (wgsl_ty, elem) = match elem_tt {
                TokenTree::Ident(i) => {
                    let name = i.to_string();
                    match name.as_str() {
                        // Scalar arrays of length 2..=4 map to WGSL vectors;
                        // longer ones are runtime-sized arrays (the length is
                        // documentation of the buffer capacity).
                        "f32" | "u32" | "i32" | "bool" => {
                            if (2..=4).contains(&len) {
                                (format!("vec{len}<{name}>"), format!("vec{len}<{name}>"))
                            } else {
                                (format!("array<{name}>"), name)
                            }
                        }
                        _ => (format!("array<{name}>"), name),
                    }
                }
                TokenTree::Group(g2) if g2.delimiter() == Delimiter::Bracket => {
                    // Nested scalar array: [[f32; 3]; 64] → array<vec3<f32>>.
                    let (inner_ty, inner_elem) =
                        parse_binding_type(&TokenTree::Group(g2.clone()))?;
                    (format!("array<{inner_ty}>"), inner_elem)
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "unsupported binding element type",
                    ));
                }
            };
            Ok((wgsl_ty, elem))
        }
        TokenTree::Ident(i) => {
            let name = i.to_string();
            Ok((name.clone(), name))
        }
        other => Err(syn::Error::new(
            other.span(),
            "expected a type: an identifier or `[Type; N]`",
        )),
    }
}

/// Parse `storage(...)` / `uniform(...)` groups into a WGSL global
/// declaration line. `binding_index` is the auto-assigned `@binding` number.
fn parse_binding_group(
    kind: &str,
    ts: TokenStream2,
    binding_index: usize,
) -> syn::Result<String> {
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
            toks.first().map_or_else(proc_macro2::Span::call_site, |t| t.span()),
            format!("expected `{kind}(name: Type, ...)`"),
        ));
    }
    let (wgsl_ty, elem) = parse_binding_type(&toks[2])?;

    // Optional `, access` tail (storage only).
    let mut access = String::new();
    if toks.len() > 3 {
        if !is_punct(&toks[3], ',') {
            return Err(syn::Error::new(
                toks[3].span(),
                "expected `,` before the access mode",
            ));
        }
        if let Some(acc_tt) = toks.get(4) {
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
            if acc == "read_write" {
                access = ", read_write".to_string();
            }
        }
    }

    let decl = match kind {
        "storage" => format!(
            "@group(0) @binding({binding_index}) var<storage{access}> {name}: array<{elem}>;"
        ),
        "uniform" => format!("@group(0) @binding({binding_index}) var<uniform> {name}: {wgsl_ty};"),
        _ => unreachable!("only storage/uniform are valid binding kinds"),
    };
    Ok(decl)
}

/// Parse one `name: semantic` pair into a `main` parameter declaration.
fn parse_builtin_pair(seg: &[TokenTree]) -> syn::Result<String> {
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
    Ok(format!("@builtin({semantic}) {name}: vec3<u32>"))
}

/// Parse `builtin(name: semantic, ...)` into `main` parameter declarations.
fn parse_builtin_group(ts: TokenStream2) -> syn::Result<Vec<String>> {
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
        bindings: Vec::new(),
        builtins: Vec::new(),
    };

    // Split the top level on commas, respecting nested groups.
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

    let mut binding_index = 0usize;
    for item in items {
        if item.is_empty() {
            continue; // trailing comma
        }
        let first = item[0].clone();
        let TokenTree::Ident(kw) = &first else {
            return Err(syn::Error::new(
                first.span(),
                "expected an option: `workgroup_size`, `storage`, `uniform` or `builtin`",
            ));
        };
        match kw.to_string().as_str() {
            "workgroup_size" => {
                if item.len() < 3 || !is_punct(&item[1], '=') {
                    return Err(syn::Error::new(
                        first.span(),
                        "expected `workgroup_size = <integer>`",
                    ));
                }
                config.workgroup_size = parse_u32(&item[2])?;
            }
            "storage" | "uniform" => {
                let group_tt = item.get(1).ok_or_else(|| {
                    syn::Error::new(first.span(), format!("expected `{kw}(name: Type, ...)`"))
                })?;
                let TokenTree::Group(group) = group_tt else {
                    return Err(syn::Error::new(
                        group_tt.span(),
                        format!("expected `{kw}(name: Type, ...)`"),
                    ));
                };
                let decl = parse_binding_group(&kw.to_string(), group.stream(), binding_index)?;
                binding_index += 1;
                config.bindings.push(decl);
            }
            "builtin" => {
                let group_tt = item.get(1).ok_or_else(|| {
                    syn::Error::new(
                        first.span(),
                        "expected `builtin(name: semantic, ...)`",
                    )
                })?;
                let TokenTree::Group(group) = group_tt else {
                    return Err(syn::Error::new(
                        group_tt.span(),
                        "expected `builtin(name: semantic, ...)`",
                    ));
                };
                config.builtins.extend(parse_builtin_group(group.stream())?);
            }
            other => {
                return Err(syn::Error::new(
                    first.span(),
                    format!(
                        "unknown gpu_pipeline option `{other}`; expected `workgroup_size`, `storage`, `uniform` or `builtin`"
                    ),
                ));
            }
        }
    }

    Ok(Some(config))
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
    let bindings_wgsl = config.bindings.join("\n");
    let builtins_wgsl = config.builtins.join(", ");
    let wgsl_source = format!(
        "{bindings_wgsl}\n@compute @workgroup_size({})\nfn main({builtins_wgsl}) {{\n{body_wgsl}\n}}\n",
        config.workgroup_size
    );
    let wgsl_lit = proc_macro2::Literal::string(&wgsl_source);

    let expanded = quote! {
        pub mod #fn_name {
            #[allow(dead_code)]
            pub fn pipeline_label() -> &'static str {
                stringify!(#fn_name)
            }

            #[allow(dead_code)]
            pub fn wgsl_source() -> &'static str {
                #wgsl_lit
            }

            #[allow(dead_code)]
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(pipeline_label()),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_source())),
                })
            }
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
