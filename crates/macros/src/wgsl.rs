use syn::{FnArg, ItemFn, Pat, Stmt};

#[allow(dead_code)]
struct WgslGen;

impl WgslGen {
    fn expr(e: &syn::Expr) -> String {
        use syn::Expr::*;
        match e {
            Lit(l) => Self::lit(l),
            Path(p) => Self::path(p),
            Field(f) => {
                let member = match &f.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                };
                format!("{}.{}", Self::expr(&f.base), member)
            }
            Binary(b) => Self::binary(b),
            Unary(u) => Self::unary(u),
            Paren(p) => format!("({})", Self::expr(&p.expr)),
            Call(c) => Self::call(c),
            MethodCall(m) => Self::method_call(m),
            If(i) => Self::if_expr(i),
            Block(b) => Self::block_expr(b),
            Assign(a) => Self::assign(a),
            Return(r) => Self::ret(r),
            _ => syn::Error::new_spanned(e, "expression not supported in WGSL kernel")
                .to_compile_error()
                .to_string(),
        }
    }

    fn lit(l: &syn::ExprLit) -> String {
        use syn::Lit::*;
        match &l.lit {
            Int(i) => i.to_string(),
            Float(f) => f.to_string(),
            Bool(b) => {
                if b.value {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            other => syn::Error::new_spanned(other, "literal type not supported in WGSL kernel")
                .to_compile_error()
                .to_string(),
        }
    }

    fn path(p: &syn::ExprPath) -> String {
        let segs: Vec<_> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
        let last = segs.last().map(|s| s.as_str());
        match last {
            Some("Vec2") | Some("vec2") => return "vec2<f32>".into(),
            Some("Vec3") | Some("vec3") => return "vec3<f32>".into(),
            Some("Vec4") | Some("vec4") => return "vec4<f32>".into(),
            Some("Mat4") | Some("mat4") => return "mat4x4<f32>".into(),
            Some("PI") => return "PI".into(),
            _ => {}
        }
        segs.join("::")
    }

    fn path_ident(p: &syn::ExprPath) -> Option<String> {
        let ident = p.path.get_ident().map(|i| i.to_string());
        if ident.is_some() {
            return ident;
        }
        p.path.segments.last().map(|s| s.ident.to_string())
    }

    fn assign(a: &syn::ExprAssign) -> String {
        format!("{} = {}", Self::expr(&a.left), Self::expr(&a.right))
    }

    fn ret(r: &syn::ExprReturn) -> String {
        match &r.expr {
            Some(expr) => format!("return {}", Self::expr(expr)),
            None => "return".to_string(),
        }
    }

    fn binary(b: &syn::ExprBinary) -> String {
        use syn::BinOp::*;
        let op = match &b.op {
            Add(_) => "+",
            Sub(_) => "-",
            Mul(_) => "*",
            Div(_) => "/",
            Rem(_) => "%",
            AddAssign(_) => "+=",
            SubAssign(_) => "-=",
            MulAssign(_) => "*=",
            DivAssign(_) => "/=",
            And(_) => "&",
            Or(_) => "|",
            Eq(_) => "==",
            Ne(_) => "!=",
            Lt(_) => "<",
            Le(_) => "<=",
            Gt(_) => ">",
            Ge(_) => ">=",
            Shl(_) => "<<",
            Shr(_) => ">>",
            BitXor(_) => "^",
            BitAnd(_) => "&",
            BitOr(_) => "|",
            _ => {
                return syn::Error::new_spanned(b.op, "operator not supported in WGSL kernel")
                    .to_compile_error()
                    .to_string()
            }
        };
        format!("{} {} {}", Self::expr(&b.left), op, Self::expr(&b.right))
    }

    fn unary(u: &syn::ExprUnary) -> String {
        use syn::UnOp::*;
        let op = match &u.op {
            Neg(_) => "-",
            Not(_) => "!",
            Deref(_) => "*",
            _ => "*",
        };
        format!("{}{}", op, Self::expr(&u.expr))
    }

    fn map_fn(fn_name: &str, args: &[String]) -> Option<String> {
        match fn_name {
            "new" => Some(args.join(", ")),
            "dot" => Some(format!("dot({})", args.join(", "))),
            "cross" => Some(format!("cross({})", args.join(", "))),
            "normalize" => Some(format!("normalize({})", args.join(", "))),
            "length" => Some(format!("length({})", args.join(", "))),
            "clamp" => Some(format!("clamp({})", args.join(", "))),
            "saturate" => Some(format!("saturate({})", args.join(", "))),
            "select" => Some(format!("select({})", args.join(", "))),
            "sqrt" | "abs" | "floor" | "ceil" | "round" | "fract" | "exp"
            | "log" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "atan2" | "pow" | "max" | "min" | "step" | "smoothstep"
            | "mix" | "reflect" | "refract" | "transpose" | "determinant"
            | "inverse" | "sign" | "dpdx" | "dpdy" | "fwidth" => {
                Some(format!("{}({})", fn_name, args.join(", ")))
            }
            _ => None,
        }
    }

    fn call(c: &syn::ExprCall) -> String {
        let args: Vec<String> = c.args.iter().map(Self::expr).collect();

        if let syn::Expr::Path(p) = c.func.as_ref() {
            let ident = Self::path_ident(p);
            let ident_str = ident.as_deref().unwrap_or("");
            // Handle constructors: glam::Vec3::new(a,b,c) -> vec3<f32>(a,b,c)
            if ident_str == "new" {
                let segs: Vec<_> = p.path.segments.iter().collect();
                if let Some(type_seg) = segs.iter().rev().nth(1) {
                    let type_name = type_seg.ident.to_string();
                    if let Some(wgsl_ty) = Self::wgsl_type(&type_name) {
                        return format!("{}({})", wgsl_ty, args.join(", "));
                    }
                }
            }
            if let Some(ident) = ident {
                if let Some(mapped) = Self::map_fn(&ident, &args) {
                    return mapped;
                }
                // Known WGSL built-ins that Rust spells differently
                match ident.as_str() {
                    "signum" => return format!("sign({})", args.join(", ")),
                    "stepf" => return format!("step({})", args.join(", ")),
                    "lerp" => return format!("mix({})", args.join(", ")),
                    _ => {}
                }
            }
        }

        let callee = Self::expr(&c.func);
        format!("{}({})", callee, args.join(", "))
    }

    fn wgsl_type(ty: &str) -> Option<String> {
        match ty {
            "Vec2" | "vec2" => Some("vec2<f32>".into()),
            "Vec3" | "vec3" => Some("vec3<f32>".into()),
            "Vec4" | "vec4" => Some("vec4<f32>".into()),
            "Mat4" | "mat4" => Some("mat4x4<f32>".into()),
            "BVec2" => Some("vec2<bool>".into()),
            "BVec3" => Some("vec3<bool>".into()),
            "BVec4" => Some("vec4<bool>".into()),
            "UVec2" => Some("vec2<u32>".into()),
            "UVec3" => Some("vec3<u32>".into()),
            "UVec4" => Some("vec4<u32>".into()),
            "IVec2" => Some("vec2<i32>".into()),
            "IVec3" => Some("vec3<i32>".into()),
            "IVec4" => Some("vec4<i32>".into()),
            _ => None,
        }
    }

    fn method_call(m: &syn::ExprMethodCall) -> String {
        let receiver = Self::expr(&m.receiver);
        let method = m.method.to_string();
        let args: Vec<String> = std::iter::once(receiver.clone())
            .chain(m.args.iter().map(Self::expr))
            .collect();

        // Handle powi specially
        if method == "powi" {
            if let Some(arg) = m.args.first() {
                if let syn::Expr::Lit(lit) = arg {
                    if let syn::Lit::Int(int_lit) = &lit.lit {
                        let p = int_lit.base10_parse::<i32>().unwrap_or(0);
                        if p == 2 {
                            return format!("{} * {}", receiver, receiver);
                        } else if p == 3 {
                            return format!("{} * {} * {}", receiver, receiver, receiver);
                        } else if p == 4 {
                            return format!("{} * {} * {} * {}", receiver, receiver, receiver, receiver);
                        } else if p > 0 {
                            // Generate repeated multiplication for small positive ints
                            let parts: Vec<String> = std::iter::repeat(receiver.clone()).take(p as usize).collect();
                            return parts.join(" * ");
                        }
                    }
                }
            }
            return format!("pow({}, {})", receiver, args[1..].join(", "));
        }

        if let Some(mapped) = Self::map_fn(&method, &args) {
            return mapped;
        }

        // Map Rust methods to WGSL functions
        match method.as_str() {
            "signum" => return format!("sign({})", args[0]),
            "abs" | "sqrt" | "sin" | "cos" | "tan" | "floor" | "ceil"
            | "round" | "fract" | "exp" | "log" | "normalize"
            | "length" | "saturate" | "asin" | "acos" | "atan" => {
                return format!("{}({})", method, args[0]);
            }
            "dot" | "cross" | "pow" | "max" | "min" | "step" | "smoothstep"
            | "reflect" | "refract" | "atan2" => {
                return format!("{}({})", method, args.join(", "));
            }
            "clamp" => {
                return format!("clamp({})", args.join(", "));
            }
            _ => {}
        }

        format!("{}.{}({})", args[0], method, args[1..].join(", "))
    }

    fn if_expr(i: &syn::ExprIf) -> String {
        let cond = Self::expr(&i.cond);
        let then_body = Self::block(&i.then_branch);
        let then_trimmed = then_body.trim();
        
        if let Some(ref else_branch) = i.else_branch {
            let else_body = match &*else_branch.1 {
                syn::Expr::Block(block) => Self::block(&block.block),
                syn::Expr::If(inner_if) => Self::if_expr(inner_if),
                _ => "".to_string(),
            };
            let else_trimmed = else_body.trim();
            format!("if ({}) {{ {} }} else {{ {} }}", cond, then_trimmed, else_trimmed)
        } else {
            format!("if ({}) {{ {} }}", cond, then_trimmed)
        }
    }

    fn block_expr(b: &syn::ExprBlock) -> String {
        Self::block(&b.block)
    }

    fn block(block: &syn::Block) -> String {
        block.stmts.iter().map(|s| Self::stmt(s)).collect::<Vec<_>>().join(" ")
    }

    fn stmt(s: &Stmt) -> String {
        match s {
            Stmt::Local(local) => {
                let pat = Self::pat(&local.pat);
                let init = local.init.as_ref().map(|init| Self::expr(&init.expr));
                if let Some(init_val) = init {
                    format!("let {} = {}; ", pat, init_val)
                } else {
                    format!("let {}; ", pat)
                }
            }
            Stmt::Expr(expr, semi) => {
                let e = Self::expr(expr);
                if semi.is_some() {
                    format!("{}; ", e)
                } else {
                    format!("{} ", e)
                }
            }
            _ => String::new(),
        }
    }

    fn pat(p: &syn::Pat) -> String {
        match p {
            syn::Pat::Ident(pi) => pi.ident.to_string(),
            _ => "_".to_string(),
        }
    }
}

#[allow(dead_code)]
pub fn rust_to_wgsl(expr: &syn::Expr) -> String {
    WgslGen::expr(expr)
}

pub fn rust_type_to_wgsl(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(tp) => {
            let segs: Vec<_> = tp
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            let last = segs.last().map(|s| s.as_str());
            match last {
                Some("f32") => "f32",
                Some("i32") => "i32",
                Some("u32") => "u32",
                Some("bool") => "bool",
                Some("Vec2") | Some("Vec2A") => "vec2<f32>",
                Some("Vec3") | Some("Vec3A") => "vec3<f32>",
                Some("Vec4") => "vec4<f32>",
                Some("Mat4") => "mat4x4<f32>",
                Some("Quat") => "vec4<f32>",
                _ => "f32",
            }
            .to_string()
        }
        _ => "f32".to_string(),
    }
}

fn param_name(pat: &Pat) -> String {
    match pat {
        Pat::Ident(i) => i.ident.to_string(),
        _ => "arg".to_string(),
    }
}

fn extract_and_convert_body(func: &ItemFn) -> String {
    let stmts = &func.block.stmts;
    if let Some(syn::Stmt::Expr(expr, _)) = stmts.last() {
        return rust_to_wgsl(expr);
    }
    "0.0".to_string()
}

/// Convert the entire function body to WGSL statements
fn convert_body_to_wgsl(func: &ItemFn) -> String {
    let mut out = String::new();
    for stmt in &func.block.stmts {
        out.push_str(&WgslGen::stmt(stmt));
    }
    out
}

/// Generate a standalone WGSL function definition (not a compute shader).
/// Returns the function body as WGSL.
pub fn wgsl_body_from_fn(func: &ItemFn) -> String {
    convert_body_to_wgsl(func)
}

/// Generate the full WGSL function signature + body as a string.
pub fn wgsl_fn_source(func: &ItemFn) -> String {
    let fn_name = func.sig.ident.to_string();
    let params: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_ty) = arg {
                Some((pat_ty.pat.as_ref(), pat_ty.ty.as_ref()))
            } else {
                None
            }
        })
        .collect();

    let return_ty = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => rust_type_to_wgsl(ty),
        syn::ReturnType::Default => "void".to_string(),
    };

    let param_wgsl: Vec<String> = params
        .iter()
        .map(|(pat, ty)| format!("{}: {}", param_name(pat), rust_type_to_wgsl(ty)))
        .collect();

    let body = convert_body_to_wgsl(func);

    format!("fn {}({}) -> {} {{ {} }}", fn_name, param_wgsl.join(", "), return_ty, body)
}

pub fn wgsl_source_from_fn(func: &ItemFn) -> String {
    let params: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_ty) = arg {
                Some((pat_ty.pat.as_ref(), pat_ty.ty.as_ref()))
            } else {
                None
            }
        })
        .collect();

    let return_ty = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => rust_type_to_wgsl(ty),
        syn::ReturnType::Default => "f32".to_string(),
    };

    let param_names: Vec<String> = params.iter().map(|(pat, _)| param_name(pat)).collect();
    let param_types: Vec<String> = params.iter().map(|(_, ty)| rust_type_to_wgsl(ty)).collect();
    let body_wgsl = extract_and_convert_body(func);

    let mut wgsl = String::new();
    for (i, (pname, ptype)) in param_names.iter().zip(param_types.iter()).enumerate() {
        wgsl.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> {pname}: array<{ptype}>;\n"
        ));
    }
    let output_idx = param_names.len();
    wgsl.push_str(&format!(
        "@group(0) @binding({output_idx}) var<storage, read_write> output: array<{return_ty}>;\n\n"
    ));
    wgsl.push_str(
        "@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
         let i = id.x;\n\
         output[i] = ",
    );
    wgsl.push_str(&body_wgsl);
    wgsl.push_str(";\n}\n");
    wgsl
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn simple_arithmetic() {
        let expr: syn::Expr = parse_quote!(a + b * c);
        assert_eq!(rust_to_wgsl(&expr), "a + b * c");
    }

    #[test]
    fn parens() {
        let expr: syn::Expr = parse_quote!((a + b) * c);
        assert_eq!(rust_to_wgsl(&expr), "(a + b) * c");
    }

    #[test]
    fn function_call() {
        let expr: syn::Expr = parse_quote!(sin(x));
        assert_eq!(rust_to_wgsl(&expr), "sin(x)");
    }

    #[test]
    fn vec3_construction() {
        let expr: syn::Expr = parse_quote!(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(rust_to_wgsl(&expr), "vec3<f32>(1.0, 2.0, 3.0)");
    }

    #[test]
    fn fields_access_path() {
        let expr: syn::Expr = parse_quote!(pos.x);
        assert_eq!(rust_to_wgsl(&expr), "pos.x");
    }

    #[test]
    fn method_call() {
        let expr: syn::Expr = parse_quote!(a.dot(b));
        assert_eq!(rust_to_wgsl(&expr), "dot(a, b)");
    }

    #[test]
    fn nested_math() {
        let expr: syn::Expr = parse_quote!(dot(normalize(a), cross(b, c)));
        assert_eq!(rust_to_wgsl(&expr), "dot(normalize(a), cross(b, c))");
    }

    #[test]
    fn field_chain() {
        let expr: syn::Expr = parse_quote!(a.b.c);
        assert_eq!(rust_to_wgsl(&expr), "a.b.c");
    }

    #[test]
    fn powi_2_expands_to_multiplication() {
        let expr: syn::Expr = parse_quote!(x.powi(2));
        assert_eq!(rust_to_wgsl(&expr), "x * x");
    }

    #[test]
    fn powi_3_expands_to_multiplication() {
        let expr: syn::Expr = parse_quote!(x.powi(3));
        assert_eq!(rust_to_wgsl(&expr), "x * x * x");
    }

    #[test]
    fn signum_maps_to_sign() {
        let expr: syn::Expr = parse_quote!(x.signum());
        assert_eq!(rust_to_wgsl(&expr), "sign(x)");
    }

    #[test]
    fn if_expr_translates() {
        let expr: syn::Expr = parse_quote!(if a > b { c } else { d });
        assert_eq!(rust_to_wgsl(&expr), "if (a > b) { c } else { d }");
    }

    #[test]
    fn let_stmt_translates() {
        let func: ItemFn = parse_quote! {
            fn test(a: f32) -> f32 {
                let x = a + 1.0;
                return x;
            }
        };
        let body = wgsl_body_from_fn(&func);
        assert!(body.contains("let x = a + 1.0;"));
        assert!(body.contains("return x;"));
    }

    #[test]
    fn fn_source_generates_wgsl_function() {
        let func: ItemFn = parse_quote! {
            fn ggx_distribution(NoH: f32, alpha: f32) -> f32 {
                let alpha2 = alpha * alpha;
                let denom = 3.14159 * (NoH * NoH * (alpha2 - 1.0) + 1.0) * (NoH * NoH * (alpha2 - 1.0) + 1.0);
                return alpha2 / denom;
            }
        };
        let source = wgsl_fn_source(&func);
        assert!(source.starts_with("fn ggx_distribution"));
        assert!(source.contains("NoH: f32"));
        assert!(source.contains("alpha: f32"));
        assert!(source.contains("-> f32"));
        assert!(source.contains("let alpha2 = alpha * alpha;"));
        assert!(source.contains("return alpha2 / denom;"));
    }

    #[test]
    fn if_else_in_fn_body() {
        let func: ItemFn = parse_quote! {
            fn test(x: f32) -> f32 {
                let y = if x > 0.0 { x } else { -x };
                return y;
            }
        };
        let source = wgsl_fn_source(&func);
        assert!(source.contains("if (x > 0.0)"));
        assert!(source.contains("else"));
    }

    #[test]
    fn vec_type_path() {
        let expr: syn::Expr = parse_quote!(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(rust_to_wgsl(&expr), "vec3<f32>(1.0, 2.0, 3.0)");
    }

    #[test]
    fn pi_constant_passes_through() {
        let expr: syn::Expr = parse_quote!(PI);
        assert_eq!(rust_to_wgsl(&expr), "PI");
    }
}
