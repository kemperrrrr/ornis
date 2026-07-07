use syn::{FnArg, ItemFn, Pat};

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

    fn binary(b: &syn::ExprBinary) -> String {
        use syn::BinOp::*;
        let op = match &b.op {
            Add(_) => "+",
            Sub(_) => "-",
            Mul(_) => "*",
            Div(_) => "/",
            Rem(_) => "%",
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
            "sin" | "cos" | "tan" | "atan2" | "sqrt" | "abs" | "floor"
            | "ceil" | "round" | "fract" | "exp" | "log" | "pow" | "max"
            | "min" | "step" | "smoothstep" | "mix" | "reflect" | "refract"
            | "transpose" | "determinant" | "inverse" => {
                Some(format!("{}({})", fn_name, args.join(", ")))
            }
            _ => None,
        }
    }

    fn call(c: &syn::ExprCall) -> String {
        let args: Vec<String> = c.args.iter().map(Self::expr).collect();

        if let syn::Expr::Path(p) = c.func.as_ref() {
            if let Some(ident) = Self::path_ident(p) {
                if let Some(mapped) = Self::map_fn(&ident, &args) {
                    return mapped;
                }
            }
        }

        let callee = Self::expr(&c.func);
        format!("{}({})", callee, args.join(", "))
    }

    fn method_call(m: &syn::ExprMethodCall) -> String {
        let receiver = Self::expr(&m.receiver);
        let method = m.method.to_string();
        let args: Vec<String> = std::iter::once(receiver)
            .chain(m.args.iter().map(Self::expr))
            .collect();

        if let Some(mapped) = Self::map_fn(&method, &args) {
            return mapped;
        }

        format!("{}.{}({})", args[0], method, args[1..].join(", "))
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
        assert_eq!(rust_to_wgsl(&expr), "1.0, 2.0, 3.0");
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
}
