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
            Index(ix) => format!("{}[{}]", Self::expr(&ix.expr), Self::expr(&ix.index)),
            // Rust casts are type-coercion hints for the DSL; WGSL infers the
            // type from context, so the cast itself is dropped.
            Cast(c) => Self::expr(&c.expr),
            Return(r) => Self::ret(r),
            _ => syn::Error::new_spanned(e, "expression not supported in WGSL kernel")
                .to_compile_error()
                .to_string(),
        }
    }

    fn lit(l: &syn::ExprLit) -> String {
        use syn::Lit::*;
        match &l.lit {
            Int(i) => {
                // Hex/octal/binary literals: pass through unchanged.
                let repr = i.to_string();
                if repr.starts_with("0x") || repr.starts_with("0o") || repr.starts_with("0b") {
                    return repr;
                }
                // Rust integer suffixes are not valid WGSL; map the common
                // ones to the WGSL `u`/`i` suffixes and drop the rest.
                let digits = i.base10_digits();
                match i.suffix() {
                    "u32" | "u64" | "usize" | "u16" | "u8" => format!("{digits}u"),
                    "i32" | "i64" | "isize" | "i16" | "i8" => format!("{digits}i"),
                    _ => digits.to_string(),
                }
            }
            Float(f) => {
                let base = f.base10_digits();
                if base.contains('.') || base.contains('e') || base.contains('E') {
                    base.to_string()
                } else {
                    format!("{}.0", base)
                }
            }
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
        let segs: Vec<_> = p
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        let last = segs.last().map(|s| s.as_str());

        // Map glam type constants to WGSL constructors
        let parent_type = segs
            .len()
            .checked_sub(2)
            .and_then(|i| segs.get(i))
            .map(|s| s.as_str());
        match (parent_type, last) {
            (Some("Vec2"), Some("ZERO")) => return "vec2<f32>(0.0)".into(),
            (Some("Vec2"), Some("ONE")) => return "vec2<f32>(1.0)".into(),
            (Some("Vec3"), Some("ZERO")) => return "vec3<f32>(0.0)".into(),
            (Some("Vec3"), Some("ONE")) => return "vec3<f32>(1.0)".into(),
            (Some("Vec4"), Some("ZERO")) => return "vec4<f32>(0.0)".into(),
            (Some("Vec4"), Some("ONE")) => return "vec4<f32>(1.0)".into(),
            _ => {}
        }

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

    fn assign_op(b: &syn::ExprBinary) -> String {
        use syn::BinOp::*;
        let op = match &b.op {
            AddAssign(_) => "+=",
            SubAssign(_) => "-=",
            MulAssign(_) => "*=",
            DivAssign(_) => "/=",
            RemAssign(_) => "%=",
            ShlAssign(_) => "<<=",
            ShrAssign(_) => ">>=",
            BitAndAssign(_) => "&=",
            BitOrAssign(_) => "|=",
            BitXorAssign(_) => "^=",
            other => {
                return syn::Error::new_spanned(other, "operator not supported in WGSL kernel")
                    .to_compile_error()
                    .to_string();
            }
        };
        format!("{} {} {}", Self::expr(&b.left), op, Self::expr(&b.right))
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
            // syn 2 parses compound assignments (`a += b`) as binary
            // expressions with an assign-op `BinOp`.
            AddAssign(_) | SubAssign(_) | MulAssign(_) | DivAssign(_) | RemAssign(_)
            | ShlAssign(_) | ShrAssign(_) | BitAndAssign(_) | BitOrAssign(_) | BitXorAssign(_) => {
                return Self::assign_op(b);
            }
            // Rust `&&`/`||` are WGSL `&&`/`||` (short-circuit, not bitwise).
            And(_) => "&&",
            Or(_) => "||",
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
                    .to_string();
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
            "sqrt" | "abs" | "floor" | "ceil" | "round" | "fract" | "exp" | "log" | "sin"
            | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" | "pow" | "max" | "min"
            | "step" | "smoothstep" | "mix" | "reflect" | "refract" | "transpose"
            | "determinant" | "inverse" | "sign" | "dpdx" | "dpdy" | "fwidth" => {
                Some(format!("{}({})", fn_name, args.join(", ")))
            }
            // Rust glam `length_squared` → WGSL dot(x, x).
            "length_sq" => Some(format!("dot({0}, {0})", args[0])),
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
            // splat: glam::Vec3::splat(1.0) -> vec3<f32>(1.0)
            if ident_str == "splat" {
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
            if let Some(arg) = m.args.first()
                && let syn::Expr::Lit(lit) = arg
                && let syn::Lit::Int(int_lit) = &lit.lit
            {
                let p = int_lit.base10_parse::<i32>().unwrap_or(0);
                if p == 2 {
                    return format!("{} * {}", receiver, receiver);
                } else if p == 3 {
                    return format!("{} * {} * {}", receiver, receiver, receiver);
                } else if p == 4 {
                    return format!("{} * {} * {} * {}", receiver, receiver, receiver, receiver);
                } else if p > 0 {
                    // Generate repeated multiplication for small positive ints
                    let parts: Vec<String> =
                        std::iter::repeat_n(receiver.clone(), p as usize).collect();
                    return parts.join(" * ");
                }
            }
            return format!("pow({}, {})", receiver, args[1..].join(", "));
        }

        if let Some(mapped) = Self::map_fn(&method, &args) {
            return mapped;
        }

        // Map Rust methods to WGSL functions
        match method.as_str() {
            // Swizzle methods - convert to field access in WGSL
            "x" | "y" | "z" | "w" | "r" | "g" | "b" | "a" | "xy" | "xz" | "xw" | "yx" | "yz"
            | "yw" | "zx" | "zy" | "zw" | "wx" | "wy" | "wz" | "xyz" | "xyw" | "xzy" | "xzw"
            | "yxz" | "yxw" | "yzx" | "yzw" | "zxy" | "zxw" | "zyx" | "zyw" | "wxy" | "wxz"
            | "wyz" | "wzx" | "wzy" | "xyzw" | "xywz" | "xzyw" | "xzwy" | "xwyz" | "xwzy"
            | "yxzw" | "yxwz" | "yzxw" | "yzwx" | "ywxz" | "ywzx" | "zxyw" | "zxwy" | "zyxw"
            | "zywx" | "zwxy" | "zwyx" | "wxyz" | "wxzy" | "wyxz" | "wyzx" | "wzxy" | "wzyx" => {
                return format!("{}.{}", receiver, method);
            }
            "signum" => return format!("sign({})", args[0]),
            "abs" | "sqrt" | "sin" | "cos" | "tan" | "floor" | "ceil" | "round" | "fract"
            | "exp" | "log" | "normalize" | "length" | "saturate" | "asin" | "acos" | "atan" => {
                return format!("{}({})", method, args[0]);
            }
            "dot" | "cross" | "pow" | "max" | "min" | "step" | "smoothstep" | "reflect"
            | "refract" | "atan2" => {
                return format!("{}({})", method, args.join(", "));
            }
            "clamp" => {
                return format!("clamp({})", args.join(", "));
            }
            "lerp" => {
                return format!("mix({})", args.join(", "));
            }
            "ln" => {
                return format!("log({})", args.join(", "));
            }
            "powf" => {
                return format!("pow({})", args.join(", "));
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
            format!(
                "if ({}) {{ {} }} else {{ {} }}",
                cond, then_trimmed, else_trimmed
            )
        } else {
            format!("if ({}) {{ {} }}", cond, then_trimmed)
        }
    }

    fn block_expr(b: &syn::ExprBlock) -> String {
        Self::block(&b.block)
    }

    fn block(block: &syn::Block) -> String {
        let count = block.stmts.len();
        block
            .stmts
            .iter()
            .enumerate()
            .map(|(i, s)| Self::stmt(s, i == count - 1))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn stmt(s: &Stmt, is_tail: bool) -> String {
        match s {
            Stmt::Local(local) => {
                // Handle `let x = if/else { .. } else { .. };` — wrap if/else into var+if
                if let Some(init) = &local.init
                    && let syn::Expr::If(if_expr) = init.expr.as_ref()
                {
                    let pat = Self::pat(&local.pat);
                    let cond = Self::expr(&if_expr.cond);
                    let then_val = Self::last_expr_in_block(&if_expr.then_branch);
                    let else_val = if_expr
                        .else_branch
                        .as_ref()
                        .and_then(|(_, else_expr)| {
                            if let syn::Expr::Block(block) = else_expr.as_ref() {
                                Some(Self::last_expr_in_block(&block.block))
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    // Convert: let x = if c { a } else { b };
                    // To WGSL: var x = b; if (c) { x = a; }
                    return format!(
                        "var {} = {}; if ({}) {{ {} = {}; }} ",
                        pat, else_val, cond, pat, then_val
                    );
                }
                let pat = Self::pat(&local.pat);
                let init = local.init.as_ref().map(|init| Self::expr(&init.expr));
                // Rust `let mut` becomes WGSL `var` — the only WGSL binding
                // kind that can be reassigned.
                let kw = if let syn::Pat::Ident(pi) = &local.pat
                    && pi.mutability.is_some()
                {
                    "var"
                } else {
                    "let"
                };
                if let Some(init_val) = init {
                    format!("{kw} {pat} = {init_val}; ")
                } else {
                    format!("{kw} {pat}; ")
                }
            }
            Stmt::Expr(expr, semi) => {
                let e = Self::expr(expr);
                // `return`/`return x` always emit as a terminated statement,
                // even when they are the trailing (tail) expression.
                if matches!(expr, syn::Expr::Return(_)) {
                    format!("{e}; ")
                // if/block expressions in WGSL are statements, not values
                // — never wrap in `return` at this level; return is handled
                // inside the branches via block() → is_tail
                } else if matches!(expr, syn::Expr::If(_) | syn::Expr::Block(_)) {
                    if semi.is_some() {
                        format!("{}; ", e)
                    } else {
                        format!("{} ", e)
                    }
                } else if semi.is_some() || !is_tail {
                    format!("{}; ", e)
                } else {
                    format!("return {}; ", e)
                }
            }
            _ => String::new(),
        }
    }

    fn last_expr_in_block(block: &syn::Block) -> String {
        block
            .stmts
            .last()
            .and_then(|s| {
                if let Stmt::Expr(e, _) = s {
                    Some(Self::expr(e))
                } else {
                    None
                }
            })
            .unwrap_or_default()
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
    let count = func.block.stmts.len();
    for (i, stmt) in func.block.stmts.iter().enumerate() {
        out.push_str(&WgslGen::stmt(stmt, i == count - 1));
    }
    out
}

/// Generate a standalone WGSL function definition (not a compute shader).
/// Returns the function body as WGSL.
// Helper wrapper, used by this module's unit tests.
#[allow(dead_code)]
pub fn wgsl_body_from_fn(func: &ItemFn) -> String {
    convert_body_to_wgsl(func)
}

/// Generate just the statement list of a function body as WGSL. Used by
/// `#[gpu_pipeline]` in full-shader mode, where the body is embedded into a
/// generated `fn main(...) { ... }` compute entry point.
pub fn wgsl_main_body(func: &ItemFn) -> String {
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

    format!(
        "fn {}({}) -> {} {{ {} }}",
        fn_name,
        param_wgsl.join(", "),
        return_ty,
        body
    )
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
        // WGSL requires return in blocks used as expressions
        assert_eq!(
            rust_to_wgsl(&expr),
            "if (a > b) { return c; } else { return d; }"
        );
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
        // let y = if/else → var y = else_val; if (cond) { y = if_val; }
        assert!(source.contains("var y = -x;"));
        assert!(source.contains("if (x > 0.0)"));
        assert!(source.contains("y = x;"));
        assert!(source.contains("return y;"));
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

    #[test]
    fn index_expr_translates() {
        let expr: syn::Expr = parse_quote!(buf[i]);
        assert_eq!(rust_to_wgsl(&expr), "buf[i]");
    }

    #[test]
    fn nested_index_field_translates() {
        let expr: syn::Expr = parse_quote!(batch_buf[gid.x].acc[l]);
        assert_eq!(rust_to_wgsl(&expr), "batch_buf[gid.x].acc[l]");
    }

    #[test]
    fn cast_is_dropped() {
        let expr: syn::Expr = parse_quote!(x as u32);
        assert_eq!(rust_to_wgsl(&expr), "x");
    }

    #[test]
    fn assign_op_translates() {
        let expr: syn::Expr = parse_quote!(ba.velocity -= delta * n);
        assert_eq!(rust_to_wgsl(&expr), "ba.velocity -= delta * n");
    }

    #[test]
    fn logical_and_or_translate_to_short_circuit() {
        let and: syn::Expr = parse_quote!(a > 0 && b < 1);
        assert_eq!(rust_to_wgsl(&and), "a > 0 && b < 1");
        let or: syn::Expr = parse_quote!(a > 0 || b < 1);
        assert_eq!(rust_to_wgsl(&or), "a > 0 || b < 1");
    }

    #[test]
    fn int_suffix_maps_to_wgsl() {
        let u: syn::Expr = parse_quote!(0u32);
        assert_eq!(rust_to_wgsl(&u), "0u");
        let i: syn::Expr = parse_quote!(7i32);
        assert_eq!(rust_to_wgsl(&i), "7i");
        let plain: syn::Expr = parse_quote!(42);
        assert_eq!(rust_to_wgsl(&plain), "42");
    }

    #[test]
    fn mut_local_becomes_var() {
        let func: ItemFn = parse_quote! {
            fn test() {
                let mut x = 1.0;
                x += 2.0;
            }
        };
        let body = wgsl_body_from_fn(&func);
        assert!(body.contains("var x = 1.0;"));
        assert!(body.contains("x += 2.0;"));
    }

    #[test]
    fn early_return_in_if_block() {
        let func: ItemFn = parse_quote! {
            fn test(l: u32, count: u32) {
                if l >= count { return; }
                let y = 1.0;
                y
            }
        };
        let body = wgsl_body_from_fn(&func);
        assert!(body.contains("if (l >= count) { return; }"));
        assert!(body.contains("let y = 1.0;"));
    }

    #[test]
    fn length_sq_maps_to_dot() {
        let expr: syn::Expr = parse_quote!(length_sq(v));
        assert_eq!(rust_to_wgsl(&expr), "dot(v, v)");
    }
}
