//! Shader language registry: the explicit set of primitive types and
//! built-in functions the DSL maps — no string-table guessing.
//!
//! Two enums replace six scattered match-tables in `ornis-macros`
//! (`wgsl_type`, the scalar arms of `glam_type_to_wgsl`/`var_type`/
//! `rust_type_to_wgsl`, `map_fn`, `named_builtin`,
//! `renamed_unary`/`passthrough_math`/`renamed_multi`):
//!
//! - [`ShaderType`]: glam/scalar names → WGSL spellings. Anything else
//!   (mirrors, bundles, markers — and bare `bool`, which is not a shader
//!   type) is NOT a primitive: [`from_rust`](ShaderType::from_rust)
//!   returns `None` and each call site keeps its previous fallback, so
//!   behavior on real inputs is unchanged.
//! - [`ShaderBuiltin`]: Rust-spelled math built-ins → WGSL calls, each
//!   with a fixed [`arity`](ShaderBuiltin::arity). Unknown names pass
//!   through verbatim downstream (kernels/helpers share WGSL spellings by
//!   design); the enum owns only names the DSL renames or validates.
//!   Notably `lerp → mix` lived in TWO tables (free-fn and method
//!   positions could drift) — now it is one variant.
//!
//! Arity is enforced by [`check_builtin_arity`], a pre-translation AST
//! walk: translation itself stays `String`-based (no `Result` ripple
//! through `WgslGen`), while wrong-arity calls fail with spanned errors
//! instead of reaching naga (or panicking, as zero-arg `length_sq` did).

use syn::visit::Visit;

/// Shader primitive types: the DSL's closed type world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    F32,
    I32,
    U32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    UVec2,
    UVec3,
    UVec4,
    IVec2,
    IVec3,
    IVec4,
    BVec2,
    BVec3,
    BVec4,
}

impl ShaderType {
    /// Map a Rust type name to a primitive, both casings where the old
    /// tables accepted them (`Vec2`/`vec2`, `Mat4`/`mat4`). `None` for
    /// everything else — mirrors, bundles, markers, bare `bool`.
    pub fn from_rust(name: &str) -> Option<Self> {
        Some(match name {
            "f32" => Self::F32,
            "i32" => Self::I32,
            "u32" => Self::U32,
            "Vec2" | "vec2" | "Vec2A" => Self::Vec2,
            "Vec3" | "vec3" | "Vec3A" => Self::Vec3,
            "Vec4" | "vec4" | "Quat" => Self::Vec4,
            "Mat4" | "mat4" => Self::Mat4,
            "UVec2" => Self::UVec2,
            "UVec3" => Self::UVec3,
            "UVec4" => Self::UVec4,
            "IVec2" => Self::IVec2,
            "IVec3" => Self::IVec3,
            "IVec4" => Self::IVec4,
            "BVec2" => Self::BVec2,
            "BVec3" => Self::BVec3,
            "BVec4" => Self::BVec4,
            _ => return None,
        })
    }

    /// WGSL spelling of the primitive.
    pub fn wgsl(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
            Self::Mat4 => "mat4x4<f32>",
            Self::UVec2 => "vec2<u32>",
            Self::UVec3 => "vec3<u32>",
            Self::UVec4 => "vec4<u32>",
            Self::IVec2 => "vec2<i32>",
            Self::IVec3 => "vec3<i32>",
            Self::IVec4 => "vec4<i32>",
            Self::BVec2 => "vec2<bool>",
            Self::BVec3 => "vec3<bool>",
            Self::BVec4 => "vec4<bool>",
        }
    }
}

/// Math built-ins the DSL renames or validates, by WGSL target.
/// Several Rust spellings can share one variant (`mix`/`lerp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderBuiltin {
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Floor,
    Ceil,
    Round,
    Fract,
    Exp,
    Log,
    Normalize,
    Length,
    Saturate,
    Transpose,
    Determinant,
    Inverse,
    Dpdx,
    Dpdy,
    Fwidth,
    Sign,
    LengthSq,
    Dot,
    Cross,
    Pow,
    Max,
    Min,
    Step,
    Reflect,
    Atan2,
    Mix,
    Smoothstep,
    Clamp,
    Select,
    Refract,
}

impl ShaderBuiltin {
    /// Map a Rust function/method name to a built-in. `None` for WGSL-
    /// native spellings (`textureSample`, …) and kernels/helpers, which
    /// pass through verbatim downstream.
    pub fn from_rust(name: &str) -> Option<Self> {
        Some(match name {
            "abs" => Self::Abs,
            "sqrt" => Self::Sqrt,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "round" => Self::Round,
            "fract" => Self::Fract,
            "exp" => Self::Exp,
            "log" | "ln" => Self::Log,
            "normalize" => Self::Normalize,
            "length" => Self::Length,
            "saturate" => Self::Saturate,
            "transpose" => Self::Transpose,
            "determinant" => Self::Determinant,
            "inverse" => Self::Inverse,
            "dpdx" => Self::Dpdx,
            "dpdy" => Self::Dpdy,
            "fwidth" => Self::Fwidth,
            "sign" | "signum" => Self::Sign,
            "length_sq" | "length_squared" => Self::LengthSq,
            "dot" => Self::Dot,
            "cross" => Self::Cross,
            "pow" | "powf" => Self::Pow,
            "max" => Self::Max,
            "min" => Self::Min,
            "step" | "stepf" => Self::Step,
            "reflect" => Self::Reflect,
            "atan2" => Self::Atan2,
            "mix" | "lerp" => Self::Mix,
            "smoothstep" => Self::Smoothstep,
            "clamp" => Self::Clamp,
            "select" => Self::Select,
            "refract" => Self::Refract,
            _ => return None,
        })
    }

    /// Fixed argument count (method receivers included: `a.lerp(b, t)`
    /// is three). Checked pre-translation by [`check_builtin_arity`].
    pub fn arity(&self) -> usize {
        match self {
            Self::Abs
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Fract
            | Self::Exp
            | Self::Log
            | Self::Normalize
            | Self::Length
            | Self::Saturate
            | Self::Transpose
            | Self::Determinant
            | Self::Inverse
            | Self::Dpdx
            | Self::Dpdy
            | Self::Fwidth
            | Self::Sign
            | Self::LengthSq => 1,
            Self::Dot
            | Self::Cross
            | Self::Pow
            | Self::Max
            | Self::Min
            | Self::Step
            | Self::Reflect
            | Self::Atan2 => 2,
            Self::Mix | Self::Smoothstep | Self::Clamp | Self::Select | Self::Refract => 3,
        }
    }

    /// WGSL function name (renames resolved: `lerp` → `mix`).
    pub fn wgsl(&self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Fract => "fract",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Normalize => "normalize",
            Self::Length => "length",
            Self::Saturate => "saturate",
            Self::Transpose => "transpose",
            Self::Determinant => "determinant",
            Self::Inverse => "inverse",
            Self::Dpdx => "dpdx",
            Self::Dpdy => "dpdy",
            Self::Fwidth => "fwidth",
            Self::Sign => "sign",
            Self::LengthSq => "dot",
            Self::Dot => "dot",
            Self::Cross => "cross",
            Self::Pow => "pow",
            Self::Max => "max",
            Self::Min => "min",
            Self::Step => "step",
            Self::Reflect => "reflect",
            Self::Atan2 => "atan2",
            Self::Mix => "mix",
            Self::Smoothstep => "smoothstep",
            Self::Clamp => "clamp",
            Self::Select => "select",
            Self::Refract => "refract",
        }
    }

    /// Lower already-translated arguments (arity checked upstream):
    /// `wgsl(a, …)`, except `length_sq(x)` → `dot(x, x)`.
    pub fn lower(&self, args: &[String]) -> String {
        if *self == Self::LengthSq {
            return format!("dot({0}, {0})", args[0]);
        }
        format!("{}({})", self.wgsl(), args.join(", "))
    }
}

/// Swizzle method names (`v.xyz()`), lowered to field access elsewhere.
/// Kept here so the arity walk and the translator share one list.
pub fn is_swizzle_method(name: &str) -> bool {
    matches!(
        name,
        "x" | "y"
            | "z"
            | "w"
            | "r"
            | "g"
            | "b"
            | "a"
            | "xy"
            | "xz"
            | "xw"
            | "yx"
            | "yz"
            | "yw"
            | "zx"
            | "zy"
            | "zw"
            | "wx"
            | "wy"
            | "wz"
            | "xyz"
            | "xyw"
            | "xzy"
            | "xzw"
            | "yxz"
            | "yxw"
            | "yzx"
            | "yzw"
            | "zxy"
            | "zxw"
            | "zyx"
            | "zyw"
            | "wxy"
            | "wxz"
            | "wyz"
            | "wzx"
            | "wzy"
            | "xyzw"
            | "xywz"
            | "xzyw"
            | "xzwy"
            | "xwyz"
            | "xwzy"
            | "yxzw"
            | "yxwz"
            | "yzxw"
            | "yzwx"
            | "ywxz"
            | "ywzx"
            | "zxyw"
            | "zxwy"
            | "zyxw"
            | "zywx"
            | "zwxy"
            | "zwyx"
            | "wxyz"
            | "wxzy"
            | "wyxz"
            | "wyzx"
            | "wzxy"
            | "wzyx"
            | "rgb"
            | "rgba"
    )
}

/// Constructors (`Type::new(…)`/`splat(…)`) and the `powi` expansion keep
/// their dedicated paths; everything else named that the registry knows
/// is arity-checked here, before `String`-based translation runs.
pub fn check_builtin_arity(func: &syn::ItemFn) -> syn::Result<()> {
    use std::cell::RefCell;
    struct Walk {
        errors: RefCell<Vec<(proc_macro2::Span, String)>>,
    }
    impl Walk {
        fn fail(&self, span: proc_macro2::Span, msg: String) {
            self.errors.borrow_mut().push((span, msg));
        }
    }
    impl<'ast> Visit<'ast> for Walk {
        fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
            let method = m.method.to_string();
            if method == "powi" {
                return;
            }
            if is_swizzle_method(&method) {
                if !m.args.is_empty() {
                    // Translation would otherwise silently drop the args.
                    self.fail(
                        m.method.span(),
                        format!("stage: swizzle `.{method}()` takes no arguments"),
                    );
                }
                return;
            }
            if let Some(builtin) = ShaderBuiltin::from_rust(&method) {
                let got = m.args.len() + 1;
                if got != builtin.arity() {
                    self.fail(
                        m.method.span(),
                        format!(
                            "stage: `{method}` expects {} arguments (receiver included), got {got}",
                            builtin.arity()
                        ),
                    );
                }
            }
        }

        fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
            if let syn::Expr::Path(p) = c.func.as_ref()
                && let Some(last) = p.path.segments.last()
            {
                let name = last.ident.to_string();
                if name == "new" || name == "splat" {
                    return;
                }
                if let Some(builtin) = ShaderBuiltin::from_rust(&name) {
                    let got = c.args.len();
                    if got != builtin.arity() {
                        self.fail(
                            last.ident.span(),
                            format!(
                                "stage: `{name}` expects {} arguments, got {got}",
                                builtin.arity()
                            ),
                        );
                    }
                }
            }
        }
    }
    let mut walk = Walk {
        errors: RefCell::new(Vec::new()),
    };
    walk.visit_item_fn(func);
    let errors = walk.errors.borrow();
    if let Some((span, msg)) = errors.first() {
        return Err(syn::Error::new(*span, msg.clone()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_cover_glam_and_scalars() {
        let cases = [
            ("f32", "f32"),
            ("i32", "i32"),
            ("u32", "u32"),
            ("Vec2", "vec2<f32>"),
            ("vec3", "vec3<f32>"),
            ("Vec4", "vec4<f32>"),
            ("Quat", "vec4<f32>"),
            ("Vec2A", "vec2<f32>"),
            ("Vec3A", "vec3<f32>"),
            ("Mat4", "mat4x4<f32>"),
            ("mat4", "mat4x4<f32>"),
            ("UVec2", "vec2<u32>"),
            ("UVec4", "vec4<u32>"),
            ("IVec3", "vec3<i32>"),
            ("BVec4", "vec4<bool>"),
        ];
        for (rust, wgsl) in cases {
            assert_eq!(
                ShaderType::from_rust(rust)
                    .unwrap_or_else(|| panic!("{rust}"))
                    .wgsl(),
                wgsl
            );
        }
        // Not primitives: mirrors pass through, bare bool is not a type.
        for other in ["VertexOutput", "CameraUniform", "bool", "Texture2d", ""] {
            assert_eq!(ShaderType::from_rust(other), None, "{other}");
        }
    }

    #[test]
    fn builtins_unify_aliases() {
        // One variant per WGSL target, however Rust spells it.
        assert_eq!(
            ShaderBuiltin::from_rust("mix"),
            ShaderBuiltin::from_rust("lerp")
        );
        assert_eq!(
            ShaderBuiltin::from_rust("log"),
            ShaderBuiltin::from_rust("ln")
        );
        assert_eq!(
            ShaderBuiltin::from_rust("pow"),
            ShaderBuiltin::from_rust("powf")
        );
        assert_eq!(
            ShaderBuiltin::from_rust("sign"),
            ShaderBuiltin::from_rust("signum")
        );
        assert_eq!(
            ShaderBuiltin::from_rust("step"),
            ShaderBuiltin::from_rust("stepf")
        );
        assert_eq!(
            ShaderBuiltin::from_rust("length_sq"),
            ShaderBuiltin::from_rust("length_squared")
        );
        assert_eq!(ShaderBuiltin::from_rust("lerp").unwrap().wgsl(), "mix");
        assert_eq!(ShaderBuiltin::from_rust("ln").unwrap().wgsl(), "log");
        // Kernels and WGSL-native spellings are not builtins.
        for other in [
            "luminance",
            "textureSample",
            "octahedral_encode",
            "vs_main",
            "",
        ] {
            assert_eq!(ShaderBuiltin::from_rust(other), None, "{other}");
        }
    }

    #[test]
    fn builtin_arity_and_lowering() {
        assert_eq!(ShaderBuiltin::from_rust("dot").unwrap().arity(), 2);
        assert_eq!(ShaderBuiltin::from_rust("mix").unwrap().arity(), 3);
        assert_eq!(ShaderBuiltin::from_rust("normalize").unwrap().arity(), 1);
        let args = ["a".to_string(), "b".to_string(), "t".to_string()];
        assert_eq!(
            ShaderBuiltin::from_rust("lerp").unwrap().lower(&args),
            "mix(a, b, t)"
        );
        assert_eq!(
            ShaderBuiltin::from_rust("length_sq")
                .unwrap()
                .lower(&["x".to_string()]),
            "dot(x, x)"
        );
    }

    #[test]
    fn arity_walk_accepts_good_bodies() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn vs_main(idx: u32) -> u32 {
                let a = normalize(v);
                let b = a.lerp(c, t);
                let d = dot(a, b);
                return max(d, 0.0);
            }
        };
        assert!(check_builtin_arity(&func).is_ok());
    }

    #[test]
    fn arity_walk_rejects_bad_counts() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn fs_main(x: u32) -> u32 {
                return dot(x);
            }
        };
        let err = check_builtin_arity(&func).expect_err("dot(x) must fail");
        assert!(err.to_string().contains("expects 2 arguments"), "{err}");
    }
}
