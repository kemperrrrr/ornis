//! `Option<T>` lowering shared by the WGSL façade and the lowerer:
//! the Rust type → IR mapping and the generated-struct declarations a
//! kernel source needs in scope wherever `Some`/`None` spell.

use ornis_shader_lang::ShaderType;
use ornis_shader_lang::ir::IrType;
use ornis_shader_lang::writer::{option_struct_decl, print_ty};
use syn::{FnArg, ItemFn};

/// Lower `Option<T>` to its generated-struct IR type. `None` when the
/// path is not a bare `Option` with exactly one registry-representable
/// payload (the validator rejects those forms with a spanned error).
pub(crate) fn option_ir_ty(ty: &syn::Type) -> Option<IrType> {
    let syn::Type::Path(tp) = ty else {
        return None;
    };
    let seg = tp.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    let syn::GenericArgument::Type(inner) = args.args.first()? else {
        return None;
    };
    let syn::Type::Path(inner_tp) = inner else {
        return None;
    };
    let inner_seg = inner_tp.path.segments.last()?;
    if inner_seg.ident == "bool" {
        // Registry-excluded for call classification; printable as an
        // `Option` payload like it is an array element.
        return Some(IrType::Bool);
    }
    ShaderType::from_rust(&inner_seg.ident.to_string()).map(IrType::Scalar)
}

/// `Option<T>` as the lowered struct type, ready for `print_ty`.
pub(crate) fn option_struct_ty(ty: &syn::Type) -> Option<IrType> {
    option_ir_ty(ty).map(|inner| IrType::Optional(Box::new(inner)))
}

/// Every `Option<T>` a kernel/helper signature or `let` annotation
/// mentions, deduplicated by WGSL struct name. The generated struct
/// declarations are prepended to the source so the spelling used in the
/// body always has a definition in scope.
pub(crate) fn collect_option_decls(func: &ItemFn) -> String {
    use syn::visit::Visit;
    struct Collector(Vec<IrType>);
    impl<'ast> Visit<'ast> for Collector {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let syn::Pat::Type(pt) = &local.pat
                && let Some(ty) = option_struct_ty(&pt.ty)
            {
                self.0.push(ty);
            }
            syn::visit::visit_local(self, local);
        }
    }
    let mut collector = Collector(Vec::new());
    for arg in &func.sig.inputs {
        if let FnArg::Typed(pat_ty) = arg
            && let Some(ty) = option_struct_ty(&pat_ty.ty)
        {
            collector.0.push(ty);
        }
    }
    if let syn::ReturnType::Type(_, ty) = &func.sig.output
        && let Some(ty) = option_struct_ty(ty)
    {
        collector.0.push(ty);
    }
    collector.visit_block(&func.block);

    // One declaration per distinct struct name. The declaration is
    // written from the *payload* spelling: `option_struct_decl` itself
    // adds the `Option_` prefix, so passing the full name would emit
    // `Option_Option_f32` and leave `Option_f32` undefined.
    let mut decls: Vec<(String, String)> = collector
        .0
        .iter()
        .filter_map(|ty| {
            let IrType::Optional(inner) = ty else {
                return None;
            };
            Some((print_ty(ty), option_struct_decl(&print_ty(inner))))
        })
        .collect();
    decls.sort();
    decls.dedup_by(|a, b| a.0 == b.0);
    decls
        .into_iter()
        .map(|(_, decl)| decl)
        .collect::<Vec<_>>()
        .join("\n")
}
