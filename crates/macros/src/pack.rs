use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, FieldsNamed, Generics, Ident, Type, TypePath, WhereClause,
    parse_macro_input,
};

#[derive(Clone)]
struct FieldInfo {
    name: Ident,
    lane_ty: Type,
    // Unique wrapper type for this field's lane
    wrapper_name: Ident,
}

#[derive(Clone)]
struct LaneInfo {
    wrapper_ty: Type,
    inner_ty: Type,
}

struct PackInfo {
    struct_name: Ident,
    generics: Generics,
    fields: Vec<FieldInfo>,
    lanes: Vec<LaneInfo>,
}

fn extract_field_info(struct_name: &Ident, fields: &FieldsNamed) -> Vec<FieldInfo> {
    fields
        .named
        .iter()
        .enumerate()
        .filter_map(|(idx, f)| {
            let name = f.ident.as_ref()?.clone();
            let ty = f.ty.clone();
            let lane_ty = extract_lane_type(&ty);
            let wrapper_name = format_ident!("{}__{}__PackLane__{}", struct_name, name, idx);
            Some(FieldInfo {
                name,
                lane_ty,
                wrapper_name,
            })
        })
        .collect()
}

fn extract_lane_type(ty: &Type) -> Type {
    match ty {
        Type::Path(TypePath { path, .. }) => {
            let mut path = path.clone();
            if let Some(last) = path.segments.last_mut()
                && last.ident == "Option"
                && let syn::PathArguments::AngleBracketed(args) = &last.arguments
                && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
            {
                return inner.clone();
            }
            ty.clone()
        }
        _ => ty.clone(),
    }
}

fn generate_lanes(fields: &[FieldInfo]) -> Vec<LaneInfo> {
    fields
        .iter()
        .map(|field| {
            let wrapper_name = &field.wrapper_name;
            let wrapper_ty: Type = syn::parse_quote!(#wrapper_name);
            LaneInfo {
                wrapper_ty,
                inner_ty: field.lane_ty.clone(),
            }
        })
        .collect()
}

fn extract_pack_info(input: &DeriveInput) -> Result<PackInfo, syn::Error> {
    let struct_name = input.ident.clone();
    let generics = input.generics.clone();

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &s.fields,
                    "#[derive(Pack)] requires named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &struct_name,
                "#[derive(Pack)] only works on structs",
            ));
        }
    };

    let field_infos = extract_field_info(&struct_name, fields);
    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(
            &struct_name,
            "#[derive(Pack)] requires at least one field",
        ));
    }

    let lanes = generate_lanes(&field_infos);

    Ok(PackInfo {
        struct_name,
        generics,
        fields: field_infos,
        lanes,
    })
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let info = match extract_pack_info(&input) {
        Ok(info) => info,
        Err(e) => return e.to_compile_error().into(),
    };

    let struct_name = &info.struct_name;
    let (impl_generics, ty_generics, where_clause) = info.generics.split_for_impl();

    let pack_mut_name = format_ident!("PackMut{}", struct_name);

    // Generate wrapper type definitions
    let wrapper_defs = generate_wrapper_defs(&info.lanes);
    let pack_register = generate_pack_register(&info.lanes);
    let pack_insert = generate_pack_insert(&info.fields);
    let pack_get = generate_pack_get(struct_name, &info.fields, &ty_generics);
    let pack_get_mut = generate_pack_get_mut(&info.fields, &ty_generics, &pack_mut_name);
    let (pack_mut_struct, pack_mut_methods) = generate_pack_mut_struct(
        struct_name,
        &info.fields,
        &ty_generics,
        where_clause,
        &pack_mut_name,
    );

    let expanded = quote! {
        #wrapper_defs

        #pack_mut_struct

        #pack_mut_methods

        impl #impl_generics ornis_core::Pack for #struct_name #ty_generics #where_clause {
            type PackMut<'a> = #pack_mut_name<'a, #ty_generics> #where_clause
            where
                Self: 'a;

            fn pack_register(store: &mut ornis_core::SmartStore) {
                #pack_register
            }

            fn pack_insert(&self, store: &mut ornis_core::SmartStore, entity: ornis_core::Entity) {
                #pack_insert
            }

            fn pack_get(store: &ornis_core::SmartStore, entity: ornis_core::Entity) -> Option<Self> {
                #pack_get
            }

            fn pack_get_mut<'a>(store: &'a mut ornis_core::SmartStore, entity: ornis_core::Entity) -> Option<Self::PackMut<'a>> {
                #pack_get_mut
            }
        }
    };

    expanded.into()
}

fn generate_wrapper_defs(lanes: &[LaneInfo]) -> TokenStream2 {
    let defs = lanes.iter().map(|lane| {
        let wrapper_name = &lane.wrapper_ty;
        let inner_ty = &lane.inner_ty;
        quote! {
            #[derive(Clone, Debug, PartialEq)]
            #[repr(transparent)]
            // Имя генерируется как Struct__field__PackLane__N — осознанно не CamelCase.
            #[allow(non_camel_case_types)]
            pub struct #wrapper_name(pub #inner_ty);
        }
    });
    quote! { #(#defs)* }
}

fn generate_pack_register(lanes: &[LaneInfo]) -> TokenStream2 {
    let lane_registers = lanes.iter().map(|lane| {
        let wrapper_ty = &lane.wrapper_ty;
        quote! { store.register::<#wrapper_ty>(); }
    });

    quote! { #(#lane_registers)* }
}

fn generate_pack_insert(fields: &[FieldInfo]) -> TokenStream2 {
    let inserts = fields.iter().map(|f| {
        let name = &f.name;
        let wrapper_name = &f.wrapper_name;
        quote! {
            store.insert(entity, #wrapper_name(self.#name.clone()));
        }
    });

    quote! { #(#inserts)* }
}

fn generate_pack_get(
    struct_name: &Ident,
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
) -> TokenStream2 {
    let field_reads = fields.iter().map(|f| {
        let name = &f.name;
        let wrapper_name = &f.wrapper_name;
        quote! {
            let #name = store.read_lane::<#wrapper_name>()?.get(entity)?.0.clone();
        }
    });

    let field_names = fields.iter().map(|f| &f.name);

    quote! {
        #(#field_reads)*
        Some(#struct_name #ty_generics { #(#field_names),* })
    }
}

fn generate_pack_get_mut(
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
    pack_mut_name: &Ident,
) -> TokenStream2 {
    let mut lane_guards = Vec::new();
    let mut guard_fields_init = Vec::new();

    for field in fields {
        let name = &field.name;
        let wrapper_name = &field.wrapper_name;
        let guard_name = format_ident!("_guard_{}", name);
        lane_guards.push(quote! {
            let #guard_name = store.write_lane::<#wrapper_name>()?;
        });
        guard_fields_init.push(quote! {
            #guard_name,
        });
    }

    quote! {
        #(#lane_guards)*
        Some(#pack_mut_name #ty_generics {
            #(#guard_fields_init)*
            _entity: entity,
        })
    }
}

fn generate_pack_mut_struct(
    _struct_name: &Ident,
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&WhereClause>,
    pack_mut_name: &Ident,
) -> (TokenStream2, TokenStream2) {
    let guard_fields = fields.iter().map(|f| {
        let name = &f.name;
        let wrapper_name = &f.wrapper_name;
        let guard_name = format_ident!("_guard_{}", name);
        quote! { #guard_name: std::sync::RwLockWriteGuard<'a, ornis_core::ComponentStore<#wrapper_name>>, }
    });

    let accessor_methods = fields.iter().map(|f| {
        let name = &f.name;
        let inner_ty = &f.lane_ty;
        let guard_name = format_ident!("_guard_{}", name);
        quote! {
            pub fn #name(&mut self) -> &mut #inner_ty {
                &mut self.#guard_name.get_mut(self._entity).unwrap().0
            }
        }
    });

    let struct_def = quote! {
        pub struct #pack_mut_name<'a, #ty_generics> #where_clause {
            #(#guard_fields)*
            _entity: ornis_core::Entity,
        }
    };

    let methods = quote! {
        impl<'a, #ty_generics> #pack_mut_name<'a, #ty_generics> #where_clause {
            #(#accessor_methods)*
        }
    };

    (struct_def, methods)
}
