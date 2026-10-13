# ornis-shader-lang

Closed shader type world shared by the `#[kernel]` DSL: `ShaderType`
scalars/vectors/matrices, `ShaderBuiltin` math names with fixed arity,
the structural `ir` nodes and the `writer` that renders them to WGSL.

## `#[kernel]` type rule

`#[kernel]` (in `ornis-macros`) accepts only this closed world in
signatures, `let` annotations and `as` casts: `f32`/`u32`/`i32` (plus
pass-through `bool`), registry vectors/matrices (`Vec3`, `Mat4`, …),
transparent `u32` newtypes (`MaterialIdx`, `TextureHandle`) and
fixed-size arrays of those.

Everything else is rejected by the validator at macro time instead of
failing in naga: narrow/wide scalars (`u8`, `f64`, …), heap types,
closures, scalar-namespaced paths (`u8::MAX`), and in particular
user-defined `struct` types — interface and layout mirrors live outside
kernels. WGSL spellings for the accepted types come from
`ShaderType::wgsl`; see the crate docs (`src/lib.rs`) for details.
