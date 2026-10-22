//! UI test: `#[derive(AutoPipeline, Pack)]` cannot serve both derives
//! from one attribute — a `pack(...)` payload is Pack's GPU opt-in,
//! while AutoPipeline reads only bare `#[pack]`, so AutoPipeline
//! rejects the payload with a spanned error instead of letting its
//! inherent `pack_register` silently shadow `Pack`'s.

use ornis_macros::{AutoPipeline, Pack};

#[derive(Clone, AutoPipeline, Pack)]
#[pack(gpu)]
struct Both {
    x: f32,
}

fn main() {}
