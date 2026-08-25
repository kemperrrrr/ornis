use ornis_macros::gpu_pipeline;

// `#[gpu_pipeline]` only accepts `workgroup_size`, `storage`, `uniform`,
// `builtin` options; anything else is rejected.
#[gpu_pipeline(unknown_option)]
fn add_one(x: f32) -> f32 {
    x + 1.0
}

fn main() {}
