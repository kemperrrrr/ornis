//! GPU timings for issue #201. The bench binary calls [`run_cull_ab`].
//!
//! Each sample is one compute dispatch, bracketed by timestamp queries
//! when the device has them. Host wall time is the submit plus the
//! blocking poll, measured beside the timestamps. Software adapters are
//! named in the header so their numbers are not read as discrete-GPU cost.

use std::time::Duration;
use std::time::Instant;

use super::super::CULL_COMPUTE_WGSL;
use super::ir::IrStats;
use super::ir::ir_stats;

mod clock;
mod dispatch;
mod pass;
mod pipeline;
mod readback;
mod session;
mod upload;

/// Dispatches the bench: IR table first, then GPU rows when an adapter exists.
///
/// The guest wasm build is inside this call. The bench prints the returned
/// text; this function does not write to stdout.
pub fn run_cull_ab() -> String {
    let lowered = load_lowered();
    let mut report = static_report(&lowered);
    if let Some(session) = pass::open() {
        report.push('\n');
        report.push_str(&pass::time_scenes(&session, &lowered.wgsl));
    }
    report
}

struct Lowered {
    wgsl: String,
    elapsed: Duration,
    saved: String,
}

fn load_lowered() -> Lowered {
    let started = Instant::now();
    let wgsl = ornis_wasm2gpu_testkit::cull::lowered_kernel();
    let saved = remember_lowered(&wgsl);
    Lowered {
        wgsl,
        elapsed: started.elapsed(),
        saved,
    }
}

fn remember_lowered(wgsl: &str) -> String {
    let path = std::env::temp_dir().join("ornis-issue-201-cull.wgsl");
    match std::fs::write(&path, wgsl) {
        Ok(()) => format!("lowered_wgsl_path: {}", path.display()),
        Err(err) => {
            eprintln!("lowered wgsl not saved: {err}");
            format!("lowered_wgsl_path: not saved ({err})")
        }
    }
}

fn static_report(lowered: &Lowered) -> String {
    let production = ir_stats(CULL_COMPUTE_WGSL);
    let wasm = ir_stats(&lowered.wgsl);
    let twin = ir_stats(pipeline::TWIN_WGSL);
    format!(
        "wasm2gpu cull A/B (issue #201)\n\
         {saved}\n\
         guest build + lift: {elapsed:?} (warm nested cargo of the guest plus the lift, not a cold translate)\n\
         IR rows count the compute entry and helper functions. Vertex and fragment stubs the translator requires are left out of those rows. spirv is the whole module.\n\
         {table}\
         {notes}",
        saved = lowered.saved,
        elapsed = lowered.elapsed,
        table = ir_table(&[
            ("production", &production),
            ("wasm2gpu", &wasm),
            ("hand_twin", &twin),
        ]),
        notes = ir_notes(&production, &wasm, &twin),
    )
}

fn ir_table(rows: &[(&str, &IrStats)]) -> String {
    let mut text = String::from(
        "shader\tbytes\tfuncs\tstmts\texprs\tif\tloop\tswitch\tstore\tatomic\tcall\tret\talu\tload\taccess\tspirv\twgx\n",
    );
    rows.iter()
        .for_each(|(name, stats)| text.push_str(&ir_row(name, stats)));
    text
}

fn ir_row(name: &str, stats: &IrStats) -> String {
    format!(
        "{name}\t{bytes}\t{funcs}\t{stmts}\t{exprs}\t{ifs}\t{loops}\t{switches}\t{stores}\t{atomics}\t{calls}\t{returns}\t{alu}\t{loads}\t{accesses}\t{spirv}\t{wgx}\n",
        bytes = stats.bytes,
        funcs = stats.functions,
        stmts = stats.statements,
        exprs = stats.expressions,
        ifs = stats.ifs,
        loops = stats.loops,
        switches = stats.switches,
        stores = stats.stores,
        atomics = stats.atomics,
        calls = stats.calls,
        returns = stats.returns,
        alu = stats.alu,
        loads = stats.loads,
        accesses = stats.accesses,
        spirv = spirv_cell(stats.spirv_ops),
        wgx = stats.workgroup_x,
    )
}

fn spirv_cell(ops: Option<u32>) -> String {
    ops.map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn ir_notes(production: &IrStats, wasm: &IrStats, twin: &IrStats) -> String {
    format!(
        "bytes wasm/prod {bytes:.2}  wasm/twin {twin_bytes:.2}\n\
         stmts wasm/prod {stmts:.2}  exprs wasm/prod {exprs:.2}  alu wasm/prod {alu:.2}\n\
         spirv wasm/prod {spirv}  spirv wasm/twin {spirv_twin}",
        bytes = ratio(wasm.bytes as f64, production.bytes as f64),
        twin_bytes = ratio(wasm.bytes as f64, twin.bytes as f64),
        stmts = ratio(wasm.statements as f64, production.statements as f64),
        exprs = ratio(wasm.expressions as f64, production.expressions as f64),
        alu = ratio(wasm.alu as f64, production.alu as f64),
        spirv = opt_ratio(wasm.spirv_ops, production.spirv_ops),
        spirv_twin = opt_ratio(wasm.spirv_ops, twin.spirv_ops),
    )
}

fn ratio(left: f64, right: f64) -> f64 {
    if right == 0.0 { 0.0 } else { left / right }
}

fn opt_ratio(left: Option<u32>, right: Option<u32>) -> String {
    match (left, right) {
        (Some(lhs), Some(rhs)) => format!("{:.2}", ratio(f64::from(lhs), f64::from(rhs))),
        _ => "n/a".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::run_cull_ab;
    use crate::gpu_culling::CULL_COMPUTE_WGSL;
    use crate::gpu_culling::shader_ab::ir::ir_stats;

    /// Keeps the bench entry in the call graph. Ignored because it builds
    /// the wasm guest and dispatches the full timing matrix.
    #[test]
    #[ignore = "full GPU A/B; use the wasm2gpu_cull_ab bench"]
    fn run_cull_ab_reaches_the_timing_path() {
        let stats = ir_stats(CULL_COMPUTE_WGSL);
        assert!(stats.atomics >= 1);
        run_cull_ab();
    }
}
