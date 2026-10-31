//! Instruction and branch counts after naga parses WGSL.
//!
//! Counts come from the naga IR (statements and expressions) and, when the
//! SPIR-V backend accepts the module, from the encoded instruction stream.
//! Source length is reported beside them because the naga WGSL printer
//! expands lets; the IR counts are the comparison that survives that printer.

use super::stats::Summary;

#[path = "walk.rs"]
mod walk;

/// Words in a SPIR-V header before the first instruction.
const SPIRV_HEADER_WORDS: usize = 5;
/// Shift of the word-count field in a SPIR-V instruction.
const SPIRV_WORD_COUNT_SHIFT: u32 = 16;

/// IR shape of one parsed shader. `spirv_ops` is `None` when validation or
/// the SPIR-V backend rejects the module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IrStats {
    /// Source length in bytes.
    pub bytes: usize,
    /// Non-entry functions plus entry points.
    pub functions: u32,
    /// Statements, including those nested in `if` / `loop` / `switch`.
    pub statements: u32,
    /// Expressions in every function arena.
    pub expressions: u32,
    /// `Statement::If` nodes.
    pub ifs: u32,
    /// `Statement::Loop` nodes.
    pub loops: u32,
    /// `Statement::Switch` nodes.
    pub switches: u32,
    /// `Statement::Store` nodes.
    pub stores: u32,
    /// Atomic statements (`Atomic` and `ImageAtomic`).
    pub atomics: u32,
    /// `Statement::Call` nodes.
    pub calls: u32,
    /// `Statement::Return` nodes.
    pub returns: u32,
    /// Binary, unary, math, and relational expressions.
    pub alu: u32,
    /// `Expression::Load` nodes.
    pub loads: u32,
    /// `Access` and `AccessIndex` expressions.
    pub accesses: u32,
    /// SPIR-V instructions after the header, or `None` if emit failed.
    pub spirv_ops: Option<u32>,
    /// Compute workgroup width of the first entry point (`0` when absent).
    pub workgroup_x: u32,
}

/// Parse `source` and count the IR. A parse error yields a zero record with
/// `spirv_ops = None` and the original byte length, so a broken shader is
/// visible as a row of zeros rather than a panic in the bench header.
pub(crate) fn ir_stats(source: &str) -> IrStats {
    naga::front::wgsl::parse_str(source)
        .map(|module| from_module(&module, source.len()))
        .unwrap_or_else(|_| IrStats::unparsed(source.len()))
}

impl IrStats {
    fn unparsed(bytes: usize) -> Self {
        Self {
            bytes,
            functions: 0,
            statements: 0,
            expressions: 0,
            ifs: 0,
            loops: 0,
            switches: 0,
            stores: 0,
            atomics: 0,
            calls: 0,
            returns: 0,
            alu: 0,
            loads: 0,
            accesses: 0,
            spirv_ops: None,
            workgroup_x: 0,
        }
    }
}

fn from_module(module: &naga::Module, bytes: usize) -> IrStats {
    let statements = walk::statement_counts(module);
    let expressions = walk::expression_counts(module);
    IrStats {
        bytes,
        functions: function_count(module),
        statements: statements.statements,
        expressions: expressions.expressions,
        ifs: statements.ifs,
        loops: statements.loops,
        switches: statements.switches,
        stores: statements.stores,
        atomics: statements.atomics,
        calls: statements.calls,
        returns: statements.returns,
        alu: expressions.alu,
        loads: expressions.loads,
        accesses: expressions.accesses,
        spirv_ops: spirv_ops(module),
        workgroup_x: workgroup_x(module),
    }
}

fn function_count(module: &naga::Module) -> u32 {
    walk::counted_functions(module)
}

fn workgroup_x(module: &naga::Module) -> u32 {
    walk::compute_workgroup_x(module)
}

fn spirv_ops(module: &naga::Module) -> Option<u32> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .ok()?;
    let words =
        naga::back::spv::write_vec(module, &info, &naga::back::spv::Options::default(), None)
            .ok()?;
    Some(encoded_ops(&words))
}

fn encoded_ops(words: &[u32]) -> u32 {
    let mut index = SPIRV_HEADER_WORDS;
    let mut ops = 0u32;
    while index < words.len() {
        let count = (words[index] >> SPIRV_WORD_COUNT_SHIFT) as usize;
        if count == 0 {
            break;
        }
        ops += 1;
        index += count;
    }
    ops
}

/// One markdown row. `summary` is accepted so the bench printer and this
/// walker share the quartile type without a second formatter owning it.
pub(crate) fn format_summary_us(summary: &Summary) -> String {
    format!(
        "{:.3} [{:.3}–{:.3}] iqr {:.3}",
        ns_to_us(summary.median),
        ns_to_us(summary.q1),
        ns_to_us(summary.q3),
        ns_to_us(summary.iqr),
    )
}

fn ns_to_us(ns: f64) -> f64 {
    ns / US_IN_NS
}

const US_IN_NS: f64 = 1_000.0;

#[cfg(test)]
mod tests {
    use super::ir_stats;
    use crate::gpu_culling::CULL_COMPUTE_WGSL;

    #[test]
    fn production_cull_has_a_loop_an_atomic_and_spirv() {
        let stats = ir_stats(CULL_COMPUTE_WGSL);
        assert!(stats.bytes > 1, "source bytes");
        assert!(stats.functions >= 1, "{stats:?}");
        assert!(stats.loops >= 1, "sphere_visible for-loop: {stats:?}");
        assert!(stats.ifs >= 1, "early-outs: {stats:?}");
        assert!(stats.atomics >= 1, "atomicAdd: {stats:?}");
        assert!(stats.stores >= 1, "struct copy: {stats:?}");
        assert!(stats.workgroup_x == crate::gpu_culling::CULL_WORKGROUP_SIZE);
        assert!(stats.spirv_ops.unwrap_or(0) > stats.statements);
    }

    #[test]
    fn compute_workgroup_ignores_an_earlier_vertex_entry() {
        let source = r#"
@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
@compute @workgroup_size(4, 1, 1)
fn cs() {}
"#;
        let stats = ir_stats(source);
        assert_eq!(stats.workgroup_x, 4);
        assert_eq!(stats.functions, 1);
    }

    #[test]
    fn unparsable_source_is_a_zero_row() {
        let stats = ir_stats("this is not wgsl");
        assert_eq!(stats.statements, 0);
        assert_eq!(stats.spirv_ops, None);
        assert_eq!(stats.bytes, "this is not wgsl".len());
    }
}
