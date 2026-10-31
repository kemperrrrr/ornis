//! Statement and expression tallies over a parsed naga module.
//!
//! Entry points keep their own function body, separate from
//! `module.functions`, so both arenas are walked. Nested blocks are
//! pushed onto an explicit stack: the walk does not recurse.

pub(super) struct StmtCounts {
    pub statements: u32,
    pub ifs: u32,
    pub loops: u32,
    pub switches: u32,
    pub stores: u32,
    pub atomics: u32,
    pub calls: u32,
    pub returns: u32,
}

pub(super) struct ExprCounts {
    pub expressions: u32,
    pub alu: u32,
    pub loads: u32,
    pub accesses: u32,
}

pub(super) fn statement_counts(module: &naga::Module) -> StmtCounts {
    let mut counts = StmtCounts {
        statements: 0,
        ifs: 0,
        loops: 0,
        switches: 0,
        stores: 0,
        atomics: 0,
        calls: 0,
        returns: 0,
    };
    let mut stack = Vec::new();
    push_bodies(module, &mut stack);
    drain_statements(&mut counts, &mut stack);
    counts
}

pub(super) fn expression_counts(module: &naga::Module) -> ExprCounts {
    let mut counts = ExprCounts {
        expressions: 0,
        alu: 0,
        loads: 0,
        accesses: 0,
    };
    module
        .functions
        .iter()
        .for_each(|(_, func)| add_function(&mut counts, func));
    counted_entries(module).for_each(|entry| add_function(&mut counts, &entry.function));
    counts
}

/// Non-entry functions plus the compute entry, when the module has one.
///
/// The translator's required vertex and fragment stubs sit in the same
/// module as `cs_main`. When a compute entry exists, only that entry is
/// counted, so the stubs do not inflate the kernel comparison.
pub(super) fn counted_functions(module: &naga::Module) -> u32 {
    let entries = counted_entries(module).count();
    (module.functions.len() + entries) as u32
}

/// Workgroup width of the compute entry. `0` when the module has none.
pub(super) fn compute_workgroup_x(module: &naga::Module) -> u32 {
    module
        .entry_points
        .iter()
        .find(|entry| entry.stage == naga::ShaderStage::Compute)
        .map(|entry| entry.workgroup_size[0])
        .unwrap_or(0)
}

fn counted_entries(module: &naga::Module) -> impl Iterator<Item = &naga::EntryPoint> {
    let compute = has_compute(module);
    module
        .entry_points
        .iter()
        .filter(move |entry| keep_entry(entry, compute))
}

fn has_compute(module: &naga::Module) -> bool {
    module
        .entry_points
        .iter()
        .any(|entry| entry.stage == naga::ShaderStage::Compute)
}

fn keep_entry(entry: &naga::EntryPoint, compute: bool) -> bool {
    !compute || entry.stage == naga::ShaderStage::Compute
}

fn push_bodies<'a>(module: &'a naga::Module, stack: &mut Vec<&'a naga::Block>) {
    for (_, func) in module.functions.iter() {
        stack.push(&func.body);
    }
    for entry in counted_entries(module) {
        stack.push(&entry.function.body);
    }
}

fn drain_statements(counts: &mut StmtCounts, stack: &mut Vec<&naga::Block>) {
    while let Some(block) = stack.pop() {
        for statement in block.iter() {
            note_statement(counts, stack, statement);
        }
    }
}

fn note_statement<'a>(
    counts: &mut StmtCounts,
    stack: &mut Vec<&'a naga::Block>,
    statement: &'a naga::Statement,
) {
    counts.statements += 1;
    match statement {
        naga::Statement::Emit(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::ImageStore { .. }
        | naga::Statement::WorkGroupUniformLoad { .. }
        | naga::Statement::RayQuery { .. }
        | naga::Statement::RayPipelineFunction(_)
        | naga::Statement::SubgroupBallot { .. }
        | naga::Statement::SubgroupGather { .. }
        | naga::Statement::SubgroupCollectiveOperation { .. }
        | naga::Statement::CooperativeStore { .. } => {}
        naga::Statement::Block(body) => stack.push(body),
        naga::Statement::If { accept, reject, .. } => {
            counts.ifs += 1;
            stack.push(accept);
            stack.push(reject);
        }
        naga::Statement::Switch { cases, .. } => {
            counts.switches += 1;
            cases.iter().for_each(|case| stack.push(&case.body));
        }
        naga::Statement::Loop {
            body, continuing, ..
        } => {
            counts.loops += 1;
            stack.push(body);
            stack.push(continuing);
        }
        naga::Statement::Return { .. } => counts.returns += 1,
        naga::Statement::Store { .. } => counts.stores += 1,
        naga::Statement::Atomic { .. } | naga::Statement::ImageAtomic { .. } => counts.atomics += 1,
        naga::Statement::Call { .. } => counts.calls += 1,
    }
}

fn add_function(counts: &mut ExprCounts, func: &naga::Function) {
    counts.expressions += func.expressions.len() as u32;
    for (_, expr) in func.expressions.iter() {
        match expr {
            naga::Expression::Binary { .. }
            | naga::Expression::Unary { .. }
            | naga::Expression::Math { .. }
            | naga::Expression::Relational { .. } => counts.alu += 1,
            naga::Expression::Load { .. } => counts.loads += 1,
            naga::Expression::Access { .. } | naga::Expression::AccessIndex { .. } => {
                counts.accesses += 1
            }
            _ => {}
        }
    }
}
