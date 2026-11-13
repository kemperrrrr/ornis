//! Textual LLVM IR -> small AST covering the spike-B subset.
//!
//! The parser accepts the shape stable `rustc` emits for a scalar compute
//! kernel after `SROA`: basic blocks, `phi`, conditional/unconditional
//! branches, scalar arithmetic/comparisons, and `ornis_gpu` ABI calls.
//! `phi`/`br` are structured by the lowerer into Naga `If`/`Loop`; linear
//! memory (`load`, `store`, `getelementptr`) and unsupported opcodes are
//! rejected with `Unsupported`, so an unknown shape never silently becomes
//! a wrong shader.

use crate::error::TranslateError;

/// One LLVM operation result name without the leading `%`.
pub(super) type Name = String;

/// A local name after inlining's renaming and parameter substitution.
///
/// The name is deliberately not `local`: the analyzer collects nested functions
/// into the project scope by name, and a short generic name there would make
/// unrelated call sites look like project calls.
fn namespaced(
    name: &str,
    prefix: &str,
    subst: &std::collections::HashMap<String, Operand>,
) -> Result<String, TranslateError> {
    match subst.get(name) {
        Some(Operand::Local(replacement)) => Ok(replacement.clone()),
        Some(_) => Err(TranslateError::Unsupported),
        None => Ok(format!("{prefix}{name}")),
    }
}

/// A scalar operand: a `%local` or a literal.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Operand {
    /// `%x` without the percent sign.
    Local(Name),
    /// Float literal.
    F32(f32),
    /// Unsigned integer literal.
    U32(u32),
    /// Iota of one bit.
    Bool(bool),
    /// A module symbol (`@alloc_…`, a string constant).
    ///
    /// It only appears as the location argument of a panic call, which the
    /// translator drops: a shader has no diagnostic buffer to write it to.
    Symbol(String),
}

/// Value type of an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    /// `float`.
    Float,
    /// `i32`.
    I32,
    /// `i1`.
    I1,
    /// `void`.
    Void,
    /// `ptr` (pointer, opaque).
    Ptr,
}

/// Where a pointer value starts.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum PtrBase {
    /// A module-level constant, by symbol name without `@`.
    Global(String),
    /// A local `alloca`, by SSA name.
    Local(Name),
}

/// One internal constant the module indexes through memory.
#[derive(Clone, Debug)]
pub(super) struct GlobalConst {
    /// Symbol name without `@`.
    pub(super) name: String,
    /// Raw contents, little-endian.
    pub(super) bytes: Vec<u8>,
}

/// One supported instruction.
#[derive(Clone, Debug)]
pub(super) enum Instr {
    /// `fadd float %a, %b`
    FAdd { dst: Name, a: Operand, b: Operand },
    /// `fmul float %a, %b`
    FMul { dst: Name, a: Operand, b: Operand },
    /// `fsub float %a, %b`
    FSub { dst: Name, a: Operand, b: Operand },
    /// `fdiv float %a, %b`
    FDiv { dst: Name, a: Operand, b: Operand },
    /// `fneg float %a`
    FNeg { dst: Name, a: Operand },
    /// `add i32 %a, %b`
    Add { dst: Name, a: Operand, b: Operand },
    /// `sub i32 %a, %b`
    Sub { dst: Name, a: Operand, b: Operand },
    /// `mul i32 %a, %b`
    Mul { dst: Name, a: Operand, b: Operand },
    /// `udiv i32 %a, %b`
    UDiv { dst: Name, a: Operand, b: Operand },
    /// `sdiv i32 %a, %b`
    SDiv { dst: Name, a: Operand, b: Operand },
    /// `urem i32 %a, %b`
    URem { dst: Name, a: Operand, b: Operand },
    /// `srem i32 %a, %b`
    SRem { dst: Name, a: Operand, b: Operand },
    /// `and i32 %a, %b`
    And { dst: Name, a: Operand, b: Operand },
    /// `or i32 %a, %b`
    Or { dst: Name, a: Operand, b: Operand },
    /// `xor i32 %a, %b`
    Xor { dst: Name, a: Operand, b: Operand },
    /// `shl i32 %a, %b`
    Shl { dst: Name, a: Operand, b: Operand },
    /// `lshr i32 %a, %b`
    LShr { dst: Name, a: Operand, b: Operand },
    /// `ashr i32 %a, %b`
    AShr { dst: Name, a: Operand, b: Operand },
    /// `fcmp <pred> float %a, %b`
    Fcmp {
        dst: Name,
        pred: String,
        a: Operand,
        b: Operand,
    },
    /// `icmp <pred> i32 %a, %b`
    Icmp {
        dst: Name,
        pred: String,
        a: Operand,
        b: Operand,
    },
    /// `select i1 %c, float %a, float %b`
    Select {
        dst: Name,
        c: Operand,
        a: Operand,
        b: Operand,
    },
    /// `uitofp i32 %a to float`.
    Uitofp { dst: Name, a: Operand },
    /// `sitofp i32 %a to float` — the value is a two's-complement `i32`.
    Sitofp { dst: Name, a: Operand },
    /// `fptoui float %a to i32`. Pixel coordinates arrive as `f32`.
    Fptoui { dst: Name, a: Operand },
    /// `bitcast i32 %a to float` / `bitcast float %a to i32`
    Bitcast {
        dst: Name,
        from: Ty,
        to: Ty,
        a: Operand,
    },
    /// `call float @name(...)`
    Call {
        dst: Name,
        callee: String,
        args: Vec<Operand>,
    },
    /// `call void @name(...)` without assignment.
    CallVoid { callee: String, args: Vec<Operand> },
    /// `ret float %a` (kept only for single-block compatibility; actual
    /// function exit is the block terminator)
    Ret { a: Operand },
    /// `ret void`
    RetVoid,
    /// `%x = phi float [ %a, %pred1 ], [ %b, %pred2 ]`
    Phi {
        dst: Name,
        ty: Ty,
        incoming: Vec<(Operand, String)>,
    },
    /// `%p = alloca [4 x float], align 4` — `bytes` is the allocation size.
    ///
    /// The allocation becomes a private array of 4-byte words; `bytes` is
    /// therefore a multiple of four.
    Alloca { dst: Name, bytes: u32 },
    /// `%p = getelementptr <ty>, ptr <base>, i32 <index>`.
    ///
    /// `stride` is the size of `<ty>` in bytes, so the pointer moves
    /// `index * stride` bytes. An aggregate type with an unknown layout is
    /// rejected at parse time.
    GetElementPtr {
        dst: Name,
        base: PtrBase,
        stride: u32,
        index: Operand,
    },
    /// `%x = load float, ptr %p`
    Load { dst: Name, ptr: Name, ty: Ty },
    /// `store float %v, ptr %p`
    Store { ptr: Name, value: Operand, ty: Ty },
}

/// Block terminator.
#[derive(Clone, Debug)]
pub(super) enum Terminator {
    /// Return with optional scalar value.
    Ret(Option<Operand>),
    /// Unconditional branch to a label.
    BrUncond(String),
    /// Conditional branch.
    BrCond {
        cond: Operand,
        true_label: String,
        false_label: String,
    },
    /// `unreachable`: the panic path of the kernel.
    ///
    /// The path is undefined behaviour in LLVM; the translator ends it with a
    /// `return` instead of pretending it cannot happen.
    Unreachable,
    /// `switch i32 %x, label %default [ i32 0, label %a i32 1, label %b ]`
    ///
    /// The selector is an `i32` whose bits the lowerer keeps as `u32`, so the
    /// case values are the same bits.
    Switch {
        /// Value the switch dispatches on.
        selector: Operand,
        /// Label taken when no case matches.
        default: String,
        /// Case value and its label, in IR order.
        cases: Vec<(u32, String)>,
    },
}

/// One basic block.
#[derive(Clone, Debug)]
pub(super) struct BasicBlock {
    /// Label without the trailing `:` (e.g. `start`, `bb2`).
    pub(super) label: String,
    /// Instructions before the terminator (including `phi` at the head).
    pub(super) instrs: Vec<Instr>,
    /// Terminator; `None` until closed (fallthrough to next block).
    pub(super) term: Terminator,
}

/// One LLVM function definition.
#[derive(Clone, Debug)]
pub(super) struct LlvmFunc {
    /// Function name without `@`.
    pub(super) name: String,
    /// Ret type: `float` or `void`.
    pub(super) ret_ty: Ty,
    /// Parameters: (type, name without `%`).
    pub(super) params: Vec<(Ty, Name)>,
    /// Basic blocks in declaration order.
    pub(super) blocks: Vec<BasicBlock>,
    /// Module-level constants the function can index.
    pub(super) globals: Vec<GlobalConst>,
}

/// Everything one module's functions say about themselves.
pub(super) struct Module {
    /// Functions that parsed, entries and helpers alike.
    pub(super) funcs: Vec<LlvmFunc>,
    /// Functions outside the subset and why, so a call can name the fix.
    pub(super) skipped: Vec<(String, String)>,
    /// Declared functions LLVM marks `noreturn` (the panic shims).
    pub(super) noreturn: Vec<String>,
}

/// Parse every function the module defines, entries and helpers alike.
///
/// A helper the kernel calls is lowered into the same Naga module; one that is
/// outside the subset is skipped here, with the reason, and only reported if a
/// kernel calls it.
pub(super) fn parse_module(text: &str) -> Result<Module, TranslateError> {
    let globals = parse_globals(text);
    let mut module = parse_header(text, &globals)?;
    module.noreturn = parse_noreturn(text);
    Ok(module)
}

/// Names of declared functions LLVM marks `noreturn`.
///
/// Rust's panic shims are external declarations with a `ptr` argument, so the
/// translator cannot lower them; `noreturn` is what says a call never comes
/// back, which is enough to drop the call and let the caller's `unreachable`
/// end the invocation.
fn parse_noreturn(text: &str) -> Vec<String> {
    let groups: Vec<(&str, bool)> = text
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("attributes #")?;
            let (group, body) = rest.split_once('=')?;
            Some((group.trim(), body.contains("noreturn")))
        })
        .collect();
    text.lines()
        .filter(|line| line.trim_start().starts_with("declare "))
        .filter_map(|line| {
            let name = line.find('@').map(|at| &line[at + 1..])?;
            let name = name.split('(').next()?.trim().trim_matches('"').to_string();
            let inline = line.rsplit(')').next().unwrap_or("").contains("noreturn");
            let group = line
                .split_whitespace()
                .last()
                .and_then(|last| last.strip_prefix('#'))
                .and_then(|id| groups.iter().find(|(group, _)| *group == id))
                .is_some_and(|(_, noreturn)| *noreturn);
            (inline || group).then_some(name)
        })
        .collect()
}

/// Whether `name` is a shader entry point rather than a helper.
pub(super) fn is_entry(name: &str) -> bool {
    name == ornis_gpu_abi::symbols::CS_MAIN
        || name == ornis_gpu_abi::symbols::VS_MAIN
        || name == ornis_gpu_abi::symbols::FS_MAIN
}

/// Decode one `c"…"` LLVM byte string, escapes included.
fn decode_byte_string(line: &str) -> Option<Vec<u8>> {
    let rest = line.strip_prefix("c\"")?;
    let mut bytes = Vec::new();
    let mut chars = rest.bytes();
    loop {
        match chars.next()? {
            b'"' => return Some(bytes),
            b'\\' => {
                let high = chars.next()?;
                let low = chars.next()?;
                let text = [high, low];
                let text = core::str::from_utf8(&text).ok()?;
                bytes.push(u8::from_str_radix(text, HEX_RADIX).ok()?);
            }
            byte => bytes.push(byte),
        }
    }
}

/// Module-level constants that hold a byte initializer.
fn parse_globals(text: &str) -> Vec<GlobalConst> {
    let mut globals = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('@') else {
            continue;
        };
        let Some((name, tail)) = rest.split_once('=') else {
            continue;
        };
        if !tail.contains("constant ") {
            continue;
        }
        let Some(start) = line.find("c\"") else {
            continue;
        };
        let Some(bytes) = decode_byte_string(&line[start..]) else {
            continue;
        };
        globals.push(GlobalConst {
            name: name.trim().trim_matches('"').to_string(),
            bytes,
        });
    }
    globals
}

/// Parse every `define ... { ... }` block.
///
/// A function outside the subset is skipped unless it is an entry point: a
/// helper the kernel never calls (`rust_begin_unwind`, a panic shim) must not
/// fail the module, while an entry that does not parse is an error.
fn parse_header(text: &str, globals: &[GlobalConst]) -> Result<Module, TranslateError> {
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if !trimmed.starts_with("define") {
            continue;
        }
        // The body is read first: it has to be consumed even when the header
        // is outside the subset.
        let mut body = Vec::new();
        loop {
            let Some(inner) = lines.next() else {
                return Err(TranslateError::Parse("unterminated function".to_string()));
            };
            if inner.trim() == "}" {
                break;
            }
            body.push(inner.to_string());
        }
        let parsed = parse_define_header(trimmed)
            .and_then(|(name, ret_ty, params)| parse_body(&name, ret_ty, &params, &body, globals));
        match parsed {
            Ok(func) => out.push(func),
            Err(error) if is_entry(&define_name(trimmed)) => return Err(error),
            Err(error) => skipped.push((define_name(trimmed), error.to_string())),
        }
    }
    if out.is_empty() {
        return Err(TranslateError::Parse("no define in llvm ir".to_string()));
    }
    Ok(Module {
        funcs: out,
        skipped,
        noreturn: Vec::new(),
    })
}

/// One terminator line, joining a `switch` case list that spans lines.
///
/// The list has to reach the parser as one line: rustc prints one `case` per
/// line and the closing `]` on the last one.
fn collect_terminator<'a>(
    first: &str,
    lines: &mut impl Iterator<Item = &'a String>,
) -> Result<String, TranslateError> {
    let mut term_line = first.to_string();
    if !term_line.starts_with("switch ") {
        return Ok(term_line);
    }
    while !term_line.contains(']') {
        let Some(next) = lines.next() else {
            return Err(TranslateError::Parse("unterminated switch".to_string()));
        };
        term_line.push(' ');
        term_line.push_str(next.split(';').next().unwrap_or("").trim());
    }
    Ok(term_line)
}

/// Parse one function body: blocks, instructions, and terminators.
/// Close a basic block: its label, the instructions collected so far, its term.
fn closed_block(label: String, instrs: Vec<Instr>, term: Terminator) -> BasicBlock {
    BasicBlock {
        label,
        instrs,
        term,
    }
}

fn parse_body(
    name: &str,
    ret_ty: Ty,
    params: &[(Ty, Name)],
    body: &[String],
    globals: &[GlobalConst],
) -> Result<LlvmFunc, TranslateError> {
    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut cur_label = "start".to_string();
    let mut cur_instrs: Vec<Instr> = Vec::new();
    let mut cur_term: Option<Terminator> = None;
    let mut saw_ret = false;
    let mut lines = body.iter();
    while let Some(line) = lines.next() {
        // Strip an inline comment (`; preds = %a, %b`) before parsing.
        let inner = line.split(';').next().unwrap_or("").trim();
        if inner.is_empty() {
            continue;
        }
        // A basic-block label.
        if inner.ends_with(':') && !inner.contains('=') {
            // Close previous block if needed.
            if let Some(term) = cur_term.take() {
                // previous block already terminated; start new
                let instrs = std::mem::take(&mut cur_instrs);
                blocks.push(closed_block(cur_label.clone(), instrs, term));
            } else if !cur_instrs.is_empty() {
                // Fallthrough without explicit terminator: synthesize ret.
                return Err(TranslateError::Unsupported);
            }
            cur_label = inner.trim_end_matches(':').to_string();
            continue;
        }
        let term_line = collect_terminator(inner, &mut lines)?;
        if let Some(term) = parse_term(&term_line)? {
            cur_term = Some(term);
            // `unreachable` ends a path just like `ret`: the panic shim is a
            // function whose only terminator is `unreachable`.
            saw_ret |= matches!(cur_term, Some(Terminator::Ret(_) | Terminator::Unreachable));
            // Terminator closes the block; the next label begins a new one.
            continue;
        }
        if let Some(instr) = parse_instr(inner)? {
            cur_instrs.push(instr);
        }
    }
    // Close the last block.
    if let Some(term) = cur_term.take() {
        saw_ret |= matches!(term, Terminator::Ret(_) | Terminator::Unreachable);
        blocks.push(closed_block(cur_label, cur_instrs, term));
    } else if !cur_instrs.is_empty() {
        return Err(TranslateError::Parse(
            "block without terminator".to_string(),
        ));
    }
    if !saw_ret {
        return Err(TranslateError::Parse("function without ret".to_string()));
    }
    Ok(LlvmFunc {
        name: name.to_string(),
        ret_ty,
        params: params.to_vec(),
        blocks,
        globals: globals.to_vec(),
    })
}

/// Name a `define` declares, without the `@`.
///
/// Best effort: a header the parser rejects still has to be reported by name.
fn define_name(line: &str) -> String {
    let Some(at) = line.find('@') else {
        return line.to_string();
    };
    let after = &line[at + 1..];
    after
        .find('(')
        .map_or(after, |open| &after[..open])
        .trim()
        .trim_matches('"')
        .to_string()
}

/// `define dso_local noundef float @cs_main(i32 noundef %id) ... {`
type Header = (String, Ty, Vec<(Ty, Name)>);

#[allow(clippy::type_complexity)]
fn parse_define_header(line: &str) -> Result<Header, TranslateError> {
    let at = line
        .find('@')
        .ok_or_else(|| TranslateError::Parse("define without @".to_string()))?;
    let after_at = &line[at + 1..];
    let open = after_at
        .find('(')
        .ok_or_else(|| TranslateError::Parse("define without (".to_string()))?;
    let name = after_at[..open].trim().trim_matches('"').to_string();
    // Ret type is the last word before the `@`, stripping linkage/attrs.
    let before_at = &line[..at];
    let mut words = before_at.split_whitespace().rev();
    let mut ret_ty = None;
    for w in words.by_ref() {
        // strip attributes that are not types (`hidden`, `dso_local`, …)
        if w == "float" || w == "i32" || w == "i1" || w == "void" {
            ret_ty = Some(parse_ty(w)?);
            break;
        }
        if w == "define" {
            break;
        }
    }
    let ret_ty =
        ret_ty.ok_or_else(|| TranslateError::Parse("define without ret type".to_string()))?;
    // Params inside parentheses (handle nested `captures(none)` parens).
    let close = find_matching_paren(after_at, open)
        .ok_or_else(|| TranslateError::Parse("define without )".to_string()))?;
    let inner = &after_at[open + 1..close];
    let mut params = Vec::new();
    if !inner.trim().is_empty() {
        for param in split_top(inner) {
            let param = param.trim();
            let mut pct = None;
            for token in param.split_whitespace() {
                let t = token.trim_matches(',');
                if t.starts_with('%') {
                    pct = Some(t.trim_start_matches('%').to_string());
                }
            }
            let first = param
                .split_whitespace()
                .next()
                .ok_or_else(|| TranslateError::Parse("empty param".to_string()))?;
            let ty = parse_ty(first)?;
            let name =
                pct.ok_or_else(|| TranslateError::Parse(format!("param without %: {param}")))?;
            params.push((ty, name));
        }
    }
    Ok((name, ret_ty, params))
}

/// Index of the `)` matching the `(` at `open`.
fn find_matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, ch) in text.char_indices() {
        if i < open {
            continue;
        }
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_ty(word: &str) -> Result<Ty, TranslateError> {
    match word {
        "float" => Ok(Ty::Float),
        "i32" => Ok(Ty::I32),
        "i1" => Ok(Ty::I1),
        "void" => Ok(Ty::Void),
        "ptr" => Ok(Ty::Ptr),
        _ => Err(TranslateError::Parse(format!("unknown llvm type {word}"))),
    }
}

fn parse_ret(rhs: &str) -> Result<Instr, TranslateError> {
    let rhs = rhs.trim();
    if rhs == "void" {
        return Ok(Instr::RetVoid);
    }
    let mut parts = rhs.splitn(2, ' ');
    let ty = parts
        .next()
        .ok_or(TranslateError::Parse("ret without type".to_string()))?;
    if ty == "void" {
        return Ok(Instr::RetVoid);
    }
    let _ = parse_ty(ty)?;
    let rest = parts
        .next()
        .ok_or(TranslateError::Parse("ret without value".to_string()))?;
    Ok(Instr::Ret {
        a: parse_operand(rest.trim())?,
    })
}

/// Parse a terminator line: `ret ...`, `br label ...`, `br i1 ..., label ..., label ...`.
fn parse_term(line: &str) -> Result<Option<Terminator>, TranslateError> {
    let trimmed = line.trim();
    if let Some(rhs) = trimmed.strip_prefix("ret ") {
        let instr = parse_ret(rhs)?;
        let value = match instr {
            Instr::Ret { a } => Some(a),
            Instr::RetVoid => None,
            _ => return Err(TranslateError::Unsupported),
        };
        return Ok(Some(Terminator::Ret(value)));
    }
    if let Some(rest) = trimmed.strip_prefix("br ") {
        let rest = rest.trim();
        // `br label %merge`
        if let Some(label) = rest.strip_prefix("label %") {
            let label = label.trim().trim_matches(',').to_string();
            return Ok(Some(Terminator::BrUncond(label)));
        }
        // `br i1 %c, label %then, label %else`
        let mut parts = rest.split(',');
        let cond_part = parts
            .next()
            .ok_or(TranslateError::Parse("br cond".to_string()))?;
        let cond_tokens: Vec<&str> = cond_part.split_whitespace().collect();
        if cond_tokens.len() < 2 || cond_tokens[0] != "i1" {
            return Err(TranslateError::Parse("br cond type".to_string()));
        }
        let cond = parse_operand(cond_tokens[1].trim())?;
        let true_part = parts
            .next()
            .ok_or(TranslateError::Parse("br true".to_string()))?;
        let false_part = parts
            .next()
            .ok_or(TranslateError::Parse("br false".to_string()))?;
        let parse_label = |s: &str| -> Result<String, TranslateError> {
            let t = s.trim();
            let t = t.strip_prefix("label %").unwrap_or(t);
            Ok(t.trim().trim_matches('"').to_string())
        };
        return Ok(Some(Terminator::BrCond {
            cond,
            true_label: parse_label(true_part)?,
            false_label: parse_label(false_part)?,
        }));
    }
    if let Some(rest) = trimmed.strip_prefix("switch ") {
        return Ok(Some(parse_switch(rest)?));
    }
    if trimmed == "unreachable" {
        return Ok(Some(Terminator::Unreachable));
    }
    Ok(None)
}

/// `i32 %x, label %default [ i32 0, label %a i32 1, label %b ]`.
///
/// The case list is scanned token by token: `i32 <value>, label %<name>`
/// pairs are separated by whitespace, and the last one may or may not carry a
/// trailing comma, which is how rustc and hand-written IR both spell it.
fn parse_switch(rest: &str) -> Result<Terminator, TranslateError> {
    let (head, cases) = rest
        .split_once('[')
        .ok_or_else(|| TranslateError::Parse("switch without case list".to_string()))?;
    let cases = cases
        .trim_end()
        .strip_suffix(']')
        .ok_or_else(|| TranslateError::Parse("switch case list is not closed".to_string()))?;
    let mut head = head.trim().splitn(2, ',');
    let selector = head
        .next()
        .and_then(|part| part.split_whitespace().nth(1))
        .ok_or_else(|| TranslateError::Parse("switch selector".to_string()))?;
    let selector = parse_operand(selector)?;
    let default = head
        .next()
        .ok_or_else(|| TranslateError::Parse("switch default".to_string()))?;
    let mut parsed = Vec::new();
    let mut tokens = cases.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "i32" {
            return Err(TranslateError::Parse(format!(
                "switch case starts with `{token}`"
            )));
        }
        let value = tokens
            .next()
            .ok_or_else(|| TranslateError::Parse("switch case value".to_string()))?;
        let value: i32 = value
            .trim_end_matches(',')
            .parse()
            .map_err(|_| TranslateError::Parse(format!("switch case value `{value}`")))?;
        if tokens.next() != Some("label") {
            return Err(TranslateError::Parse(
                "switch case target is not a label".to_string(),
            ));
        }
        let label = tokens
            .next()
            .ok_or_else(|| TranslateError::Parse("switch case label".to_string()))?;
        let label = label
            .strip_prefix('%')
            .ok_or_else(|| TranslateError::Parse("switch case label".to_string()))?;
        parsed.push((value as u32, label.to_string()));
    }
    Ok(Terminator::Switch {
        selector,
        default: switch_label(default)?,
        cases: parsed,
    })
}

/// `label %name` → `name`.
fn switch_label(text: &str) -> Result<String, TranslateError> {
    let label = text
        .trim()
        .strip_prefix("label %")
        .ok_or_else(|| TranslateError::Parse("switch label".to_string()))?;
    Ok(label.trim().to_string())
}

/// `alloca [4 x float], align 4` → the allocation size in bytes.
fn parse_alloca(dst: &str, rhs: &str) -> Result<Instr, TranslateError> {
    let rest = rhs
        .strip_prefix("alloca ")
        .ok_or_else(|| TranslateError::Parse("alloca without a type".to_string()))?;
    let ty = rest.split(',').next().unwrap_or(rest).trim();
    let bytes = type_bytes(ty)?;
    Ok(Instr::Alloca {
        dst: dst.to_string(),
        bytes,
    })
}

/// `getelementptr inbounds nuw [4 x float], ptr @G, i32 %i`.
fn parse_gep(dst: &str, rhs: &str) -> Result<Instr, TranslateError> {
    let (ty, base, index) = split_gep(rhs)?;
    Ok(Instr::GetElementPtr {
        dst: dst.to_string(),
        base: pointer_base(&base)?,
        stride: type_bytes(&ty)?,
        index: parse_operand(&index)?,
    })
}

/// `(source type, base, index)` of a `getelementptr`.
fn split_gep(rhs: &str) -> Result<(String, String, String), TranslateError> {
    let rest = rhs
        .strip_prefix("getelementptr")
        .ok_or_else(|| TranslateError::Parse("getelementptr".to_string()))?;
    // Drop the `inbounds`/`nuw`/`nusw` markers before the source type.
    let rest = rest.trim();
    let rest = rest
        .strip_prefix("inbounds ")
        .unwrap_or(rest)
        .trim_start_matches("nuw ")
        .trim_start_matches("nusw ")
        .trim();
    let (ty, tail) = rest
        .split_once(',')
        .ok_or_else(|| TranslateError::Parse("getelementptr without a base".to_string()))?;
    let tail = tail.trim();
    let tail = tail
        .strip_prefix("ptr ")
        .ok_or_else(|| TranslateError::Parse("getelementptr base is not a pointer".to_string()))?;
    let (base, index) = tail
        .split_once(',')
        .ok_or_else(|| TranslateError::Parse("getelementptr without an index".to_string()))?;
    let index = index
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| TranslateError::Parse("getelementptr index".to_string()))?;
    Ok((
        ty.trim().to_string(),
        base.trim().to_string(),
        index.to_string(),
    ))
}

/// `@name` / `%name` of a `getelementptr` base.
fn pointer_base(base: &str) -> Result<PtrBase, TranslateError> {
    if let Some(name) = base.strip_prefix('@') {
        return Ok(PtrBase::Global(name.trim_matches('"').to_string()));
    }
    if let Some(name) = base.strip_prefix('%') {
        return Ok(PtrBase::Local(name.to_string()));
    }
    Err(TranslateError::Parse(format!(
        "getelementptr base `{base}` is not a symbol"
    )))
}

/// A line without a result: `store`, `call void`, or nothing this subset knows.
fn parse_void_line(line: &str) -> Result<Option<Instr>, TranslateError> {
    if let Some(rest) = line.strip_prefix("store ") {
        return Ok(Some(parse_store(rest)?));
    }
    if line.starts_with("call ") {
        let (callee, args) = parse_call(line)?;
        return Ok(Some(Instr::CallVoid { callee, args }));
    }
    if line.starts_with("load ") || line.contains("getelementptr") || line.contains("call") {
        return Err(TranslateError::Unsupported);
    }
    Ok(None)
}

/// `load float, ptr %p, align 4`.
fn parse_load(dst: &str, rhs: &str) -> Result<Instr, TranslateError> {
    let rest = rhs
        .strip_prefix("load ")
        .ok_or_else(|| TranslateError::Parse("load".to_string()))?;
    let (ty, tail) = rest
        .split_once(',')
        .ok_or_else(|| TranslateError::Parse("load without a pointer".to_string()))?;
    let ty = parse_ty(ty.trim())?;
    let ptr = tail
        .trim()
        .strip_prefix("ptr ")
        .ok_or_else(|| TranslateError::Parse("load source is not a pointer".to_string()))?;
    let ptr = ptr
        .split(',')
        .next()
        .unwrap_or(ptr)
        .trim()
        .trim_start_matches('%')
        .to_string();
    Ok(Instr::Load {
        dst: dst.to_string(),
        ptr,
        ty,
    })
}

/// `store float %v, ptr %p, align 4`.
fn parse_store(rhs: &str) -> Result<Instr, TranslateError> {
    let (ty, tail) = rhs
        .split_once(',')
        .ok_or_else(|| TranslateError::Parse("store without a pointer".to_string()))?;
    let mut parts = ty.split_whitespace();
    let ty = parse_ty(
        parts
            .next()
            .ok_or_else(|| TranslateError::Parse("store without a type".to_string()))?,
    )?;
    let value = parts
        .next()
        .ok_or_else(|| TranslateError::Parse("store without a value".to_string()))?;
    let ptr = tail
        .trim()
        .strip_prefix("ptr ")
        .ok_or_else(|| TranslateError::Parse("store target is not a pointer".to_string()))?;
    let ptr = ptr
        .split(',')
        .next()
        .unwrap_or(ptr)
        .trim()
        .trim_start_matches('%')
        .to_string();
    Ok(Instr::Store {
        ptr,
        value: parse_operand(value)?,
        ty,
    })
}

/// Byte size of the scalar types the memory subset accepts.
const WORD_BYTES: u32 = 4;
/// Byte size of a 16-bit scalar.
const HALF_WORD_BYTES: u32 = 2;
/// Byte size of an 8-bit scalar.
const BYTE_BYTES: u32 = 1;

/// Size of a type the translator can lay out, in bytes.
///
/// Scalars are one, two, or four bytes, `[N x T]` is `N` times `T`, and
/// `<N x T>` is a vector of `N` lanes. A struct or an opaque pointer has no
/// layout here. The loop walks nested aggregates without recursing, so a
/// deeply nested type cannot overflow the stack.
fn type_bytes(ty: &str) -> Result<u32, TranslateError> {
    let mut rest = ty.trim();
    let mut total = 1u32;
    loop {
        let (inner, close) = match rest.strip_prefix('[') {
            Some(inner) => (inner, ']'),
            None => match rest.strip_prefix('<') {
                Some(inner) => (inner, '>'),
                None => {
                    let size = match rest {
                        "float" | "i32" | "u32" => WORD_BYTES,
                        "i16" | "u16" => HALF_WORD_BYTES,
                        "i8" | "u8" => BYTE_BYTES,
                        _ => return Err(TranslateError::Unsupported),
                    };
                    return total
                        .checked_mul(size)
                        .ok_or_else(|| TranslateError::Parse(format!("type `{ty}` overflows")));
                }
            },
        };
        let (count, element) = inner
            .split_once(" x ")
            .ok_or_else(|| TranslateError::Parse(format!("aggregate type `{ty}`")))?;
        let count: u32 = count
            .trim()
            .parse()
            .map_err(|_| TranslateError::Parse(format!("length in `{ty}`")))?;
        total = total
            .checked_mul(count)
            .ok_or_else(|| TranslateError::Parse(format!("type `{ty}` overflows")))?;
        rest = element
            .trim_end()
            .strip_suffix(close)
            .ok_or_else(|| TranslateError::Parse(format!("aggregate type `{ty}`")))?
            .trim();
    }
}

/// `%x = phi float [ %a, %pred ], [ 1.0, %pred2 ]`
fn parse_phi(rhs: &str, ty: Ty) -> Result<Vec<(Operand, String)>, TranslateError> {
    let mut out = Vec::new();
    for part in split_top(rhs) {
        let part = part.trim().trim_start_matches('[').trim_end_matches(']');
        if part.is_empty() {
            continue;
        }
        let mut sub = part.split(',');
        let value = sub
            .next()
            .ok_or(TranslateError::Parse("phi val".to_string()))?;
        let pred = sub
            .next()
            .ok_or(TranslateError::Parse("phi pred".to_string()))?;
        let pred = pred.trim().trim_start_matches('%').trim().to_string();
        out.push((parse_typed_operand(value.trim(), ty)?, pred));
    }
    if out.is_empty() {
        return Err(TranslateError::Parse("phi without incoming".to_string()));
    }
    Ok(out)
}

fn parse_instr(line: &str) -> Result<Option<Instr>, TranslateError> {
    // `call void @name(...)` without assignment.
    let stripped = line.strip_prefix("tail ").unwrap_or(line).trim();
    if stripped.starts_with("call ") {
        let (callee, args) = parse_call(stripped)?;
        return Ok(Some(Instr::CallVoid { callee, args }));
    }
    // `%dst = ...`
    let Some((lhs, rhs)) = line.split_once('=') else {
        return parse_void_line(stripped);
    };
    let dst = lhs.trim().trim_start_matches('%').to_string();
    let rhs = rhs.trim();
    // Strip `tail` marker.
    let rhs = rhs.strip_prefix("tail ").unwrap_or(rhs).trim();
    if rhs.starts_with("call ") {
        let (callee, args) = parse_call(rhs)?;
        return Ok(Some(Instr::Call { dst, callee, args }));
    }
    let tokens: Vec<String> = rhs.split_whitespace().map(String::from).collect();
    if tokens.is_empty() {
        return Err(TranslateError::Parse("empty instruction".to_string()));
    }
    match tokens[0].as_str() {
        "alloca" => Ok(Some(parse_alloca(&dst, rhs)?)),
        "getelementptr" => Ok(Some(parse_gep(&dst, rhs)?)),
        "load" => Ok(Some(parse_load(&dst, rhs)?)),
        "phi" => {
            let ty = parse_ty(&tokens[1])?;
            // `parse_phi` expects the tail after `phi <type>`.
            let tail = rhs.split_whitespace().skip(2).collect::<Vec<_>>().join(" ");
            let incoming = parse_phi(&tail, ty)?;
            Ok(Some(Instr::Phi { dst, ty, incoming }))
        }
        "fadd" | "fmul" | "fsub" | "fdiv" => {
            let (a, b) = parse_bin_operands(&tokens)?;
            let op = |a: Operand, b: Operand| match tokens[0].as_str() {
                "fadd" => Instr::FAdd {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "fmul" => Instr::FMul {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "fsub" => Instr::FSub {
                    dst: dst.clone(),
                    a,
                    b,
                },
                _ => Instr::FDiv {
                    dst: dst.clone(),
                    a,
                    b,
                },
            };
            Ok(Some(op(a, b)))
        }
        "fneg" => {
            let a = parse_operand(tokens[operand_index(&tokens, 1)?].trim_end_matches(','))?;
            Ok(Some(Instr::FNeg { dst, a }))
        }
        "add" | "sub" | "mul" | "udiv" | "sdiv" | "urem" | "srem" | "and" | "or" | "xor"
        | "shl" | "lshr" | "ashr" => {
            let (a, b) = parse_bin_operands(&tokens)?;
            let op = |a: Operand, b: Operand| match tokens[0].as_str() {
                "add" => Instr::Add {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "sub" => Instr::Sub {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "mul" => Instr::Mul {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "udiv" => Instr::UDiv {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "sdiv" => Instr::SDiv {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "urem" => Instr::URem {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "srem" => Instr::SRem {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "and" => Instr::And {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "or" => Instr::Or {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "xor" => Instr::Xor {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "shl" => Instr::Shl {
                    dst: dst.clone(),
                    a,
                    b,
                },
                "lshr" => Instr::LShr {
                    dst: dst.clone(),
                    a,
                    b,
                },
                _ => Instr::AShr {
                    dst: dst.clone(),
                    a,
                    b,
                },
            };
            Ok(Some(op(a, b)))
        }
        "fcmp" => {
            let (pred, a, b) = parse_compare(&tokens)?;
            Ok(Some(Instr::Fcmp { dst, pred, a, b }))
        }
        "icmp" => {
            let (pred, a, b) = parse_compare(&tokens)?;
            Ok(Some(Instr::Icmp { dst, pred, a, b }))
        }
        "select" => {
            // `select i1 %c, float %a, float %b`
            let value_ty = parse_ty(&tokens[3])?;
            let c = parse_operand(tokens[2].trim_end_matches(','))?;
            let a = parse_typed_operand(tokens[4].trim_end_matches(','), value_ty)?;
            let b = parse_typed_operand(tokens[6].trim_end_matches(','), value_ty)?;
            Ok(Some(Instr::Select { dst, c, a, b }))
        }
        "uitofp" | "sitofp" => {
            // `uitofp nneg i32 %a to float` — rustc marks unsigned casts `nneg`.
            let a = parse_operand(tokens[operand_index(&tokens, 1)?].trim_end_matches(','))?;
            if tokens[0] == "sitofp" {
                return Ok(Some(Instr::Sitofp { dst, a }));
            }
            Ok(Some(Instr::Uitofp { dst, a }))
        }
        "fptoui" => {
            let a = parse_operand(tokens[operand_index(&tokens, 1)?].trim_end_matches(','))?;
            Ok(Some(Instr::Fptoui { dst, a }))
        }
        "zext" => parse_zext(&dst, &tokens),
        "bitcast" => {
            // `bitcast i32 %a to float`
            let from = parse_ty(&tokens[1])?;
            let a = parse_operand(tokens[2].trim_end_matches(','))?;
            let to = parse_ty(&tokens[4])?;
            Ok(Some(Instr::Bitcast { dst, from, to, a }))
        }
        "br" | "switch" => Err(TranslateError::Unsupported),
        "ptrtoint" | "inttoptr" => Err(TranslateError::UnsupportedType(
            "a GPU buffer address is not a number: `ptrtoint`/`inttoptr` are outside the \
             subset; index the buffer by element instead of casting a pointer"
                .to_string(),
        )),
        // Any other opcode is outside the translated subset.
        _ => Err(TranslateError::Unsupported),
    }
}

/// Instruction flags rustc/LLVM may print between an opcode and its type.
fn is_flag(token: &str) -> bool {
    matches!(
        token.trim_end_matches(','),
        "nuw"
            | "nsw"
            | "exact"
            | "fast"
            | "nnan"
            | "ninf"
            | "nsz"
            | "arcp"
            | "contract"
            | "afn"
            | "reassoc"
            | "samesign"
            | "disjoint"
            | "nneg"
    )
}

fn is_type_token(token: &str) -> bool {
    matches!(
        token.trim_end_matches(','),
        "float" | "i1" | "i32" | "void" | "ptr"
    )
}

/// Index of the first operand token after `from`, skipping flags and types.
fn operand_index(tokens: &[String], from: usize) -> Result<usize, TranslateError> {
    let mut index = from;
    while index < tokens.len() && (is_flag(&tokens[index]) || is_type_token(&tokens[index])) {
        index += 1;
    }
    if index >= tokens.len() {
        return Err(TranslateError::Parse(
            "instruction without operand".to_string(),
        ));
    }
    Ok(index)
}

fn parse_bin_operands(tokens: &[String]) -> Result<(Operand, Operand), TranslateError> {
    let index = operand_index(tokens, 1)?;
    let a = parse_operand(tokens[index].trim_end_matches(','))?;
    let b = parse_operand(
        tokens
            .get(index + 1)
            .ok_or(TranslateError::Parse("binary without rhs".to_string()))?
            .trim_end_matches(','),
    )?;
    Ok((a, b))
}

/// `fcmp <flags> <pred> <type> %a, %b` / `icmp <flags> <pred> <type> …`.
fn parse_compare(tokens: &[String]) -> Result<(String, Operand, Operand), TranslateError> {
    let mut index = 1;
    while index < tokens.len() && is_flag(&tokens[index]) {
        index += 1;
    }
    let pred = tokens
        .get(index)
        .ok_or(TranslateError::Parse(
            "compare without predicate".to_string(),
        ))?
        .clone();
    index += 1;
    while index < tokens.len() && (is_flag(&tokens[index]) || is_type_token(&tokens[index])) {
        index += 1;
    }
    let a = parse_operand(
        tokens
            .get(index)
            .ok_or(TranslateError::Parse("compare without lhs".to_string()))?
            .trim_end_matches(','),
    )?;
    let b = parse_operand(
        tokens
            .get(index + 1)
            .ok_or(TranslateError::Parse("compare without rhs".to_string()))?
            .trim_end_matches(','),
    )?;
    Ok((pred, a, b))
}

/// Parse `call float @name(float %a, float %b)`.
fn parse_call(rhs: &str) -> Result<(String, Vec<Operand>), TranslateError> {
    let at = rhs
        .find('@')
        .ok_or(TranslateError::Parse("call without @".to_string()))?;
    let after = &rhs[at + 1..];
    let open = after
        .find('(')
        .ok_or(TranslateError::Parse("call without (".to_string()))?;
    let callee = after[..open].trim().trim_matches('"').to_string();
    let close = after
        .rfind(')')
        .ok_or(TranslateError::Parse("call without )".to_string()))?;
    let inner = &after[open + 1..close];
    let mut args = Vec::new();
    if inner.trim().is_empty() {
        return Ok((callee, args));
    }
    for arg in split_top(inner) {
        // `i32 noundef %id` and `float 1.0`: the value is the last token.
        if let Some(value) = arg.split_whitespace().next_back() {
            args.push(parse_operand(value)?);
        }
    }
    Ok((callee, args))
}

/// An operand whose `poison`/`undef` becomes a zero of the expected type.
///
/// LLVM marks values it proved unobservable as `poison`; a shader has to pick
/// something, and any value is as good as another because observing one is
/// undefined behaviour in LLVM too.
fn parse_typed_operand(word: &str, ty: Ty) -> Result<Operand, TranslateError> {
    let word = word.trim().trim_matches(',');
    if word != "poison" && word != "undef" {
        return parse_operand(word);
    }
    Ok(match ty {
        Ty::Float => Operand::F32(0.0),
        Ty::I1 => Operand::Bool(false),
        _ => Operand::U32(0),
    })
}

fn parse_operand(word: &str) -> Result<Operand, TranslateError> {
    let w = word.trim().trim_matches(',');
    if let Some(name) = w.strip_prefix('%') {
        return Ok(Operand::Local(name.to_string()));
    }
    if let Some(name) = w.strip_prefix('@') {
        return Ok(Operand::Symbol(name.trim_matches('"').to_string()));
    }
    if w == "true" {
        return Ok(Operand::Bool(true));
    }
    if w == "false" {
        return Ok(Operand::Bool(false));
    }
    if let Some(value) = llvm_hex_f32(w) {
        return Ok(Operand::F32(value));
    }
    if let Ok(value) = w.parse::<f32>()
        && (w.contains('.') || w.contains('e') || w.contains('E'))
    {
        return Ok(Operand::F32(value));
    }
    if let Ok(value) = w.parse::<i64>() {
        // LLVM prints i32 masks as negative decimal; keep the bit pattern.
        return Ok(Operand::U32(value as u32));
    }
    Err(TranslateError::Parse(format!("unknown operand {w}")))
}

/// Hex digits in the `f64` pattern LLVM prints for an `f32` immediate.
const LLVM_F32_HEX_DIGITS: usize = 16;
/// Radix of that pattern.
const HEX_RADIX: u32 = 16;

/// Token index of the source type in `zext <ty> <value> to <ty>`.
const ZEXT_FROM: usize = 1;
/// Token index of the widened value.
const ZEXT_VALUE: usize = 2;
/// Token index of the destination type (`to` sits between the value and it).
const ZEXT_TO: usize = 4;

/// `zext i1 %c to i32`: rustc counts a bool by widening it to 0 or 1.
fn parse_zext(dst: &str, tokens: &[String]) -> Result<Option<Instr>, TranslateError> {
    let (Some(from_tok), Some(value_tok), Some(to_tok)) = (
        tokens.get(ZEXT_FROM),
        tokens.get(ZEXT_VALUE),
        tokens.get(ZEXT_TO),
    ) else {
        return Err(TranslateError::Parse("zext".to_string()));
    };
    let from = parse_ty(from_tok)?;
    let value = parse_operand(value_tok.trim_end_matches(','))?;
    let to = parse_ty(to_tok)?;
    if from != Ty::I1 || to != Ty::I32 {
        return Err(TranslateError::Unsupported);
    }
    Ok(Some(Instr::Select {
        dst: dst.to_string(),
        c: value,
        a: Operand::U32(1),
        b: Operand::U32(0),
    }))
}

/// LLVM prints an `f32` immediate as the 64-bit pattern of that value widened
/// to `f64` (`0x401ABD0AE0000000`). Narrowing recovers the original `f32`.
fn llvm_hex_f32(word: &str) -> Option<f32> {
    let hex = word.strip_prefix("0x")?;
    if hex.len() != LLVM_F32_HEX_DIGITS || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let bits = u64::from_str_radix(hex, HEX_RADIX).ok()?;
    Some(f64::from_bits(bits) as f32)
}

impl Instr {
    /// Rewrite the names an inlined body defines and the values it reads.
    ///
    /// `prefix` namespaces every local of the inlined function so it cannot
    /// collide with the caller's, and `subst` replaces the inlined parameters
    /// with the arguments at the call site (a parameter is never reassigned, so
    /// substitution is sound).
    ///
    /// # Errors
    ///
    /// [`TranslateError::Unsupported`] when a parameter the body dereferences
    /// was passed as something other than a local name, because only a name can
    /// be a pointer here.
    // qual:allow(srp, nms) reason: "the method rewrites the instruction it is called on; the mutation happens through the destructured fields, which the needless-&mut-self check does not see"
    pub(super) fn inline_rename(
        &mut self,
        prefix: &str,
        subst: &std::collections::HashMap<String, Operand>,
    ) -> Result<(), TranslateError> {
        let mut apply = |operand: &mut Operand| -> Result<(), TranslateError> {
            let Operand::Local(name) = operand else {
                return Ok(());
            };
            *name = namespaced(name, prefix, subst)?;
            Ok(())
        };
        let define = |dst: &mut String| *dst = format!("{prefix}{dst}");
        let pointer = |name: &mut String| -> Result<(), TranslateError> {
            *name = namespaced(name, prefix, subst)?;
            Ok(())
        };
        match self {
            Self::FAdd { dst, a, b }
            | Self::FMul { dst, a, b }
            | Self::FSub { dst, a, b }
            | Self::FDiv { dst, a, b }
            | Self::Add { dst, a, b }
            | Self::Sub { dst, a, b }
            | Self::Mul { dst, a, b }
            | Self::UDiv { dst, a, b }
            | Self::SDiv { dst, a, b }
            | Self::URem { dst, a, b }
            | Self::SRem { dst, a, b }
            | Self::And { dst, a, b }
            | Self::Or { dst, a, b }
            | Self::Xor { dst, a, b }
            | Self::Shl { dst, a, b }
            | Self::LShr { dst, a, b }
            | Self::AShr { dst, a, b }
            | Self::Fcmp { dst, a, b, .. }
            | Self::Icmp { dst, a, b, .. } => {
                define(dst);
                apply(a)?;
                apply(b)?;
            }
            Self::FNeg { dst, a }
            | Self::Uitofp { dst, a }
            | Self::Sitofp { dst, a }
            | Self::Fptoui { dst, a }
            | Self::Bitcast { dst, a, .. } => {
                define(dst);
                apply(a)?;
            }
            Self::Select { dst, c, a, b } => {
                define(dst);
                apply(c)?;
                apply(a)?;
                apply(b)?;
            }
            Self::Call { dst, args, .. } => {
                define(dst);
                args.iter_mut().try_for_each(&mut apply)?;
            }
            Self::CallVoid { args, .. } => args.iter_mut().try_for_each(&mut apply)?,
            Self::Ret { a } => apply(a)?,
            Self::RetVoid => {}
            Self::Phi { dst, incoming, .. } => {
                define(dst);
                for (value, _) in incoming {
                    apply(value)?;
                }
            }
            Self::Alloca { dst, .. } => define(dst),
            Self::GetElementPtr {
                dst, base, index, ..
            } => {
                define(dst);
                apply(index)?;
                if let PtrBase::Local(name) = base {
                    pointer(name)?;
                }
            }
            Self::Load { dst, ptr, .. } => {
                define(dst);
                pointer(ptr)?;
            }
            Self::Store { ptr, value, .. } => {
                pointer(ptr)?;
                apply(value)?;
            }
        }
        Ok(())
    }

    /// Every value this instruction reads.
    pub(super) fn each_operand<'a>(&'a self, mut visit: impl FnMut(&'a Operand)) {
        match self {
            Self::FAdd { a, b, .. }
            | Self::FMul { a, b, .. }
            | Self::FSub { a, b, .. }
            | Self::FDiv { a, b, .. }
            | Self::Add { a, b, .. }
            | Self::Sub { a, b, .. }
            | Self::Mul { a, b, .. }
            | Self::UDiv { a, b, .. }
            | Self::SDiv { a, b, .. }
            | Self::URem { a, b, .. }
            | Self::SRem { a, b, .. }
            | Self::And { a, b, .. }
            | Self::Or { a, b, .. }
            | Self::Xor { a, b, .. }
            | Self::Shl { a, b, .. }
            | Self::LShr { a, b, .. }
            | Self::AShr { a, b, .. }
            | Self::Fcmp { a, b, .. }
            | Self::Icmp { a, b, .. } => visit_pair(a, b, &mut visit),
            Self::FNeg { a, .. }
            | Self::Uitofp { a, .. }
            | Self::Sitofp { a, .. }
            | Self::Fptoui { a, .. }
            | Self::Bitcast { a, .. }
            | Self::Ret { a } => {
                visit(a);
            }
            Self::Select { c, a, b, .. } => visit_triple(c, a, b, &mut visit),
            Self::GetElementPtr { index, .. } => visit(index),
            Self::Store { value, .. } => visit(value),
            Self::Alloca { .. } | Self::Load { .. } => {}
            Self::Call { args, .. } | Self::CallVoid { args, .. } => {
                args.iter().for_each(&mut visit);
            }
            Self::Phi { incoming, .. } => {
                incoming.iter().for_each(|(value, _)| visit(value));
            }
            Self::RetVoid => {}
        }
    }

    /// Name this instruction writes, when it writes one.
    pub(super) fn defined(&self) -> Option<&str> {
        match self {
            Self::CallVoid { .. } | Self::Store { .. } | Self::Ret { .. } | Self::RetVoid => None,
            Self::FAdd { dst, .. }
            | Self::FMul { dst, .. }
            | Self::FSub { dst, .. }
            | Self::FDiv { dst, .. }
            | Self::FNeg { dst, .. }
            | Self::Add { dst, .. }
            | Self::Sub { dst, .. }
            | Self::Mul { dst, .. }
            | Self::UDiv { dst, .. }
            | Self::SDiv { dst, .. }
            | Self::URem { dst, .. }
            | Self::SRem { dst, .. }
            | Self::And { dst, .. }
            | Self::Or { dst, .. }
            | Self::Xor { dst, .. }
            | Self::Shl { dst, .. }
            | Self::LShr { dst, .. }
            | Self::AShr { dst, .. }
            | Self::Fcmp { dst, .. }
            | Self::Icmp { dst, .. }
            | Self::Select { dst, .. }
            | Self::Uitofp { dst, .. }
            | Self::Sitofp { dst, .. }
            | Self::Fptoui { dst, .. }
            | Self::Bitcast { dst, .. }
            | Self::Call { dst, .. }
            | Self::Phi { dst, .. }
            | Self::Alloca { dst, .. }
            | Self::GetElementPtr { dst, .. }
            | Self::Load { dst, .. } => Some(dst),
        }
    }

    /// Whether this instruction writes `name`.
    pub(super) fn defines(&self, name: &str) -> bool {
        match self {
            Self::CallVoid { .. } | Self::Store { .. } | Self::Ret { .. } | Self::RetVoid => false,
            Self::FAdd { dst, .. }
            | Self::FMul { dst, .. }
            | Self::FSub { dst, .. }
            | Self::FDiv { dst, .. }
            | Self::FNeg { dst, .. }
            | Self::Add { dst, .. }
            | Self::Sub { dst, .. }
            | Self::Mul { dst, .. }
            | Self::UDiv { dst, .. }
            | Self::SDiv { dst, .. }
            | Self::URem { dst, .. }
            | Self::SRem { dst, .. }
            | Self::And { dst, .. }
            | Self::Or { dst, .. }
            | Self::Xor { dst, .. }
            | Self::Shl { dst, .. }
            | Self::LShr { dst, .. }
            | Self::AShr { dst, .. }
            | Self::Fcmp { dst, .. }
            | Self::Icmp { dst, .. }
            | Self::Select { dst, .. }
            | Self::Uitofp { dst, .. }
            | Self::Sitofp { dst, .. }
            | Self::Fptoui { dst, .. }
            | Self::Bitcast { dst, .. }
            | Self::Call { dst, .. }
            | Self::Phi { dst, .. }
            | Self::Alloca { dst, .. }
            | Self::GetElementPtr { dst, .. }
            | Self::Load { dst, .. } => dst == name,
        }
    }

    /// Result type, when the opcode determines it.
    pub(super) fn value_ty(&self) -> Option<Ty> {
        match self {
            Self::FAdd { .. }
            | Self::FMul { .. }
            | Self::FSub { .. }
            | Self::FDiv { .. }
            | Self::FNeg { .. }
            | Self::Uitofp { .. }
            | Self::Sitofp { .. } => Some(Ty::Float),
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::UDiv { .. }
            | Self::SDiv { .. }
            | Self::URem { .. }
            | Self::SRem { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Shl { .. }
            | Self::LShr { .. }
            | Self::AShr { .. }
            | Self::Fptoui { .. } => Some(Ty::I32),
            Self::Fcmp { .. } | Self::Icmp { .. } => Some(Ty::I1),
            Self::Phi { ty, .. } => Some(*ty),
            Self::Bitcast { to, .. } => Some(*to),
            Self::Select { .. } => None,
            Self::Call { callee, .. } => call_value_ty(callee),
            Self::Load { ty, .. } => Some(*ty),
            Self::Alloca { .. } | Self::GetElementPtr { .. } | Self::Store { .. } => None,
            Self::CallVoid { .. } | Self::Ret { .. } | Self::RetVoid => None,
        }
    }
}

fn visit_pair<'a>(a: &'a Operand, b: &'a Operand, visit: &mut impl FnMut(&'a Operand)) {
    visit(a);
    visit(b);
}

fn visit_triple<'a>(
    a: &'a Operand,
    b: &'a Operand,
    c: &'a Operand,
    visit: &mut impl FnMut(&'a Operand),
) {
    visit(a);
    visit(b);
    visit(c);
}

fn call_value_ty(callee: &str) -> Option<Ty> {
    if int_call(callee) {
        Some(Ty::I32)
    } else {
        // Every other value call in this subset returns `f32`, including
        // `llvm.maximumnum` and `gpu_clamp`, which the older name list missed.
        Some(Ty::Float)
    }
}

fn int_call(callee: &str) -> bool {
    callee.contains("umin")
        || callee.contains("umax")
        || callee.contains("smin")
        || callee.contains("smax")
        || callee.contains("atomic")
        || callee.contains("_u32")
        || callee.contains("fptoui")
        || callee.contains("texture_width_2d")
        || callee.contains("texture_height_2d")
}

/// Split on `,` ignoring parentheses depth.
fn split_top(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in text.chars() {
        match ch {
            '(' | '<' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | '>' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                out.push(current);
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}
