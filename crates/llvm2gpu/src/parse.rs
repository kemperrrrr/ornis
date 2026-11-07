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
    /// `uitofp i32 %a to float` and `sitofp i32 %a to float`.
    Uitofp { dst: Name, a: Operand },
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
}

/// Parse the whole textual IR, returning `cs_main` or `fs_main`.
/// Declarations are skipped.
pub(super) fn parse(text: &str) -> Result<LlvmFunc, TranslateError> {
    let header = parse_header(text)?;
    let Some(func) = header.into_iter().find(|f| is_entry(&f.name)) else {
        return Err(TranslateError::Unsupported);
    };
    Ok(func)
}

fn is_entry(name: &str) -> bool {
    name == "cs_main" || name == "fs_main"
}

/// Parse every `define ... { ... }` block.
fn parse_header(text: &str) -> Result<Vec<LlvmFunc>, TranslateError> {
    let mut out = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if !trimmed.starts_with("define") {
            continue;
        }
        let (name, ret_ty, params) = parse_define_header(trimmed)?;
        // Only the shader entry is of interest; other defines
        // (`rust_begin_unwind`, …) may use trap/ptr shapes outside the subset.
        if !is_entry(&name) {
            loop {
                let Some(inner) = lines.next() else {
                    return Err(TranslateError::Parse("unterminated function".to_string()));
                };
                if inner.trim() == "}" {
                    break;
                }
            }
            continue;
        }
        let mut blocks: Vec<BasicBlock> = Vec::new();
        let mut cur_label = "start".to_string();
        let mut cur_instrs: Vec<Instr> = Vec::new();
        let mut cur_term: Option<Terminator> = None;
        let mut saw_ret = false;
        loop {
            let Some(inner) = lines.next() else {
                return Err(TranslateError::Parse("unterminated function".to_string()));
            };
            // Strip an inline comment (`; preds = %a, %b`) before parsing.
            let inner = inner.split(';').next().unwrap_or("").trim();
            if inner == "}" {
                break;
            }
            if inner.is_empty() {
                continue;
            }
            // A basic-block label.
            if inner.ends_with(':') && !inner.contains('=') {
                // Close previous block if needed.
                if cur_term.is_some() {
                    // previous block already terminated; start new
                    blocks.push(BasicBlock {
                        label: cur_label.clone(),
                        instrs: std::mem::take(&mut cur_instrs),
                        term: cur_term.take().unwrap(),
                    });
                } else if !cur_instrs.is_empty() {
                    // Fallthrough without explicit terminator: synthesize ret.
                    return Err(TranslateError::Unsupported);
                }
                cur_label = inner.trim_end_matches(':').to_string();
                continue;
            }
            if let Some(term) = parse_term(inner)? {
                cur_term = Some(term);
                saw_ret |= matches!(cur_term, Some(Terminator::Ret(_)));
                // Terminator closes the block; the next label begins a new one.
                continue;
            }
            if let Some(instr) = parse_instr(inner)? {
                cur_instrs.push(instr);
            }
        }
        // Close the last block.
        if let Some(term) = cur_term.take() {
            saw_ret |= matches!(term, Terminator::Ret(_));
            blocks.push(BasicBlock {
                label: cur_label,
                instrs: cur_instrs,
                term,
            });
        } else if !cur_instrs.is_empty() {
            return Err(TranslateError::Parse(
                "block without terminator".to_string(),
            ));
        }
        if !saw_ret {
            return Err(TranslateError::Parse("function without ret".to_string()));
        }
        out.push(LlvmFunc {
            name,
            ret_ty,
            params,
            blocks,
        });
    }
    if out.is_empty() {
        return Err(TranslateError::Parse("no define in llvm ir".to_string()));
    }
    Ok(out)
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
    if trimmed == "unreachable" || trimmed.starts_with("switch") {
        return Err(TranslateError::Unsupported);
    }
    Ok(None)
}

/// `%x = phi float [ %a, %pred ], [ 1.0, %pred2 ]`
fn parse_phi(rhs: &str) -> Result<Vec<(Operand, String)>, TranslateError> {
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
        out.push((parse_operand(value.trim())?, pred));
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
        // Non-assignment lines: reject stores/loads/traces.
        if line.starts_with("store ") || line.starts_with("load ") || line.contains("getelementptr")
        {
            return Err(TranslateError::Unsupported);
        }
        // `tail call` without a result is a trap/assume; reject.
        if line.contains("call") {
            return Err(TranslateError::Unsupported);
        }
        return Ok(None);
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
        "phi" => {
            let ty = parse_ty(&tokens[1])?;
            // `parse_phi` expects the tail after `phi <type>`.
            let tail = rhs.split_whitespace().skip(2).collect::<Vec<_>>().join(" ");
            let incoming = parse_phi(&tail)?;
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
            let c = parse_operand(tokens[2].trim_end_matches(','))?;
            let a = parse_operand(tokens[4].trim_end_matches(','))?;
            let b = parse_operand(tokens[6].trim_end_matches(','))?;
            Ok(Some(Instr::Select { dst, c, a, b }))
        }
        "uitofp" | "sitofp" => {
            // `uitofp nneg i32 %a to float` — rustc marks unsigned casts `nneg`.
            let a = parse_operand(tokens[operand_index(&tokens, 1)?].trim_end_matches(','))?;
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
        "load" | "store" | "getelementptr" => Err(TranslateError::Unsupported),
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

fn parse_operand(word: &str) -> Result<Operand, TranslateError> {
    let w = word.trim().trim_matches(',');
    if let Some(name) = w.strip_prefix('%') {
        return Ok(Operand::Local(name.to_string()));
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
            | Self::Fptoui { a, .. }
            | Self::Bitcast { a, .. }
            | Self::Ret { a } => {
                visit(a);
            }
            Self::Select { c, a, b, .. } => visit_triple(c, a, b, &mut visit),
            Self::Call { args, .. } | Self::CallVoid { args, .. } => {
                args.iter().for_each(&mut visit);
            }
            Self::Phi { incoming, .. } => {
                incoming.iter().for_each(|(value, _)| visit(value));
            }
            Self::RetVoid => {}
        }
    }

    /// Whether this instruction writes `name`.
    pub(super) fn defines(&self, name: &str) -> bool {
        match self {
            Self::CallVoid { .. } | Self::Ret { .. } | Self::RetVoid => false,
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
            | Self::Fptoui { dst, .. }
            | Self::Bitcast { dst, .. }
            | Self::Call { dst, .. }
            | Self::Phi { dst, .. } => dst == name,
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
            | Self::Uitofp { .. } => Some(Ty::Float),
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
