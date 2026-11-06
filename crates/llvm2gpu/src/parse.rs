//! Textual LLVM IR -> small AST covering the spike-B subset.
//!
//! The parser only needs the shape stable `rustc` emits for a scalar
//! compute kernel after `SROA` at `opt-level=3`: one `define` with a few
//! operations and a `ret`. Anything with control flow (`phi`, `br`,
//! `load`, `store`, `getelementptr`, trap/assume calls) parses but is
//! rejected by the lowerer with `Unsupported`, so a failed spike never
//! silently becomes a wrong shader.

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
    /// `uitofp i32 %a to float`
    Uitofp { dst: Name, a: Operand },
    /// `call float @name(...)`
    Call {
        dst: Name,
        callee: String,
        args: Vec<Operand>,
    },
    /// `ret float %a`
    Ret { a: Operand },
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
    /// Body instructions in order, no labels.
    pub(super) body: Vec<Instr>,
}

/// Parse the whole textual IR, returning the first defined function whose
/// name is `cs_main` (the spike entry). Declarations are skipped.
pub(super) fn parse(text: &str) -> Result<LlvmFunc, TranslateError> {
    let header = parse_header(text)?;
    let Some(func) = header.into_iter().find(|f| f.name == "cs_main") else {
        return Err(TranslateError::Unsupported);
    };
    Ok(func)
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
        // Only `cs_main` is of interest; other defines (`rust_begin_unwind`, …)
        // may use trap/ptr shapes that are not part of the subset.
        if name != "cs_main" {
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
        let mut body = Vec::new();
        let mut saw_ret = false;
        loop {
            let Some(inner) = lines.next() else {
                return Err(TranslateError::Parse("unterminated function".to_string()));
            };
            let inner = inner.trim();
            if inner == "}" {
                break;
            }
            if inner.is_empty() || inner.starts_with(';') {
                continue;
            }
            // A basic-block label.
            if inner.ends_with(':') && !inner.contains('=') && !inner.contains(" preds") {
                continue;
            }
            if inner.starts_with("phi ") || inner.contains(" = phi ") {
                return Err(TranslateError::Unsupported);
            }
            if inner.starts_with("br ") || inner == "unreachable" || inner.starts_with("switch") {
                return Err(TranslateError::Unsupported);
            }
            if inner.starts_with("ret ") {
                let (_, rhs) = inner.split_once("ret ").unwrap_or(("", ""));
                body.push(parse_ret(rhs)?);
                saw_ret = true;
                continue;
            }
            if let Some(instr) = parse_instr(inner)? {
                if matches!(instr, Instr::Ret { .. }) {
                    saw_ret = true;
                }
                body.push(instr);
            }
        }
        if !saw_ret {
            return Err(TranslateError::Parse("function without ret".to_string()));
        }
        out.push(LlvmFunc {
            name,
            ret_ty,
            params,
            body,
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
    let mut parts = rhs.splitn(2, ' ');
    let ty = parts
        .next()
        .ok_or(TranslateError::Parse("ret without type".to_string()))?;
    if ty == "void" {
        return Err(TranslateError::Unsupported);
    }
    let _ = parse_ty(ty)?;
    let rest = parts
        .next()
        .ok_or(TranslateError::Parse("ret without value".to_string()))?;
    Ok(Instr::Ret {
        a: parse_operand(rest.trim())?,
    })
}

fn parse_instr(line: &str) -> Result<Option<Instr>, TranslateError> {
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
        "fadd" | "fmul" | "fsub" | "fdiv" => {
            // `fadd float %a, %b`
            let a = parse_operand(tokens[2].trim_end_matches(','))?;
            let b = parse_operand(tokens[3].trim_end_matches(','))?;
            let instr = match tokens[0].as_str() {
                "fadd" => Instr::FAdd {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
                "fmul" => Instr::FMul {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
                "fsub" => Instr::FSub {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
                _ => Instr::FDiv {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
            };
            Ok(Some(instr))
        }
        "fneg" => {
            let a = parse_operand(tokens[2].trim_end_matches(','))?;
            Ok(Some(Instr::FNeg { dst, a }))
        }
        "add" | "sub" | "mul" => {
            let a = parse_operand(tokens[2].trim_end_matches(','))?;
            let b = parse_operand(tokens[3].trim_end_matches(','))?;
            let instr = match tokens[0].as_str() {
                "add" => Instr::Add {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
                "sub" => Instr::Sub {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
                _ => Instr::Mul {
                    dst: dst.clone(),
                    a: a.clone(),
                    b,
                },
            };
            Ok(Some(instr))
        }
        "fcmp" => {
            let pred = tokens[1].clone();
            let a = parse_operand(tokens[3].trim_end_matches(','))?;
            let b = parse_operand(tokens[4].trim_end_matches(','))?;
            Ok(Some(Instr::Fcmp { dst, pred, a, b }))
        }
        "icmp" => {
            let pred = tokens[1].clone();
            let a = parse_operand(tokens[3].trim_end_matches(','))?;
            let b = parse_operand(tokens[4].trim_end_matches(','))?;
            Ok(Some(Instr::Icmp { dst, pred, a, b }))
        }
        "select" => {
            // `select i1 %c, float %a, float %b`
            let c = parse_operand(tokens[2].trim_end_matches(','))?;
            let a = parse_operand(tokens[4].trim_end_matches(','))?;
            let b = parse_operand(tokens[6].trim_end_matches(','))?;
            Ok(Some(Instr::Select { dst, c, a, b }))
        }
        "uitofp" => {
            // `uitofp i32 %a to float`
            let a = parse_operand(tokens[2].trim_end_matches(','))?;
            Ok(Some(Instr::Uitofp { dst, a }))
        }
        "phi" | "br" | "switch" => Err(TranslateError::Unsupported),
        "load" | "store" | "getelementptr" => Err(TranslateError::Unsupported),
        // Any other opcode (and, or, shl, …) is out of the spike subset.
        _ => Err(TranslateError::Unsupported),
    }
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
        let arg = arg.trim();
        let mut parts = arg.split_whitespace();
        let _ty = parts.next();
        if let Some(value) = parts.next() {
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
    if let Ok(value) = w.parse::<f32>()
        && (w.contains('.') || w.contains('e') || w.contains('E'))
    {
        return Ok(Operand::F32(value));
    }
    if let Ok(value) = w.parse::<i64>() {
        if value < 0 {
            return Err(TranslateError::Parse(format!("negative literal {w}")));
        }
        return Ok(Operand::U32(value as u32));
    }
    Err(TranslateError::Parse(format!("unknown operand {w}")))
}

/// Split on `,` ignoring parentheses depth.
fn split_top(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in text.chars() {
        match ch {
            '(' | '<' => {
                depth += 1;
                current.push(ch);
            }
            ')' | '>' => {
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
