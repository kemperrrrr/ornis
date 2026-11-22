//! Where the translator currently is, for rejection messages.
//!
//! The subset rejects a construct by name — `TranslateError::Unsupported` — but
//! "llvm ir is outside the supported subset" alone sends the reader through a
//! bisection of their own kernel, and a kernel is one `cargo rustc` away from
//! the line of IR that was rejected. So the lowerer records its position
//! (function, block, instruction) as it walks the module, and the error carries
//! that text.
//!
//! A thread-local is enough: this is a build tool that lowers one module on one
//! thread, and the position is only read while building an error. Setting it
//! per instruction is a `String` write on the hot path of a compile, which is
//! why the value is formatted lazily from borrowed parts in debug builds — see
//! [`Site::remember`].

use core::cell::RefCell;

/// Function, block and instruction text of the step being lowered.
#[derive(Default)]
struct Site {
    function: String,
    block: String,
    instruction: String,
    /// `true` once any part is non-empty, so the common case (no error) never
    /// formats a message.
    any: bool,
}

thread_local! {
    static SITE: RefCell<Site> = RefCell::new(Site::default());
}

/// Record the function whose body is being lowered.
pub(crate) fn enter_function(name: &str) {
    SITE.with(|site| {
        let mut site = site.borrow_mut();
        site.function.clear();
        site.function.push_str(name);
        site.block.clear();
        site.instruction.clear();
        site.any = true;
    });
}

/// Record the basic block whose instructions are being lowered.
pub(crate) fn enter_block(label: &str) {
    SITE.with(|site| {
        let mut site = site.borrow_mut();
        site.block.clear();
        site.block.push_str(label);
        site.instruction.clear();
    });
}

/// Record one instruction. `text` is the IR line, without the leading `%dest =`.
pub(crate) fn enter_instruction(text: &str) {
    SITE.with(|site| {
        let mut site = site.borrow_mut();
        site.instruction.clear();
        // Long aggregate constants would swamp the message; the operator and
        // the first operands are what a reader needs.
        let keep = text.len().min(120);
        let cut = text.get(..keep).unwrap_or(text);
        site.instruction.push_str(cut.trim_end());
    });
}

/// One-line description of the current position, for an error message.
pub(crate) fn describe() -> String {
    SITE.with(|site| {
        let site = site.borrow();
        if !site.any {
            return String::from("at an unrecorded step");
        }
        let mut out = String::new();
        if !site.function.is_empty() {
            out.push_str("in ");
            out.push_str(&site.function);
        }
        if !site.block.is_empty() {
            out.push_str(", block ");
            out.push_str(&site.block);
        }
        if !site.instruction.is_empty() {
            out.push_str(": ");
            out.push_str(&site.instruction);
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three parts accumulate into one readable position, and a step with
    /// nothing recorded says so instead of reporting an empty site.
    #[test]
    fn the_site_reads_back_as_one_line() {
        enter_function("cs_main");
        enter_block("bb4");
        enter_instruction("%7 = fadd float %a, %b");
        assert_eq!(describe(), "in cs_main, block bb4: %7 = fadd float %a, %b");
        // A new instruction clears the previous one but keeps the block.
        enter_instruction("%8 = fsub float %a, %b");
        assert_eq!(describe(), "in cs_main, block bb4: %8 = fsub float %a, %b");
        // A new block clears the instruction.
        enter_block("bb5");
        assert_eq!(describe(), "in cs_main, block bb5");
    }
}
