//! UI test: `kernel` requires single-expression arms in a value
//! `match` (issue #110, feature 3).

use ornis_macros::kernel;

// A value-position `match` expands to a temporary plus an if-chain of
// assignments, so each arm must contribute exactly one expression.
// Bind intermediate results with `let` outside the arm, or move the
// `match` into statement position.
#[kernel]
fn multi_stmt(x: u32) -> u32 {
    let out = match x {
        1u32 => {
            let doubled = 2u32;
            doubled + 1u32
        }
        _ => 0u32,
    };
    out
}

fn main() {}
