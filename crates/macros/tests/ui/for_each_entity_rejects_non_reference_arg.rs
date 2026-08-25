use ornis_macros::for_each_entity;

// `for_each_entity!` requires every closure argument to be `&T` or `&mut T`.
// A bare `T` is rejected at macro-expansion time.
fn main() {
    let store = ();
    for_each_entity!(store, |pos: Position| {
        let _ = pos;
    });
}

struct Position;
