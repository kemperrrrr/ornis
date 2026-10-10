//! UI test: `#[smart_system]` rejects `Res<SmartStore>`.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

// Components are reached through the first `&T`/`&mut T` parameter.
#[smart_system]
fn sys(u: &mut Unit, store: Res<SmartStore>) {
    let _ = (u.hp, store);
}

fn main() {}
