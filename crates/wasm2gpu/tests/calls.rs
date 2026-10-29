//! Direct calls, `br_table` is covered next door, and the compute entry.
//!
//! Each `translate` runs naga's validator, reprints WGSL, and round-trips
//! SPIR-V. A constant group or binding is specialized into the callee.
//! A returned immediate can be the next import's binding.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

fn vertex() -> String {
    format!(
        r#"(func (export "{vs}") (param f32 f32 f32 f32) (result f32 f32 f32 f32 f32 f32)
            local.get 0 local.get 1 f32.const 0 f32.const 1 local.get 2 local.get 3)"#,
        vs = symbols::VS_MAIN,
    )
}

fn fragment(body: &str) -> String {
    format!(
        r#"(func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32) {body})"#,
        fs = symbols::FS_MAIN,
    )
}

fn wasm(imports: &str, funcs: &str) -> Vec<u8> {
    wat::parse_str(format!("(module {imports} {funcs} {vs})", vs = vertex(),)).expect("wat")
}

fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

fn wgsl(imports: &str, funcs: &str) -> String {
    translate(&wasm(imports, funcs)).expect("translate").wgsl
}

fn err(imports: &str, funcs: &str) -> TranslateError {
    translate(&wasm(imports, funcs)).expect_err("reject")
}

fn read_f32() -> String {
    import(
        symbols::STORAGE_READ_F32,
        "(param i32 i32 i32) (result f32)",
    )
}

fn rgba_tail() -> &'static str {
    "f32.const 0 f32.const 0 f32.const 1"
}

#[test]
fn call_specializes_constant_provenance() {
    let load = format!(
        r#"(func $load (param i32 i32 i32) (result f32)
            local.get 0 local.get 1 local.get 2 call ${read})"#,
        read = symbols::STORAGE_READ_F32,
    );
    let fs = fragment(&format!(
        "i32.const 0 i32.const 1 i32.const 2 call $load {tail}",
        tail = rgba_tail(),
    ));
    let got = wgsl(&read_f32(), &format!("{load} {fs}"));
    assert!(got.contains("fn fn0_"), "{got}");
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains("2u"), "{got}");
}

#[test]
fn returned_immediate_is_a_binding() {
    let slot = "(func $slot (result i32) i32.const 1)";
    let fs = fragment(&format!(
        "i32.const 0 call $slot i32.const 0 call ${read} {tail}",
        read = symbols::STORAGE_READ_F32,
        tail = rgba_tail(),
    ));
    let got = wgsl(&read_f32(), &format!("{slot} {fs}"));
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn offset_passes_through_an_argument_and_a_result() {
    let plus = r#"(func $plus (param i32) (result i32)
        local.get 0 i32.const 1 i32.add)"#;
    let fs = format!(
        r#"(func (export "{fs}") (param f32 f32 i32) (result f32 f32 f32 f32)
            i32.const 0 i32.const 1 local.get 2 call $plus
            call ${read}
            {tail})"#,
        fs = symbols::FS_MAIN,
        read = symbols::STORAGE_READ_F32,
        tail = rgba_tail(),
    );
    let got = wgsl(&read_f32(), &format!("{plus} {fs}"));
    assert!(got.contains("fn fn0_"), "{got}");
    assert!(got.contains("storage_0_1_"), "{got}");
}

#[test]
fn recursion_is_rejected() {
    let rec = "(func $rec (param i32) (result i32) local.get 0 call $rec)";
    let fs = fragment("i32.const 0 call $rec drop f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    assert!(matches!(
        err("", &format!("{rec} {fs}")),
        TranslateError::Recursion
    ));
}

#[test]
fn mutual_recursion_is_rejected() {
    let funcs = "(func $a (result i32) call $b) (func $b (result i32) call $a)";
    let fs = fragment("call $a drop f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    assert!(matches!(
        err("", &format!("{funcs} {fs}")),
        TranslateError::Recursion
    ));
}

#[test]
fn indirect_call_is_rejected() {
    let table = "(table 1 funcref)";
    let fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    assert!(matches!(err(table, &fs), TranslateError::IndirectCall));
}

#[test]
fn compute_entry_has_workgroup_and_invocation_id() {
    let body = format!(
        r#"(func (export "{cs}") (param i32)
            i32.const 0 i32.const 1 local.get 0 f32.const 1 call ${write})
           (func (export "{fs}") (param f32 f32) (result f32 f32 f32 f32)
            f32.const 0 f32.const 0 f32.const 0 f32.const 1)"#,
        cs = symbols::CS_MAIN,
        fs = symbols::FS_MAIN,
        write = symbols::STORAGE_WRITE_F32,
    );
    let imports = import(symbols::STORAGE_WRITE_F32, "(param i32 i32 i32 f32)");
    let translated = translate(&wasm(&imports, &body)).expect("compute");
    let size = symbols::CS_WORKGROUP;
    assert!(
        translated.wgsl.contains(&format!(
            "@workgroup_size({}, {}, {})",
            size[0], size[1], size[2]
        )),
        "{}",
        translated.wgsl
    );
    assert!(
        translated.wgsl.contains("@builtin(global_invocation_id)"),
        "{}",
        translated.wgsl
    );
    assert!(translated.wgsl.contains("@compute"), "{}", translated.wgsl);
    assert!(
        translated.wgsl.contains("storage_0_1_"),
        "{}",
        translated.wgsl
    );
    assert!(!translated.compute_spirv.is_empty());
    let parsed = naga::front::wgsl::parse_str(&translated.wgsl).expect("wgsl");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .unwrap_or_else(|err| panic!("naga: {err}\n{}", translated.wgsl));
}

const DEPTH: usize = {
    const LEVELS: usize = 20;
    LEVELS
};

const FORK: usize = {
    const LEVELS: usize = 18;
    LEVELS
};

const FAN: usize = {
    const CALLS: usize = 80;
    CALLS
};

const WIDE: usize = {
    const PARAMS: usize = 12;
    PARAMS
};

const BULK: usize = {
    const OPS: usize = 12000;
    OPS
};

#[test]
fn straight_line_body_is_not_charged_against_the_call_budget() {
    let fs = fragment(&format!(
        "{}f32.const 0 f32.const 0 f32.const 0 f32.const 1",
        dropped(BULK)
    ));
    let started = std::time::Instant::now();
    let got = wgsl("", &fs);
    assert!(
        started.elapsed().as_secs() < 2,
        "straight line took too long"
    );
    assert!(got.contains("fn fs_main"), "{got}");
}

#[test]
fn loop_body_is_not_charged_against_the_call_budget() {
    let fs = fragment(&format!(
        "(local i32) (block (loop local.get 2 i32.const 1 i32.add local.tee 2 {} br_if 0)) f32.const 0 f32.const 0 f32.const 0 f32.const 1",
        dropped(BULK)
    ));
    let started = std::time::Instant::now();
    let got = wgsl("", &fs);
    assert!(started.elapsed().as_secs() < 2, "loop took too long");
    assert!(got.contains("loop"), "{got}");
}

#[test]
fn deep_call_chain_reuses_one_specialization_per_function() {
    let funcs = chain(DEPTH, false);
    let fs = fragment("i32.const 1 call $f0 drop f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    let started = std::time::Instant::now();
    let got = wgsl("", &format!("{funcs} {fs}"));
    assert!(started.elapsed().as_secs() < 2, "chain took too long");
    assert_eq!(got.matches("fn fn").count(), DEPTH, "{got}");
}

#[test]
fn distinct_constant_tree_hits_the_specialization_budget() {
    let funcs = chain(FORK, true);
    let fs = fragment("i32.const 1 call $f0 drop f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    let started = std::time::Instant::now();
    let got = err("", &format!("{funcs} {fs}"));
    assert!(started.elapsed().as_secs() < 2, "budget took too long");
    assert!(matches!(got, TranslateError::Budget), "{got}");
}

#[test]
fn many_distinct_calls_of_one_function_hit_the_budget() {
    let id = "(func $id (param i32) (result i32) local.get 0)";
    let mut body = String::new();
    let mut index = 0u32;
    while (index as usize) < FAN {
        body.push_str(&format!("i32.const {index} call $id drop "));
        index += 1;
    }
    body.push_str("f32.const 0 f32.const 0 f32.const 0 f32.const 1");
    let started = std::time::Instant::now();
    let got = err("", &format!("{id} {}", fragment(&body)));
    assert!(started.elapsed().as_secs() < 2, "fan-out took too long");
    assert!(matches!(got, TranslateError::Budget), "{got}");
}

#[test]
fn f32_argument_is_shared_across_constants() {
    let scale = "(func $scale (param f32) (result f32) local.get 0 f32.const 2 f32.mul)";
    let fs = fragment(
        "f32.const 1 call $scale f32.const 3 call $scale f32.add f32.const 0 f32.const 0 f32.const 1",
    );
    let got = wgsl("", &format!("{scale} {fs}"));
    assert_eq!(got.matches("fn fn0_").count(), 1, "{got}");
    assert!(got.contains("fn fn0_0_("), "{got}");
    assert!(got.contains("1f"), "{got}");
    assert!(got.contains("3f"), "{got}");
}

#[test]
fn tail_constant_does_not_collide_in_the_specialization_key() {
    let mut params = String::new();
    let mut index = 0usize;
    while index < WIDE {
        params.push_str("i32 ");
        index += 1;
    }
    let id = format!(
        "(func $id (param {params}) (result i32) local.get {})",
        WIDE - 1
    );
    let ones = wide_consts(1);
    let mut twos = wide_consts(1);
    // The last immediate differs. Truncating the key would merge the two calls.
    let needle = "i32.const 1";
    let end = twos.rfind(&needle).expect("tail");
    twos.replace_range(end..end + needle.len(), "i32.const 2");
    let fs = fragment(&format!(
        "{ones} call $id drop {twos} call $id drop {ones} call $id drop f32.const 0 f32.const 0 f32.const 0 f32.const 1"
    ));
    let got = wgsl("", &format!("{id} {fs}"));
    assert_eq!(got.matches("fn fn0_").count(), 2, "{got}");
}

fn chain(depth: usize, fork: bool) -> String {
    let mut out = String::new();
    let mut index = 0usize;
    while index < depth {
        let body = if index + 1 == depth {
            "local.get 0".to_string()
        } else if fork {
            format!(
                "local.get 0 i32.const 2 i32.mul call $f{next} drop local.get 0 i32.const 2 i32.mul i32.const 1 i32.add call $f{next}",
                next = index + 1,
            )
        } else {
            format!(
                "local.get 0 call $f{next} drop local.get 0 call $f{next}",
                next = index + 1,
            )
        };
        out.push_str(&format!("(func $f{index} (param i32) (result i32) {body})"));
        index += 1;
    }
    out
}

fn dropped(ops: usize) -> String {
    let mut out = String::new();
    let mut left = ops;
    while left > 1 {
        out.push_str("f32.const 1 drop ");
        left -= 2;
    }
    out
}

fn wide_consts(value: u32) -> String {
    let mut out = String::new();
    let mut index = 0usize;
    while index < WIDE {
        out.push_str(&format!("i32.const {value} "));
        index += 1;
    }
    out
}
