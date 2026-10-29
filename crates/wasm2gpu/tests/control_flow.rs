//! Control-flow lowering: multi-level `br`, joins, and struct layout.
//!
//! A side-effect-free `if` with no results is dropped in `pure_if`: the
//! condition is consumed and neither arm is emitted.

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

fn wasm(custom: &str, imports: &str, fs: &str) -> Vec<u8> {
    wat::parse_str(format!(
        "(module {custom} {imports} {vs} (func (export \"{name}\") {fs}))",
        vs = vertex(),
        name = symbols::FS_MAIN,
    ))
    .expect("wat")
}

fn import(name: &str, sig: &str) -> String {
    format!(
        "(import \"{m}\" \"{name}\" (func ${name} {sig}))",
        m = symbols::MODULE,
    )
}

fn wgsl(custom: &str, imports: &str, fs: &str) -> String {
    translate(&wasm(custom, imports, fs))
        .expect("translate")
        .wgsl
}

fn err(custom: &str, imports: &str, fs: &str) -> TranslateError {
    translate(&wasm(custom, imports, fs)).expect_err("reject")
}

fn write_f32() -> String {
    import(symbols::STORAGE_WRITE_F32, "(param i32 i32 i32 f32)")
}

fn rgba() -> &'static str {
    "f32.const 0 f32.const 0 f32.const 0 f32.const 1"
}

fn write_call() -> String {
    format!(
        "i32.const 0 i32.const 1 i32.const 0 f32.const 1 call ${write}",
        write = symbols::STORAGE_WRITE_F32,
    )
}

fn contains(got: &str, snapshot: &str) {
    assert!(got.contains(snapshot.trim()), "{got}");
}

#[test]
fn br_from_if_arm_skips_later_write() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (if (local.get 2)
             (then
               (block (br 2))
               {write})
             (else)))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/if_arm_leave.wgsl"),
    );
}

#[test]
fn br_from_if_arm_continues_outer_loop() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (loop
           (if (local.get 2)
             (then
               (block (br 2))
               {write})
             (else))
           br 0)
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/if_arm_again.wgsl"),
    );
}

#[test]
fn br_through_three_blocks_skips_the_middle_write() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (block
           (block
             (block (br 2))
             {write}))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/br_three.wgsl"),
    );
}

#[test]
fn loop_br_if_to_outer_block_skips_the_write() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (loop
             local.get 2
             br_if 1
             {write}
             br 0))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    contains(
        &wgsl("", &write_f32(), &fs),
        include_str!("snapshots/loop_exit.wgsl"),
    );
}

#[test]
fn loop_back_edge_joins_provenance() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (local i32)
         i32.const 0
         local.set 2
         (loop
           i32.const 0
           local.get 2
           i32.const 0
           call ${read}
           drop
           i32.const 1
           local.set 2
           br 0)
         {ret}",
        read = symbols::STORAGE_READ,
        ret = rgba(),
    );
    let err = err(
        "",
        &import(
            symbols::STORAGE_READ,
            "(param i32 i32 i32) (result f32 f32 f32 f32)",
        ),
        &fs,
    );
    assert!(matches!(err, TranslateError::AmbiguousProvenance));
}

#[test]
fn branch_to_the_function_is_an_error() {
    let fs = "(param f32 f32) (result f32 f32 f32 f32)
         (block
           f32.const 0 f32.const 0 f32.const 0 f32.const 1
           br 1)";
    let err = err("", "", fs);
    assert!(matches!(err, TranslateError::FunctionBranch));
    assert!(
        err.to_string()
            .contains("branch to the function is not lowered")
    );
}

#[test]
fn pure_if_without_results_is_dropped() {
    let fs = "(param f32 f32 i32) (result f32 f32 f32 f32)
         (if (local.get 2) (then i32.const 1) (else i32.const 2))
         f32.const 0 f32.const 0 f32.const 0 f32.const 1";
    let got = wgsl("", "", fs);
    assert!(!got.contains("if ("), "{got}");
    assert!(!got.contains("select("), "{got}");
}

#[test]
fn unset_local_is_constant_zero() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (local i32)
         local.get 2
         i32.const 1
         i32.const 0
         call ${read}
         f32.const 0 f32.const 0 f32.const 0",
        read = symbols::STORAGE_READ_F32,
    );
    let got = wgsl(
        "",
        &import(
            symbols::STORAGE_READ_F32,
            "(param i32 i32 i32) (result f32)",
        ),
        &fs,
    );
    assert!(got.contains("storage_0_1_"), "{got}");
    assert!(got.contains("array<f32>"), "{got}");
}

#[test]
fn two_vec3_and_integer_fields_lower() {
    let custom = format!(
        "(@custom \"{name}\" \"\\4f\\47\\54\\31\\01\\00\\04\\00\\05\\05\\02\\03\")",
        name = symbols::TYPES_SECTION,
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 0 i32.const 0 call ${read}
         drop drop drop drop drop drop drop drop
         {ret}",
        read = symbols::STORAGE_READ_STRUCT,
        ret = rgba(),
    );
    let got = wgsl(
        &custom,
        &import(
            symbols::STORAGE_READ_STRUCT,
            "(param i32 i32 i32 i32) (result f32 f32 f32 f32 f32 f32 i32 i32)",
        ),
        &fs,
    );
    contains(&got, include_str!("snapshots/struct_two_vec3.wgsl"));
    assert!(got.contains("array<S0_>"), "{got}");
}

#[test]
fn struct_id_out_of_range_is_rejected() {
    let custom = format!(
        "(@custom \"{name}\" \"\\4f\\47\\54\\31\\01\\00\\01\\00\\01\")",
        name = symbols::TYPES_SECTION,
    );
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         i32.const 0 i32.const 1 i32.const 0 i32.const 1 call ${read}
         drop
         {ret}",
        read = symbols::STORAGE_READ_STRUCT,
        ret = rgba(),
    );
    let err = err(
        &custom,
        &import(
            symbols::STORAGE_READ_STRUCT,
            "(param i32 i32 i32 i32) (result f32)",
        ),
        &fs,
    );
    match err {
        TranslateError::UnsupportedType(msg) => {
            assert!(msg.contains("struct type id is out of range"), "{msg}");
        }
        other => panic!("unexpected {other}"),
    }
}

#[test]
fn br_table_lowers_to_switch() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (block
             (block
               (br_table 0 1 2 (local.get 2))
             )
             i32.const 0 i32.const 1 i32.const 0 f32.const 1
             call ${write}
             br 1
           )
           i32.const 0 i32.const 1 i32.const 0 f32.const 2
           call ${write}
           br 0
         )
         i32.const 0 i32.const 1 i32.const 0 f32.const 3
         call ${write}
         {ret}",
        write = symbols::STORAGE_WRITE_F32,
        ret = rgba(),
    );
    let got = wgsl("", &write_f32(), &fs);
    assert!(got.contains("switch "), "{got}");
    // Depth 0 sets one leave flag, depth 1 sets two, default (depth 2) sets three.
    assert_eq!(trues(arm(&got, "case 0u:", "case 1u:")), 1, "{got}");
    assert_eq!(trues(arm(&got, "case 1u:", "default:")), 2, "{got}");
    assert_eq!(trues(arm(&got, "default:", "}")), 3, "{got}");
    let tail = got.split("switch ").nth(1).expect("switch");
    let first = tail.find("= 1f").expect("store 1");
    let second = tail.find("= 2f").expect("store 2");
    let third = tail.find("= 3f").expect("store 3");
    assert!(first < second && second < third, "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_table_continue_inside_a_loop() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (loop
           (br_table 0 (local.get 2)))
         {ret}",
        ret = rgba(),
    );
    let got = wgsl("", "", &fs);
    assert!(got.contains("switch "), "{got}");
    assert!(got.contains("continue;"), "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_table_arms_span_three_nested_labels() {
    // Arm 0 leaves the inner block, arm 1 continues the loop, default
    // leaves the loop and the outer block. Every arm crosses at least one
    // label, so each one runs through the leave/again flags, not the
    // depth-0 `continue` fast path.
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (loop
             (block
               (br_table 0 1 2 (local.get 2)))
             i32.const 0 i32.const 1 i32.const 0 f32.const 1
             call ${write}
             br 0)
           i32.const 0 i32.const 1 i32.const 0 f32.const 2
           call ${write})
         i32.const 0 i32.const 1 i32.const 0 f32.const 3
         call ${write}
         {ret}",
        write = symbols::STORAGE_WRITE_F32,
        ret = rgba(),
    );
    let got = wgsl("", &write_f32(), &fs);
    assert!(got.contains("switch "), "{got}");
    // Depth 0 sets the inner leave, depth 1 the inner leave plus the loop
    // again flag, default three flags up to the outermost block.
    assert_eq!(trues(arm(&got, "case 0u:", "case 1u:")), 1, "{got}");
    assert_eq!(trues(arm(&got, "case 1u:", "default:")), 2, "{got}");
    assert_eq!(trues(arm(&got, "default:", "}")), 3, "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_table_continues_a_loop_nested_in_a_block() {
    // The loop sits at depth 1 relative to the function. Arm 1 continues
    // it from inside an inner block (two labels up), which is the shape
    // that used to be rejected as unsupported.
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (loop
             (block
               (br_table 1 (local.get 2)))
             {write}
             br 0))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    let got = wgsl("", &write_f32(), &fs);
    assert!(got.contains("switch "), "{got}");
    assert!(got.contains("continue;"), "{got}");
    // The continue arm sets the inner block's leave flag so the write
    // after the block is skipped.
    assert_eq!(trues(arm(&got, "default:", "}")), 2, "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_table_leaves_a_loop_and_its_outer_block() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (loop
             (br_table 1 0 (local.get 2))))
         {ret}",
        ret = rgba(),
    );
    let got = wgsl("", "", &fs);
    assert!(got.contains("switch "), "{got}");
    assert!(got.contains("continue;"), "{got}");
    // Case 0 leaves both the loop and the block (two flags); the default
    // arm continues the loop with a direct `continue` and no flags.
    assert_eq!(trues(arm(&got, "case 0u:", "default:")), 2, "{got}");
    assert_eq!(trues(arm(&got, "default:", "}")), 0, "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_table_in_an_if_arm_continues_the_loop() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (loop
           (if (local.get 2)
             (then
               (br_table 1 (local.get 2)))
             (else)))
         {ret}",
        ret = rgba(),
    );
    let got = wgsl("", "", &fs);
    assert!(got.contains("switch "), "{got}");
    assert!(got.contains("continue;"), "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_if_through_three_blocks_skips_the_write() {
    let fs = format!(
        "(param f32 f32 i32) (result f32 f32 f32 f32)
         (block
           (block
             (block
               (local.get 2)
               (br_if 2))
             {write}))
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    let got = wgsl("", &write_f32(), &fs);
    // All three blocks get a leave flag so the branch exits the outermost
    // one and skips the write in the middle.
    assert_eq!(trues(&got), 3, "{got}");
    let _ = parsed_switch(&got);
}

#[test]
fn br_from_an_inner_loop_continues_the_outer_loop() {
    let fs = format!(
        "(param f32 f32) (result f32 f32 f32 f32)
         (loop
           (loop
             (br 1))
           {write})
         {ret}",
        write = write_call(),
        ret = rgba(),
    );
    let got = wgsl("", &write_f32(), &fs);
    // The inner loop leaves its own flag and sets the outer again flag, so
    // the write between the loops never runs.
    assert_eq!(trues(&got), 2, "{got}");
    let _ = parsed_switch(&got);
}

fn arm<'a>(wgsl: &'a str, start: &str, end: &str) -> &'a str {
    wgsl.split(start)
        .nth(1)
        .unwrap_or_else(|| panic!("{start} missing"))
        .split(end)
        .next()
        .unwrap_or_else(|| panic!("{end} missing"))
}

fn trues(body: &str) -> usize {
    body.matches("= true").count()
}

fn parsed_switch(wgsl: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(wgsl).expect("wgsl");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|err| panic!("naga: {err}\n{wgsl}"));
    module
}
