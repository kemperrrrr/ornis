//! Criterion benchmarks for the MaterialX pipeline: XML parsing of a large
//! document and full `.mtlx` → `OpenPBRMaterial` conversion of a math chain.

use criterion::{Criterion, criterion_group, criterion_main};

use ornis_materialx::{MaterialXParser, materialx_to_openpbr};

/// Estimated bytes per constant node in [`large_document`] (capacity hint).
const BYTES_PER_CONSTANT: usize = 128;
/// Estimated bytes per math node in [`math_chain`] (capacity hint).
const BYTES_PER_MATH_NODE: usize = 200;
/// Constant-node count for the parser-bound bench.
const PARSE_CONSTANTS: usize = 1000;
/// Math-chain length for the convert bench.
const CONVERT_CHAIN: usize = 100;

/// Minimal nodedef set, named realistically as in real .mtlx libraries; the
/// evaluator resolves definitions by the `node` attribute.
const NODEDEFS: &str = r#"
  <nodedef name="ND_output" node="output" />
  <nodedef name="ND_constant" node="constant" />
  <nodedef name="ND_add" node="add" />
  <nodedef name="ND_multiply" node="multiply" />
  <nodedef name="ND_open_pbr_surface" node="open_pbr_surface" />
"#;

/// A document with `n` constant nodes in one nodegraph — parser-bound.
fn large_document(n: usize) -> String {
    let mut body = String::with_capacity(n * BYTES_PER_CONSTANT);
    for i in 0..n {
        body.push_str(&format!(
            r#"    <constant name="c{i}" type="color3"><input name="value" type="color3" value="0.1, 0.2, 0.3" /></constant>
"#,
        ));
    }
    format!(
        r#"<?xml version="1.0"?>
<materialx version="1.39">
{NODEDEFS}
  <nodegraph name="bench" nodedef="ND_open_pbr_surface_surfaceshader">
{body}  </nodegraph>
</materialx>"#
    )
}

/// A chain of `n` multiply/add color3 nodes feeding an `open_pbr_surface`
/// output — exercises parsing, graph evaluation and material extraction.
fn math_chain(n: usize) -> String {
    let mut body = String::with_capacity(n * BYTES_PER_MATH_NODE);
    body.push_str(
        r#"    <constant name="seed" type="color3"><input name="value" type="color3" value="0.5, 0.5, 0.5" /></constant>
"#,
    );
    let mut prev = "seed".to_string();
    for i in 0..n {
        let (op, name) = if i % 2 == 0 {
            ("multiply", format!("m{i}"))
        } else {
            ("add", format!("a{i}"))
        };
        body.push_str(&format!(
            r#"    <{op} name="{name}" type="color3"><input name="in1" type="color3" nodename="{prev}" /><input name="in2" type="color3" nodename="seed" /></{op}>
"#,
        ));
        prev = name;
    }
    format!(
        r#"<?xml version="1.0"?>
<materialx version="1.39">
{NODEDEFS}
  <nodegraph name="bench" nodedef="ND_open_pbr_surface_surfaceshader">
{body}    <output name="out" type="surfaceshader">
      <input name="base_color" type="color3" nodename="{prev}" />
    </output>
  </nodegraph>
</materialx>"#
    )
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("materialx_parse");
    let doc = large_document(PARSE_CONSTANTS);
    group.bench_function("constants_1000", |b| {
        b.iter(|| {
            let parsed = MaterialXParser::new().parse(std::hint::black_box(&doc));
            std::hint::black_box(parsed.ok())
        });
    });
    group.finish();
}

fn bench_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("materialx_convert");
    let doc = math_chain(CONVERT_CHAIN);
    group.bench_function("math_chain_100", |b| {
        b.iter(|| {
            let converted = materialx_to_openpbr(std::hint::black_box(&doc));
            std::hint::black_box(converted.ok())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_parse, bench_convert);
criterion_main!(benches);
