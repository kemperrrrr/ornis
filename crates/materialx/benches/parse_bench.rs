//! Criterion benchmarks for the MaterialX pipeline: XML parsing of a large
//! document and full `.mtlx` → `OpenPBRMaterial` conversion of a math chain.

use criterion::{Criterion, criterion_group, criterion_main};

use ornis_materialx::{MaterialXParser, materialx_to_openpbr};

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
    let mut body = String::with_capacity(n * 128);
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
    let mut body = String::with_capacity(n * 200);
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
    let doc = large_document(1000);
    group.bench_function("constants_1000", |b| {
        b.iter(|| {
            MaterialXParser::new()
                .parse(std::hint::std::hint::black_box(&doc))
                .unwrap()
        });
    });
    group.finish();
}

fn bench_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("materialx_convert");
    let doc = math_chain(100);
    group.bench_function("math_chain_100", |b| {
        b.iter(|| materialx_to_openpbr(std::hint::std::hint::black_box(&doc)).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_parse, bench_convert);
criterion_main!(benches);
