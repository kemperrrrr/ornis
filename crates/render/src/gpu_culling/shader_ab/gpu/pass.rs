//! One device, both pipelines, and the timed dispatches.
//!
//! A set comparison runs before the clock starts. A mismatch aborts the
//! process so a bad pack cannot publish times.

use super::super::super::cull_workgroup_count;
use super::super::scene::Kind;
use super::super::scene::Scene;
use super::super::scene::kind_name;
use super::super::scene::visible_ids;
use super::clock;
use super::dispatch;
use super::pipeline;
use super::pipeline::FLAG_ENTRY;
use super::pipeline::PROD_ENTRY;
use super::pipeline::Pipe;
use super::readback;
use super::session;
use super::session::Session;
use super::upload;

pub(super) fn open() -> Option<Session> {
    session::open()
}

pub(super) fn time_scenes(session: &Session, lowered: &str) -> String {
    let prod = pipeline::build(&session.device, pipeline::prod(), PROD_ENTRY, true);
    let flag_layout = pipeline::flag_layout_of(&session.device);
    let wasm = pipeline::compile_with(&session.device, lowered, FLAG_ENTRY, flag_layout.clone());
    let twin = pipeline::compile_with(
        &session.device,
        pipeline::TWIN_WGSL,
        FLAG_ENTRY,
        flag_layout,
    );
    let pipes = Pipes {
        prod: &prod,
        wasm: &wasm,
        twin: &twin,
        lowered,
    };
    let mut text = String::from(&session.header);
    text.push('\n');
    super::super::scene::scenes().iter().for_each(|scene| {
        text.push_str(&time_scene(session, scene, &pipes));
        text.push('\n');
    });
    text
}

struct Pipes<'a> {
    prod: &'a Pipe,
    wasm: &'a Pipe,
    twin: &'a Pipe,
    lowered: &'a str,
}

fn time_scene(session: &Session, scene: &Scene, pipes: &Pipes<'_>) -> String {
    let prod_set = upload::upload_prod(session, scene, pipes.prod);
    let flag_set = upload::upload_flag(session, scene, pipes.wasm);
    let shaders = Shaders {
        prod: pipes.prod,
        wasm: pipes.wasm,
        twin: pipes.twin,
        widths: widths(scene, pipes.lowered),
    };
    check_sets(session, scene, &prod_set, &flag_set, &shaders);
    let prod_samples = clock::measure(|| {
        dispatch::submit(
            session,
            pipes.prod,
            dispatch::prod_bind(&prod_set),
            shaders.widths.prod,
            Some(dispatch::prod_args(&prod_set)),
        )
    });
    let wasm_samples = clock::measure(|| {
        dispatch::submit(
            session,
            pipes.wasm,
            dispatch::flag_bind(&flag_set),
            shaders.widths.wasm,
            None,
        )
    });
    let twin_samples = clock::measure(|| {
        dispatch::submit(
            session,
            pipes.twin,
            dispatch::flag_bind(&flag_set),
            shaders.widths.twin,
            None,
        )
    });
    clock::scene_report(scene, &prod_samples, &wasm_samples, &twin_samples)
}

struct Shaders<'a> {
    prod: &'a Pipe,
    wasm: &'a Pipe,
    twin: &'a Pipe,
    widths: Widths,
}

struct Widths {
    prod: u32,
    wasm: u32,
    twin: u32,
}

fn widths(scene: &Scene, lowered: &str) -> Widths {
    Widths {
        prod: cull_workgroup_count(scene.spheres.len() as u32),
        wasm: pipeline::groups(scene.spheres.len(), lowered),
        twin: pipeline::groups(scene.spheres.len(), pipeline::TWIN_WGSL),
    }
}

fn check_sets(
    session: &Session,
    scene: &Scene,
    prod_set: &dispatch::ProdSet,
    flag_set: &dispatch::FlagSet,
    shaders: &Shaders<'_>,
) {
    let _ = dispatch::submit(
        session,
        shaders.prod,
        dispatch::prod_bind(prod_set),
        shaders.widths.prod,
        Some(dispatch::prod_args(prod_set)),
    );
    let prod_ids = readback::prod_ids(session, prod_set, scene.spheres.len());
    let _ = dispatch::submit(
        session,
        shaders.wasm,
        dispatch::flag_bind(flag_set),
        shaders.widths.wasm,
        None,
    );
    let wasm_ids = readback::flag_ids(session, flag_set, scene.spheres.len());
    let _ = dispatch::submit(
        session,
        shaders.twin,
        dispatch::flag_bind(flag_set),
        shaders.widths.twin,
        None,
    );
    let twin_ids = readback::flag_ids(session, flag_set, scene.spheres.len());
    let cpu = visible_ids(scene);
    if let Some(text) = mismatch(scene.kind, &cpu, &prod_ids, &wasm_ids, &twin_ids) {
        abort(&text);
    }
}

fn mismatch(kind: Kind, cpu: &[u32], prod: &[u32], wasm: &[u32], twin: &[u32]) -> Option<String> {
    if agrees(cpu, prod, wasm, twin) {
        None
    } else {
        Some(format!(
            "visible-set mismatch on {}: cpu {} prod {} wasm {} twin {}",
            kind_name(kind),
            cpu.len(),
            prod.len(),
            wasm.len(),
            twin.len(),
        ))
    }
}

fn agrees(cpu: &[u32], prod: &[u32], wasm: &[u32], twin: &[u32]) -> bool {
    cpu == prod && cpu == wasm && cpu == twin
}

fn abort(text: &str) -> ! {
    eprintln!("{text}");
    std::process::exit(1)
}
