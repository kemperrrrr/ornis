//! Debug inspector for the animation track: loads one `.glb`/`.gltf`
//! file and prints what the importer assembled (nodes, skins, clips) and
//! what it honestly skipped. Usage:
//! `cargo run -p ornis-gltf --example inspect -- <path>`.

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inspect <path-to-glb-or-gltf>");
    let model = ornis_gltf::load_path(std::path::Path::new(&path)).expect("import");
    println!("scene: {}", model.name);
    println!(
        "nodes: {} roots: {} primitives: {}",
        model.nodes.len(),
        model.roots.len(),
        model.primitives.len()
    );
    let meshless = model
        .nodes
        .iter()
        .filter(|node| node.primitives.is_empty())
        .count();
    println!("mesh-less nodes: {meshless}");
    println!("skins: {}", model.skins.len());
    println!("skel_clips: {}", model.skel_clips.len());
    for clip in model.skel_clips.iter().take(5) {
        let keys: usize = clip
            .tracks
            .iter()
            .map(|track| {
                track.translation.keys.len() + track.rotation.keys.len() + track.scale.keys.len()
            })
            .sum();
        println!(
            "  skel clip: {} tracks, {keys} keys, duration {:.2}s",
            clip.tracks.len(),
            clip.duration
        );
    }
    if model.skel_clips.len() > 5 {
        println!("  ... and {} more", model.skel_clips.len() - 5);
    }
    println!("anim_clips: {}", model.anim_clips.len());
    let stats = &model.stats;
    println!(
        "skipped_clips: {} skipped_cubicspline: {}",
        stats.skipped_clips, stats.skipped_cubicspline
    );
}
