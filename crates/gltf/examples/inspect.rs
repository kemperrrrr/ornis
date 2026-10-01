//! Debug inspector for the animation track: loads one `.glb`/`.gltf`
//! file and prints what the importer assembled (skins, clips) and what
//! it honestly skipped. Usage:
//! `cargo run -p ornis-gltf --example inspect -- <path>`.

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inspect <path-to-glb-or-gltf>");
    let scene = ornis_gltf::load_path(std::path::Path::new(&path)).expect("import");
    println!("scene: {}", scene.name);
    println!("entities: {}", scene.entities.len());
    println!("skins: {}", scene.skins.len());
    println!("skel_clips: {}", scene.skel_clips.len());
    for clip in scene.skel_clips.iter().take(5) {
        let keys: usize = clip
            .tracks
            .iter()
            .map(|t| t.translation.keys.len() + t.rotation.keys.len() + t.scale.keys.len())
            .sum();
        println!(
            "  skel clip: {} tracks, {keys} keys, duration {:.2}s",
            clip.tracks.len(),
            clip.duration
        );
    }
    if scene.skel_clips.len() > 5 {
        println!("  ... and {} more", scene.skel_clips.len() - 5);
    }
    println!("anim_clips: {}", scene.anim_clips.len());
    let stats = &scene.stats;
    println!(
        "skipped_clips: {} skipped_cubicspline: {}",
        stats.skipped_clips, stats.skipped_cubicspline
    );
}
