//! `Animator` playback: named clips, unknown names, pause/stop/looping.
//!
//! The character root holds the map; skeleton roots hold the cursors.
//! Nothing is selected until `play`.

use std::collections::HashMap;

use ornis_animation::{Animator, AnimatorAccess, AnimatorError, ClipId, SkelPlayer, try_animator};
use ornis_core::{Clamped01, Entity, Seconds, SmartStore};

struct Handle<'a> {
    store: &'a mut SmartStore,
    entity: Entity,
}

impl AnimatorAccess for Handle<'_> {
    fn anim_store_mut(&mut self) -> &mut SmartStore {
        self.store
    }

    fn anim_entity(&self) -> Entity {
        self.entity
    }
}

fn setup() -> (SmartStore, Entity, Entity, Entity, ClipId) {
    let mut store = SmartStore::new();
    let hero = store.create_entity();
    let skeleton = store.create_entity();
    let other = store.create_entity();
    let playlist = store.create_entity();
    let clip = ClipId(playlist);
    let mut clips = HashMap::new();
    clips.insert("Walk_Loop".to_string(), clip);
    store.insert(hero, Animator::new(clips, vec![skeleton, other]));
    (store, hero, skeleton, other, clip)
}

fn player(store: &SmartStore, entity: Entity) -> SkelPlayer {
    *store
        .read_lane::<SkelPlayer>()
        .expect("players")
        .get(entity)
        .expect("cursor")
}

#[test]
fn play_known_clip_starts_every_target() {
    let (mut store, hero, skeleton, other, clip) = setup();
    {
        let mut handle = Handle {
            store: &mut store,
            entity: hero,
        };
        handle.animator().unwrap().play("Walk_Loop").unwrap();
    }
    for target in [skeleton, other] {
        let cursor = player(&store, target);
        assert_eq!(cursor.clip, clip);
        assert_eq!(cursor.time, Seconds::ZERO);
        assert_eq!(cursor.speed, Seconds::new(1.0));
        assert_eq!(cursor.weight, Clamped01::ONE);
        assert!(cursor.playing);
        assert!(cursor.looping);
    }
}

#[test]
fn play_unknown_clip_is_an_error_and_leaves_cursors() {
    let (mut store, hero, skeleton, _, clip) = setup();
    try_animator(&mut store, hero)
        .unwrap()
        .play("Walk_Loop")
        .unwrap();
    {
        let mut players = store.write_lane::<SkelPlayer>().unwrap();
        players.get_mut(skeleton).unwrap().time = Seconds::new(0.4);
    }
    let err = try_animator(&mut store, hero)
        .unwrap()
        .play("Crouch_Fwd_Loop")
        .unwrap_err();
    assert_eq!(
        err,
        AnimatorError::UnknownClip {
            name: "Crouch_Fwd_Loop".to_string(),
        }
    );
    let cursor = player(&store, skeleton);
    assert_eq!(cursor.clip, clip);
    assert_eq!(cursor.time, Seconds::new(0.4));
    assert!(cursor.playing);
}

#[test]
fn missing_animator_is_an_error() {
    let mut store = SmartStore::new();
    let hero = store.create_entity();
    assert!(matches!(
        try_animator(&mut store, hero),
        Err(AnimatorError::Missing)
    ));
}

#[test]
fn pause_holds_time_and_stop_resets_it() {
    let (mut store, hero, skeleton, _, clip) = setup();
    try_animator(&mut store, hero)
        .unwrap()
        .play("Walk_Loop")
        .unwrap();
    {
        let mut players = store.write_lane::<SkelPlayer>().unwrap();
        players.get_mut(skeleton).unwrap().time = Seconds::new(0.75);
    }
    try_animator(&mut store, hero).unwrap().pause().unwrap();
    let paused = player(&store, skeleton);
    assert!(!paused.playing);
    assert_eq!(paused.time, Seconds::new(0.75));
    assert_eq!(paused.clip, clip);

    try_animator(&mut store, hero).unwrap().stop().unwrap();
    let stopped = player(&store, skeleton);
    assert!(!stopped.playing);
    assert_eq!(stopped.time, Seconds::ZERO);
    assert_eq!(stopped.clip, clip);
}

#[test]
fn looping_flag_applies_to_current_and_future_plays() {
    let (mut store, hero, skeleton, _, _) = setup();
    try_animator(&mut store, hero)
        .unwrap()
        .set_looping(false)
        .unwrap();
    assert!(
        !store
            .read_lane::<Animator>()
            .unwrap()
            .get(hero)
            .unwrap()
            .is_looping()
    );
    try_animator(&mut store, hero)
        .unwrap()
        .play("Walk_Loop")
        .unwrap();
    assert!(!player(&store, skeleton).looping);

    try_animator(&mut store, hero)
        .unwrap()
        .set_looping(true)
        .unwrap();
    assert!(player(&store, skeleton).looping);
    assert!(
        store
            .read_lane::<Animator>()
            .unwrap()
            .get(hero)
            .unwrap()
            .is_looping()
    );
}
