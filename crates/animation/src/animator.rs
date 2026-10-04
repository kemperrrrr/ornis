//! Named-clip playback on the character root.
//!
//! [`Animator`] indexes skeletal clip names to playlist [`ClipId`]s and
//! drives the [`SkelPlayer`](crate::SkelPlayer) cursors on the skeleton
//! roots. Nothing plays until [`AnimatorMut::play`]: wiring does not pick
//! a clip. [`AnimatorAccess`] is the hook for an entity-mutation handle
//! (`world.entity_mut(hero).animator()?.play("Walk_Loop")?`) once that
//! handle exists.

use std::collections::HashMap;

use ornis_core::{Clamped01, Entity, Seconds, SmartStore};

use crate::{ClipId, SkelPlayer};

/// Clip-name index on the character root.
///
/// The map is the playback catalog (glTF animation name → playlist
/// entity). `targets` are the skeleton roots whose [`SkelPlayer`] this
/// animator writes. Looping starts enabled (the glTF clip convention)
/// and applies to every target on [`AnimatorMut::play`] and
/// [`AnimatorMut::set_looping`].
///
/// Targets have no [`SkelPlayer`] until the first successful `play`.
#[derive(Debug, Clone, PartialEq)]
pub struct Animator {
    clips: HashMap<String, ClipId>,
    targets: Vec<Entity>,
    looping: bool,
}

impl Animator {
    /// Builds an animator that resolves `clips` and drives `targets`.
    ///
    /// Looping starts on. An empty `clips` map makes every [`AnimatorMut::play`]
    /// fail with [`AnimatorError::UnknownClip`]. An empty `targets` list
    /// accepts known names and writes no cursor.
    pub fn new(clips: HashMap<String, ClipId>, targets: Vec<Entity>) -> Self {
        Self {
            clips,
            targets,
            looping: true,
        }
    }

    /// Playlist for `name`, if this animator indexes it.
    pub fn clip(&self, name: &str) -> Option<ClipId> {
        self.clips.get(name).copied()
    }

    /// Clip name → playlist entity. First name wins when the map was built.
    pub fn clips(&self) -> &HashMap<String, ClipId> {
        &self.clips
    }

    /// Skeleton roots this animator writes [`SkelPlayer`] cursors onto.
    pub fn targets(&self) -> &[Entity] {
        &self.targets
    }

    /// Whether the next [`AnimatorMut::play`] (and current cursors, after
    /// [`AnimatorMut::set_looping`]) wrap at the clip duration.
    pub fn is_looping(&self) -> bool {
        self.looping
    }
}

/// Why named playback could not run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnimatorError {
    /// `play` was asked for a name that is not in the animator's map.
    #[error("unknown clip `{name}`")]
    UnknownClip {
        /// Requested name.
        name: String,
    },
    /// The entity has no [`Animator`] component.
    #[error("entity has no Animator")]
    Missing,
}

/// Mutable view of one entity's [`Animator`].
///
/// Drives every target [`SkelPlayer`]: `play` restarts that clip from
/// time zero at real-time speed and full weight; `pause` holds the cursor;
/// `stop` holds it at time zero. Sampling itself stays in `skel_sample`.
pub struct AnimatorMut<'a> {
    store: &'a mut SmartStore,
    entity: Entity,
}

impl AnimatorMut<'_> {
    /// Starts `name` on every target.
    ///
    /// Restarts at time zero, real-time speed (`Seconds::new(1.0)` — the
    /// raw value is the playback rate, not a duration), full weight, and
    /// the animator's current looping flag. A second `play` of the same
    /// name restarts. Targets that do not exist yet still receive a cursor.
    ///
    /// # Errors
    ///
    /// [`AnimatorError::UnknownClip`] when `name` is not in the map.
    /// [`AnimatorError::Missing`] when the [`Animator`] was removed.
    pub fn play(&mut self, name: &str) -> Result<(), AnimatorError> {
        let (clip, targets, looping) = {
            let lane = self
                .store
                .read_lane::<Animator>()
                .ok_or(AnimatorError::Missing)?;
            let anim = lane.get(self.entity).ok_or(AnimatorError::Missing)?;
            let clip = anim
                .clips
                .get(name)
                .copied()
                .ok_or_else(|| AnimatorError::UnknownClip {
                    name: name.to_owned(),
                })?;
            Ok((clip, anim.targets.clone(), anim.looping))
        }?;
        let player = SkelPlayer {
            clip,
            time: Seconds::ZERO,
            speed: Seconds::new(1.0),
            weight: Clamped01::ONE,
            playing: true,
            looping,
        };
        for target in targets {
            self.store.insert(target, player);
        }
        Ok(())
    }

    /// Holds every target cursor: time stops advancing and the pose is kept.
    ///
    /// No cursor (never played) is success.
    ///
    /// # Errors
    ///
    /// [`AnimatorError::Missing`] when the [`Animator`] was removed.
    pub fn pause(&mut self) -> Result<(), AnimatorError> {
        self.map_players(|player| player.playing = false)
    }

    /// Holds every target cursor at time zero.
    ///
    /// The clip assignment stays, so the next frame does not sample a
    /// different clip; [`Self::play`] restarts it. No cursor is success.
    ///
    /// # Errors
    ///
    /// [`AnimatorError::Missing`] when the [`Animator`] was removed.
    pub fn stop(&mut self) -> Result<(), AnimatorError> {
        self.map_players(|player| {
            player.playing = false;
            player.time = Seconds::ZERO;
        })
    }

    /// Sets whether target cursors wrap (`true`) or clamp (`false`).
    ///
    /// Stored on the [`Animator`] for the next [`Self::play`] and written
    /// through to cursors that already exist.
    ///
    /// # Errors
    ///
    /// [`AnimatorError::Missing`] when the [`Animator`] was removed.
    pub fn set_looping(&mut self, looping: bool) -> Result<(), AnimatorError> {
        let targets = {
            let mut lane = self
                .store
                .write_lane::<Animator>()
                .ok_or(AnimatorError::Missing)?;
            let anim = lane.get_mut(self.entity).ok_or(AnimatorError::Missing)?;
            anim.looping = looping;
            anim.targets.clone()
        };
        if let Some(mut players) = self.store.write_lane::<SkelPlayer>() {
            for target in targets {
                if let Some(player) = players.get_mut(target) {
                    player.looping = looping;
                }
            }
        }
        Ok(())
    }

    fn map_players(&mut self, mut apply: impl FnMut(&mut SkelPlayer)) -> Result<(), AnimatorError> {
        let targets = {
            let lane = self
                .store
                .read_lane::<Animator>()
                .ok_or(AnimatorError::Missing)?;
            lane.get(self.entity)
                .ok_or(AnimatorError::Missing)?
                .targets
                .clone()
        };
        if let Some(mut players) = self.store.write_lane::<SkelPlayer>() {
            for target in targets {
                if let Some(player) = players.get_mut(target) {
                    apply(player);
                }
            }
        }
        Ok(())
    }
}

/// Opens the [`Animator`] on `entity`.
///
/// # Errors
///
/// [`AnimatorError::Missing`] when `entity` has no [`Animator`].
pub fn try_animator<'a>(
    store: &'a mut SmartStore,
    entity: Entity,
) -> Result<AnimatorMut<'a>, AnimatorError> {
    let present = store
        .read_lane::<Animator>()
        .is_some_and(|lane| lane.get(entity).is_some());
    if !present {
        return Err(AnimatorError::Missing);
    }
    Ok(AnimatorMut { store, entity })
}

/// `animator()` for an entity-mutation handle that can see the store.
///
/// `GameWorld::entity_mut` is owned by the Core track and is not on this
/// branch. Implement the two accessors on that handle and the default
/// method is `world.entity_mut(hero).animator()?.play("Walk_Loop")?`.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use ornis_animation::{Animator, AnimatorAccess, ClipId};
/// use ornis_core::{Entity, SmartStore};
///
/// struct Handle<'a> {
///     store: &'a mut SmartStore,
///     entity: Entity,
/// }
///
/// impl AnimatorAccess for Handle<'_> {
///     fn anim_store_mut(&mut self) -> &mut SmartStore {
///         self.store
///     }
///
///     fn anim_entity(&self) -> Entity {
///         self.entity
///     }
/// }
///
/// let mut store = SmartStore::new();
/// let hero = store.create_entity();
/// let clip = store.create_entity();
/// let mut clips = HashMap::new();
/// clips.insert("Walk_Loop".to_string(), ClipId(clip));
/// store.insert(hero, Animator::new(clips, Vec::new()));
///
/// let mut handle = Handle {
///     store: &mut store,
///     entity: hero,
/// };
/// handle.animator().unwrap().play("Walk_Loop").unwrap();
/// ```
pub trait AnimatorAccess {
    /// Store the handle mutates.
    fn anim_store_mut(&mut self) -> &mut SmartStore;

    /// Character-root entity that carries [`Animator`].
    fn anim_entity(&self) -> Entity;

    /// Borrows the [`Animator`] on [`Self::anim_entity`].
    ///
    /// # Errors
    ///
    /// [`AnimatorError::Missing`] when that entity has no [`Animator`].
    fn animator(&mut self) -> Result<AnimatorMut<'_>, AnimatorError> {
        let entity = self.anim_entity();
        try_animator(self.anim_store_mut(), entity)
    }
}
