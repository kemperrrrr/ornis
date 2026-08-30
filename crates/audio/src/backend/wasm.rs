//! Web Audio backend for the wasm target.
//!
//! Each [`MixInput`] becomes an `AudioBufferSourceNode` → `GainNode` →
//! optional `PannerNode` chain on a shared `AudioContext`. All state lives
//! in `Rc<RefCell<..>>` — the backend is single-threaded, matching the
//! browser event loop.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioScheduledSourceNode, GainNode,
    PannerNode,
};

use crate::source::{MixInput, SpatialParams};

type SourceId = usize;

pub struct AudioBackend {
    context: AudioContext,
    active: Rc<
        RefCell<
            Vec<(
                SourceId,
                AudioBufferSourceNode,
                GainNode,
                Option<PannerNode>,
            )>,
        >,
    >,
    next_id: Rc<RefCell<SourceId>>,
}

impl AudioBackend {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let context = AudioContext::new().map_err(|_| "failed to create AudioContext")?;
        Ok(Self {
            context,
            active: Rc::new(RefCell::new(Vec::new())),
            next_id: Rc::new(RefCell::new(0)),
        })
    }

    pub fn play(&self, input: MixInput) {
        let Some(audio_buffer) = self.decode_stereo(&input) else {
            return;
        };
        let Some((source, gain, panner)) = self.build_source_chain(&input, &audio_buffer) else {
            return;
        };
        self.connect_chain(&source, &gain, panner.as_ref());

        let id = {
            let mut n = self.next_id.borrow_mut();
            let id = *n;
            *n += 1;
            id
        };

        if input.looping {
            source.set_loop(true);
        }

        if let Err(e) = source.start() {
            let _ = web_sys::console::error_1(&JsValue::from_str(&format!(
                "Audio start error: {:?}",
                e
            )));
            return;
        }

        attach_on_ended(&source, &self.active, id);
        self.active.borrow_mut().push((id, source, gain, panner));
    }

    /// Interleave-decode the input samples into a stereo `AudioBuffer`.
    /// Returns `None` when there is no audio or the context rejects the buffer.
    fn decode_stereo(&self, input: &MixInput) -> Option<AudioBuffer> {
        let sample_rate = self.context.sample_rate();
        let channels = 2u32;
        let frame_count = input.samples.len() as u32 / channels;
        if frame_count == 0 {
            return None;
        }

        let audio_buffer = self
            .context
            .create_buffer(channels, frame_count, sample_rate)
            .ok()?;

        let mut left_channel: Vec<f32> = Vec::with_capacity(frame_count as usize);
        let mut right_channel: Vec<f32> = Vec::with_capacity(frame_count as usize);

        for i in 0..frame_count as usize {
            let idx = i * channels as usize;
            let l = input.samples.get(idx).copied().unwrap_or(0.0) * input.volume;
            let r = input.samples.get(idx + 1).copied().unwrap_or(l) * input.volume;
            left_channel.push(l);
            right_channel.push(r);
        }

        audio_buffer.copy_to_channel(&left_channel, 0).ok()?;
        audio_buffer.copy_to_channel(&right_channel, 1).ok()?;
        Some(audio_buffer)
    }

    /// Create the source → gain (→ panner) node chain for a playback.
    fn build_source_chain(
        &self,
        input: &MixInput,
        audio_buffer: &AudioBuffer,
    ) -> Option<(AudioBufferSourceNode, GainNode, Option<PannerNode>)> {
        let source = self.context.create_buffer_source().ok()?;
        source.set_buffer(Some(audio_buffer));

        let gain = self.context.create_gain().ok()?;
        let _ = gain.gain().set_value(1.0);

        let panner = input.spatial.as_ref().and_then(|sp| {
            let panner = self.context.create_panner().ok();
            if let Some(p) = &panner {
                configure_panner(p, sp);
            }
            panner
        });

        Some((source, gain, panner))
    }

    /// Wire the node chain into the destination graph:
    /// source → gain → (panner | destination).
    fn connect_chain(
        &self,
        source: &AudioBufferSourceNode,
        gain: &GainNode,
        panner: Option<&PannerNode>,
    ) {
        let _ = source.connect_with_audio_node(gain);
        match panner {
            Some(p) => {
                let _ = gain.connect_with_audio_node(p);
                let _ = p.connect_with_audio_node(&self.context.destination());
            }
            None => {
                let _ = gain.connect_with_audio_node(&self.context.destination());
            }
        }
    }

    pub fn stop(&self, index: usize) {
        let active = self.active.borrow();
        if let Some((_, source, _, _)) = active.get(index) {
            // Use the AudioScheduledSourceNode method — the
            // AudioBufferSourceNode inherent one is deprecated in web-sys.
            AudioScheduledSourceNode::stop_with_when(source, 0.0).ok();
        }
    }

    pub fn set_volume(&self, index: usize, volume: f32) {
        let active = self.active.borrow();
        if let Some((_, _, gain, _)) = active.get(index) {
            let _ = gain.gain().set_value(volume.max(0.0).min(1.0));
        }
    }

    pub fn clear(&self) {
        let mut active = self.active.borrow_mut();
        for (_, source, _, _) in active.iter() {
            AudioScheduledSourceNode::stop_with_when(source, 0.0).ok();
        }
        active.clear();
    }
}

impl Drop for AudioBackend {
    fn drop(&mut self) {
        self.clear();
        let _ = self.context.close();
    }
}

/// Map a `SpatialParams` to a Web Audio `PannerNode` configuration.
///
/// `azimuth` is a true geometric angle in radians ([-π/2, π/2], 0 = ahead);
/// the listener faces +z, so a source sits at
/// `(d·sin az·cos el, d·sin el, -d·cos az·cos el)` — matching the native
/// backend's geometry. Distance attenuation uses the LINEAR distance model
/// with the same rolloff/reference values as native `distance_attenuation`,
/// so browser and native builds hear the same falloff.
fn configure_panner(p: &web_sys::PannerNode, sp: &SpatialParams) {
    let cos_el = sp.elevation.cos();
    let x = sp.distance * sp.azimuth.sin() * cos_el;
    let y = sp.distance * sp.elevation.sin();
    let z = -sp.distance * sp.azimuth.cos() * cos_el;
    let _ = p.position_x().set_value(x);
    let _ = p.position_y().set_value(y);
    let _ = p.position_z().set_value(z);

    // Linear model mirrors native distance_attenuation():
    // gain = ref / (ref + rolloff * max(0, dist - ref)) is close to linear
    // falloff from full gain at <= reference_distance.
    let _ = p.set_distance_model(web_sys::DistanceModelType::Linear);
    let _ = p.set_ref_distance(sp.reference_distance as f64);
    let _ = p.set_rolloff_factor(sp.rolloff_factor as f64);
}

/// Register an `onended` callback that removes the finished source from the
/// active list (identified by `stop_id`).
fn attach_on_ended(
    source: &AudioBufferSourceNode,
    active: &Rc<
        RefCell<
            Vec<(
                SourceId,
                AudioBufferSourceNode,
                GainNode,
                Option<PannerNode>,
            )>,
        >,
    >,
    stop_id: SourceId,
) {
    let active_weak = Rc::downgrade(active);
    let on_ended = Closure::wrap(Box::new(move || {
        if let Some(active) = active_weak.upgrade() {
            active.borrow_mut().retain(|(sid, _, _, _)| *sid != stop_id);
        }
    }) as Box<dyn FnMut()>);
    // set_onended is deprecated in web-sys; the modern path is
    // add_event_listener_with_callback("ended", ...).
    let _ = source.add_event_listener_with_callback("ended", on_ended.as_ref().unchecked_ref());
    on_ended.forget();
}
