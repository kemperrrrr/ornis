use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::{AudioBufferSourceNode, AudioContext, GainNode, PannerNode};

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
        let sample_rate = self.context.sample_rate();
        let channels = 2u32;
        let frame_count = input.samples.len() as u32 / channels;
        if frame_count == 0 {
            return;
        }

        let audio_buffer = match self
            .context
            .create_buffer(channels, frame_count, sample_rate)
        {
            Ok(buf) => buf,
            Err(_) => return,
        };

        let mut left_channel: Vec<f32> = Vec::with_capacity(frame_count as usize);
        let mut right_channel: Vec<f32> = Vec::with_capacity(frame_count as usize);

        for i in 0..frame_count as usize {
            let idx = i * channels as usize;
            let l = input.samples.get(idx).copied().unwrap_or(0.0) * input.volume;
            let r = input.samples.get(idx + 1).copied().unwrap_or(l) * input.volume;
            left_channel.push(l);
            right_channel.push(r);
        }

        if audio_buffer.copy_to_channel(&left_channel, 0).is_err() {
            return;
        }
        if audio_buffer.copy_to_channel(&right_channel, 1).is_err() {
            return;
        }

        let source = match self.context.create_buffer_source() {
            Ok(s) => s,
            Err(_) => return,
        };
        source.set_buffer(Some(&audio_buffer));

        let gain = match self.context.create_gain() {
            Ok(g) => g,
            Err(_) => return,
        };
        let _ = gain.gain().set_value(1.0);

        let panner = if input.spatial.is_some() {
            match self.context.create_panner() {
                Ok(p) => {
                    if let Some(sp) = &input.spatial {
                        let (x, z) = panner_position(sp);
                        let _ = p.position_x().set_value(x);
                        let _ = p.position_z().set_value(z);
                    }
                    Some(p)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let _ = source.connect_with_audio_node(&gain);
        match &panner {
            Some(p) => {
                let _ = gain.connect_with_audio_node(p);
                let _ = p.connect_with_audio_node(&self.context.destination());
            }
            None => {
                let _ = gain.connect_with_audio_node(&self.context.destination());
            }
        }

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

    pub fn stop(&self, index: usize) {
        let active = self.active.borrow();
        if let Some((_, source, _, _)) = active.get(index) {
            let _ = source.stop_with_when(0.0);
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
            let _ = source.stop_with_when(0.0);
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

/// Map a `SpatialParams` to a Web Audio `PannerNode` world position.
///
/// `azimuth` is a true geometric angle in radians ([-π/2, π/2], 0 = ahead).
/// The listener faces +z, so a source at `azimuth` with `distance` sits at
/// `(distance·sin azimuth, -distance·cos azimuth)` — matching the native
/// backend's equal-power pan.
fn panner_position(sp: &SpatialParams) -> (f32, f32) {
    (
        sp.distance * sp.azimuth.sin(),
        -sp.distance * sp.azimuth.cos(),
    )
}

/// Register an `onended` callback that removes the finished source from the
/// active list (identified by `stop_id`).
fn attach_on_ended(
    source: &AudioBufferSourceNode,
    active: &Rc<RefCell<Vec<(SourceId, AudioBufferSourceNode, GainNode, Option<PannerNode>)>>>,
    stop_id: SourceId,
) {
    let active_weak = Rc::downgrade(active);
    let on_ended = Closure::wrap(Box::new(move || {
        if let Some(active) = active_weak.upgrade() {
            active.borrow_mut().retain(|(sid, _, _, _)| *sid != stop_id);
        }
    }) as Box<dyn FnMut()>);
    let _ = source.set_onended(Some(on_ended.as_ref().unchecked_ref()));
    on_ended.forget();
}
