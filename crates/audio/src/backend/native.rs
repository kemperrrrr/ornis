//! Native audio output backend built on cpal.
//!
//! Spawns a dedicated mixer thread owning the device stream and mixing
//! queued [`MixInput`]s at 48 kHz; control goes through a crossbeam command
//! channel, so `play`/`stop`/`set_volume` never block the calling thread.
//! Dropping the backend signals shutdown and joins the thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};

use crate::source::MixInput;

const SAMPLE_RATE: u32 = 48000;
const BUFFER_FRAMES: usize = 512;

enum AudioCommand {
    Play(MixInput),
    Stop(usize),
    SetVolume(usize, f32),
    Clear,
    Shutdown,
}

/// Native (cpal) output backend.
///
/// Spawns a dedicated `ornis-audio` thread owning the cpal device stream;
/// commands are exchanged via a crossbeam channel so `play`/`stop` are safe
/// from any thread. Dropping the backend signals shutdown and joins the thread.
pub struct AudioBackend {
    command_tx: Sender<AudioCommand>,
    handle: Option<thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl AudioBackend {
    /// Spawn the mixer thread.
    ///
    /// # Errors
    /// Returns an error when the OS thread cannot be spawned. Device problems
    /// surface asynchronously as stderr messages from the mixer thread.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let handle = thread::Builder::new()
            .name("ornis-audio".into())
            .spawn(move || {
                if let Err(e) = run_audio_thread(command_rx, running_clone) {
                    eprintln!("Audio thread error: {}", e);
                }
            })?;

        Ok(Self {
            command_tx,
            handle: Some(handle),
            running,
        })
    }

    /// Queue a sound for mixing; never blocks on the audio thread.
    pub fn play(&self, input: MixInput) {
        let _ = self.command_tx.send(AudioCommand::Play(input));
    }

    /// Remove the queued sound at `index` (index into the mixer's active list).
    pub fn stop(&self, index: usize) {
        let _ = self.command_tx.send(AudioCommand::Stop(index));
    }

    /// Change the volume of the queued sound at `index`; clamped to [0, 1] at mix time.
    pub fn set_volume(&self, index: usize, volume: f32) {
        let _ = self.command_tx.send(AudioCommand::SetVolume(index, volume));
    }

    /// Drop every queued sound immediately.
    pub fn clear(&self) {
        let _ = self.command_tx.send(AudioCommand::Clear);
    }
}

impl Drop for AudioBackend {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.command_tx.send(AudioCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_audio_thread(
    command_rx: Receiver<AudioCommand>,
    running: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;

    let config = device.default_output_config()?;
    let output_channels = config.channels() as u16;

    let mut active_sounds: Vec<MixInput> = Vec::new();
    let samples_arc = Arc::new(Mutex::new(Vec::new()));

    let stream_samples = samples_arc.clone();

    let err_fn = move |err| eprintln!("Audio stream error: {}", err);

    let stream = device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
            let mut guard = match stream_samples.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let frame_count = data.len() / output_channels as usize;
            if guard.len() < frame_count {
                guard.resize(frame_count, 0.0);
            }
            for (i, frame) in guard.iter().enumerate() {
                let ch = i % output_channels as usize;
                let frame_idx = i / output_channels as usize;
                if frame_idx < frame_count {
                    data[frame_idx * output_channels as usize + ch] = *frame;
                }
            }
        },
        err_fn,
        Some(std::time::Duration::from_millis(5)),
    )?;

    stream.play()?;

    let mut temp_mix = vec![0.0f32; BUFFER_FRAMES * output_channels as usize];

    while running.load(Ordering::SeqCst) {
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                AudioCommand::Play(input) => {
                    active_sounds.push(input);
                }
                AudioCommand::Stop(idx) => {
                    if idx < active_sounds.len() {
                        active_sounds.remove(idx);
                    }
                }
                AudioCommand::SetVolume(idx, volume) => {
                    if let Some(sound) = active_sounds.get_mut(idx) {
                        sound.volume = volume;
                    }
                }
                AudioCommand::Clear => {
                    active_sounds.clear();
                }
                AudioCommand::Shutdown => return Ok(()),
            }
        }

        mix(&mut active_sounds, &mut temp_mix, output_channels);

        if let Ok(mut guard) = samples_arc.lock() {
            *guard = temp_mix.clone();
        }

        thread::sleep(std::time::Duration::from_micros(
            (BUFFER_FRAMES as u64 * 1_000_000) / SAMPLE_RATE as u64 / 2,
        ));
    }

    Ok(())
}

fn pan_sample(sample: f32, azimuth: f32) -> (f32, f32) {
    // azimuth is a true geometric angle in radians, [-π/2, π/2], where
    // -π/2 = hard left, 0 = center, +π/2 = hard right.
    //
    // Equal-GAIN linear pan (deliberate product choice, see
    // docs/quality/audio-panning-bug-2026-08-25.md): normalize
    // t = az/(π/2) ∈ [-1, 1], then L = clamp(1 - t) * sample,
    // R = clamp(1 + t) * sample. Center gives L = R = sample (0 dB, full
    // loudness); hard left gives (sample, 0); no channel ever inverts.
    // Tradeoff vs equal-power: the channel SUM dips slightly toward the
    // center — accepted in exchange for an exact center level.
    let t = (azimuth / std::f32::consts::FRAC_PI_2).clamp(-1.0, 1.0);
    let l_gain = (1.0 - t).clamp(0.0, 1.0);
    let r_gain = (1.0 + t).clamp(0.0, 1.0);
    let left = (l_gain * sample).clamp(-1.0, 1.0);
    let right = (r_gain * sample).clamp(-1.0, 1.0);
    (left, right)
}

fn distance_attenuation(distance: f32, rolloff: f32, reference: f32) -> f32 {
    if distance <= reference {
        return 1.0;
    }
    let atten = reference / (reference + rolloff * (distance - reference));
    atten.clamp(0.0, 1.0)
}

fn mix(active_sounds: &mut Vec<MixInput>, output: &mut [f32], output_channels: u16) {
    for frame in output.chunks_exact_mut(output_channels as usize) {
        let mut left: f32 = 0.0;
        let mut right: f32 = 0.0;

        active_sounds.retain_mut(|sound| {
            if sound.cursor >= sound.samples.len() {
                if sound.looping {
                    sound.cursor = 0;
                } else {
                    return false;
                }
            }

            let remaining = sound.samples.len() - sound.cursor;
            let sample = if remaining > 0 {
                sound.samples[sound.cursor]
            } else {
                0.0
            };
            sound.cursor += 1;

            let vol = sound.volume.clamp(0.0, 1.0);
            let s = sample * vol;

            if let Some(sp) = sound.spatial {
                let atten =
                    distance_attenuation(sp.distance, sp.rolloff_factor, sp.reference_distance);
                let (l, r) = pan_sample(s * atten, sp.azimuth);
                left += l;
                right += r;
            } else {
                left += s;
                right += s;
            }

            true
        });

        left = left.clamp(-1.0, 1.0);
        right = right.clamp(-1.0, 1.0);
        frame[0] = left;
        if output_channels > 1 {
            frame[1] = right;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{AudioClip, AudioSource};

    #[test]
    fn pan_sample_centered() {
        // Equal-gain: center gives FULL gain on both channels (product choice).
        let (l, r) = pan_sample(0.5, 0.0);
        assert!((l - 0.5).abs() < 1e-6, "left = {l}");
        assert!((r - 0.5).abs() < 1e-6, "right = {r}");
        assert!((l - r).abs() < 1e-6, "L != R: {l} vs {r}");
    }

    #[test]
    fn pan_sample_full_left() {
        // azimuth -π/2 -> hard left (L = sample, R = 0).
        let (l, r) = pan_sample(0.5, -std::f32::consts::FRAC_PI_2);
        assert!((l - 0.5).abs() < 1e-6, "left = {l}");
        assert!(r.abs() < 1e-6, "right = {r}");
    }

    #[test]
    fn pan_sample_full_right() {
        // azimuth +π/2 -> hard right (L = 0, R = sample).
        let (l, r) = pan_sample(0.5, std::f32::consts::FRAC_PI_2);
        assert!(l.abs() < 1e-6, "left = {l}");
        assert!((r - 0.5).abs() < 1e-6, "right = {r}");
    }

    #[test]
    fn pan_sample_never_inverts_and_monotonic() {
        // Across the full range: no channel goes negative for a positive
        // sample; L decreases and R increases monotonically with azimuth;
        // max(L, R) == sample everywhere (at least one channel at full gain
        // only at the extremes — otherwise both <= sample).
        let sample = 0.5f32;
        let mut prev_l = f32::INFINITY;
        let mut prev_r = f32::NEG_INFINITY;
        for i in 0..=20 {
            let az = -std::f32::consts::FRAC_PI_2 + (i as f32 / 20.0) * std::f32::consts::PI;
            let (l, r) = pan_sample(sample, az);
            assert!(l >= -1e-6, "L inverted at azimuth={az}: {l}");
            assert!(r >= -1e-6, "R inverted at azimuth={az}: {r}");
            assert!(
                l <= prev_l + 1e-6,
                "L not monotonic at {az}: {prev_l} -> {l}"
            );
            assert!(
                r >= prev_r - 1e-6,
                "R not monotonic at {az}: {prev_r} -> {r}"
            );
            prev_l = l;
            prev_r = r;
        }
    }

    #[test]
    fn distance_attenuation_inside_reference() {
        // At or below reference distance -> full gain (1.0).
        assert_eq!(distance_attenuation(0.0, 1.0, 1.0), 1.0);
        assert_eq!(distance_attenuation(1.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn distance_attenuation_falls_off() {
        // Beyond reference, gain decreases but stays in [0, 1].
        let near = distance_attenuation(2.0, 1.0, 1.0);
        let far = distance_attenuation(10.0, 1.0, 1.0);
        assert!(near < 1.0 && near > far, "near={near}, far={far}");
        assert!(far > 0.0 && far < 1.0, "far = {far}");
        // Clamped to [0, 1].
        let clamped = distance_attenuation(1e9, 1.0, 1.0);
        assert!((0.0..=1.0).contains(&clamped));
    }

    #[test]
    fn mix_non_spatial_stereo() {
        let clip = AudioClip {
            sample_rate: 48000,
            channels: 1,
            samples: std::sync::Arc::new(vec![0.5f32; 4]),
        };
        let src = AudioSource::new();
        let input = MixInput::new(&clip, &src, None);

        let mut out = vec![0.0f32; 4]; // 2 frames x 2 channels
        mix(&mut vec![input], &mut out, 2);

        // Non-spatial: same sample on both channels, volume 1.0.
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 0.5);
        assert_eq!(out[3], 0.5);
    }

    #[test]
    fn mix_volume_applied() {
        let clip = AudioClip {
            sample_rate: 48000,
            channels: 1,
            samples: std::sync::Arc::new(vec![1.0f32; 2]),
        };
        let mut src = AudioSource::new();
        src.volume = 0.5;
        let input = MixInput::new(&clip, &src, None);

        let mut out = vec![0.0f32; 2];
        mix(&mut vec![input], &mut out, 1);
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn mix_looping_restarts_cursor() {
        let clip = AudioClip {
            sample_rate: 48000,
            channels: 1,
            samples: std::sync::Arc::new(vec![0.25f32; 2]),
        };
        let mut src = AudioSource::new();
        src.looping = true;
        let input = MixInput::new(&clip, &src, None);

        // Output longer than the clip: looping must wrap the cursor and
        // keep producing samples instead of dropping the sound.
        let mut out = vec![0.0f32; 4];
        mix(&mut vec![input], &mut out, 1);
        assert_eq!(out[0], 0.25);
        assert_eq!(out[1], 0.25);
        assert_eq!(out[2], 0.25);
        assert_eq!(out[3], 0.25);
    }

    #[test]
    fn mix_non_looping_stops_at_end() {
        let clip = AudioClip {
            sample_rate: 48000,
            channels: 1,
            samples: std::sync::Arc::new(vec![0.25f32; 2]),
        };
        let src = AudioSource::new(); // looping = false
        let input = MixInput::new(&clip, &src, None);

        // Output longer than clip: sound should be removed after 2 frames.
        let mut out = vec![0.0f32; 4];
        let mut sounds = vec![input];
        mix(&mut sounds, &mut out, 1);
        assert_eq!(out[0], 0.25);
        assert_eq!(out[1], 0.25);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
        assert!(sounds.is_empty(), "sound should have been dropped");
    }
}
