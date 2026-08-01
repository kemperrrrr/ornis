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

pub struct AudioBackend {
    command_tx: Sender<AudioCommand>,
    handle: Option<thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl AudioBackend {
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

    pub fn play(&self, input: MixInput) {
        let _ = self.command_tx.send(AudioCommand::Play(input));
    }

    pub fn stop(&self, index: usize) {
        let _ = self.command_tx.send(AudioCommand::Stop(index));
    }

    pub fn set_volume(&self, index: usize, volume: f32) {
        let _ = self.command_tx.send(AudioCommand::SetVolume(index, volume));
    }

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
    let angle = (azimuth + 1.0) * std::f32::consts::FRAC_PI_4;
    let left = (angle.cos() * sample).max(-1.0).min(1.0);
    let right = (angle.sin() * sample).max(-1.0).min(1.0);
    (left, right)
}

fn distance_attenuation(distance: f32, rolloff: f32, reference: f32) -> f32 {
    if distance <= reference {
        return 1.0;
    }
    let atten = reference / (reference + rolloff * (distance - reference));
    atten.max(0.0).min(1.0)
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

            let vol = sound.volume.max(0.0).min(1.0);
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

        left = left.max(-1.0).min(1.0);
        right = right.max(-1.0).min(1.0);
        frame[0] = left;
        if output_channels > 1 {
            frame[1] = right;
        }
    }
}
