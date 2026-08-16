//! Audio merging and M4B encoding.
//!
//! ## Strategy
//!
//! All per-chunk WAV files are:
//!   1. Read into memory as i16 PCM samples
//!   2. Resampled (if necessary) to a common 44 100 Hz using `rubato`
//!   3. Concatenated with configurable silence gaps between speakers
//!   4. Written to a single temporary WAV
//!   5. Passed to `ffmpeg` (if available) to produce an M4B with AAC audio
//!      and chapter markers embedded via an ffmetadata file.
//!
//! ## Fallback (no ffmpeg)
//!
//! When `ffmpeg` is not on PATH the final combined WAV is written next to the
//! requested `.m4b` path with a `.wav` extension, accompanied by a `.cue`
//! chapter sheet.  The user is informed that they can install ffmpeg to get a
//! proper M4B, but they already have fully usable audio.
//!
//! This means the binary itself has **zero runtime dependencies** — ffmpeg is
//! purely optional for the final packaging step.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use tracing::{info, warn};

const TARGET_SAMPLE_RATE: u32 = 44_100;
const TARGET_CHANNELS: u16 = 1;

pub struct AudioMerger {
    pub pause_between_speakers_ms: u64,
    pub pause_same_speaker_ms: u64,
    /// Linear cross-fade duration (ms) applied at each segment join.
    pub crossfade_ms: u64,
    /// Feed-forward dynamics compressor settings. `None` disables compression.
    pub compressor: Option<CompressorSettings>,
}

/// Feed-forward RMS compressor settings.
///
/// Signal flow: gain computer → gain smoothing → makeup gain → limiter.
/// All level values are in dBFS.
#[derive(Debug, Clone)]
pub struct CompressorSettings {
    /// Level above which gain reduction begins (dBFS). Default −18.0.
    pub threshold_db: f32,
    /// Compression ratio above threshold (e.g. 4.0 = 4:1). Default 4.0.
    pub ratio: f32,
    /// Gain applied after compression to restore perceived loudness (dB). Default 6.0.
    pub makeup_db: f32,
    /// Attack time constant (ms) — how fast the compressor clamps down. Default 10.0.
    pub attack_ms: f32,
    /// Release time constant (ms) — how fast gain recovers. Default 100.0.
    pub release_ms: f32,
    /// Hard limiter ceiling (dBFS). Prevents any sample exceeding this. Default −1.0.
    pub limit_db: f32,
    /// RMS window length (ms) for level detection. Default 50.0.
    pub rms_window_ms: f32,
}

impl AudioMerger {
    /// Merge wav files → M4B (or WAV+CUE fallback when ffmpeg is absent).
    ///
    /// `wav_paths`: ordered `(wav_file, speaker_name)` pairs.
    /// `chapters`:  `(title, chunk_index)` — chunk_index points into `wav_paths`.
    pub async fn merge_to_m4b(
        &self,
        wav_paths: &[(PathBuf, String)],
        chapters: &[(String, usize)],
        output: &Path,
        title: &str,
        author: &str,
        cover: Option<&Path>,
    ) -> Result<()> {
        if wav_paths.is_empty() {
            anyhow::bail!("No WAV files to merge");
        }

        // ── 1. Load + normalise all WAV segments ─────────────────────────────
        info!("Loading {} WAV segments…", wav_paths.len());
        let mut segments: Vec<Segment> = Vec::with_capacity(wav_paths.len());

        for (path, speaker) in wav_paths {
            let seg = load_wav(path)
                .with_context(|| format!("Failed to load WAV: {}", path.display()))?;
            let seg = if seg.sample_rate != TARGET_SAMPLE_RATE || seg.channels != TARGET_CHANNELS {
                resample(seg)?
            } else {
                seg
            };
            // Convert f32 [-1,1] → i16 for concatenation
            let mut samples_i16: Vec<i16> = seg.samples.iter().map(|&s| f32_to_i16(s)).collect();
            // Per-segment compression: evens out level differences between
            // independently synthesised chunks before they are joined.
            if let Some(ref cfg) = self.compressor {
                compress_i16(&mut samples_i16, TARGET_SAMPLE_RATE, cfg);
            }
            segments.push(Segment {
                samples: samples_i16,
                speaker: speaker.clone(),
            });
        }

        // ── 2. Compute chapter start offsets (in samples) ────────────────────
        //      before concatenation so we know exact positions.
        let pause_diff = samples_for_ms(
            self.pause_between_speakers_ms,
            TARGET_SAMPLE_RATE,
        );
        let pause_same = samples_for_ms(
            self.pause_same_speaker_ms,
            TARGET_SAMPLE_RATE,
        );

        let mut chapter_starts_samples: Vec<(String, u64)> = Vec::new();
        let mut cursor: u64 = 0;

        for (i, seg) in segments.iter().enumerate() {
            // Record chapter start at this segment if its index is a chapter boundary
            for (ch_title, ch_idx) in chapters {
                if *ch_idx == i {
                    chapter_starts_samples.push((ch_title.clone(), cursor));
                }
            }

            let gap = if i + 1 < segments.len() {
                if segments[i + 1].speaker == seg.speaker {
                    pause_same
                } else {
                    pause_diff
                }
            } else {
                0
            };

            cursor += seg.samples.len() as u64 + gap as u64;
        }
        let total_samples = cursor;
        let fade_samples = samples_for_ms(self.crossfade_ms, TARGET_SAMPLE_RATE);

        // ── 3. Build combined PCM ─────────────────────────────────────────────
        info!("Concatenating audio ({total_samples} samples at {TARGET_SAMPLE_RATE} Hz)…");
        let mut combined: Vec<i16> = Vec::with_capacity(total_samples as usize);

        for (i, seg) in segments.iter().enumerate() {
            let samples = &seg.samples;
            // Apply fade-in on every segment except the first
            if i > 0 && fade_samples > 0 {
                let n = fade_samples.min(samples.len());
                for (j, &s) in samples.iter().take(n).enumerate() {
                    let gain = j as f32 / n as f32;
                    combined.push((s as f32 * gain).round() as i16);
                }
                combined.extend_from_slice(&samples[n..]);
            } else {
                combined.extend_from_slice(samples);
            }
            if i + 1 < segments.len() {
                // Apply fade-out on the tail before the gap
                if fade_samples > 0 {
                    let tail_len = fade_samples.min(combined.len());
                    let tail_start = combined.len() - tail_len;
                    for j in 0..tail_len {
                        let gain = 1.0 - (j as f32 / tail_len as f32);
                        combined[tail_start + j] = (combined[tail_start + j] as f32 * gain).round() as i16;
                    }
                }
                let gap = if segments[i + 1].speaker == seg.speaker {
                    pause_same
                } else {
                    pause_diff
                };
                combined.extend(std::iter::repeat(0i16).take(gap));
            }
        }

        // ── 4. Write temp WAV ─────────────────────────────────────────────────
        // ── 4. Compress ───────────────────────────────────────────────────────────────
        if let Some(ref cfg) = self.compressor {
            info!(
                "Final-pass compression (threshold {:.0} dBFS, ratio {:.0}:1, makeup +{:.0} dB)…",
                cfg.threshold_db, cfg.ratio, cfg.makeup_db
            );
            compress_i16(&mut combined, TARGET_SAMPLE_RATE, cfg);
        }

        let tmp_wav = output.with_extension("_tmp_combined.wav");
        write_wav_i16(&combined, TARGET_SAMPLE_RATE, TARGET_CHANNELS, &tmp_wav)?;
        info!("Temp WAV written: {}", tmp_wav.display());

        // ── 5. Encode to M4B ─────────────────────────────────────────────────
        let result = encode_m4b(
            &tmp_wav,
            output,
            title,
            author,
            cover,
            &chapter_starts_samples,
            total_samples,
            TARGET_SAMPLE_RATE,
        )
        .await;

        // Clean up temp WAV
        let _ = std::fs::remove_file(&tmp_wav);

        match result {
            Ok(()) => {
                info!("M4B written: {}", output.display());
            }
            Err(e) => {
                // ffmpeg unavailable or failed — emit WAV+CUE fallback
                warn!("M4B encoding failed ({e}), writing WAV+CUE fallback");
                let wav_out = output.with_extension("wav");
                write_wav_i16(&combined, TARGET_SAMPLE_RATE, TARGET_CHANNELS, &wav_out)?;
                write_cue_sheet(
                    &wav_out,
                    &chapter_starts_samples,
                    total_samples,
                    TARGET_SAMPLE_RATE,
                    title,
                    author,
                )?;
                eprintln!(
                    "\n⚠  ffmpeg not found — wrote WAV fallback:\n   {}\n   {}\n\
                     Install ffmpeg and re-run `antioch render` to get an M4B.",
                    wav_out.display(),
                    wav_out.with_extension("cue").display()
                );
            }
        }

        Ok(())
    }
}

// ─── WAV loading ─────────────────────────────────────────────────────────────

struct RawWav {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

struct Segment {
    samples: Vec<i16>,
    speaker: String,
}

fn load_wav(path: &Path) -> Result<RawWav> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("WavReader::open {}", path.display()))?;
    let spec = reader.spec();

    let samples_f32: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| Ok(s? as f32 / i16::MAX as f32))
            .collect::<hound::Result<_>>()?,
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| Ok(s? as f32 / i32::MAX as f32))
            .collect::<hound::Result<_>>()?,
        (SampleFormat::Float, _) => reader
            .samples::<f32>()
            .collect::<hound::Result<_>>()?,
        _ => anyhow::bail!(
            "Unsupported WAV format: {:?} {}bit in {}",
            spec.sample_format,
            spec.bits_per_sample,
            path.display()
        ),
    };

    Ok(RawWav {
        samples: samples_f32,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

// ─── Resampling + channel conversion ─────────────────────────────────────────

fn resample(wav: RawWav) -> Result<RawWav> {
    use rubato::{FftFixedIn, Resampler};

    // ── 1. Downmix to mono ────────────────────────────────────────────────────
    let mono: Vec<f32> = if wav.channels == 1 {
        wav.samples
    } else {
        let ch = wav.channels as usize;
        wav.samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    };

    // ── 2. Resample if needed ─────────────────────────────────────────────────
    if wav.sample_rate == TARGET_SAMPLE_RATE {
        return Ok(RawWav {
            samples: mono,
            sample_rate: TARGET_SAMPLE_RATE,
            channels: 1,
        });
    }

    let chunk_size = 1024usize;
    let mut resampler = FftFixedIn::<f32>::new(
        wav.sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        chunk_size,
        2,
        1,
    )
    .context("Failed to create resampler")?;

    let mut out_samples: Vec<f32> = Vec::new();
    let mut pos = 0;

    while pos < mono.len() {
        let end = (pos + chunk_size).min(mono.len());
        let mut chunk = mono[pos..end].to_vec();
        // Pad last chunk if necessary
        chunk.resize(chunk_size, 0.0);

        let output = resampler
            .process(&[&chunk], None)
            .context("Resampler process failed")?;
        out_samples.extend_from_slice(&output[0]);
        pos += chunk_size;
    }

    Ok(RawWav {
        samples: out_samples,
        sample_rate: TARGET_SAMPLE_RATE,
        channels: 1,
    })
}

// ─── WAV writing ─────────────────────────────────────────────────────────────

fn write_wav_i16(samples: &[i16], sample_rate: u32, channels: u16, path: &Path) -> Result<()> {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Cannot create WAV: {}", path.display()))?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn f32_to_i16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

// ─── Dynamics compressor ─────────────────────────────────────────────────────────────

/// Feed-forward RMS compressor applied in-place to i16 PCM.
///
/// Algorithm:
///   1. Convert window of i16 samples to f32, compute RMS level in dBFS.
///   2. Gain computer: below threshold, gain = 0 dB; above, reduce by (R-1)/R.
///   3. Smooth gain with independent attack/release time constants (1-pole IIR).
///   4. Apply smoothed gain + makeup gain to each sample.
///   5. Hard-limit output to `limit_db` ceiling.
fn compress_i16(samples: &mut [i16], sample_rate: u32, cfg: &CompressorSettings) {
    if samples.is_empty() { return; }

    let sr = sample_rate as f32;
    let attack_coeff  = 1.0 - (-1.0_f32 / (cfg.attack_ms  * 0.001 * sr)).exp();
    let release_coeff = 1.0 - (-1.0_f32 / (cfg.release_ms * 0.001 * sr)).exp();
    let rms_len = ((cfg.rms_window_ms * 0.001 * sr) as usize).max(1);

    let scale = 1.0_f32 / i16::MAX as f32;
    let limit_linear = db_to_linear(cfg.limit_db);
    let makeup_linear = db_to_linear(cfg.makeup_db);

    let n = samples.len();

    // Pre-compute normalised float samples once — avoids repeated int→float casts
    // and eliminates the catastrophic cancellation that caused drift in the
    // incremental sum-of-squares when applied to 50+ million samples.
    let floats: Vec<f32> = samples.iter().map(|&s| s as f32 * scale).collect();

    // Precompute RMS per sample using an exact sliding window.
    // Every `rms_len` samples we fully re-accumulate sum_sq from scratch to
    // flush any residual f32 rounding error.
    let mut rms_values: Vec<f32> = Vec::with_capacity(n);
    {
        let mut sum_sq = 0.0_f32;
        // Seed with first window
        let seed_end = rms_len.min(n);
        for &f in &floats[..seed_end] { sum_sq += f * f; }

        for i in 0..n {
            // Slide: add incoming sample, remove outgoing
            if i + rms_len < n { sum_sq += floats[i + rms_len] * floats[i + rms_len]; }
            if i > 0           { sum_sq = (sum_sq - floats[i - 1] * floats[i - 1]).max(0.0); }

            // Periodic exact re-accumulation every `rms_len` samples to prevent drift
            if i > 0 && i % rms_len == 0 {
                let lo = i.min(n.saturating_sub(rms_len));
                let hi = (lo + rms_len).min(n);
                sum_sq = floats[lo..hi].iter().map(|&f| f * f).sum();
            }

            rms_values.push((sum_sq / rms_len as f32).sqrt());
        }
    }

    let mut gain_db_smooth = 0.0_f32;

    for i in 0..n {
        let rms = rms_values[i];
        let level_db = if rms > 1e-9 { 20.0 * rms.log10() } else { -120.0 };

        let over = level_db - cfg.threshold_db;
        let target_gain_db = if over > 0.0 {
            -over * (1.0 - 1.0 / cfg.ratio)
        } else {
            0.0
        };

        let coeff = if target_gain_db < gain_db_smooth { attack_coeff } else { release_coeff };
        gain_db_smooth += coeff * (target_gain_db - gain_db_smooth);

        let total_linear = db_to_linear(gain_db_smooth) * makeup_linear;
        let f_out = (floats[i] * total_linear).clamp(-limit_linear, limit_linear);
        samples[i] = (f_out * i16::MAX as f32) as i16;
    }
}

#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ─── Silence helpers ─────────────────────────────────────────────────────────

fn samples_for_ms(ms: u64, sample_rate: u32) -> usize {
    (ms as usize * sample_rate as usize) / 1000
}

// ─── ffmpeg M4B encoding ─────────────────────────────────────────────────────

async fn encode_m4b(
    input_wav: &Path,
    output: &Path,
    title: &str,
    author: &str,
    cover: Option<&Path>,
    chapter_starts: &[(String, u64)],
    total_samples: u64,
    sample_rate: u32,
) -> Result<()> {
    // Check ffmpeg is available
    let ffmpeg = find_ffmpeg().ok_or_else(|| anyhow::anyhow!("ffmpeg not found on PATH"))?;

    // Write ffmetadata file
    let meta_path = output.with_extension("_ffmeta.txt");
    write_ffmetadata(
        &meta_path,
        title,
        author,
        chapter_starts,
        total_samples,
        sample_rate,
    )?;

    // Build ffmpeg command
    // Input 0: audio WAV
    // Input 1: ffmetadata
    // Input 2 (optional): cover image
    let mut cmd = tokio::process::Command::new(&ffmpeg);
    cmd.arg("-y")
        .arg("-i")
        .arg(input_wav);

    if let Some(cover_path) = cover {
        cmd.arg("-i").arg(cover_path);
    }

    cmd.arg("-i").arg(&meta_path);

    let meta_input_idx = if cover.is_some() { 2 } else { 1 };
    cmd.arg("-map_metadata").arg(meta_input_idx.to_string());

    // Audio stream
    cmd.arg("-map").arg("0:a");

    // Cover art
    if cover.is_some() {
        cmd.arg("-map")
            .arg("1:v")
            .arg("-c:v")
            .arg("copy")
            .arg("-disposition:v:0")
            .arg("attached_pic");
    }

    cmd.arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("128k")
        .arg("-movflags")
        .arg("+faststart")
        .arg(output);

    let output_result = cmd
        .output()
        .await
        .context("Failed to spawn ffmpeg process")?;

    let _ = std::fs::remove_file(&meta_path);

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        anyhow::bail!(
            "ffmpeg exited with {}: {}",
            output_result.status,
            &stderr[stderr.len().saturating_sub(500)..]
        );
    }

    Ok(())
}

fn find_ffmpeg() -> Option<PathBuf> {
    // Check bundled binary first (same directory as the executable)
    if let Ok(exe) = std::env::current_exe() {
        let bundled = exe.parent()?.join("ffmpeg");
        if bundled.exists() {
            return Some(bundled);
        }
        #[cfg(windows)]
        {
            let bundled_win = exe.parent()?.join("ffmpeg.exe");
            if bundled_win.exists() {
                return Some(bundled_win);
            }
        }
    }
    // Fall back to PATH
    which_ffmpeg()
}

fn which_ffmpeg() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join("ffmpeg");
            if candidate.exists() {
                return Some(candidate);
            }
            #[cfg(windows)]
            {
                let win = dir.join("ffmpeg.exe");
                if win.exists() {
                    return Some(win);
                }
            }
            None
        })
    })
}

// ─── ffmetadata writer ────────────────────────────────────────────────────────

fn write_ffmetadata(
    path: &Path,
    title: &str,
    author: &str,
    chapter_starts: &[(String, u64)],
    total_samples: u64,
    sample_rate: u32,
) -> Result<()> {
    use std::fmt::Write as FmtWrite;

    let mut buf = String::new();
    writeln!(buf, ";FFMETADATA1")?;
    writeln!(buf, "title={}", escape_ffmeta(title))?;
    writeln!(buf, "artist={}", escape_ffmeta(author))?;
    writeln!(buf, "genre=Audiobook")?;
    writeln!(buf)?;

    let samples_to_ms = |s: u64| -> u64 { s * 1000 / sample_rate as u64 };
    let total_ms = samples_to_ms(total_samples);

    for (i, (ch_title, start_sample)) in chapter_starts.iter().enumerate() {
        let start_ms = samples_to_ms(*start_sample);
        let end_ms = if i + 1 < chapter_starts.len() {
            samples_to_ms(chapter_starts[i + 1].1)
        } else {
            total_ms
        };

        writeln!(buf, "[CHAPTER]")?;
        writeln!(buf, "TIMEBASE=1/1000")?;
        writeln!(buf, "START={start_ms}")?;
        writeln!(buf, "END={end_ms}")?;
        writeln!(buf, "title={}", escape_ffmeta(ch_title))?;
        writeln!(buf)?;
    }

    std::fs::write(path, buf)?;
    Ok(())
}

fn escape_ffmeta(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('=', "\\=")
        .replace(';', "\\;")
        .replace('#', "\\#")
        .replace('\n', " ")
}

// ─── CUE sheet fallback ───────────────────────────────────────────────────────

fn write_cue_sheet(
    wav_path: &Path,
    chapter_starts: &[(String, u64)],
    _total_samples: u64,
    sample_rate: u32,
    title: &str,
    author: &str,
) -> Result<()> {
    use std::fmt::Write as FmtWrite;

    let cue_path = wav_path.with_extension("cue");
    let wav_name = wav_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    let mut buf = String::new();
    writeln!(buf, "PERFORMER \"{}\"", author)?;
    writeln!(buf, "TITLE \"{}\"", title)?;
    writeln!(buf, "FILE \"{}\" WAVE", wav_name)?;

    let samples_to_frames = |s: u64| -> u64 {
        // CUE time format: MM:SS:FF where FF = 1/75 second frames
        s * 75 / sample_rate as u64
    };

    for (i, (ch_title, start_sample)) in chapter_starts.iter().enumerate() {
        let frames = samples_to_frames(*start_sample);
        let mm = frames / (75 * 60);
        let ss = (frames / 75) % 60;
        let ff = frames % 75;

        writeln!(buf, "  TRACK {:02} AUDIO", i + 1)?;
        writeln!(buf, "    TITLE \"{}\"", ch_title)?;
        writeln!(buf, "    INDEX 01 {:02}:{:02}:{:02}", mm, ss, ff)?;
    }

    std::fs::write(&cue_path, buf)?;
    Ok(())
}
