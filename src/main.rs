#![allow(unused_imports, dead_code)]

mod audio;
mod llm;
mod script;
mod tts;

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing::{debug, info};

/// `MakeWriter` that routes each log line through `MultiProgress::suspend()`.
/// This ensures tracing output appears *above* the progress bars rather than
/// interleaving with spinner frames mid-redraw.
struct MpWriter(MultiProgress);
impl io::Write for MpWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.suspend(|| io::stderr().write(buf))
    }
    fn flush(&mut self) -> io::Result<()> { io::stderr().flush() }
}
struct MpMakeWriter(MultiProgress);
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for MpMakeWriter {
    type Writer = MpWriter;
    fn make_writer(&'a self) -> Self::Writer { MpWriter(self.0.clone()) }
}

use crate::{
    audio::merger::{AudioMerger, CompressorSettings},
    llm::{api::LlmApiClient, LlmBackend},
    script::{annotator::ScriptAnnotator, chunker::Chunker, markdown::MarkdownParser},
};

#[cfg(feature = "ggml")]
use crate::tts::TtsEngine;

#[cfg(feature = "ggml")]
use crate::tts::CppTts;
#[cfg(feature = "ggml")]
use crate::tts::cpp::{Mode, CppBaseTts, GgmlIclRef, read_wav_f32};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Anchor text used to generate a reference voice clip in two-phase GGML mode
/// (must match the constant in `tts/cpp.rs::REF_ANCHOR_TEXT`).
#[cfg(feature = "ggml")]
const GGML_REF_ANCHOR_TEXT: &str =
    "Such a Fire existeth extending through the rushings of Air or even a Fire Formless \
     whence comes the Image of a Voice or even a flashing Light abounding, revolving, \
     whirling forth, crying aloud.";

// ─── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "antioch",
    about = "Markdown → M4B audiobook generator (llama.cpp + qwentts.cpp)",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Full pipeline: markdown → annotated script → audio → m4b
    Generate(GenerateArgs),

    /// Only parse markdown and annotate script, save as JSON for inspection
    Annotate(AnnotateArgs),

    /// Convert an existing annotated script JSON to M4B
    Render(RenderArgs),

    /// Download required models to local cache (run once before offline use)
    Fetch(FetchArgs),
}

// ── generate ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub struct GenerateArgs {
    /// Input markdown file
    #[arg(value_name = "INPUT.md")]
    pub input: PathBuf,

    /// Output M4B file  [default: <input stem>.m4b]
    #[arg(short, long, value_name = "OUTPUT.m4b")]
    pub output: Option<PathBuf>,

    /// Audiobook title (embedded in M4B metadata)
    #[arg(long, default_value = "Audiobook")]
    pub title: String,

    /// Audiobook author
    #[arg(long, default_value = "")]
    pub author: String,

    #[command(flatten)]
    pub llm: LlmArgs,

    #[command(flatten)]
    pub tts: TtsArgs,

    /// Skip LLM annotation; attribute every line to a single narrator voice
    #[arg(long)]
    pub single_speaker: bool,

    /// Anatomy-only voice identity for VoiceDesign mode.
    /// Use register + timbre/texture terms ONLY (e.g. "male baritone, rich chest
    /// resonance, warm smooth timbre, slight gravelly texture").
    /// Do NOT include delivery, pacing, or emotion words here -- those belong in
    /// --narrator-style. Mixing acoustic and behavioral terms causes voice drift
    /// between segments.
    #[arg(long, default_value = "male baritone, rich chest resonance, warm smooth timbre",
          env = "ANTIOCH_NARRATOR_VOICE")]
    pub narrator_voice: String,

    /// Delivery and style direction for VoiceDesign mode.
    /// Use emotion/pacing/attitude words (e.g. "measured, scholarly, calm and
    /// authoritative"). This is passed as the per-generation instruct string
    /// and shapes how the voice reads each segment.
    #[arg(long, default_value = "measured, calm, authoritative",
          env = "ANTIOCH_NARRATOR_STYLE")]
    pub narrator_style: String,

    /// Save intermediate annotated script JSON alongside the output
    #[arg(long)]
    pub save_script: bool,

    /// Cover image for M4B (JPEG or PNG)
    #[arg(long, value_name = "COVER.jpg")]
    pub cover: Option<PathBuf>,
}

// ── annotate ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub struct AnnotateArgs {
    #[arg(value_name = "INPUT.md")]
    pub input: PathBuf,

    #[arg(short, long, value_name = "SCRIPT.json")]
    pub output: Option<PathBuf>,

    #[command(flatten)]
    pub llm: LlmArgs,

    #[arg(long)]
    pub single_speaker: bool,

    #[arg(long, default_value = "male baritone, rich chest resonance, warm smooth timbre",
          env = "ANTIOCH_NARRATOR_VOICE")]
    pub narrator_voice: String,

    #[arg(long, default_value = "measured, calm, authoritative",
          env = "ANTIOCH_NARRATOR_STYLE")]
    pub narrator_style: String,
}

// ── render ────────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub struct RenderArgs {
    #[arg(value_name = "SCRIPT.json")]
    pub script: PathBuf,

    #[arg(short, long, value_name = "OUTPUT.m4b")]
    pub output: Option<PathBuf>,

    #[arg(long, default_value = "Audiobook")]
    pub title: String,

    #[arg(long, default_value = "")]
    pub author: String,

    #[command(flatten)]
    pub tts: TtsArgs,

    #[arg(long, value_name = "COVER.jpg")]
    pub cover: Option<PathBuf>,
}

// ── fetch ─────────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub struct FetchArgs {
    /// Fetch LLM model weights (pass --no-llm to skip)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub llm: bool,

    /// Fetch TTS model weights (pass --no-tts to skip)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub tts: bool,
}

// ── shared sub-arg groups ─────────────────────────────────────────────────────

#[derive(clap::Args, Debug, Clone)]
pub struct LlmArgs {
    /// LLM backend: "llama" (local GGUF via llama.cpp, requires --features llama)
    /// or "api" (OpenAI-compatible HTTP endpoint)
    #[arg(long, default_value = "llama", env = "ANTIOCH_LLM_BACKEND")]
    pub llm_backend: String,

    /// GGUF model path for --llm-backend=llama.
    /// If omitted, resolved to unsloth/Qwen3.5-2B-GGUF / Qwen3.5-2B-Q8_0.gguf
    /// from the HuggingFace cache, downloading it if not already present.
    #[arg(long, default_value = "", env = "ANTIOCH_LLM_GGUF")]
    pub llm_gguf: String,

    /// GPU layers to offload for --llm-backend=llama (negative = all, 0 = none)
    #[arg(long, default_value_t = -1, env = "ANTIOCH_LLM_GPU_LAYERS")]
    pub llm_gpu_layers: i32,

    /// OpenAI-compatible API base URL (for --llm-backend=api)
    #[arg(long, default_value = "http://localhost:11434/v1", env = "ANTIOCH_LLM_URL")]
    pub llm_url: String,

    /// API key (for --llm-backend=api)
    #[arg(long, default_value = "local", env = "ANTIOCH_LLM_KEY")]
    pub llm_key: String,

    /// Model name (for --llm-backend=api)
    #[arg(long, default_value = "qwen2.5:14b", env = "ANTIOCH_LLM_MODEL")]
    pub llm_model: String,

    /// Text chunk size (chars) for LLM annotation
    #[arg(long, default_value_t = 3000)]
    pub chunk_size: usize,

    /// Max new tokens per LLM call
    #[arg(long, default_value_t = 4096)]
    pub max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.6)]
    pub temperature: f64,
}

#[derive(clap::Args, Debug, Clone)]
pub struct TtsArgs {
    /// TTS backend: "ggml" (qwentts.cpp, requires --features ggml) or
    /// "none" (skip audio, script-only)
    #[arg(long, default_value = "ggml", env = "ANTIOCH_TTS_BACKEND")]
    pub tts_backend: String,

    /// GGML synthesis mode for --tts-backend=ggml:
    /// "twophase" (default; VoiceDesign reference gen + Base ICL voice
    /// cloning, the ONNX-equivalent pipeline), "voicedesign" (attribute
    /// instruction drives the voice), or "customvoice" (fixed named speaker
    /// table).  Must match the GGUF model(s) loaded.
    #[arg(long, default_value = "twophase", env = "ANTIOCH_TTS_GGML_MODE")]
    pub tts_ggml_mode: String,

    /// Quantization precision for the GGUF models fetched by default:
    /// q4 (Q4_K_M), q8 (Q8_0), f32, or bf16.  Only used when the user does not
    /// specify model paths explicitly (--tts-ggml-talker / -base-talker /
    /// -codec) — those take precedence.  [default: q4]
    #[arg(long, default_value = "q4", env = "ANTIOCH_TTS_GGML_PRECISION")]
    pub tts_ggml_precision: String,

    /// Talker GGUF path for --tts-backend=ggml.  If omitted, resolved to the
    /// `voice_design` model for the selected precision (--tts-ggml-precision)
    /// and fetched into the HuggingFace cache if not already present.
    #[arg(long, env = "ANTIOCH_TTS_GGML_TALKER")]
    pub tts_ggml_talker: Option<String>,

    /// Codec/tokenizer GGUF path for --tts-backend=ggml.  If omitted, resolved
    /// to the tokenizer for the selected precision and fetched if not cached.
    #[arg(long, env = "ANTIOCH_TTS_GGML_CODEC")]
    pub tts_ggml_codec: Option<String>,

    /// Base talker GGUF for --tts-backend=ggml in two-phase mode.  If omitted,
    /// resolved to the `base` model for the selected precision and fetched if
    /// not cached.
    #[arg(long, env = "ANTIOCH_TTS_GGML_BASE_TALKER")]
    pub tts_ggml_base_talker: Option<String>,

    /// Max audio tokens per TTS call
    #[arg(long, default_value_t = 2048)]
    pub tts_max_tokens: usize,

    /// Pause between different speakers (ms)
    #[arg(long, default_value_t = 500)]
    pub pause_between_speakers_ms: u64,

    /// Pause when same speaker continues (ms)
    #[arg(long, default_value_t = 250)]
    pub pause_same_speaker_ms: u64,

    /// Fixed random seed for TTS sampling (omit for a random seed).
    /// Using a fixed seed keeps the voice character consistent across all
    /// segments — without it each chunk seeds from the system clock and the
    /// voice can drift noticeably between segments.
    #[arg(long, env = "ANTIOCH_TTS_SEED")]
    pub tts_seed: Option<u64>,

    /// Sampling temperature for TTS token generation (0.0–1.0).
    /// Lower values produce a more stable, consistent voice; higher values
    /// add more expressive variation.
    #[arg(long, default_value_t = 0.7, env = "ANTIOCH_TTS_TEMPERATURE")]
    pub tts_temperature: f64,

    /// Sliding-window size for the talker KV cache (number of tokens).
    /// Older tokens are evicted once the window fills, keeping memory O(window)
    /// rather than O(sequence length). 512 ≈ 40s of speech context.
    /// Set 0 to disable (unbounded pre-allocated cache).
    #[arg(long, default_value_t = 512, env = "ANTIOCH_KV_WINDOW")]
    pub kv_window: usize,

    /// GPU device index for the TTS process.  In a process-per-GPU parallel
    /// run, launch one process per device with a distinct --device-id and a
    /// matching shard (see --shard-count).
    #[arg(long, default_value_t = 0, env = "ANTIOCH_TTS_DEVICE_ID")]
    pub device_id: i32,

    /// Total number of parallel shard processes.  Combine with --shard-id to
    /// split TTS synthesis across multiple processes/GPUs.  All shards share
    /// the same output WAV directory (files are named by chunk index), so they
    /// can run concurrently and be merged afterwards with --merge-only.
    #[arg(long, default_value_t = 1, env = "ANTIOCH_TTS_SHARD_COUNT")]
    pub shard_count: usize,

    /// Zero-based index of THIS process's shard (0..shard-count).  Each shard
    /// synthesises chunks where `chunk_index % shard_count == shard_id`, so the
    /// `NNNN.wav` files written are disjoint across shards.
    #[arg(long, default_value_t = 0, env = "ANTIOCH_TTS_SHARD_ID")]
    pub shard_id: usize,

    /// Skip synthesis entirely and just merge the already-generated WAV files
    /// in the output directory into the final M4B.  Used after all shard
    /// processes have finished.
    #[arg(long)]
    pub merge_only: bool,

    /// Cross-fade duration (ms) applied between consecutive segments.
    /// A short fade-out/fade-in at each join hides any residual level
    /// discontinuity between independently synthesised chunks.
    #[arg(long, default_value_t = 20)]
    pub crossfade_ms: u64,

    /// Enable dynamics compression on the final mixed audio.
    /// Evens out volume differences between segments. On by default.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub compress: bool,

    /// Compressor threshold in dBFS. Gain reduction begins above this level.
    #[arg(long, default_value_t = -18.0)]
    pub compress_threshold_db: f32,

    /// Compressor ratio (e.g. 4 = 4:1 above threshold).
    #[arg(long, default_value_t = 4.0)]
    pub compress_ratio: f32,

    /// Makeup gain in dB applied after compression.
    #[arg(long, default_value_t = 6.0)]
    pub compress_makeup_db: f32,

    /// Compressor attack time constant (ms).
    #[arg(long, default_value_t = 10.0)]
    pub compress_attack_ms: f32,

    /// Compressor release time constant (ms).
    #[arg(long, default_value_t = 100.0)]
    pub compress_release_ms: f32,

    /// Hard limiter ceiling in dBFS (prevents clipping after makeup gain).
    #[arg(long, default_value_t = -1.0)]
    pub compress_limit_db: f32,
}


// ─── entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());

    // Single shared MultiProgress that owns every spinner and bar.
    // All tracing log lines are routed through MpMakeWriter which calls
    // mp.suspend() before writing to stderr — no interleaving with spinners.
    let mp = MultiProgress::new();
    if !is_tty {
        mp.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }

    {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(MpMakeWriter(mp.clone()))
                    .without_time()
                    .with_ansi(is_tty)
                    .with_filter(
                        tracing_subscriber::EnvFilter::from_default_env()
                            .add_directive("antioch=info".parse().unwrap()),
                    ),
            )
            .init();
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => cmd_generate(args, mp).await,
        Commands::Annotate(args) => cmd_annotate(args, mp).await,
        Commands::Render(args)   => cmd_render(args, mp).await,
        Commands::Fetch(args)    => cmd_fetch(args).await,
    }
}

// ─── command: generate ────────────────────────────────────────────────────────

async fn cmd_generate(args: GenerateArgs, mp: MultiProgress) -> Result<()> {
    let output = args.output.clone().unwrap_or_else(|| {
        args.input
            .with_extension("m4b")
    });

    banner();
    println!(
        "  {} {}",
        style("Input:").bold().cyan(),
        args.input.display()
    );
    println!(
        "  {} {}",
        style("Output:").bold().cyan(),
        output.display()
    );
    println!();

    // ── 1. Parse markdown ─────────────────────────────────────────────────────
    let step = step_bar(&mp, "Parsing markdown");
    let raw = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Cannot read input file: {}", args.input.display()))?;
    let text = MarkdownParser::extract_plain_text(&raw);
    step.finish_and_clear();

    // ── 2. Annotate script ────────────────────────────────────────────────────
    let script_entries = if args.single_speaker {
        let step = step_bar(&mp, "Chunking (single-speaker mode)");
        let entries = Chunker::single_speaker(&text, &args.narrator_voice, &args.narrator_style);
        step.finish_and_clear();
        entries
    } else {
        let step = step_bar(&mp, "Annotating script via LLM");
        let backend = build_llm_backend(&args.llm).await?;
        let annotator = ScriptAnnotator::new(backend);
        let entries = annotator
            .annotate(&text, args.llm.chunk_size)
            .await
            .context("LLM script annotation failed")?;
        step.finish_and_clear();
        entries
    };

    if args.save_script {
        let script_path = output.with_extension("script.json");
        let json = serde_json::to_string_pretty(&script_entries)?;
        std::fs::write(&script_path, json)?;
        info!("Script saved to {}", script_path.display());
    }

    // ── 3. Group into TTS chunks ──────────────────────────────────────────────
    let chunks = Chunker::group_by_speaker(script_entries, 500);
    info!("{} TTS chunks after grouping", chunks.len());

    // ── 4. Synthesise audio ───────────────────────────────────────────────────
    let wav_dir = output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(format!(
            ".antioch_{}",
            output.file_stem().unwrap_or_default().to_string_lossy()
        ));
    std::fs::create_dir_all(&wav_dir)?;

    #[allow(unused_mut)]
    let mut wav_paths: Vec<(PathBuf, String)> = Vec::new(); // (path, speaker)

    // ── Merge-only: recombine already-generated WAV files into the M4B ──────
    // Used after a multi-GPU sharded run: once every shard has written its
    // NNNN.wav files to wav_dir, this rebuilds the full ordered list and merges.
    if args.tts.merge_only {
        let wav_paths = collect_all_wavs(&chunks, &wav_dir);
        if wav_paths.is_empty() {
            println!("{}", style("No WAV files found in output dir — exiting.").yellow());
            return Ok(());
        }
        let chapters = build_chapters(&chunks);
        merge_wavs(
            &wav_paths,
            &chapters,
            &output,
            &args.title,
            &args.author,
            args.cover.as_deref(),
            &args.tts,
            &mp,
        ).await?;
        let _ = std::fs::remove_dir_all(&wav_dir);
        println!("\n{} {}", style("✔ Done!").bold().green(), output.display());
        return Ok(());
    }

    if args.tts.tts_backend != "none" {
        // Dispatch to the selected backend.
        #[cfg(feature = "ggml")]
        { wav_paths = run_tts_backend(
            &args.tts, &chunks, &wav_dir, &mp,
        ).await?; }
        #[cfg(not(feature = "ggml"))]
        anyhow::bail!("TTS requires building with --features ggml (qwentts.cpp)");
    } else {
        println!("{}", style("TTS skipped (--tts-backend=none)").yellow());
    }

    if wav_paths.is_empty() {
        println!("{}", style("No audio generated — exiting.").yellow());
        return Ok(());
    }

    // ── 5. Build chapter list from chunks ─────────────────────────────────────
    let chapters = build_chapters(&chunks);

    // ── 6. Merge + encode M4B ────────────────────────────────────────────────
    // In a sharded run each process exits after writing its own WAVs; the user
    // runs a final `--merge-only` pass once every shard has finished.  A single
    // (non-sharded) process merges inline as before.
    if args.tts.shard_count > 1 {
        println!(
            "{}\n  Shard {}/{} complete. Once all shards have finished, run the \
             merge with `--merge-only` to produce the M4B.",
            style("✔ Shard synthesised").green().bold(),
            args.tts.shard_id,
            args.tts.shard_count,
        );
        return Ok(());
    }

    merge_wavs(
        &wav_paths,
        &chapters,
        &output,
        &args.title,
        &args.author,
        args.cover.as_deref(),
        &args.tts,
        &mp,
    ).await?;

    // ── Cleanup temp WAVs ────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&wav_dir);

    println!(
        "\n{} {}",
        style("✔ Done!").bold().green(),
        output.display()
    );
    Ok(())
}

// ─── command: annotate ────────────────────────────────────────────────────────

async fn cmd_annotate(args: AnnotateArgs, _mp: MultiProgress) -> Result<()> {
    let output = args.output.clone().unwrap_or_else(|| {
        args.input.with_extension("script.json")
    });

    let raw = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Cannot read: {}", args.input.display()))?;
    let text = MarkdownParser::extract_plain_text(&raw);

    let entries = if args.single_speaker {
        Chunker::single_speaker(&text, &args.narrator_voice, &args.narrator_style)
    } else {
        let backend = build_llm_backend(&args.llm).await?;
        let annotator = ScriptAnnotator::new(backend);
        annotator.annotate(&text, args.llm.chunk_size).await?
    };

    let json = serde_json::to_string_pretty(&entries)?;
    std::fs::write(&output, json)?;
    println!(
        "{} {} entries → {}",
        style("✔").green(),
        entries.len(),
        output.display()
    );
    Ok(())
}

// ─── command: render ─────────────────────────────────────────────────────────

async fn cmd_render(args: RenderArgs, mp: MultiProgress) -> Result<()> {
    let output = args.output.clone().unwrap_or_else(|| {
        args.script.with_extension("m4b")
    });

    let json = std::fs::read_to_string(&args.script)?;
    let entries: Vec<script::ScriptEntry> = serde_json::from_str(&json)?;
    let chunks = Chunker::group_by_speaker(entries, 500);

    let wav_dir = output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(format!(
            ".antioch_{}",
            output.file_stem().unwrap_or_default().to_string_lossy()
        ));
    std::fs::create_dir_all(&wav_dir)?;

    #[allow(unused_mut)]
    let mut wav_paths: Vec<(PathBuf, String)> = Vec::new();

    // ── Merge-only: recombine already-generated WAV files into the M4B ──────
    if args.tts.merge_only {
        let wav_paths = collect_all_wavs(&chunks, &wav_dir);
        if wav_paths.is_empty() {
            println!("{}", style("No WAV files found in output dir — exiting.").yellow());
            return Ok(());
        }
        let chapters = build_chapters(&chunks);
        merge_wavs(
            &wav_paths,
            &chapters,
            &output,
            &args.title,
            &args.author,
            args.cover.as_deref(),
            &args.tts,
            &mp,
        ).await?;
        let _ = std::fs::remove_dir_all(&wav_dir);
        println!("{} {}", style("✔").green(), output.display());
        return Ok(());
    }

    if args.tts.tts_backend != "none" {
        #[cfg(feature = "ggml")]
        { wav_paths = run_tts_backend(
            &args.tts, &chunks, &wav_dir, &mp,
        ).await?; }
        #[cfg(not(feature = "ggml"))]
        anyhow::bail!("TTS requires building with --features ggml (qwentts.cpp)");
    }

    let chapters = build_chapters(&chunks);

    // In a sharded run each process exits after writing its own WAVs; a final
    // `--merge-only` pass produces the M4B once every shard has finished.
    if args.tts.shard_count > 1 {
        println!(
            "{}\n  Shard {}/{} complete. Once all shards have finished, run the \
             merge with `--merge-only` to produce the M4B.",
            style("✔ Shard synthesised").green().bold(),
            args.tts.shard_id,
            args.tts.shard_count,
        );
        return Ok(());
    }

    merge_wavs(
        &wav_paths,
        &chapters,
        &output,
        &args.title,
        &args.author,
        args.cover.as_deref(),
        &args.tts,
        &mp,
    ).await?;

    let _ = std::fs::remove_dir_all(&wav_dir);
    println!("{} {}", style("✔").green(), output.display());
    Ok(())
}

// ─── command: fetch ───────────────────────────────────────────────────────────

async fn cmd_fetch(args: FetchArgs) -> Result<()> {
    if args.llm {
        #[cfg(feature = "llama")] {
            println!("Fetching LLM GGUF ({LLM_DEFAULT_REPO}/{LLM_DEFAULT_FILE})…");
            tokio::task::spawn_blocking(fetch_llm_gguf_sync)
                .await
                .context("LLM GGUF fetch thread panicked")??
                .display()
                .to_string();
            println!("{} LLM weights cached", style("✔").green());
        }
        #[cfg(not(feature = "llama"))]
        anyhow::bail!("fetch --llm requires building with --features llama");
    }
    if args.tts {
        #[cfg(feature = "ggml")] {
            println!("Fetching TTS GGUF model weights…");
            for kind in &["voice_design", "base", "tokenizer"] {
                resolve_ggml_model(None, "q4", kind).await
                    .with_context(|| format!("fetch ggml {kind}"))?;
            }
            println!("{} TTS weights cached", style("✔").green());
        }
        #[cfg(not(feature = "ggml"))]
        anyhow::bail!("fetch --tts requires building with --features ggml");
    }
    Ok(())
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn banner() {
    println!(
        "{}",
        style(
            r#"
  ╔═══════════════════════════════════╗
  ║     A N T I O C H                ║
  ║     Markdown → M4B Audiobook     ║
  ╚═══════════════════════════════════╝"#
        )
        .cyan()
        .bold()
    );
    println!();
}

// ─── synthesis with rich progress ───────────────────────────────────────────

/// Run TTS on every chunk, printing a rich multi-line progress display:
///
/// ```
///  Synthesising  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  312/5010  6.2%

/// qwentts.cpp two-phase pipeline, phase 1: generate one reference voice clip
/// per unique speaker with the voice_design GGUF, and return a map ready for
/// `CppBaseTts` (reference PCM + anchor transcript).
///
/// Skips speakers whose `.ref.wav` already exists (checkpoint resume).
#[cfg(feature = "ggml")]
async fn generate_ref_voices_ggml(
    vd:          &CppTts,
    chunks:      &[script::Chunk],
    wav_dir:     &PathBuf,
    seed:        Option<u64>,
    temperature: f64,
    max_frames:  usize,
) -> Result<HashMap<String, GgmlIclRef>> {
    use std::collections::BTreeMap;

    let structural = ["TITLE", "CHAPTER", "SECTION"];
    let narrator_speaker = chunks.iter()
        .find(|c| !structural.contains(&c.speaker.as_str()))
        .map(|c| c.speaker.clone())
        .unwrap_or_else(|| "NARRATOR".to_string());

    // Map speaker -> delivery style taken from the first chunk that uses it.
    // In single-speaker mode the speaker key *is* the literal --narrator-voice
    // description and `instruct` is the literal --narrator-style, so both must
    // be honoured verbatim instead of being re-derived from the palette.
    let mut speakers: BTreeMap<String, String> = BTreeMap::new();
    for chunk in chunks {
        let is_structural = structural.contains(&chunk.speaker.as_str());
        let s = if is_structural { narrator_speaker.clone() } else { chunk.speaker.clone() };
        // Only body chunks contribute the delivery style; structural headings
        // carry their own one-off instruct ("Slow, atmospheric.") which must not
        // become the narrator's reference-voice style.
        match speakers.entry(s) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(if is_structural { String::new() } else { chunk.instruct.clone() });
            }
            std::collections::btree_map::Entry::Occupied(mut o) => {
                if o.get().is_empty() && !is_structural {
                    o.insert(chunk.instruct.clone());
                }
            }
        }
    }

    let mut ref_voices: HashMap<String, GgmlIclRef> = HashMap::new();

    for (idx, (speaker, chunk_style)) in speakers.iter().enumerate() {
        let (voice, style) = if is_literal_voice_description(speaker) {
            // User-supplied acoustic description: use it as-is.
            (speaker.clone(), chunk_style.clone())
        } else {
            voice_profile_for_speaker(speaker, idx)
        };
        let ref_wav = wav_dir.join(format!("{}.ref.wav", speaker_slug(speaker)));

        // Checkpoint: reuse an existing ref WAV from a prior run / shard.
        if !ref_wav.exists() {
            info!("Generating ref voice: '{}' ({})…", speaker, voice);
            vd.synthesise_ref_voice(speaker, &voice, &style, &ref_wav, max_frames, seed, temperature)
                .await
                .with_context(|| format!("qwentts.cpp VoiceDesign ref failed for speaker '{speaker}'"))?;
            debug!("  Saved reference WAV: {}", ref_wav.display());
        } else {
            info!("Reusing existing ref voice: '{}'", speaker);
        }

        // Load the ref WAV into mono float32 PCM for ICL cloning.
        let ref_audio = read_wav_f32(&ref_wav)
            .with_context(|| format!("qwentts.cpp ref WAV read failed for speaker '{speaker}'"))?;
        if ref_audio.is_empty() {
            anyhow::bail!("qwentts.cpp ref WAV is empty: {}", ref_wav.display());
        }

        ref_voices.insert(speaker.clone(), GgmlIclRef {
            ref_audio,
            ref_text: GGML_REF_ANCHOR_TEXT.to_string(),
        });
    }

    Ok(ref_voices)
}

/// True when a "speaker" string is actually a literal acoustic voice
/// description supplied by the user (via `--narrator-voice`) rather than a
/// character name produced by the LLM annotator.
///
/// Annotator speakers are single UPPERCASE tokens (`NARRATOR`, `ELENA`).
/// Voice descriptions contain commas and/or spaces, so they must be passed to
/// VoiceDesign verbatim — never re-derived from the palette (doing so is what
/// made `--narrator-voice "female alto, …"` come out as a male baritone).
#[cfg(feature = "ggml")]
fn is_literal_voice_description(speaker: &str) -> bool {
    speaker.contains(',') || speaker.contains(char::is_whitespace)
}

/// Stable, filesystem-safe short slug for a speaker key.
///
/// Voice descriptions can be arbitrarily long, so anything over 32 chars is
/// truncated and suffixed with a hash of the full key to stay unique.
#[cfg(feature = "ggml")]
fn speaker_slug(speaker: &str) -> String {
    let clean: String = speaker
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
    if clean.len() <= 32 {
        return clean;
    }
    let hash: u64 = speaker.bytes().fold(1469598103934665603u64, |acc, b| {
        (acc ^ b as u64).wrapping_mul(1099511628211)
    });
    let head: String = clean.chars().take(32).collect();
    format!("{head}_{hash:016x}")
}

/// Derive a concrete acoustic voice description and delivery style for a
/// speaker, purely from their name.
///
/// VoiceDesign needs real anatomy terms (register, timbre, texture) to
/// produce a distinct voice.  Passing the raw speaker name ("ELENA") gives
/// it no acoustic signal and everything collapses to the same default.
///
/// Strategy:
/// - NARRATOR always gets a fixed neutral narrator profile.
/// - Character names are mapped to a voice from a small palette that covers
///   a wide acoustic spread (deep male, mid male, light male, contralto,
///   mezzo, soprano-adjacent).  The palette index is derived from the name
///   so the same name always gets the same voice across runs.
#[cfg(feature = "ggml")]
fn voice_profile_for_speaker(speaker: &str, insertion_order: usize) -> (String, String) {
    if speaker == "NARRATOR" || speaker == "UNKNOWN" {
        return (
            "male baritone, warm chest resonance, smooth slightly gravelly timbre".into(),
            "measured, calm, authoritative".into(),
        );
    }

    // Heuristic gender guess from common name endings / known names.
    let name_lower = speaker.to_lowercase();
    let name_lower = name_lower.trim_end_matches('_');
    let probably_female = matches!(
        name_lower.as_ref(),
        "elena" | "sarah" | "mary" | "anne" | "anna" | "emma" | "lucy" | "kate" | "clara"
        | "alice" | "julia" | "lydia" | "grace" | "ruth" | "helen" | "rose" | "jane"
        | "sophie" | "lily" | "eva" | "nina" | "nora" | "vera" | "iris" | "ada"
    ) || name_lower.ends_with('a')
      || name_lower.ends_with("ine")
      || name_lower.ends_with("elle")
      || name_lower.ends_with("ette");

    // Palette: 4 female + 4 male profiles, chosen for maximum acoustic spread.
    let female_profiles: &[(&str, &str)] = &[
        ("female mezzo-soprano, warm mid-range, smooth clear timbre, slight husky edge",
         "composed, direct, quietly emotional"),
        ("female contralto, deep rich chest voice, resonant and velvety timbre",
         "calm, deliberate, low and measured"),
        ("female soprano-adjacent, bright clear tone, light airy quality, precise articulation",
         "crisp, energetic, articulate"),
        ("female mezzo, slightly breathy timbre, warm and intimate, conversational register",
         "soft, thoughtful, understated"),
    ];
    let male_profiles: &[(&str, &str)] = &[
        ("male tenor, light bright tone, clear articulation, youthful quality",
         "earnest, expressive, forward"),
        ("male baritone, mid-low register, slightly rough textured timbre, grounded",
         "steady, dry, understated"),
        ("male bass-baritone, deep resonant chest voice, dark velvety timbre",
         "slow, deliberate, commanding"),
        ("male tenor-baritone, smooth warm mid-register, clean clear timbre",
         "conversational, relaxed, natural"),
    ];

    // Pick index: combine insertion order with a simple name hash so two
    // characters with the same insertion slot still get different voices.
    let name_hash: usize = speaker.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
    let palette = if probably_female { female_profiles } else { male_profiles };
    let idx = (insertion_order.wrapping_add(name_hash)) % palette.len();
    let (voice, style) = palette[idx];
    (voice.to_string(), style.to_string())
}

///  Elapsed 00:04:12  │  ETA ~01:07:43  │  2.4 chunks/min  │  ~38.6 min audio
///  Now: [NARRATOR] The aspirant who enters the path of the mysteries…
/// ```
/// Build the TTS engine selected by `--tts-backend` and run synthesis on every
/// chunk.  Returns the ordered list of (wav path, speaker) produced.
///
/// - `ggml` — the qwentts.cpp native backend (requires `--features ggml`).
/// - `none` — callers skip this entirely.
#[cfg(feature = "ggml")]
async fn run_tts_backend(
    tts: &TtsArgs,
    chunks: &[script::Chunk],
    wav_dir: &PathBuf,
    mp: &MultiProgress,
) -> Result<Vec<(PathBuf, String)>> {
    match tts.tts_backend.as_str() {
        "ggml" => run_ggml_tts(tts, chunks, wav_dir, mp).await,
        other => anyhow::bail!("Unknown TTS backend '{other}'"),
    }
}

/// The GGUF precision quantizer suffix for `--tts-ggml-precision`.
///
/// `q4` → `Q4_K_M`, `q8` → `Q8_0`, `f32` → `F32`, `bf16` → `BF16`.
#[cfg(feature = "ggml")]
fn ggml_precision_suffix(precision: &str) -> Result<&'static str> {
    match precision.to_ascii_lowercase().as_str() {
        "q4" | "q4_k_m" => Ok("Q4_K_M"),
        "q8" | "q8_0" => Ok("Q8_0"),
        "f32" => Ok("F32"),
        "bf16" => Ok("BF16"),
        other => anyhow::bail!(
            "Unknown ggml precision '{other}' (expected q4, q8, f32, or bf16)"
        ),
    }
}

/// Resolve a GGUF model path for the ggml backend.
///
/// * If the user supplied an explicit path, it is used verbatim.
/// * Otherwise the default filename for `kind` at the selected precision is
///   constructed, looked up in the HuggingFace cache (`~/.cache/huggingface/`),
///   and downloaded from `Serveurperso/Qwen3-TTS-GGUF` if not present.
///
/// `kind` selects which model: `voice_design`, `base`, or `tokenizer`.
#[cfg(feature = "ggml")]
async fn resolve_ggml_model(
    user_path: Option<String>,
    precision: &str,
    kind: &str,
) -> Result<std::path::PathBuf> {
    if let Some(p) = user_path {
        return Ok(std::path::PathBuf::from(p));
    }

    let suffix = ggml_precision_suffix(precision)?;
    let filename = match kind {
        "voice_design" => format!("qwen-talker-1.7b-voicedesign-{suffix}.gguf"),
        "base"         => format!("qwen-talker-1.7b-base-{suffix}.gguf"),
        "tokenizer"    => format!("qwen-tokenizer-12hz-{suffix}.gguf"),
        other          => anyhow::bail!("Unknown ggml model kind '{other}'"),
    };

    // Look the file up in the HF cache; download if missing.
    let path = tokio::task::spawn_blocking(move || {
        fetch_ggml_sync(&filename)
    })
    .await
    .context("ggml model fetch thread panicked")??;
    Ok(path)
}

/// Synchronously resolve (and, if necessary, download) a GGUF file into the
/// HuggingFace cache via hf-hub, returning the local cache path.
#[cfg(feature = "ggml")]
fn fetch_ggml_sync(filename: &str) -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    const REPO: &str = "Serveurperso/Qwen3-TTS-GGUF";
    let api = ApiBuilder::new().with_progress(false).build()
        .context("Failed to init HF Hub API")?;
    let repo = api.repo(Repo::new(REPO.to_string(), RepoType::Model));
    repo.get(filename)
        .with_context(|| format!("Failed to fetch {filename} from {REPO}"))
}

/// Default LLM GGUF: `unsloth/Qwen3.5-2B-GGUF / Qwen3.5-2B-Q8_0.gguf`.
const LLM_DEFAULT_REPO: &str = "unsloth/Qwen3.5-2B-GGUF";
const LLM_DEFAULT_FILE: &str = "Qwen3.5-2B-Q8_0.gguf";

/// Resolve the LLM GGUF path.
///
/// If `user_path` is non-empty it is used verbatim.  Otherwise the default
/// (`unsloth/Qwen3.5-2B-GGUF / Qwen3.5-2B-Q8_0.gguf`) is looked up in the
/// HuggingFace cache and downloaded if absent.
#[cfg(feature = "llama")]
async fn resolve_llm_gguf(user_path: &str) -> Result<std::path::PathBuf> {
    if !user_path.is_empty() {
        return Ok(std::path::PathBuf::from(user_path));
    }
    info!("--llm-gguf not specified; resolving default {LLM_DEFAULT_REPO}/{LLM_DEFAULT_FILE}");
    tokio::task::spawn_blocking(fetch_llm_gguf_sync)
        .await
        .context("LLM GGUF fetch thread panicked")?
}

/// Synchronously resolve (and, if necessary, download) the default LLM GGUF.
#[cfg(feature = "llama")]
fn fetch_llm_gguf_sync() -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let api = ApiBuilder::new().with_progress(false).build()
        .context("Failed to init HF Hub API")?;
    let repo = api.repo(Repo::new(LLM_DEFAULT_REPO.to_string(), RepoType::Model));
    repo.get(LLM_DEFAULT_FILE)
        .with_context(|| format!("Failed to fetch {LLM_DEFAULT_FILE} from {LLM_DEFAULT_REPO}"))
}

/// The qwentts.cpp backend dispatch (requires `--features ggml`).
///
/// Mode selection via `--tts-ggml-mode`:
/// * `voicedesign` — one `voice_design` GGUF drives every chunk via `instruct`.
/// * `customvoice` — one `custom_voice` GGUF selects named speakers.
/// * `twophase`    — the ONNX-equivalent two-phase pipeline (default): phase 1
///   uses the `voice_design` talker (--tts-ggml-talker) to generate a reference
///   clip per speaker, phase 2 uses the `base` talker (--tts-ggml-base-talker)
///   to clone that voice for every chunk via ICL.
#[cfg(feature = "ggml")]
async fn run_ggml_tts(
    tts: &TtsArgs,
    chunks: &[script::Chunk],
    wav_dir: &PathBuf,
    mp: &MultiProgress,
) -> Result<Vec<(PathBuf, String)>> {
    match tts.tts_ggml_mode.as_str() {
        "twophase" => run_ggml_twophase(tts, chunks, wav_dir, mp).await,
        // "voicedesign" / "customvoice" (and anything unknown) — single model.
        _ => {
            let talker = resolve_ggml_model(
                tts.tts_ggml_talker.clone(), &tts.tts_ggml_precision, "voice_design",
            ).await
            .context("resolve ggml voice_design talker")?;
            let codec = resolve_ggml_model(
                tts.tts_ggml_codec.clone(), &tts.tts_ggml_precision, "tokenizer",
            ).await
            .context("resolve ggml tokenizer")?;

            let (label, mode) = if tts.tts_ggml_mode.as_str() == "customvoice" {
                ("custom_voice", Mode::CustomVoice)
            } else {
                ("voice_design", Mode::VoiceDesign)
            };
            let step = step_bar(mp, &format!("Loading qwentts.cpp {label} model"));
            let engine = CppTts::load(
                talker.to_string_lossy().into_owned(),
                codec.to_string_lossy().into_owned(),
                mode,
            )
                .await
                .context("qwentts.cpp load failed")?;
            step.finish_and_clear();

            synthesise_chunks(
                &engine, chunks, wav_dir,
                tts.tts_max_tokens, tts.kv_window, tts.tts_seed, tts.tts_temperature,
                tts.shard_id, tts.shard_count, mp,
            ).await
        }
    }
}

/// The qwentts.cpp two-phase pipeline (generate + render):
/// 1. VoiceDesign — generate a reference clip per unique speaker (`.ref.wav`).
/// 2. Base — clone each speaker's reference voice for every chunk via ICL.
#[cfg(feature = "ggml")]
async fn run_ggml_twophase(
    tts: &TtsArgs,
    chunks: &[script::Chunk],
    wav_dir: &PathBuf,
    mp: &MultiProgress,
) -> Result<Vec<(PathBuf, String)>> {
    // Resolve / fetch the three GGUF files (unless the user gave explicit paths).
    let vd_talker = resolve_ggml_model(
        tts.tts_ggml_talker.clone(), &tts.tts_ggml_precision, "voice_design",
    ).await
    .context("resolve ggml voice_design talker")?;
    let base_talker = resolve_ggml_model(
        tts.tts_ggml_base_talker.clone(), &tts.tts_ggml_precision, "base",
    ).await
    .context("resolve ggml base talker")?;
    let codec = resolve_ggml_model(
        tts.tts_ggml_codec.clone(), &tts.tts_ggml_precision, "tokenizer",
    ).await
    .context("resolve ggml tokenizer")?;

    // ── Phase 1: VoiceDesign reference generation ─────────────────────────
    let step = step_bar(mp, "Loading qwentts.cpp voice_design model");
    let vd = CppTts::load(
        vd_talker.to_string_lossy().into_owned(),
        codec.clone().to_string_lossy().into_owned(),
        Mode::VoiceDesign,
    )
        .await
        .context("qwentts.cpp voice_design load failed (phase 1)")?;
    step.finish_and_clear();

    let ref_voices = generate_ref_voices_ggml(
        &vd, chunks, wav_dir,
        tts.tts_seed, tts.tts_temperature, tts.tts_max_tokens,
    ).await?;

    drop(vd);
    debug!("qwentts.cpp voice_design model unloaded");

    // ── Phase 2: Base ICL clone synthesis ─────────────────────────────────
    let step = step_bar(mp, "Loading qwentts.cpp base model");
    let base = CppTts::load(
        base_talker.to_string_lossy().into_owned(),
        codec.to_string_lossy().into_owned(),
        Mode::Base,
    )
        .await
        .context("qwentts.cpp base load failed (phase 2)")?;
    step.finish_and_clear();
    let engine = CppBaseTts::new(base, ref_voices);

    synthesise_chunks(
        &engine, chunks, wav_dir,
        tts.tts_max_tokens, tts.kv_window, tts.tts_seed, tts.tts_temperature,
        tts.shard_id, tts.shard_count, mp,
    ).await
}



#[cfg(feature = "ggml")]
async fn synthesise_chunks<T: TtsEngine>(
    tts: &T,
    chunks: &[script::Chunk],
    wav_dir: &PathBuf,
    max_tokens: usize,
    kv_window: usize,
    seed: Option<u64>,
    temperature: f64,
    shard_id: usize,
    shard_count: usize,
    mp: &MultiProgress,
) -> Result<Vec<(PathBuf, String)>> {
    let total = chunks.len();
    debug_assert!(shard_count > 0);
    // Each process only synthesises chunks where index % shard_count == shard_id.
    // WAV files are named by chunk index (`{:04}.wav`), so shards write disjoint
    // files and can run concurrently in separate processes/GPUs.
    let mine = |i: usize| i % shard_count == shard_id;
    let my_total = (0..total).filter(|&i| mine(i)).count();

    // Determine the canonical narrator/body speaker for remapping structural entries.
    let structural_tags = ["TITLE", "CHAPTER", "SECTION"];
    let narrator_speaker = chunks.iter()
        .find(|c| !structural_tags.contains(&c.speaker.as_str()))
        .map(|c| c.speaker.as_str())
        .unwrap_or("NARRATOR");

    // ── chunk progress bar ──────────────────────────────────────────────────
    let bar = mp.add(ProgressBar::new(my_total as u64));
    bar.set_style(
        ProgressStyle::with_template(
            " {spinner:.cyan} {pos}/{len} {percent:>3}%  [{wide_bar:.cyan/238}]  {msg}",
        )
        .unwrap()
        .progress_chars("█▓░")
        .tick_strings(&["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]),
    );
    bar.enable_steady_tick(std::time::Duration::from_millis(120));
    bar.set_message("Synthesising audio");

    // ── stats line (single spinner, no extra tick chars) ──────────────────
    let stats = mp.add(ProgressBar::new_spinner());
    stats.set_style(ProgressStyle::with_template("  {msg}").unwrap());
    stats.enable_steady_tick(std::time::Duration::from_millis(500));

    let started = Instant::now();
    let mut wav_paths: Vec<(PathBuf, String)> = Vec::with_capacity(my_total);
    let mut audio_secs_total = 0f64;
    let mut skipped = 0usize;
    let mut done = 0usize; // count of this shard's chunks processed (incl. skipped)

    for (i, chunk) in chunks.iter().enumerate() {
        // Skip chunks owned by other shard processes.
        if !mine(i) {
            continue;
        }
        let wav_path = wav_dir.join(format!("{:04}.wav", i));

        // ── Checkpoint: skip already-generated WAVs ────────────────────────────
        if wav_path.exists() {
            if let Ok(reader) = hound::WavReader::open(&wav_path) {
                let spec = reader.spec();
                audio_secs_total += reader.duration() as f64 / spec.sample_rate as f64;
            }
            wav_paths.push((wav_path, chunk.speaker.clone()));
            bar.inc(1);
            skipped += 1;
            continue;
        }

        // ── Log chunk start (visible in log files, not just TTY) ───────────────
        let preview_text = chunk.text.chars().take(80).collect::<String>();
        let preview_text = if chunk.text.len() > 80 {
            format!("{preview_text}…")
        } else {
            preview_text.clone()
        };
        info!("[{}/{}] {}", i + 1, total, preview_text);

        bar.set_message(format!("[{}] {}", chunk.speaker, preview_text));

        let wav_path = wav_dir.join(format!("{:04}.wav", i));
        // Remap structural speakers to the narrator voice for TTS lookup
        let tts_speaker = if structural_tags.contains(&chunk.speaker.as_str()) {
            narrator_speaker
        } else {
            chunk.speaker.as_str()
        };
        tts.synthesise(&chunk.text, tts_speaker, &chunk.instruct, &wav_path, max_tokens, kv_window, seed, temperature)
            .await
            .with_context(|| format!("TTS failed on chunk {i}"))?;

        // Measure generated audio duration from WAV header (cheap: just reads header)
        if let Ok(reader) = hound::WavReader::open(&wav_path) {
            let spec = reader.spec();
            let n_samples = reader.duration(); // samples per channel
            audio_secs_total += n_samples as f64 / spec.sample_rate as f64;
        }

        wav_paths.push((wav_path, chunk.speaker.clone()));
        bar.inc(1);

        // Refresh stats line every chunk — rate excludes skipped (cached) chunks.
        // `done` only counts freshly-synthesised chunks (cached ones `continue`
        // above before `done += 1`), so `done` already excludes resumed chunks.
        // On a resumed run `skipped` can exceed `done`, so never subtract them
        // directly (would underflow).
        done += 1;
        let elapsed = started.elapsed().as_secs_f64();
        let synthesised = done;
        let chunks_per_min = if elapsed > 0.0 && synthesised > 0 {
            synthesised as f64 / (elapsed / 60.0)
        } else {
            0.0
        };
        let remaining = my_total.saturating_sub(done);
        let eta_secs = if chunks_per_min > 0.0 {
            (remaining as f64 / chunks_per_min) * 60.0
        } else {
            0.0
        };
        stats.set_message(format!(
            "{} {elapsed_fmt}  │  ETA ~{eta_fmt}  │  {rate:.1} chunks/min  │  ~{audio:.1} min audio done",
            style("Elapsed").dim(),
            elapsed_fmt = fmt_duration(elapsed as u64),
            eta_fmt     = fmt_duration(eta_secs as u64),
            rate        = chunks_per_min,
            audio       = audio_secs_total / 60.0,
        ));

        // ── Periodic summary log every 25 chunks ───────────────────────────────
        if synthesised % 25 == 0 {
            info!(
                "Progress: {}/{} chunks  │  {:.1} chunks/min  │  ETA ~{}  │  {:.1} min audio",
                i + 1, total,
                chunks_per_min,
                fmt_duration(eta_secs as u64),
                audio_secs_total / 60.0,
            );
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let synthesised = my_total - skipped;
    let chunks_per_min = if elapsed > 0.0 && synthesised > 0 {
        synthesised as f64 / (elapsed / 60.0)
    } else {
        0.0
    };

    stats.finish_and_clear();
    let skip_note = if skipped > 0 {
        format!("  │  {skipped} resumed from cache")
    } else {
        String::new()
    };
    bar.finish_with_message(format!(
        "{} {my_total} chunks  │  {audio:.1} min audio  │  {rate:.1} chunks/min  │  {elapsed_fmt}{skip_note}",
        style("✔ Synthesised").green().bold(),
        audio       = audio_secs_total / 60.0,
        rate        = chunks_per_min,
        elapsed_fmt = fmt_duration(elapsed as u64),
    ));

    Ok(wav_paths)
}

/// Reconstruct the full, chunk-ordered list of generated WAVs from `wav_dir`.
///
/// WAV files are named `NNNN.wav` by chunk index (see synthesise_chunks), so a
/// merge process can rebuild the complete ordered list across all shards by
/// scanning the shared output directory.  Chunks without a corresponding WAV
/// (e.g. a shard that never ran) are simply omitted.
fn collect_all_wavs(chunks: &[script::Chunk], wav_dir: &PathBuf) -> Vec<(PathBuf, String)> {
    let mut out = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let wav_path = wav_dir.join(format!("{:04}.wav", i));
        if wav_path.exists() {
            out.push((wav_path, chunk.speaker.clone()));
        }
    }
    out
}

/// Merge the (ordered) WAV files into the final M4B.
/// Shared by the normal single-process path, the per-shard merge step, and the
/// `--merge-only` recombine pass.
async fn merge_wavs(
    wav_paths: &[(PathBuf, String)],
    chapters:  &[(String, usize)],
    output:    &PathBuf,
    title:     &str,
    author:    &str,
    cover:     Option<&std::path::Path>,
    tts:       &TtsArgs,
    mp:        &MultiProgress,
) -> Result<()> {
    let step = step_bar(mp, "Merging audio & encoding M4B");
    let merger = AudioMerger {
        pause_between_speakers_ms: tts.pause_between_speakers_ms,
        pause_same_speaker_ms: tts.pause_same_speaker_ms,
        crossfade_ms: tts.crossfade_ms,
        compressor: tts.compress.then(|| CompressorSettings {
            threshold_db:  tts.compress_threshold_db,
            ratio:         tts.compress_ratio,
            makeup_db:     tts.compress_makeup_db,
            attack_ms:     tts.compress_attack_ms,
            release_ms:    tts.compress_release_ms,
            limit_db:      tts.compress_limit_db,
            rms_window_ms: 50.0,
        }),
    };
    merger
        .merge_to_m4b(wav_paths, chapters, output, title, author, cover)
        .await
        .context("M4B encoding failed")?;
    step.finish_and_clear();
    Ok(())
}

/// Format seconds as HH:MM:SS.
#[cfg(feature = "ggml")]
fn fmt_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn step_bar(mp: &MultiProgress, msg: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message(msg.to_string());
    pb
}


async fn build_llm_backend(args: &LlmArgs) -> Result<Box<dyn LlmBackend + Send + Sync>> {
    match args.llm_backend.as_str() {
        #[cfg(feature = "llama")]
        "llama" => {
            let gguf = resolve_llm_gguf(&args.llm_gguf).await
                .context("resolving LLM GGUF model")?;
            let llm = llm::llama::LlamaCpp::load(
                gguf.to_string_lossy().into_owned(),
                args.llm_gpu_layers,
                args.temperature,
            ).await?;
            Ok(Box::new(llm))
        }
        #[cfg(not(feature = "llama"))]
        "llama" => {
            anyhow::bail!("--llm-backend=llama requires building with --features llama")
        }
        // "api" or anything else
        _ => {
            let client = LlmApiClient::new(
                &args.llm_url,
                &args.llm_key,
                &args.llm_model,
                args.max_tokens,
                args.temperature,
            );
            Ok(Box::new(client))
        }
    }
}

/// Build a chapter list from TTS chunks.
/// Uses the structured TITLE/CHAPTER/SECTION speaker tags emitted by the annotator.
/// Falls back to heuristic text detection for scripts without structural tags.
fn build_chapters(chunks: &[script::Chunk]) -> Vec<(String, usize)> {
    let mut chapters: Vec<(String, usize)> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        match chunk.speaker.as_str() {
            "TITLE" | "CHAPTER" | "SECTION" => {
                chapters.push((chunk.text.trim().chars().take(120).collect(), i));
            }
            _ => {}
        }
    }
    // Fallback heuristic for legacy scripts without structural tags
    if chapters.is_empty() {
        let heading_re = regex::Regex::new(
            r"(?i)^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b",
        ).unwrap();
        for (i, chunk) in chunks.iter().enumerate() {
            let t = chunk.text.trim();
            if heading_re.is_match(t) || (t.len() < 80 && !t.ends_with('.') && !t.ends_with('?') && !t.ends_with('!')) {
                chapters.push((t.chars().take(120).collect(), i));
            }
        }
    }
    if chapters.is_empty() {
        chapters.push(("Audiobook".into(), 0));
    }
    chapters
}
