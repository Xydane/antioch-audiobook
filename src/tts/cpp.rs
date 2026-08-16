//! qwentts.cpp TTS backend — links the GGML-based Qwen3-TTS port as a Rust
//! library and exposes it through the shared `TtsEngine` trait.
//!
//! qwentts.cpp exposes a stable, single-header, plain-C99 ABI (`qwen.h`):
//! opaque `qt_context` handles, POD init/synthesis params, and `qt_*` free
//! functions.  This module mirrors the relevant subset of that ABI with
//! hand-written `#[repr(C)]` bindings (no bindgen dependency needed — the
//! header is clean C99 POD).
//!
//! Build/link is handled by `build.rs` under the `ggml` feature.  Three
//! synthesis modes are supported, selected at runtime by the GGUF model type
//! loaded into the context:
//!
//! * `voice_design`  — a free-text attribute instruction drives the voice.
//!                     This maps directly onto antioch's `voice`/`style`
//!                     parameters the same way the ONNX VoiceDesign model does.
//! * `custom_voice`  — a fixed set of named speakers (serena, vivian, ...).
//! * `base`          — voice cloning from a reference clip, used by the
//!                     two-phase pipeline (see `CppBaseTts`).
//!
//! The two-phase equivalent of the ONNX pipeline is implemented by `CppBaseTts`:
//!
//! * Phase 1 — a `voice_design` context generates a reference clip per speaker
//!             (anchor text + instruct style) and writes `<speaker>.ref.wav`.
//! * Phase 2 — a `base` context extracts the `.spk` / `.rvq` latents from that
//!             reference WAV with `qt_extract_voice_ref`, then clones the voice
//!             for every chunk via ICL (`ref_spk_emb` + `ref_codes` + `ref_text`).
//!
//! The single-model `voice_design` / `custom_voice` modes are also exposed
//! directly through `CppTts::synthesise` (the `TtsEngine` impl).

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

use crate::tts::TtsEngine;

// ─── C ABI constants (must match qwen.h) ────────────────────────────────────

/// Struct ABI version — reject headers that don't match our layout.
const QT_ABI_VERSION: c_int = 4;

/// Anchor text used to generate a reference voice clip in phase 1 (VoiceDesign).
/// Long enough to yield ≥200 codec frames for robust ICL cloning, matching the
/// constant used by the ORT backend (`ort.rs`).
const REF_ANCHOR_TEXT: &str =
    "Such a Fire existeth extending through the rushings of Air or even a Fire Formless \
     whence comes the Image of a Voice or even a flashing Light abounding, revolving, \
     whirling forth, crying aloud.";

// ─── C ABI structs (mirror qwen.h POD) ──────────────────────────────────────

/// Output audio buffer. `samples` is malloc'd by `qt_synthesize`, freed by
/// `qt_audio_free`. Must be zero-initialised before first use.
#[repr(C)]
struct QtAudio {
    samples: *mut f32,
    n_samples: c_int,
    sample_rate: c_int,
    channels: c_int,
}

/// Precomputed Base-model voice reference latents (qwen.h `struct qt_voice_ref`).
/// Both pointers are malloc'd by `qt_extract_voice_ref`, owned by the struct,
/// released by `qt_voice_ref_free`. Must be zero-initialised before first use.
///
/// `ref_spk_emb` is the speaker embedding (`.spk` equivalent), `ref_codes` is
/// the RVQ code matrix (`.rvq` equivalent) laid out `[num_codebooks, ref_T]`.
#[allow(non_snake_case)]
#[repr(C)]
struct QtVoiceRef {
    ref_spk_emb: *mut f32,
    ref_spk_dim: c_int,
    ref_codes: *mut i32,
    ref_T: c_int,
    num_codebooks: c_int,
}

/// Initialisation parameters. Both GGUF paths required.
#[repr(C)]
struct QtInitParams {
    abi_version: c_int,
    talker_path: *const c_char,
    codec_path: *const c_char,
    use_fa: bool,
    clamp_fp16: bool,
    max_batch: c_int,
    codec_chunk_sec: f32,
}

/// Synthesis parameters — mirrors qwen.h `struct qt_tts_params`.
/// Field names match the C header exactly (some use snake_case, some not).
#[allow(non_snake_case)]
#[repr(C)]
struct QtTtsParams {
    abi_version: c_int,
    text: *const c_char,
    lang: *const c_char,
    instruct: *const c_char,
    speaker: *const c_char,
    ref_audio_24k: *const f32,
    ref_n_samples: c_int,
    ref_text: *const c_char,
    seed: i64,
    max_new_tokens: c_int,
    do_sample: bool,
    temperature: f32,
    top_k: c_int,
    top_p: f32,
    repetition_penalty: f32,
    subtalker_do_sample: bool,
    subtalker_temperature: f32,
    subtalker_top_k: c_int,
    subtalker_top_p: f32,
    dump_dir: *const c_char,
    cancel: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
    cancel_user_data: *mut c_void,
    on_chunk: Option<unsafe extern "C" fn(*const f32, c_int, *mut c_void) -> bool>,
    on_chunk_user_data: *mut c_void,
    ref_spk_emb: *const f32,
    ref_spk_dim: c_int,
    ref_codes: *const i32,
    ref_T: c_int,
}

// ─── Status codes (qwen.h enum qt_status) ──────────────────────────────────

const QT_STATUS_OK: c_int = 0;

/// qwentts.cpp log severities (qwen.h `enum qt_log_level`).  Numerically
/// ordered so a filter can use `level < threshold`.
const QT_LOG_DEBUG: c_int = 0;
const QT_LOG_INFO: c_int = 1;
const QT_LOG_WARN: c_int = 2;
const QT_LOG_ERROR: c_int = 3;

/// C log callback signature (qwen.h `qt_log_cb`).
/// `user_data` is forwarded verbatim from `qt_log_set`; we ignore it.
type QtLogCb = unsafe extern "C" fn(level: c_int, msg: *const c_char, user_data: *mut c_void);

/// Bridge qwentts.cpp's `qt_log_*` output into Rust `tracing` so the C++
/// library's `[Pipeline]` / `[Perf]` / `[Sample]` chatter is filtered by
/// `RUST_LOG` like everything else, instead of spamming stderr directly.
///
/// The callback runs on whatever thread the library emits from, so it must be
/// reentrant — `tracing` macros are thread-safe, so that's fine here.
unsafe extern "C" fn qt_log_bridge(level: c_int, msg: *const c_char, _user_data: *mut c_void) {
    if msg.is_null() {
        return;
    }
    let msg = CStr::from_ptr(msg).to_string_lossy();
    match level {
        QT_LOG_DEBUG => tracing::trace!(target: "qwentts", "{msg}"),
        QT_LOG_INFO => tracing::debug!(target: "qwentts", "{msg}"),
        QT_LOG_WARN => tracing::warn!(target: "qwentts", "{msg}"),
        QT_LOG_ERROR => tracing::error!(target: "qwentts", "{msg}"),
        _ => { /* unknown level — ignore */ }
    }
}

/// Install the qwentts.cpp log bridge exactly once (process-wide, per
/// `qt_log_set`'s semantics).  INFO is forwarded at `debug` level so the
/// verbose synthesis cadence stays hidden under the default `antioch=info`
/// filter; enable with `RUST_LOG=qwentts=debug` (or `=trace` for DEBUG).
fn install_log_bridge() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        qt_log_set(Some(qt_log_bridge), std::ptr::null_mut());
    });
}

// ─── FFI declarations (from qwen.h) ─────────────────────────────────────────

// Opaque handle type — definition lives in qwen.cpp.
#[repr(C)]
struct QtContext {
    _private: [u8; 0],
}

#[allow(dead_code)]
extern "C" {
    fn qt_init_default_params(p: *mut QtInitParams);
    fn qt_init(params: *const QtInitParams) -> *mut QtContext;
    fn qt_free(q: *mut QtContext);
    fn qt_version() -> *const c_char;
    fn qt_last_error() -> *const c_char;
    fn qt_tts_default_params(p: *mut QtTtsParams);
    fn qt_synthesize(
        q: *mut QtContext,
        params: *const QtTtsParams,
        out: *mut QtAudio,
    ) -> c_int;
    fn qt_audio_free(a: *mut QtAudio);
    fn qt_n_speakers(q: *const QtContext) -> c_int;
    fn qt_speaker_name(q: *const QtContext, i: c_int) -> *const c_char;
    fn qt_extract_voice_ref(
        q: *mut QtContext,
        ref_audio_24k: *const f32,
        ref_n_samples: c_int,
        out: *mut QtVoiceRef,
    ) -> c_int;
    fn qt_voice_ref_free(ref_: *mut QtVoiceRef);
    fn qt_log_set(cb: Option<QtLogCb>, user_data: *mut c_void);
}

// ─── Synthesis mode, driven by the loaded GGUF model type ──────────────────

/// How a `qt_context` was initialised / which GGUF model it loaded.
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    /// voice_design GGUF — `instruct` drives the voice.
    VoiceDesign,
    /// custom_voice GGUF — named speakers.
    CustomVoice,
    /// base GGUF — voice cloning from a reference clip (ICL).
    Base,
}

/// Loaded qwentts.cpp library: the opaque context plus its mode.
pub struct CppTts {
    ctx: *mut QtContext,
    mode: Mode,
}

// Raw pointer to QtContext is not Send by default; the qwentts.cpp qt_* API is
// thread-safe (each handle serialises internally / the lib is thread safe).
// We only ever move the pointer into `spawn_blocking` closures one at a time.
unsafe impl Send for CppTts {}
unsafe impl Sync for CppTts {}

impl CppTts {
    /// Initialise a qwentts.cpp context from talker + codec GGUF paths.
    ///
    /// This is a heavy operation (loads both GGUF files), so it runs on a
    /// blocking thread via `spawn_blocking` rather than stalling the async
    /// executor.
    ///
    /// `mode` must match the GGUF's embedded model type:
    /// `voice_design` for a `qwen-talker-1.7b-voicedesign-*.gguf`,
    /// `custom_voice` for a `qwen-talker-1.7b-customvoice-*.gguf`.
    pub async fn load(talker_path: String, codec_path: String, mode: Mode) -> Result<Self> {
        let result: Result<CppTts> = tokio::task::spawn_blocking(move || load_sync(&talker_path, &codec_path, mode))
            .await
            .context("qwentts.cpp load thread panicked")?;
        result
    }

    /// Number of named speakers the loaded model exposes (custom_voice only).
    #[allow(dead_code)]
    pub fn n_speakers(&self) -> usize {
        let n = unsafe { qt_n_speakers(self.ctx) };
        n.max(0) as usize
    }

    /// Phase 1 of the two-phase pipeline: generate a reference voice clip for
    /// `speaker` and write it to `output_path`.
    ///
    /// Requires a `voice_design` context.  The `acoustic` voice description and
    /// `style` are combined into the `instruct` attribute string, exactly as
    /// `synthesise` does in `Mode::VoiceDesign`.  The text is the shared
    /// `REF_ANCHOR_TEXT` so the resulting clip is long enough for ICL cloning.
    pub async fn synthesise_ref_voice(
        &self,
        _speaker: &str,
        acoustic: &str,
        style: &str,
        output_path: &Path,
        max_tokens: usize,
        seed: Option<u64>,
        temperature: f64,
    ) -> Result<()> {
        let combined = if style.is_empty() {
            acoustic.to_string()
        } else {
            format!("{acoustic}, {style}")
        };
        self.synthesise_inner(
            REF_ANCHOR_TEXT,
            Some(&combined),
            None,
            None,
            output_path,
            max_tokens,
            seed,
            temperature,
        )
        .await
    }

    /// Extract reusable ICL voice-clone latents from a decoded reference WAV.
    ///
    /// Requires a `base` context (with speaker encoder weights).  `ref_audio_24k`
    /// is mono float32 PCM at 24 kHz.  Returns the owned `.spk`- and `.rvq`-
    /// equivalent buffers, to be passed back through `synthesise_inner`.
    pub async fn extract_voice_ref(&self, ref_audio_24k: Vec<f32>) -> Result<GgmlVoiceRef> {
        let ctx = self.ctx as usize;
        tokio::task::spawn_blocking(move || {
            let mut out = std::mem::MaybeUninit::<QtVoiceRef>::uninit();
            // Zero-init the struct so qt_voice_ref_free is safe even on failure.
            let mut out = unsafe {
                std::ptr::write_bytes(out.as_mut_ptr() as *mut u8, 0, std::mem::size_of::<QtVoiceRef>());
                out.assume_init()
            };
            let rc = unsafe {
                qt_extract_voice_ref(
                    ctx as *mut QtContext,
                    ref_audio_24k.as_ptr(),
                    ref_audio_24k.len() as c_int,
                    &mut out,
                )
            };
            if rc != QT_STATUS_OK {
                unsafe { qt_voice_ref_free(&mut out) };
                let msg = unsafe { cstr_to_string(qt_last_error()) };
                anyhow::bail!("qt_extract_voice_ref failed (status {rc}): {msg}");
            }
            // Copy the malloc'd buffers into owned Rust vectors, then release the
            // library's allocation (qt_extract_voice_ref malloc'd them).
            let result = GgmlVoiceRef {
                spk_emb: unsafe { slice_from_raw_parts(out.ref_spk_emb, out.ref_spk_dim.max(0) as usize) }.to_vec(),
                codes: unsafe { slice_from_raw_parts_i32(out.ref_codes, (out.ref_T.max(0) as usize) * (out.num_codebooks.max(0) as usize)) }.to_vec(),
                ref_T: out.ref_T,
                num_codebooks: out.num_codebooks,
            };
            unsafe { qt_voice_ref_free(&mut out) };
            Ok(result)
        })
        .await
        .context("qwentts.cpp voice-ref extract thread panicked")?
    }

}

impl Drop for CppTts {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { qt_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

// ─── Two-phase pipeline types ───────────────────────────────────────────────

/// Owned, Rust-side copy of the ICL voice-reference latents extracted from a
/// reference WAV by `qt_extract_voice_ref`.
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct GgmlVoiceRef {
    /// Speaker embedding (`.spk` equivalent), length `ref_spk_dim`.
    pub spk_emb: Vec<f32>,
    /// RVQ code matrix (`.rvq` equivalent), laid out `[num_codebooks, ref_T]`.
    pub codes: Vec<i32>,
    /// Number of reference audio frames (`ref_T`).
    pub ref_T: i32,
    /// Number of RVQ codebooks.
    #[allow(dead_code)]
    pub num_codebooks: i32,
}

/// A reference voice clip plus the transcript used to generate it, ready for
/// phase-2 ICL cloning on a `base` context.
///
/// Holds the raw mono float32 PCM at 24 kHz (`ref_audio`) and the anchor
/// transcript (`ref_text`).  The phase-2 backend extracts the `.spk`/`.rvq`
/// latents from `ref_audio` on each synthesis call via `qt_extract_voice_ref`.
#[derive(Clone)]
pub struct GgmlIclRef {
    /// Mono float32 PCM reference clip at 24 kHz.
    pub ref_audio: Vec<f32>,
    /// Reference transcript (`ref_text`) — the anchor text used in phase 1.
    pub ref_text: String,
}

/// Read a WAV file into mono float32 PCM.  Handles both float and integer
/// sample formats; down-mixes stereo to mono by averaging channels.
pub fn read_wav_f32(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Cannot open ref WAV: {}", path.display()))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        reader.into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        reader.into_samples::<i32>()
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|s| s as f32 / i32::MAX as f32)
            .collect()
    };
    // Down-mix to mono if needed.
    if ch == 1 {
        Ok(samples)
    } else {
        Ok(samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect())
    }
}

/// `slice::from_raw_parts` for `*mut f32`.
///
/// # Safety
/// `p` must point to at least `len` valid `f32` values, or be null when `len == 0`.
unsafe fn slice_from_raw_parts<'a>(p: *mut f32, len: usize) -> &'a [f32] {
    if p.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, len)
    }
}

/// `slice::from_raw_parts` for `*mut i32`.
///
/// # Safety
/// `p` must point to at least `len` valid `i32` values, or be null when `len == 0`.
unsafe fn slice_from_raw_parts_i32<'a>(p: *mut i32, len: usize) -> &'a [i32] {
    if p.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, len)
    }
}

// ─── Two-phase TtsEngine (base ICL clone) ──────────────────────────────────

/// Phase-2 backend: a `base` qwentts.cpp context that clones a reference voice
/// for every chunk via ICL.
///
/// Holds the base context plus a map of speaker → reference voice clip (the
/// reference PCM + anchor transcript produced in phase 1).  On each `synthesise`
/// call it extracts the `.spk`/`.rvq` latents from that clip with
/// `qt_extract_voice_ref`, then synthesises the chunk with those latents
/// (`ref_spk_emb` + `ref_codes` + `ref_text`).
pub struct CppBaseTts {
    base: CppTts,
    ref_voices: std::collections::HashMap<String, GgmlIclRef>,
}

impl CppBaseTts {
    pub fn new(base: CppTts, ref_voices: std::collections::HashMap<String, GgmlIclRef>) -> Self {
        Self { base, ref_voices }
    }
}

#[async_trait::async_trait]
impl TtsEngine for CppBaseTts {
    async fn synthesise(
        &self,
        text: &str,
        voice: &str,
        _style: &str,
        output_path: &Path,
        max_tokens: usize,
        _kv_window: usize,
        seed: Option<u64>,
        temperature: f64,
    ) -> Result<()> {
        let icl = self.ref_voices.get(voice)
            .with_context(|| format!("No reference voice for speaker '{voice}'"))?
            .clone();
        let (text, output_path) = (text.to_string(), output_path.to_path_buf());

        // Extract the `.spk`/`.rvq` latents from the speaker's reference clip,
        // then synthesise the chunk with ICL conditioning (latents + ref_text).
        // Both run on blocking threads off the async executor.
        let latents = self.base.extract_voice_ref(icl.ref_audio).await?;

        self.base
            .synthesise_inner(&text, None, None, Some((latents, icl.ref_text)), &output_path, max_tokens, seed, temperature)
            .await
    }
}

/// Synchronous context initialisation — the body of `CppTts::load`, run on a
/// blocking thread.  Returns a context handle that is safe to move between
/// threads once created (qwentts.cpp handles are thread-safe).
fn load_sync(talker_path: &str, codec_path: &str, mode: Mode) -> Result<CppTts> {
    // Route the C++ library's own logs into tracing (idempotent — installed
    // once per process).
    install_log_bridge();

    let talker = CString::new(talker_path).context("talker GGUF path contains interior NUL")?;
    let codec = CString::new(codec_path).context("codec GGUF path contains interior NUL")?;

    let mut params = std::mem::MaybeUninit::<QtInitParams>::uninit();
    unsafe {
        qt_init_default_params(params.as_mut_ptr());
    }
    let mut params = unsafe { params.assume_init() };
    params.abi_version = QT_ABI_VERSION;
    params.talker_path = talker.as_ptr();
    params.codec_path = codec.as_ptr();

    let ctx = unsafe { qt_init(&params) };
    if ctx.is_null() {
        let msg = unsafe { cstr_to_string(qt_last_error()) };
        anyhow::bail!("qt_init failed for {talker_path}: {msg}");
    }

    info!(
        "Loaded qwentts.cpp backend (qt_version={})",
        unsafe { cstr_to_string(qt_version()) }
    );

    Ok(CppTts { ctx, mode })
}

/// Convert a `*const c_char` NUL-terminated string to a Rust `String`.
///
/// # Safety
/// `p` must be a valid NUL-terminated C string, or null (returns empty).
unsafe fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

// ─── TtsEngine impl ───────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl TtsEngine for CppTts {
    async fn synthesise(
        &self,
        text: &str,
        voice: &str,
        style: &str,
        output_path: &Path,
        max_tokens: usize,
        _kv_window: usize,
        seed: Option<u64>,
        temperature: f64,
    ) -> Result<()> {
        // Resolve the per-call instruct / speaker from the mode, mirroring how
        // the ONNX VoiceDesign / CustomVoice backends consume `voice` + `style`.
        let (instruct, speaker) = match self.mode {
            Mode::VoiceDesign => {
                // instruct = "<acoustic voice>, <delivery style>".
                // If `voice` is already a literal acoustic description (from
                // `--narrator-voice`), use it verbatim; only bare character
                // names get mapped through the palette.
                let acoustic = if voice.contains(',') || voice.contains(char::is_whitespace) {
                    voice.to_string()
                } else {
                    voice_profile_for_speaker(voice).0
                };
                let combined = if style.is_empty() {
                    acoustic
                } else {
                    format!("{acoustic}, {style}")
                };
                (Some(combined), None)
            }
            Mode::CustomVoice => {
                // speaker = named speaker from the custom_voice table; style is
                // optional (passed as the optional instruct for custom_voice).
                let spk = customvoice_speaker(voice);
                (if style.is_empty() { None } else { Some(style.to_string()) }, Some(spk))
            }
            Mode::Base => {
                anyhow::bail!(
                    "CppTts in Base mode cannot synthesise without ICL latents; \
                     use the two-phase CppBaseTts backend"
                );
            }
        };

        self.synthesise_inner(text, instruct.as_deref(), speaker.as_deref(), None, output_path, max_tokens, seed, temperature)
            .await
    }
}

impl CppTts {
    /// Low-level shared synthesis core.
    ///
    /// `instruct` / `speaker` are passed verbatim to the engine.  `icls` carries
    /// the pre-extracted ICL clone latents plus the reference transcript for
    /// base-mode voice cloning (phase 2); when `Some`, `instruct`/`speaker`
    /// must be empty.
    pub(crate) async fn synthesise_inner(
        &self,
        text: &str,
        instruct: Option<&str>,
        speaker: Option<&str>,
        icls: Option<(GgmlVoiceRef, String)>,
        output_path: &Path,
        max_tokens: usize,
        seed: Option<u64>,
        temperature: f64,
    ) -> Result<()> {
        let text = text.to_string();
        let instruct = instruct.map(|s| s.to_string());
        let speaker = speaker.map(|s| s.to_string());
        let output_path = output_path.to_path_buf();
        let ctx = self.ctx as usize;

        tokio::task::spawn_blocking(move || {
            let mut params = std::mem::MaybeUninit::<QtTtsParams>::uninit();
            unsafe { qt_tts_default_params(params.as_mut_ptr()) };
            let mut params = unsafe { params.assume_init() };
            params.abi_version = QT_ABI_VERSION;
            params.max_new_tokens = max_tokens as c_int;
            params.seed = seed.map(|s| s as i64).unwrap_or(-1);
            params.temperature = temperature.clamp(0.0, 2.0) as f32;

            let text_c = CString::new(text).unwrap();
            let lang_c = CString::new("English").unwrap();
            let instruct_c: Option<CString> = instruct.map(|s| CString::new(s).unwrap());
            let speaker_c: Option<CString> = speaker.map(|s| CString::new(s).unwrap());

            params.text = text_c.as_ptr();
            params.lang = lang_c.as_ptr();
            if let Some(ic) = &instruct_c { params.instruct = ic.as_ptr(); }
            if let Some(sc) = &speaker_c { params.speaker = sc.as_ptr(); }

            // ICL clone latents (phase 2).  `ref_text_c` must outlive the call.
            let ref_text_c: Option<CString> = match &icls {
                Some((_, rt)) => Some(CString::new(rt.clone()).unwrap()),
                None => None,
            };
            if let (Some((lat, _)), Some(rt)) = (&icls, &ref_text_c) {
                params.ref_spk_emb = lat.spk_emb.as_ptr();
                params.ref_spk_dim = lat.spk_emb.len() as c_int;
                params.ref_codes = lat.codes.as_ptr();
                params.ref_T = lat.ref_T;
                params.ref_text = rt.as_ptr();
            }

            let ctx = ctx as *mut QtContext;
            let mut audio = std::mem::MaybeUninit::<QtAudio>::uninit();
            let rc = unsafe { qt_synthesize(ctx, &params, audio.as_mut_ptr()) };
            let mut audio = unsafe { audio.assume_init() };
            if rc != QT_STATUS_OK {
                let msg = unsafe { cstr_to_string(qt_last_error()) };
                unsafe { qt_audio_free(&mut audio) };
                anyhow::bail!("qt_synthesize failed (status {rc}): {msg}");
            }

            let n = audio.n_samples.max(0) as usize;
            let samples = if n > 0 && !audio.samples.is_null() {
                unsafe { std::slice::from_raw_parts(audio.samples, n) }.to_vec()
            } else {
                Vec::new()
            };
            let sample_rate = audio.sample_rate;
            unsafe { qt_audio_free(&mut audio) };

            write_wav(&samples, sample_rate as u32, &output_path)
                .with_context(|| format!("Failed to write WAV: {}", output_path.display()))
        })
        .await
        .context("qwentts.cpp TTS thread panicked")??;

        Ok(())
    }
}

// ─── Voice mapping (shared with ort backend's philosophy) ──────────────────

/// Map an abstract speaker name to a concrete acoustic voice description.
///
/// Mirrors the strategy in `main.rs::voice_profile_for_speaker`: NARRATOR gets
/// a fixed neutral profile, character names are hashed into an acoustic
/// palette covering a wide spread (male/female, register, timbre).
fn voice_profile_for_speaker(speaker: &str) -> (String, String) {
    if speaker == "NARRATOR" || speaker == "UNKNOWN" {
        return (
            "male baritone, warm chest resonance, smooth slightly gravelly timbre".into(),
            "measured, calm, authoritative".into(),
        );
    }
    let name_lower = speaker.to_lowercase();
    let name_lower = name_lower.trim_end_matches('_');
    let probably_female = matches!(
        name_lower,
        "elena" | "sarah" | "mary" | "anne" | "anna" | "emma" | "lucy" | "kate" | "clara"
        | "alice" | "julia" | "lydia" | "grace" | "ruth" | "helen" | "rose" | "jane"
        | "sophie" | "lily" | "eva" | "nina" | "nora" | "vera" | "iris" | "ada"
    ) || name_lower.ends_with('a')
      || name_lower.ends_with("ine")
      || name_lower.ends_with("elle")
      || name_lower.ends_with("ette");

    let female_profiles: &[&str] = &[
        "female mezzo-soprano, warm mid-range, smooth clear timbre, slight husky edge",
        "female contralto, deep rich chest voice, resonant and velvety timbre",
        "female soprano-adjacent, bright clear tone, light airy quality, precise articulation",
        "female mezzo, slightly breathy timbre, warm and intimate, conversational register",
    ];
    let male_profiles: &[&str] = &[
        "male tenor, light bright tone, clear articulation, youthful quality",
        "male baritone, mid-low register, slightly rough textured timbre, grounded",
        "male bass-baritone, deep resonant chest voice, dark velvety timbre",
        "male tenor-baritone, smooth warm mid-register, clean clear timbre",
    ];

    let name_hash: usize = speaker.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
    let palette = if probably_female { female_profiles } else { male_profiles };
    let idx = name_hash % palette.len();
    let voice = palette[idx];
    (voice.to_string(), "measured, calm, authoritative".to_string())
}

/// Map an abstract antioch speaker name to a qwentts.cpp custom_voice name.
///
/// qwentts.cpp custom_voice ships a fixed table of named speakers.  We map
/// abstract antioch speakers onto that table deterministically by hashing the
/// name, so the same book always gets the same voice assignment.
fn customvoice_speaker(speaker: &str) -> String {
    const SPEAKERS: &[&str] = &[
        "serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan",
    ];
    let name_lower = speaker.to_lowercase();
    if SPEAKERS.contains(&name_lower.as_str()) {
        return name_lower;
    }
    let hash: usize = speaker.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
    SPEAKERS[hash % SPEAKERS.len()].to_string()
}

// ─── WAV output (float32 mono, matches ort.rs write_wav) ────────────────────

fn write_wav(samples: &[f32], sample_rate: u32, path: &Path) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("WAV create: {}", path.display()))?;
    for &s in samples {
        w.write_sample(s.clamp(-1.0, 1.0))
            .context("WAV write")?;
    }
    w.finalize().context("WAV finalize")
}
