use anyhow::Result;

#[cfg(feature = "ggml")]
pub mod cpp;

#[cfg(feature = "ggml")]
pub use cpp::CppTts;

/// Trait abstracting over TTS backends.
#[cfg(feature = "ggml")]
#[async_trait::async_trait]
pub trait TtsEngine {
    /// Synthesise `text` to a WAV file at `output_path`.
    ///
    /// `voice`       — speaker name key into the voice map.
    /// `style`       — delivery direction.
    /// `seed`        — RNG seed; same seed = same voice character.
    /// `temperature` — logit temperature for AR sampling.
    async fn synthesise(
        &self,
        text:        &str,
        voice:       &str,
        style:       &str,
        output_path: &std::path::Path,
        max_tokens:  usize,
        kv_window:   usize,
        seed:        Option<u64>,
        temperature: f64,
    ) -> Result<()>;
}
