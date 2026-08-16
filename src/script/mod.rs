use serde::{Deserialize, Serialize};

pub mod annotator;
pub mod chunker;
pub mod markdown;

fn default_instruct() -> String {
    "Neutral, even narration.".to_string()
}

/// One entry produced by the LLM annotation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptEntry {
    /// Speaker name in UPPERCASE (e.g. "NARRATOR", "ELENA")
    pub speaker: String,
    /// Verbatim text the TTS voice should speak
    pub text: String,
    /// 1-2 sentence voice direction for the TTS engine
    #[serde(default = "default_instruct")]
    pub instruct: String,
}

/// A TTS-ready chunk: consecutive same-speaker entries merged up to `max_chars`.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub speaker: String,
    pub text: String,
    pub instruct: String,
}
