use anyhow::Result;

pub mod api;

#[cfg(feature = "llama")]
pub mod llama;

/// Trait abstracting over LLM backends (local llama.cpp or remote OpenAI API).
#[async_trait::async_trait]
pub trait LlmBackend {
    /// Send a system + user prompt and return the raw completion text.
    async fn complete(&self, system: &str, user: &str) -> Result<String>;
}
