use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::llm::LlmBackend;

/// OpenAI-compatible HTTP API backend (Ollama, LM Studio, OpenAI, etc.)
pub struct LlmApiClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: usize,
    temperature: f64,
}

impl LlmApiClient {
    pub fn new(
        base_url: &str,
        api_key: &str,
        model: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            max_tokens,
            temperature,
        }
    }
}

// ─── Wire types ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: usize,
    temperature: f64,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: String,
}

// ─── Trait impl ───────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl LlmBackend for LlmApiClient {
    async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);

        let req = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage { role: "system", content: system },
                ChatMessage { role: "user",   content: user },
            ],
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .context("HTTP request to LLM API failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API returned {status}: {body}");
        }

        let body: ChatResponse = resp
            .json()
            .await
            .context("Failed to deserialise LLM API response")?;

        body.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("LLM API returned empty choices"))
    }
}
