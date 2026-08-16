//! llama.cpp LLM backend — links GGML's `libllama` as a Rust library and
//! exposes it through the shared `LlmBackend` trait.
//!
//! llama.cpp exposes a stable C ABI (`llama.h`): opaque `llama_model` /
//! `llama_context` / `llama_sampler` handles, POD params structs with
//! `llama_*_default_params` initialisers, and `llama_*` functions.  This module
//! mirrors the relevant subset of that ABI with hand-written `#[repr(C)]`
//! bindings (no bindgen dependency needed — the header is a stable C API).
//!
//! Build/link is handled by `build.rs` under the `llama` feature.  Generation
//! uses the standard llama.cpp sampler chain (temperature → top-p → penalties →
//! dist) and a single-token decode loop with a fixed KV-cache context.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};

use anyhow::{Context, Result};
use tracing::info;

use crate::llm::LlmBackend;

// ─── C ABI constants (must match llama.h) ────────────────────────────────────

/// Context size (tokens).
const N_CTX: u32 = 8192;
/// Batch size for prompt processing.
const N_BATCH: u32 = 512;
/// Max new tokens per `complete` call.
const MAX_NEW_TOKENS: i32 = 4096;
/// Repeat-penalty sampler params.
const PENALTY_REPEAT: i32 = 1;
const PENALTY_REPEAT_LAST_N: i32 = 64;
const PENALTY_REPEAT_VAL: f32 = 1.1;
const PENALTY_FREQ: f32 = 0.0;
const PENALTY_PRESENT: f32 = 0.0;

// ─── Opaque handle types (definitions live in llama.cpp) ────────────────────

#[repr(C)]
struct LlamaModel {
    _private: [u8; 0],
}
#[repr(C)]
struct LlamaContext {
    _private: [u8; 0],
}
#[repr(C)]
struct LlamaSampler {
    _private: [u8; 0],
}
#[repr(C)]
struct LlamaVocab {
    _private: [u8; 0],
}
#[repr(C)]
struct LlamaMemory {
    _private: [u8; 0],
}

// ─── C ABI structs (mirror llama.h POD) ──────────────────────────────────────

/// Model load parameters.  We start from `llama_model_default_params()` and
/// only override `n_gpu_layers`, so this mirrors the fields we touch.
#[repr(C)]
#[allow(non_snake_case)]
struct LlamaModelParams {
    devices: *mut c_void,
    tensor_buft_overrides: *const c_void,
    n_gpu_layers: c_int,
    split_mode: c_int,
    load_mode: c_int,
    main_gpu: c_int,
    tensor_split: *const c_float,
    progress_callback: Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
    load_mtp: bool,
}

/// Context parameters.  We start from `llama_context_default_params()` and
/// override `n_ctx`, `n_batch`, `n_threads*`.
#[repr(C)]
#[allow(non_snake_case)]
struct LlamaContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_rs_seq: u32,
    n_outputs_max: u32,
    n_outputs_max_per_seq: u32,
    n_threads: c_int,
    n_threads_batch: c_int,
    ctx_type: c_int,
    rope_scaling_type: c_int,
    pooling_type: c_int,
    attention_type: c_int,
    flash_attn_type: c_int,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: *const c_void,
    cb_eval_user_data: *mut c_void,
    type_k: c_int,
    type_v: c_int,
    abort_callback: *const c_void,
    abort_callback_data: *mut c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut c_void,
    n_samplers: usize,
    ctx_other: *mut LlamaContext,
}

/// A batch of tokens to decode (llama_batch).
#[repr(C)]
#[allow(non_snake_case)]
struct LlamaBatch {
    n_tokens: c_int,
    token: *mut c_int,
    embd: *mut f32,
    pos: *mut i64,
    n_seq_id: *mut c_int,
    seq_id: *mut *mut c_int,
    logits: *mut i8,
}

/// A chat message for the template.
#[repr(C)]
struct LlamaChatMessage {
    role: *const c_char,
    content: *const c_char,
}

/// Sampler chain params (only `no_perf`).
#[repr(C)]
#[allow(non_snake_case)]
struct LlamaSamplerChainParams {
    no_perf: bool,
}

// ─── FFI declarations (from llama.h) ─────────────────────────────────────────

#[allow(dead_code)]
extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();
    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    fn llama_model_load_from_file(
        path_model: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel;
    fn llama_model_free(model: *mut LlamaModel);
    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_model_meta_val_str(
        model: *const LlamaModel,
        key: *const c_char,
        buf: *mut c_char,
        buf_size: usize,
    ) -> c_int;
    fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: c_int) -> bool;
    fn llama_init_from_model(model: *mut LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);
    fn llama_get_memory(ctx: *const LlamaContext) -> *mut LlamaMemory;
    fn llama_memory_clear(mem: *mut LlamaMemory, data: bool);
    fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> c_int;
    fn llama_tokenize(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut c_int,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;
    fn llama_token_to_piece(
        vocab: *const LlamaVocab,
        token: c_int,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;
    fn llama_batch_get_one(tokens: *mut c_int, n_tokens: c_int) -> LlamaBatch;
    fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;
    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    fn llama_sampler_chain_add(smpl: *mut LlamaSampler, child: *mut LlamaSampler);
    fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSampler;
    fn llama_sampler_init_penalties(
        repeat: c_int,
        repeat_last_n: c_int,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> *mut LlamaSampler;
    fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;
    fn llama_sampler_sample(smpl: *mut LlamaSampler, ctx: *mut LlamaContext, idx: c_int) -> c_int;
    fn llama_sampler_accept(smpl: *mut LlamaSampler, token: c_int);
    fn llama_sampler_reset(smpl: *mut LlamaSampler);
    fn llama_sampler_free(smpl: *mut LlamaSampler);
    fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const LlamaChatMessage,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: c_int,
    ) -> c_int;
    fn llama_log_set(callback: Option<unsafe extern "C" fn(c_int, *const c_char, *mut c_void)>, user_data: *mut c_void);
}

// ─── Log bridge ──────────────────────────────────────────────────────────────

/// llama.cpp log severities (ggml_log_level enum): NONE=0, DEBUG=1, INFO=2,
/// WARN=3, ERROR=4, CONT=5.
const LLAMA_LOG_DEBUG: c_int = 1;
const LLAMA_LOG_INFO: c_int = 2;
const LLAMA_LOG_WARN: c_int = 3;
const LLAMA_LOG_ERROR: c_int = 4;

/// Bridge llama.cpp's log output into Rust `tracing` so the library's chatter
/// is filtered by `RUST_LOG` like everything else.
unsafe extern "C" fn llama_log_bridge(level: c_int, msg: *const c_char, _user_data: *mut c_void) {
    if msg.is_null() {
        return;
    }
    let msg = CStr::from_ptr(msg).to_string_lossy();
    match level {
        LLAMA_LOG_DEBUG => tracing::trace!(target: "llamacpp", "{msg}"),
        LLAMA_LOG_INFO => tracing::debug!(target: "llamacpp", "{msg}"),
        LLAMA_LOG_WARN => tracing::warn!(target: "llamacpp", "{msg}"),
        LLAMA_LOG_ERROR => tracing::error!(target: "llamacpp", "{msg}"),
        _ => tracing::trace!(target: "llamacpp", "{msg}"), // CONT etc.
    }
}

/// Install the llama.cpp log bridge exactly once (process-wide, per
/// `llama_log_set`'s semantics).
fn install_log_bridge() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        llama_backend_init();
        llama_log_set(Some(llama_log_bridge), std::ptr::null_mut());
    });
}

// ─── LlamaCpp ────────────────────────────────────────────────────────────────

/// Local LLM backend using llama.cpp.
pub struct LlamaCpp {
    model: *mut LlamaModel,
    vocab: *const LlamaVocab,
    ctx: *mut LlamaContext,
    sampler: *mut LlamaSampler,
    /// The model's chat template (for `llama_chat_apply_template`).
    chat_template: Option<String>,
}

// Raw pointers aren't Send by default; llama.cpp handles are thread-safe as
// long as we only touch them from one thread at a time (we do — every call is
// wrapped in `spawn_blocking`).
unsafe impl Send for LlamaCpp {}
unsafe impl Sync for LlamaCpp {}

impl LlamaCpp {
    /// Load a GGUF model file into a llama.cpp context.
    pub async fn load(path: String, n_gpu_layers: i32, temperature: f64) -> Result<Self> {
        let result = tokio::task::spawn_blocking(move || load_sync(&path, n_gpu_layers, temperature))
            .await
            .context("llama.cpp load thread panicked")?;
        result
    }

    fn generate(&self, system: &str, user: &str, max_tokens: i32) -> Result<String> {
        // Each `complete` is an independent conversation, so reset the KV cache
        // (the shared context is reused across calls) and the sampler state.
        unsafe {
            let mem = llama_get_memory(self.ctx);
            llama_memory_clear(mem, true);
            llama_sampler_reset(self.sampler);
        }

        // Build the chat-formatted prompt using the model's template.
        let prompt = apply_chat_template(&self.chat_template, system, user)?;

        let vocab = self.vocab;
        let n_vocab = unsafe { llama_vocab_n_tokens(vocab) };
        if n_vocab <= 0 {
            anyhow::bail!("llama.cpp vocab reports {n_vocab} tokens — invalid model");
        }

        // Tokenise the prompt.
        let mut prompt_tokens: Vec<c_int> = vec![0; prompt.len() + 16];
        let n_tok = unsafe {
            llama_tokenize(
                vocab,
                prompt.as_ptr() as *const c_char,
                prompt.len() as c_int,
                prompt_tokens.as_mut_ptr(),
                prompt_tokens.len() as c_int,
                true, // add_special
                true, // parse_special
            )
        };
        if n_tok < 0 {
            anyhow::bail!("llama_tokenize failed ({n_tok})");
        }
        prompt_tokens.truncate(n_tok as usize);
        info!("llama.cpp prompt: {} tokens", prompt_tokens.len());

        // Prefill: decode the whole prompt in one batch.
        let batch = unsafe { llama_batch_get_one(prompt_tokens.as_mut_ptr(), n_tok) };
        let rc = unsafe { llama_decode(self.ctx, batch) };
        if rc != 0 {
            anyhow::bail!("llama_decode prefill failed ({rc})");
        }

        // Sample the first token, then decode one token at a time.
        let mut generated: Vec<c_int> = Vec::new();
        let mut token = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1) };

        for _ in 0..max_tokens {
            if is_eog_token(vocab, token) {
                break;
            }
            generated.push(token);

            let single = [token];
            let batch = unsafe { llama_batch_get_one(single.as_ptr() as *mut c_int, 1) };
            let rc = unsafe { llama_decode(self.ctx, batch) };
            if rc != 0 {
                break;
            }
            unsafe { llama_sampler_accept(self.sampler, token) };
            token = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1) };
        }

        // Detokenise.
        let text = detokenize(vocab, &generated)?;
        Ok(text.trim().to_string())
    }
}

#[async_trait::async_trait]
impl LlmBackend for LlamaCpp {
    async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let addr = self as *const LlamaCpp as usize;
        let (system, user) = (system.to_string(), user.to_string());
        tokio::task::spawn_blocking(move || {
            let model = unsafe { &*(addr as *const LlamaCpp) };
            model.generate(&system, &user, MAX_NEW_TOKENS)
        })
        .await
        .context("llama.cpp generate thread panicked")?
    }
}

impl Drop for LlamaCpp {
    fn drop(&mut self) {
        unsafe {
            llama_sampler_free(self.sampler);
            llama_free(self.ctx);
            llama_model_free(self.model);
        }
    }
}

// ─── Model loading ───────────────────────────────────────────────────────────

fn load_sync(path: &str, n_gpu_layers: i32, temperature: f64) -> Result<LlamaCpp> {
    install_log_bridge();

    let path = CString::new(path).context("model path contains interior NUL")?;

    // Model params: defaults + GPU offload.
    let mut mparams = unsafe { llama_model_default_params() };
    mparams.n_gpu_layers = n_gpu_layers;

    let model = unsafe { llama_model_load_from_file(path.as_ptr(), mparams) };
    if model.is_null() {
        anyhow::bail!("llama_model_load_from_file failed for {path:?}");
    }

    // Context params: defaults + our sizes.
    let mut cparams = unsafe { llama_context_default_params() };
    cparams.n_ctx = N_CTX;
    cparams.n_batch = N_BATCH;
    cparams.n_ubatch = N_BATCH;
    cparams.no_perf = true;

    let ctx = unsafe { llama_init_from_model(model, cparams) };
    if ctx.is_null() {
        unsafe { llama_model_free(model) };
        anyhow::bail!("llama_init_from_model failed");
    }

    // Vocab.
    let vocab = unsafe { llama_model_get_vocab(model) };
    if vocab.is_null() {
        unsafe { llama_free(ctx); llama_model_free(model) };
        anyhow::bail!("llama_model_get_vocab returned null");
    }

    // Sampler chain: temp -> top_p -> penalties -> dist.
    let sparams = unsafe { llama_sampler_chain_default_params() };
    let chain = unsafe { llama_sampler_chain_init(sparams) };
    unsafe {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature as f32));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
        llama_sampler_chain_add(
            chain,
            llama_sampler_init_penalties(
                PENALTY_REPEAT,
                PENALTY_REPEAT_LAST_N,
                PENALTY_REPEAT_VAL,
                PENALTY_FREQ,
                PENALTY_PRESENT,
            ),
        );
        llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
    }

    // Chat template — read the model's `tokenizer.chat_template` metadata.
    let chat_template = read_chat_template(model);

    info!(
        "Loaded llama.cpp backend (ctx={N_CTX}, gpu_layers={n_gpu_layers}, temp={temperature:.2})"
    );

    Ok(LlamaCpp {
        model,
        vocab,
        ctx,
        sampler: chain,
        chat_template,
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Apply the model's chat template to a system + user message pair.
/// Falls back to a simple hardcoded ChatML-style format if the model has no
/// built-in template.
fn apply_chat_template(template: &Option<String>, system: &str, user: &str) -> Result<String> {
    let sys = CString::new(system).context("system prompt contains interior NUL")?;
    let usr = CString::new(user).context("user prompt contains interior NUL")?;
    // If the model has a template, try it; otherwise (or if it fails to match
    // a built-in template) fall back to ChatML (null template → "chatml").
    let mut n = -1;
    let mut buf = vec![0u8; 8192];

    if let Some(t) = template {
        if let Ok(c) = CString::new(t.as_str()) {
            let messages = [
                LlamaChatMessage { role: c"system".as_ptr(), content: sys.as_ptr() },
                LlamaChatMessage { role: c"user".as_ptr(), content: usr.as_ptr() },
            ];
            n = unsafe {
                llama_chat_apply_template(
                    c.as_ptr(),
                    messages.as_ptr(),
                    messages.len(),
                    true,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as c_int,
                )
            };
        }
    }

    // Fall back to ChatML when no template, or when the model's template
    // isn't recognised by llama.cpp's built-in list.
    if n < 0 {
        let messages = [
            LlamaChatMessage { role: c"system".as_ptr(), content: sys.as_ptr() },
            LlamaChatMessage { role: c"user".as_ptr(), content: usr.as_ptr() },
        ];
        n = unsafe {
            llama_chat_apply_template(
                std::ptr::null(), // → ChatML
                messages.as_ptr(),
                messages.len(),
                true,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            )
        };
    }

    if n < 0 {
        anyhow::bail!("llama_chat_apply_template failed ({n})");
    }
    if (n as usize) >= buf.len() {
        anyhow::bail!("llama_chat_apply_template needs a larger buffer ({n} bytes)");
    }
    buf.truncate(n as usize);
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Read the model's `tokenizer.chat_template` metadata string (if present).
fn read_chat_template(model: *mut LlamaModel) -> Option<String> {
    const KEY: &str = "tokenizer.chat_template";
    let mut buf = vec![0u8; 65536];
    let n = unsafe {
        llama_model_meta_val_str(model, KEY.as_ptr() as *const c_char, buf.as_mut_ptr() as *mut c_char, buf.len())
    };
    if n < 0 {
        return None;
    }
    buf.truncate(n as usize);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Convert a sequence of tokens into text by accumulating `llama_token_to_piece`.
fn detokenize(vocab: *const LlamaVocab, tokens: &[c_int]) -> Result<String> {
    let mut out = String::new();
    let mut buf = vec![0u8; 64];
    for &token in tokens {
        let mut n = unsafe {
            llama_token_to_piece(
                vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                0,  // lstrip
                false, // special
            )
        };
        // A negative return is the required buffer size (minus one); grow and retry.
        if n < 0 {
            let need = (-n) as usize;
            buf.resize(need + 1, 0);
            n = unsafe {
                llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as c_int,
                    0,
                    false,
                )
            };
        }
        if n > 0 {
            out.push_str(&String::from_utf8_lossy(&buf[..n as usize]));
        }
    }
    Ok(out)
}

/// Whether a token should end generation (EOS / EOG / EOT).
fn is_eog_token(vocab: *const LlamaVocab, token: c_int) -> bool {
    token < 0 || unsafe { llama_vocab_is_eog(vocab, token) }
}
