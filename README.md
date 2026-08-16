# Antioch

> **Markdown → M4B audiobook in a single native binary.**
> No Python. No pip. No Docker. No ONNX Runtime.

Antioch is a Rust CLI that converts a Markdown file into a fully-chaptered M4B
audiobook using local ML models. Everything runs on-device — no cloud API
required. Both ML stages use native C++ libraries:

- **LLM** — [llama.cpp](https://github.com/ggml-org/llama.cpp) (GGML) for
  speaker/script annotation, or any OpenAI-compatible HTTP endpoint.
- **TTS** — [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)
  (GGML port of Qwen3-TTS) for speech synthesis with voice cloning.

```bash
antioch generate my-book.md --title "My Book" --author "Jane Doe"
# → my-book.m4b
```

---

## Table of Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Build from source](#build-from-source)
  - [Backend feature flags](#backend-feature-flags)
  - [Build llama.cpp](#build-llamacpp)
  - [Build qwentts.cpp](#build-qwenttscpp)
  - [Build antioch](#build-antioch)
  - [The shared ggml story](#the-shared-ggml-story)
  - [Runtime library path](#runtime-library-path)
- [Usage](#usage)
  - [Full pipeline](#full-pipeline)
  - [Single-speaker mode](#single-speaker-mode)
  - [External LLM API](#external-llm-api)
  - [Annotate then render](#annotate-then-render)
  - [Pre-fetch models](#pre-fetch-models)
- [CLI reference](#cli-reference)
- [Models](#models)
- [Hardware acceleration](#hardware-acceleration)
- [Environment variables](#environment-variables)
- [Project structure](#project-structure)
- [License](#license)

---

## How it works

```
book.md
  │
  ▼  markdown parser (pulldown-cmark)
plain text paragraphs
  │
  ▼  LLM — llama.cpp (Qwen3.5-2B GGUF) or any OpenAI-compatible API
annotated script (.json)     ← speaker, text, voice instruction per line
  │
  ▼  TTS — qwentts.cpp (Qwen3-TTS GGUF)
chunk_0001.wav … chunk_N.wav ← one WAV per text chunk
  │
  ▼  hound + rubato (pure Rust)
merged.wav                   ← concatenated with configurable silence gaps
  │
  ▼  ffmpeg (optional — see below)
book.m4b                     ← AAC M4B with chapter markers + cover art
```

### No ffmpeg? No problem

When `ffmpeg` is not on `PATH`, Antioch writes a `.wav` + `.cue` pair instead.
You get full audio immediately; install ffmpeg later and run `antioch render`
to repackage into M4B without re-generating audio.

### Checkpoint and resume

Chunk WAVs are written to a hidden temp directory (`.antioch_<stem>/`) alongside
the output file. If the process is interrupted, re-running the same command
skips any chunks that already have a WAV on disk — no audio is re-generated.

### Multi-GPU (process-per-device)

The most expensive stage — segment TTS synthesis — is embarrassingly parallel:
each chunk is independent, writes a disjoint `NNNN.wav`, and the merger consumes
the ordered list afterwards.  You can therefore fan it out across multiple GPUs
by running one **process per device**, each loading the same model onto its own
device and handling a different slice of chunks.

```bash
# 2-GPU CUDA example (same model loaded on both cards)
antioch generate my-book.md --device-id 0 --shard-count 2 --shard-id 0 &

antioch generate my-book.md --device-id 1 --shard-count 2 --shard-id 1 &

wait   # both shards write disjoint .antioch_my-book/NNNN.wav files

# Recombine everything into the final M4B (no model load)
antioch generate my-book.md --merge-only --title "My Book"
```

Notes:
- `--shard-id` selects which chunks this process synthesises (chunks where
  `index % shard_count == shard_id`).  Give each process a unique `--shard-id`
  in `0..shard_count` and a matching `--device-id`.
- All shards share the same `.antioch_<stem>/` output directory, so they can run
  concurrently and are individually checkpointable/resumable.
- When `--shard-count > 1` a shard exits **without** merging; run a final
  `--merge-only` pass (which needs no GPU/model) once every shard is done.
- `--merge-only` works with `generate` (re-parses + re-annotates the input) or
  `render` (reads the script JSON) — use whichever matches your workflow.
- Each device needs enough VRAM for its own full model copy.

---

## Quick start

```bash
# 1. Download models (one-time, ~3–6 GB total)
antioch fetch

# 2. Convert a book — models resolve and download automatically if needed
antioch generate my-book.md --title "My Book" --author "Jane Doe"
```

`my-book.m4b` will appear in the same directory.

> Both the LLM GGUF and the TTS GGUFs are **auto-resolved**. If you don't pass
> explicit model paths, antioch checks `~/.cache/huggingface/` for the default
> models and downloads them on first use. See [Models](#models).

---

## Build from source

### Prerequisites

- Rust toolchain (`rustup`)
- CMake ≥ 3.14 and a C++17 compiler (for building llama.cpp / qwentts.cpp)
- (CUDA builds) NVIDIA CUDA toolkit + `nvcc` on `PATH`
- `ffmpeg` at runtime for M4B output (optional — WAV/CUE fallback otherwise)

### Backend feature flags

Antioch is feature-gated around two optional native backends:

| Feature | Backend | Cargo flag |
|---|---|---|
| `llama` | LLM via llama.cpp (GGML) | `--features llama` |
| `ggml` | TTS via qwentts.cpp (GGML) | `--features ggml` |

The two features are **independent** and **combinable**. You can build any
subset — including the full `--features "ggml llama"` combo — and the default
build (no features) produces a binary with neither native backend (only the
OpenAI-compatible API LLM path and no TTS).

| Build | LLM backend | TTS backend |
|---|---|---|
| `cargo build` | `api` only | none |
| `cargo build --features llama` | `api` + `llama` | none |
| `cargo build --features ggml` | `api` only | `ggml` |
| `cargo build --features "ggml llama"` | `api` + `llama` | `ggml` |

### Build llama.cpp

```bash
cd ../llama.cpp
git checkout 7d56da7            # ggml 0.17.0 pinned (see "shared ggml" below)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON -DGGML_BLAS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF
cmake --build build --config Release -j"$(nproc)"
```

Output lands in `../llama.cpp/build/bin/` (`libllama.so`, `libggml*.so`).

### Build qwentts.cpp

```bash
cd ../qwentts.cpp
./buildcpu.sh          # CPU-only
# or
./buildcuda.sh         # NVIDIA GPU (CUDA)
# or
./buildvulkan.sh       # Vulkan GPU
```

Output lands in `../qwentts.cpp/build/` (`libqwen.so`, `libggml*.so`).

If `buildcuda.sh` reports *CUDA Toolkit not found*, point CMake at your toolkit:

```bash
cd ../qwentts.cpp/build
cmake .. -DGGML_CUDA=ON -DQWEN_SHARED=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --config Release -j"$(nproc)"
```

### Build antioch

```bash
# LLM only (llama.cpp)
LLAMA_LIB_DIR=../llama.cpp/build/bin cargo build --release --features llama

# TTS only (qwentts.cpp)
QWENTTS_LIB_DIR=../qwentts.cpp/build cargo build --release --features ggml

# Both
LLAMA_LIB_DIR=../llama.cpp/build/bin \
QWENTTS_LIB_DIR=../qwentts.cpp/build \
cargo build --release --features "ggml llama"
```

### The shared ggml story

Both llama.cpp and qwentts.cpp build against **GGML** and produce a shared
`libggml.so.0`. If the two libraries are built against *different* ggml versions,
their `libggml.so.0` files collide at runtime and the dynamic linker resolves
both to one copy, causing symbol errors or crashes.

The two must therefore be built against the **same** ggml source. The tested
configuration pins llama.cpp to commit `7d56da7` ("sync : ggml"), which vendors
ggml **0.17.0** directly in-tree, and replaces its `ggml/` directory with
qwentts.cpp's ggml fork (also 0.17.0, plus the CUDA/Vulkan patches qwentts.cpp
needs). Both then emit an identical `libggml.so.0.17.0`:

```bash
cd ../llama.cpp
git checkout 7d56da7
rm -rf ggml && cp -r ../qwentts.cpp/ggml .
# … then run the llama.cpp CMake build above
```

> If you build **only** `--features llama` or **only** `--features ggml`, the
> ggml unification step is unnecessary — only the combined build needs matching
> ggml versions.

### Runtime library path

Put qwentts.cpp's build dir **first** on `LD_LIBRARY_PATH` so both `libllama.so`
and `libqwen.so` resolve `libggml.so.0` to the **same** file (qwentts.cpp's copy,
which is identical). Then add llama.cpp and CUDA:

```bash
export LD_LIBRARY_PATH=../qwentts.cpp/build:../llama.cpp/build/bin:/usr/local/cuda/lib64

./target/release/antioch generate my-book.md \
  --llm-backend llama \
  --tts-backend ggml
```

LLM layers are offloaded to the GPU by default (`--llm-gpu-layers` defaults
to `-1`, i.e. all layers). Pass `--llm-gpu-layers 0` to force CPU-only.

### Build environment variables

| Env var | Meaning |
|---|---|
| `LLAMA_LIB_DIR` | llama.cpp build `bin/` dir containing `libllama.so` (default `../llama.cpp/build/bin`) |
| `LLAMA_SHARED=1` | Force dynamic linking of `libllama.so` (used automatically when no `libllama.a` exists) |
| `LLAMA_CUDA=1` | Link the CUDA GGML backend (static builds only) |
| `QWENTTS_LIB_DIR` | qwentts.cpp build dir containing `libqwen.so` / `libqwen-core.a` (default `../qwentts.cpp/build`) |
| `QWENTTS_SHARED=1` | Link `libqwen.so` (dynamic) instead of the static `libqwen-core.a` |
| `QWENTTS_CUDA=1` | Also link the CUDA GGML backend (static builds) |
| `QWENTTS_VULKAN=1` | Also link the Vulkan GGML backend (static builds) |

---

## Usage

### Full pipeline

```bash
# Minimal — title defaults to "Audiobook"
antioch generate book.md

# With metadata and cover art
antioch generate book.md \
  --title "The Golden Dawn" \
  --author "Israel Regardie" \
  --cover cover.jpg

# Save the intermediate script for inspection
antioch generate book.md \
  --title "My Book" \
  --output /path/to/my-book.m4b \
  --save-script
```

### Single-speaker mode

Skips the LLM entirely. Every chunk is narrated by a single voice.
Fastest option — good for non-fiction and technical books.

```bash
antioch generate book.md \
  --single-speaker \
  --narrator-voice "warm, measured female voice, slight British accent, mature" \
  --narrator-style "calm, authoritative, unhurried"
```

`--narrator-voice` controls acoustic character (timbre, age, accent).
`--narrator-style` controls delivery (pace, emotion, energy).
Mixing acoustic terms into `--narrator-style` can cause voice drift.

### Local LLM — llama.cpp

By default antioch annotates with the local **llama.cpp** backend
(`--llm-backend llama`). It loads any `.gguf` model on CPU or GPU (CUDA).

```bash
# Default — auto-resolves unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf
antioch generate book.md --llm-backend llama

# Explicit model + GPU offload
antioch generate book.md \
  --llm-gguf models/model.gguf \
  --llm-gpu-layers 0   # override the default (force CPU-only)
```

- `--llm-gguf` is the path to a GGUF model. If omitted, it is resolved to
  `unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf` from the HuggingFace cache
  (`~/.cache/huggingface/`), downloading it if not already present.
- `--llm-gpu-layers` offloads that many layers to the GPU. Defaults to `-1`
  (all layers on GPU); `0` forces CPU-only.
- Generation uses the model's own chat template, falling back to ChatML.
- llama.cpp logs are routed through `tracing` (target `llamacpp`); hidden by
  default, enable with `RUST_LOG=llamacpp=debug`.

### External LLM API

Point Antioch at any OpenAI-compatible endpoint for better annotation quality
on complex or dialogue-heavy texts:

```bash
# Ollama running locally
antioch generate book.md \
  --llm-backend api \
  --llm-url http://localhost:11434/v1 \
  --llm-model qwen2.5:14b

# OpenAI
antioch generate book.md \
  --llm-backend api \
  --llm-url https://api.openai.com/v1 \
  --llm-key sk-... \
  --llm-model gpt-4o-mini
```

### TTS — qwentts.cpp

By default antioch synthesises audio with the native **qwentts.cpp** backend
(`--tts-backend ggml`). There are three GGML synthesis modes, selected by
`--tts-ggml-mode` (default `twophase`):

| Mode | Phase 1 | Phase 2 |
|---|---|---|
| `twophase` *(default)* | voice_design talker generates a reference clip per speaker | base talker clones that voice via ICL for every chunk |
| `voicedesign` | — | one voice_design talker drives every chunk via `instruct` |
| `customvoice` | — | one customvoice talker selects named speakers |

```bash
# Minimal — default q4 models auto-resolved and fetched into the HF cache:
antioch generate my-book.md --tts-backend ggml

# Explicit models + precision:
antioch generate my-book.md \
  --tts-backend ggml \
  --tts-ggml-mode twophase \
  --tts-ggml-precision q8 \
  --tts-ggml-talker      models/qwen-talker-1.7b-voicedesign-Q8_0.gguf \
  --tts-ggml-base-talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --tts-ggml-codec       models/qwen-tokenizer-12hz-Q8_0.gguf
```

**Automatic model resolution.** The model paths are optional. If you omit
`--tts-ggml-talker` / `--tts-ggml-base-talker` / `--tts-ggml-codec`, antioch
selects the default GGUF for the chosen precision (`--tts-ggml-precision`),
checks the HuggingFace cache (`~/.cache/huggingface/`), and downloads it from
`Serveurperso/Qwen3-TTS-GGUF` if it is not cached.

| `--tts-ggml-precision` | GGUF variant |
|---|---|
| `q4` *(default)* | `Q4_K_M` |
| `q8` | `Q8_0` |
| `f32` | `F32` |
| `bf16` | `BF16` |

Any explicitly supplied model path takes precedence over the automatic
resolution, so you can mix (e.g. a cached Q4 talker with a downloaded Q8
tokenizer).

### Annotate then render

Separate the two phases to inspect or hand-edit the script before a TTS run:

```bash
# Phase 1: produce script JSON only (no audio)
antioch annotate book.md --output book.script.json

# Edit book.script.json if desired …

# Phase 2: render audio from the script
antioch render book.script.json \
  --title "My Book" \
  --output my-book.m4b
```

The script JSON is an array of objects — one per spoken chunk:

```json
[
  {
    "speaker": "NARRATOR",
    "text": "The conference room smelled of stale coffee and carpet cleaner.",
    "instruct": "Slow, atmospheric, low energy."
  },
  {
    "speaker": "PRIYA",
    "text": "Sorry. The Piccadilly line decided today was a good day to exist as a concept.",
    "instruct": "Wry, slightly breathless, self-deprecating."
  }
]
```

### Pre-fetch models

Download all model weights ahead of time for offline use:

```bash
antioch fetch                        # both LLM and TTS
antioch fetch --llm false            # TTS only
antioch fetch --tts false            # LLM only
```

Models are stored in `~/.cache/huggingface/` via the standard HF Hub cache and
reused across runs. If you skip this step, models are downloaded automatically
on first use.

---

## CLI reference

### `antioch generate`

```
antioch generate <INPUT.md> [OPTIONS]

Arguments:
  <INPUT.md>                          Input Markdown file

Options:
  -o, --output <FILE>                 Output M4B path  [default: <input stem>.m4b]
      --title <TITLE>                 M4B title        [default: Audiobook]
      --author <AUTHOR>               M4B author
      --cover <FILE>                  Cover image (JPEG or PNG)
      --save-script                   Save intermediate script JSON next to the output

  # Speaker / voice
      --single-speaker                Skip LLM; use one narrator voice for all chunks
      --narrator-voice <DESC>         Acoustic voice description
                                      [default: "male baritone, rich chest resonance,
                                       warm smooth timbre"]  [env: ANTIOCH_NARRATOR_VOICE]
      --narrator-style <DESC>         Delivery style description
                                      [default: "measured, calm, authoritative"]
                                      [env: ANTIOCH_NARRATOR_STYLE]

  # LLM options
      --llm-backend <llama|api>       LLM backend  [default: llama]  [env: ANTIOCH_LLM_BACKEND]
      --llm-gguf <FILE>               GGUF path (llama); auto-resolves default if omitted
                                      [env: ANTIOCH_LLM_GGUF]
      --llm-gpu-layers <N>            GPU layers to offload (llama; <0 = all)
                                      [default: -1]  [env: ANTIOCH_LLM_GPU_LAYERS]
      --llm-url <URL>                 API base URL (api)   [default: http://localhost:11434/v1]
      --llm-key <KEY>                 API key (api)       [default: local]
      --llm-model <MODEL>             Model name (api)    [default: qwen2.5:14b]
      --chunk-size <N>                LLM annotation chunk size (chars)  [default: 3000]
      --max-tokens <N>                Max new tokens per LLM call        [default: 4096]
      --temperature <F>               LLM sampling temperature           [default: 0.6]

  # TTS options
      --tts-backend <ggml|none>       TTS engine  [default: ggml]  [env: ANTIOCH_TTS_BACKEND]
      --tts-ggml-mode <mode>          twophase | voicedesign | customvoice
                                      [default: twophase]  [env: ANTIOCH_TTS_GGML_MODE]
      --tts-ggml-precision <p>        q4 | q8 | f32 | bf16  (auto-resolved models)
                                      [default: q4]  [env: ANTIOCH_TTS_GGML_PRECISION]
      --tts-ggml-talker <path>        talker GGUF (auto-fetched if omitted)
                                      [env: ANTIOCH_TTS_GGML_TALKER]
      --tts-ggml-base-talker <path>   base talker GGUF, twophase (auto-fetched if omitted)
                                      [env: ANTIOCH_TTS_GGML_BASE_TALKER]
      --tts-ggml-codec <path>         codec/tokenizer GGUF (auto-fetched if omitted)
                                      [env: ANTIOCH_TTS_GGML_CODEC]
      --tts-temperature <F>           TTS sampling temperature (0–1)  [default: 0.7]
      --tts-max-tokens <N>            Max audio tokens per chunk  [default: 2048]
      --tts-seed <N>                  Fixed RNG seed for reproducibility
      --kv-window <N>                 Talker KV-cache sliding window  [default: 512]

  # Multi-GPU sharding (generate & render)
      --device-id <N>                 GPU device index for the TTS process  [default: 0]
      --shard-count <N>               Total parallel shard processes [default: 1]
      --shard-id <N>                  This process's shard index (0..count)  [default: 0]
      --merge-only                    Skip synthesis; merge existing WAVs into M4B

  # Audio pacing
      --pause-between-speakers-ms <MS>  Silence between different speakers [default: 500]
      --pause-same-speaker-ms <MS>      Silence, same speaker continuation [default: 250]
      --crossfade-ms <MS>               Cross-fade at segment joins        [default: 20]

  # Compression (final mix)
      --compress <true|false>           Dynamics compression  [default: true]
      --compress-threshold-db <DB>      [default: -18]
      --compress-ratio <R>              [default: 4]
      --compress-makeup-db <DB>         [default: 6]
      --compress-attack-ms <MS>         [default: 10]
      --compress-release-ms <MS>        [default: 100]
      --compress-limit-db <DB>          [default: -1]
```

### `antioch annotate`

```
antioch annotate <INPUT.md> [OPTIONS]

Arguments:
  <INPUT.md>                Input Markdown file

Options:
  -o, --output <FILE>       Script JSON output  [default: <input stem>.script.json]
      --single-speaker      Skip LLM; use narrator voice for every line
      --narrator-voice, --narrator-style   (same as generate)
      [all --llm-* options from generate]
```

### `antioch render`

```
antioch render <SCRIPT.json> [OPTIONS]

Arguments:
  <SCRIPT.json>             Annotated script produced by annotate (or hand-written)

Options:
  -o, --output <FILE>       Output M4B path  [default: <script stem>.m4b]
      --title, --author, --cover  (same as generate)
      [all --tts-*, --pause-*, --crossfade-*, --compress-* options from generate]
```

### `antioch fetch`

```
antioch fetch [OPTIONS]

Options:
  --llm   Fetch LLM model weights [default: true]
  --tts   Fetch TTS model weights [default: true]
  (pass --llm false / --tts false to skip one)
```

---

## Models

All models are stored in `~/.cache/huggingface/` via the standard HF Hub cache
and reused across runs. If you omit explicit paths, antioch resolves and
downloads them automatically.

### LLM — Qwen3.5-2B (GGUF)

Parses prose into an annotated speaker script. Runs once per ~3000-char chunk.

| | |
|---|---|
| **HF repo** | `unsloth/Qwen3.5-2B-GGUF` |
| **File** | `Qwen3.5-2B-Q8_0.gguf` (Q8_0 quantized) |
| **Size** | ~1.9 GB |
| **Architecture** | Standard decoder-only transformer (llama.cpp / GGUF) |

The model runs in no-think mode (` thinking\n response\n` prefix) to suppress
chain-of-thought reasoning and return JSON directly.

For higher-quality annotation on long or complex texts use `--llm-backend api`
with a larger model (e.g. Qwen2.5-14B via Ollama, or GPT-4o-mini).

### TTS — Qwen3-TTS-12Hz-1.7B (GGUF)

Synthesises natural speech with voice cloning via in-context learning.
In `twophase` mode two talker models are loaded per session:

| | VoiceDesign | Base |
|---|---|---|
| **Purpose** | Generates a reference voice clip from a text description | Synthesises final speech, cloning the reference voice |
| **GGUF** | `qwen-talker-1.7b-voicedesign-*.gguf` | `qwen-talker-1.7b-base-*.gguf` |
| **Size (Q4_K_M)** | ~1.1 GB | ~1.1 GB |

Plus the shared codec/tokenizer `qwen-tokenizer-12hz-*.gguf` (~0.25 GB).

**Pipeline per chunk (twophase):**

1. **VoiceDesign** synthesises a ~5-second reference clip from the voice
   description (once per unique speaker)
2. **tok_encoder** encodes the reference clip to codec codes for in-context
   learning
3. **speaker_encoder** extracts an x-vector from the reference clip
4. **talker** (AR decoder) generates first-codebook tokens from text + ICL context
5. **code_predictor** fills codebooks 2–16 causally (called 15× per AR frame)
6. **tok_decoder** converts codec codes to 24 kHz PCM (25-frame / 2-second windows)

**Precision variants** (selected by `--tts-ggml-precision`, applies to
auto-resolved models):

| Precision | GGUF variant | Size (talker) |
|---|---|---|
| `q4` *(default)* | `Q4_K_M` | ~1.1 GB |
| `q8` | `Q8_0` | ~1.3 GB |
| `f32` | `F32` | several GB |
| `bf16` | `BF16` | several GB |

**Sampling parameters** (matching upstream `inference.py` defaults):

| Parameter | Value |
|---|---|
| `top_k` | 50 |
| `top_p` | 1.0 |
| `temperature` | 0.7 (configurable via `--tts-temperature`) |
| `repetition_penalty` | 1.05 |

---

## Hardware acceleration

Both native backends are built with CUDA support and fall back to CPU
automatically when no compatible GPU is present.

- **llama.cpp** — offload LLM layers with `--llm-gpu-layers <N>` (`<0` = all,
  default `-1` i.e. all layers on GPU).
  Build with `-DGGML_CUDA=ON`.
- **qwentts.cpp** — runs TTS on the CUDA device selected by `--device-id`.
  Build with `./buildcuda.sh`. The CUDA device index is configurable per
  process, enabling multi-GPU sharding.

At runtime, ensure the CUDA libraries are on the loader path:

```bash
export LD_LIBRARY_PATH=../qwentts.cpp/build:../llama.cpp/build/bin:/usr/local/cuda/lib64
```

Both libraries route their C++ logs through Rust `tracing` (targets `ggml`,
`qwentts`, `llamacpp`); hidden by default, enable with `RUST_LOG=…`.

---

## Environment variables

All options marked `[env: …]` in the CLI reference can be set in the
environment:

| Variable | Equivalent flag |
|---|---|
| `ANTIOCH_LLM_BACKEND` | `--llm-backend` |
| `ANTIOCH_LLM_GGUF` | `--llm-gguf` |
| `ANTIOCH_LLM_GPU_LAYERS` | `--llm-gpu-layers` |
| `ANTIOCH_LLM_URL` | `--llm-url` |
| `ANTIOCH_LLM_KEY` | `--llm-key` |
| `ANTIOCH_LLM_MODEL` | `--llm-model` |
| `ANTIOCH_TTS_BACKEND` | `--tts-backend` |
| `ANTIOCH_TTS_GGML_MODE` | `--tts-ggml-mode` |
| `ANTIOCH_TTS_GGML_PRECISION` | `--tts-ggml-precision` |
| `ANTIOCH_TTS_GGML_TALKER` | `--tts-ggml-talker` |
| `ANTIOCH_TTS_GGML_BASE_TALKER` | `--tts-ggml-base-talker` |
| `ANTIOCH_TTS_GGML_CODEC` | `--tts-ggml-codec` |
| `ANTIOCH_TTS_TEMPERATURE` | `--tts-temperature` |
| `ANTIOCH_TTS_SEED` | `--tts-seed` |
| `ANTIOCH_KV_WINDOW` | `--kv-window` |
| `ANTIOCH_TTS_DEVICE_ID` | `--device-id` |
| `ANTIOCH_TTS_SHARD_COUNT` | `--shard-count` |
| `ANTIOCH_TTS_SHARD_ID` | `--shard-id` |
| `ANTIOCH_NARRATOR_VOICE` | `--narrator-voice` |
| `ANTIOCH_NARRATOR_STYLE` | `--narrator-style` |

**Build-time** environment variables are listed in the
[build section](#build-environment-variables).

---

## Project structure

```
src/
├── main.rs                CLI entry point (clap subcommands: generate, annotate, render, fetch)
├── script/
│   ├── mod.rs             ScriptEntry and Chunk types
│   ├── markdown.rs        Markdown → plain text (pulldown-cmark)
│   ├── chunker.rs         Sentence splitting and same-speaker merging
│   └── annotator.rs       LLM annotation pipeline + JSON repair
├── llm/
│   ├── mod.rs             LlmBackend trait
│   ├── api.rs             OpenAI-compatible HTTP client (reqwest + rustls)
│   └── llama.rs           llama.cpp FFI + LlamaCpp backend (--features llama)
├── tts/
│   ├── mod.rs             TtsEngine trait
│   └── cpp.rs             qwentts.cpp FFI + two-phase pipeline (--features ggml)
└── audio/
    ├── mod.rs
    └── merger.rs          WAV merge + resample (rubato) + compressor + M4B encode (ffmpeg)
```

---

## License

MIT
