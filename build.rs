// build.rs — links qwentts.cpp for the `ggml` TTS backend feature.
//
// Activated only when the `ggml` cargo feature is set.
//
// Link strategy (in order of preference):
//
//  1. Shared library (`libqwen.so` / `libqwen.dll` / `libqwen.dylib`) if
//     `QWENTTS_SHARED=1` is set.  The resulting binary uses dynamic linking and
//     requires the .so to be on LD_LIBRARY_PATH at runtime.
//
//  2. Static archive `libqwen-core.a` (default qwentts.cpp build artefact).
//     Produces a fully self-contained binary — the qwentts.cpp logic is baked
//     in.  The archive is a C++ library so we must also link the C++ standard
//     library and the GGML backend libs that qwentts.cpp itself uses.
//
// REQUIRED environment variables
// ───────────────────────────────
//  QWENTTS_LIB_DIR   Path to the qwentts.cpp build directory (contains
//                    libqwen-core.a or libqwen.so and the companion ggml*.a).
//                    Defaults to `../qwentts.cpp/build` relative to this
//                    Cargo.toml.
//
// OPTIONAL environment variables
// ───────────────────────────────
//  QWENTTS_SHARED    Set to `1` to link the shared library instead of the
//                    static archive.
//  QWENTTS_CUDA      Set to `1` if qwentts.cpp was built with CUDA, so that
//                    the CUDA runtime libraries are also linked.
//  QWENTTS_VULKAN    Set to `1` if qwentts.cpp was built with Vulkan.
//  QWENTTS_METAL     Set to `1` if qwentts.cpp was built with Metal (macOS).

/// Platform-correct static-archive filename for a library basename.
///
/// CMake names static archives `libfoo.a` on Unix but `foo.lib` under MSVC
/// (and qwentts.cpp/llama.cpp both clear `CMAKE_STATIC_LIBRARY_PREFIX` on
/// WIN32, so there is no `lib` prefix even under mingw).  The auto-detect
/// probes below must use the right name or they silently return `false` on
/// Windows and the GPU backends never get linked.
fn static_archive_name(base: &str) -> String {
    if cfg!(target_env = "msvc") {
        format!("{base}.lib")
    } else if cfg!(target_os = "windows") {
        format!("{base}.a")
    } else {
        format!("lib{base}.a")
    }
}

/// True when `base` exists as a static archive in `dir`.
fn has_static(dir: &std::path::Path, base: &str) -> bool {
    dir.join(static_archive_name(base)).is_file()
}

/// Resolve a backend toggle: explicit env var wins, else probe for the archive.
fn backend_enabled(env_var: &str, dir: &std::path::Path, base: &str) -> bool {
    match std::env::var(env_var) {
        Ok(v) => v == "1",
        Err(_) => has_static(dir, base),
    }
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ggml_active = std::env::var("CARGO_FEATURE_GGML").is_ok();

    // llama.cpp LLM backend — independent of the ggml feature.  When both
    // `ggml` and `llama` are active *and* both link statically, the ggml
    // static archives (libggml.a / libggml-base.a / libggml-cpu.a /
    // libggml-vulkan.a / libggml-cuda.a) are byte-identical between the two
    // builds only if both were built from the same ggml source (see the
    // "shared ggml" note in README.md — llama.cpp is pinned to a commit and
    // has its ggml/ directory replaced with qwentts.cpp's fork).  Linking
    // both copies would produce duplicate-symbol errors, so `link_llama`
    // skips the ggml archives entirely when the `ggml` feature is also
    // active — the qwentts.cpp side below links them once for both.
    link_llama(std::path::Path::new(&manifest_dir), ggml_active);

    // Only do anything when the `ggml` feature is requested.
    if !ggml_active {
        return;
    }

    // ── Locate the qwentts.cpp build directory ───────────────────────────────
    let default_lib_dir = format!("{manifest_dir}/../qwentts.cpp/build");
    let lib_dir = std::env::var("QWENTTS_LIB_DIR").unwrap_or(default_lib_dir);
    let lib_dir = std::path::PathBuf::from(&lib_dir);

    if !lib_dir.exists() {
        panic!(
            "\n\n\
             [antioch / ggml feature] Cannot find qwentts.cpp build directory:\n\
             \n\
             {}\n\
             \n\
             Build qwentts.cpp first:\n\
             \n\
             cd ../qwentts.cpp\n\
             ./buildcpu.sh          # CPU-only\n\
             ./buildcuda.sh         # CUDA\n\
             ./buildvulkan.sh       # Vulkan\n\
             \n\
             Or set QWENTTS_LIB_DIR to point at the correct build directory.\n",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rerun-if-env-changed=QWENTTS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=QWENTTS_SHARED");
    println!("cargo:rerun-if-env-changed=QWENTTS_CUDA");
    println!("cargo:rerun-if-env-changed=QWENTTS_VULKAN");
    println!("cargo:rerun-if-env-changed=QWENTTS_METAL");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let use_shared = std::env::var("QWENTTS_SHARED").map(|v| v == "1").unwrap_or(false);

    if use_shared {
        // ── Dynamic linking ───────────────────────────────────────────────────
        // cmake -DQWEN_SHARED=ON produces libqwen.so / libqwen.dll / libqwen.dylib
        // exporting only the qt_* symbols.
        println!("cargo:rustc-link-lib=dylib=qwen");
    } else {
        // ── Static linking ────────────────────────────────────────────────────
        // Link the main qwentts.cpp archive.
        println!("cargo:rustc-link-lib=static=qwen-core");

        // GGML core and CPU backend — always present.
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");

        // CUDA backend — included when qwentts.cpp was built with CUDA.
        if backend_enabled("QWENTTS_CUDA", &lib_dir, "ggml-cuda") {
            println!("cargo:rustc-link-lib=static=ggml-cuda");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cublas");
            println!("cargo:rustc-link-lib=dylib=cublasLt");
        }

        // Vulkan backend.
        if backend_enabled("QWENTTS_VULKAN", &lib_dir, "ggml-vulkan") {
            println!("cargo:rustc-link-lib=static=ggml-vulkan");
            link_vulkan();
            link_vulkan_sdk_lib_dir();
        }

        // Metal backend (macOS).  ggml-metal links Foundation / Metal /
        // MetalKit, which rustc will not infer from the static archive — they
        // must be named explicitly as framework link args.
        if backend_enabled("QWENTTS_METAL", &lib_dir, "ggml-metal") {
            println!("cargo:rustc-link-lib=static=ggml-metal");
            link_apple_frameworks();
        }

        // Accelerate — ggml-cpu links it on Apple when GGML_ACCELERATE=ON.
        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // C++ standard library (qwentts.cpp is a C++17 project).
        // On Linux/Android link libstdc++ (GCC) or libc++ (Clang).
        // On macOS link libc++ (always Clang).
        // On Windows the MSVC runtime is linked automatically by the linker.
        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=dylib=c++");
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // ── Tell cargo to re-run if the library changes ───────────────────────────
    let lib_name = if use_shared {
        if cfg!(target_os = "windows") { "qwen.dll".to_string() }
        else if cfg!(target_os = "macos") { "libqwen.dylib".to_string() }
        else { "libqwen.so".to_string() }
    } else {
        static_archive_name("qwen-core")
    };
    println!("cargo:rerun-if-changed={}", lib_dir.join(lib_name).display());
}

/// Link the platform's Vulkan loader.
///
/// The Vulkan SDK ships a differently-named import library per platform:
///   - Windows  → `vulkan-1.lib`  (the loader DLL is vulkan-1.dll)
///   - Linux     → `libvulkan.so`  (the loader SONAME is libvulkan.so.1)
///   - macOS     → no Vulkan loader (ggml uses Metal instead)
///
/// `cargo:rustc-link-lib=dylib=X` maps to `-lX`, so we must use the exact
/// name of the import library that ships with the SDK on each platform.
fn link_vulkan() {
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=dylib=vulkan-1");
    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-lib=dylib=vulkan");
}

/// Add the Vulkan SDK's import-library directory to the linker search path.
///
/// The ggml Vulkan backend is built into a static archive (ggml-vulkan.lib),
/// but that archive does not embed the platform's Vulkan loader import library
/// — the *final* Rust link must resolve it from the SDK's `Lib` directory.
///
/// The CMake step finds the SDK via `VULKAN_SDK` (ggml appends it to
/// `CMAKE_PREFIX_PATH`).  We read the same variable so the Rust link resolves
/// the loader from the identical SDK.  On Windows the import library lives in
/// `$VULKAN_SDK/Lib`; on Linux `libvulkan.so` is resolved via the system
/// linker path, so no extra search dir is required.  No-op when `VULKAN_SDK`
/// is unset (e.g. a distro Vulkan loader on Linux).
fn link_vulkan_sdk_lib_dir() {
    #[cfg(target_os = "windows")]
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        if !sdk.is_empty() {
            println!("cargo:rustc-link-search=native={}\\Lib", sdk);
        }
    }
}

/// Frameworks required by ggml's Metal backend.  Only meaningful on macOS.
fn link_apple_frameworks() {
    if !cfg!(target_os = "macos") {
        return;
    }
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
}

// llama.cpp LLM backend — links GGML's `libllama` for the `llama` feature.
//
// Link strategy (mirrors the qwentts.cpp `ggml` feature):
//
//  1. Shared library (`libllama.so` / `libllama.dll` / `libllama.dylib`) when
//     `LLAMA_SHARED=1` is set, or when no static `libllama.a` exists (the
//     llama.cpp `BUILD_SHARED_LIBS=ON` default).  Runtime requires the build
//     dir on LD_LIBRARY_PATH (libllama.so pulls in libggml*.so, and the
//     CPU/CUDA backends are loaded as plugins).
//
//  2. Static archive `libllama.a` if present.  We must also link the GGML
//     backend libs that llama.cpp uses.
fn link_llama(manifest_dir: &std::path::Path, ggml_provides_ggml: bool) {
    // Only do anything when the `llama` feature is requested.
    if std::env::var("CARGO_FEATURE_LLAMA").is_err() {
        return;
    }

    // ── Locate the llama.cpp build directory ────────────────────────────────
    let default_lib_dir = format!("{}/../llama.cpp/build/bin", manifest_dir.display());
    let lib_dir = std::env::var("LLAMA_LIB_DIR").unwrap_or(default_lib_dir);
    let lib_dir = std::path::PathBuf::from(&lib_dir);

    if !lib_dir.exists() {
        panic!(
            "\n\n\
             [antioch / llama feature] Cannot find llama.cpp build directory:\n\n\
             {}\n\n\
             Build llama.cpp first (see llama.cpp/README.md), then either:\n\n\
               - leave it at the default ../llama.cpp/build/bin, or\n\
               - set LLAMA_LIB_DIR to point at the correct directory.\n",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_SHARED");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");
    println!("cargo:rerun-if-env-changed=LLAMA_METAL");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    // Prefer shared linking when a .so is available and no static archive
    // exists (the llama.cpp BUILD_SHARED_LIBS=ON default), or when the user
    // explicitly requests it.
    let static_archive = lib_dir.join(static_archive_name("llama"));
    let shared_lib = if cfg!(target_os = "windows") {
        lib_dir.join("llama.dll")
    } else if cfg!(target_os = "macos") {
        lib_dir.join("libllama.dylib")
    } else {
        lib_dir.join("libllama.so")
    };
    let explicit_shared = std::env::var("LLAMA_SHARED").map(|v| v == "1").unwrap_or(false);
    let use_shared = explicit_shared || (!static_archive.is_file() && shared_lib.exists());

    if use_shared {
        // ── Dynamic linking ───────────────────────────────────────────────────
        // libllama.so bundles libggml.so / libggml-base.so as DT_NEEDED, and
        // loads the CPU/CUDA backends as runtime plugins.
        println!("cargo:rustc-link-lib=dylib=llama");
    } else {
        // ── Static linking ───────────────────────────────────────────────────
        println!("cargo:rustc-link-lib=static=llama");

        // Skip the ggml archives entirely when the `ggml` feature is also
        // active and will link them itself — see the comment in `main()`.
        // Linking two copies of libggml.a (one from llama.cpp's build dir,
        // one from qwentts.cpp's) produces duplicate-symbol errors even
        // when both were built from the same ggml source.
        if !ggml_provides_ggml {
            println!("cargo:rustc-link-lib=static=ggml");
            println!("cargo:rustc-link-lib=static=ggml-base");
            println!("cargo:rustc-link-lib=static=ggml-cpu");

            if backend_enabled("LLAMA_CUDA", &lib_dir, "ggml-cuda") {
                println!("cargo:rustc-link-lib=static=ggml-cuda");
                println!("cargo:rustc-link-lib=dylib=cudart");
                println!("cargo:rustc-link-lib=dylib=cublas");
                println!("cargo:rustc-link-lib=dylib=cublasLt");
            }

            if backend_enabled("LLAMA_VULKAN", &lib_dir, "ggml-vulkan") {
                println!("cargo:rustc-link-lib=static=ggml-vulkan");
                link_vulkan();
                link_vulkan_sdk_lib_dir();
            }

            if backend_enabled("LLAMA_METAL", &lib_dir, "ggml-metal") {
                println!("cargo:rustc-link-lib=static=ggml-metal");
                link_apple_frameworks();
            }

            // Accelerate — ggml-cpu links it on Apple when GGML_ACCELERATE=ON.
            #[cfg(target_os = "macos")]
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }

        // C++ standard library (llama.cpp is a C++ project).
        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=dylib=c++");
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    let lib_file = if use_shared {
        if cfg!(target_os = "windows") { "llama.dll".to_string() }
        else if cfg!(target_os = "macos") { "libllama.dylib".to_string() }
        else { "libllama.so".to_string() }
    } else {
        static_archive_name("llama")
    };
    println!("cargo:rerun-if-changed={}", lib_dir.join(lib_file).display());
}
