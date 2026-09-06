# ggml-staging-automation

Staging repository for AMD experimental ggml project branches. Projects here will be upstreamed or removed — this is not a stable distribution channel.

## Project layout

```
scripts/hrx/build/           ROCm fetch, HRX build, llama.cpp build helpers
scripts/hrx/benchmark/       Perplexity / batched / lemonade benchmarks
scripts/hrx/package/         Release artifact packaging
scripts/hrx/release/         Release source validation
build_tools/github_actions/  CI helper scripts (PR depth, benchmark selection)
scan_tools/github_actions/   Bandit / gitleaks / zizmor scanners
hrx-system/                  HRX runtime submodule (empty until initialized)
llama.cpp/                   llama.cpp submodule (empty until initialized)
build/                       Default output tree (created by build scripts)
```

## Setup commands

```bash
# Initialize required submodules
git submodule update --init hrx-system llama.cpp

# Install Python helpers
python3 -m pip install --upgrade -r requirements.txt
```

## Build/test/lint commands

```bash
# Full local Release build (Linux only)
python3 scripts/hrx/build/build_all.py

# Incremental steps
python3 scripts/hrx/build/fetch_rocm.py
python3 scripts/hrx/build/build_hrx_system.py
python3 scripts/hrx/build/build_llama_cpp.py
python3 scripts/hrx/build/validate_install.py

# Security scans
python3 scan_tools/github_actions/bandit.py
python3 scan_tools/github_actions/gitleaks.py
python3 scan_tools/github_actions/zizmor.py
```

## Key conventions

- The TheRock ROCm artifact pin is in `rocm-version.json` (`release_type`, `run_id`). Scripts do not fall back to a floating latest build.
- Default build type is `Release` for both `hrx-system` and `llama.cpp`.
- Build output layout: `build/rocm-root`, `build/downloads`, `build/hrx-system-build`, `build/hrx-system-install`, `build/llama.cpp-build`, `build/llama.cpp-install`.
- CI dispatches `build_llama_cpp_linux.yml` on PR/push and runs `release_test_bench.yml` afterward.
- Security scanning: Bandit, gitleaks, and zizmor reusable workflows run on PRs and main pushes.
- Windows support is intentionally not implemented yet.

## Gotchas

- Submodules `hrx-system` and `llama.cpp` are required; the root directory entries are empty until `git submodule update --init` runs.
- The build fetches a pinned TheRock ROCm artifact run; an invalid or missing `run_id` in `rocm-version.json` will fail.
- `GGML_HRX_BUNDLE_RUNTIME_LIBS` copies HRX, Loom, and ROCm shared libraries next to the HRX backend with `$ORIGIN` RPATHs.
- ROCm sysdeps are preserved under `rocm_sysdeps/lib`.
- Releases run nightly via `.github/workflows/release.yml`; published archives ship all required runtime libraries, so no ROCm install is needed on the target machine.
