# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-07-22

### Added
- **PR-040 HF adapter I/O**
  - `hf` module: `HfLoraConfig` (`peft_type`, `r`, `lora_alpha`, `target_modules`,
    optional `base_model_name_or_path` / `task_type`, `modules_to_save` config-only)
  - HF LoRA key styles: native, `lora_A.default.weight`, module-prefixed, `base_model.model.*`
  - `save_pretrained_hf` / `load_pretrained_hf`, `extract_lora_ab`, key rewrite helpers
  - `LoraLayer::load_state_dict` accepts HF or native keys
  - Tests: `tests/hf_roundtrip.rs` + unit tests in `hf`
- **PR-041 real inject path**
  - `LinearWithLora` — base `candle_nn::Linear` + `LoraLayer` residual with real forward
  - `PeftLinearModel` / `get_peft_model(base_modules, pattern, config, name, vb)` product path
  - Legacy name-list API renamed to `get_peft_model_registry`
  - Example `examples/lora_inject_train.rs` — multi-layer MLP + AdamW adapter updates (base frozen)
  - Integration test `tests/inject_train.rs`
  - `modules_to_save` policy documented as config-only non-goal for auto-training
- **PR-042 parity fixtures**
  - `tests/parity/fixtures/lora_fwd_merge.json` + `tests/parity_lora.rs` (atol/rtol `1e-5`)
  - Offline generator `scripts/gen_lora_parity_fixture.py` (Python peft optional)
  - `METRICS.md` correctness section filled for LoRA forward/merge

### Changed
- Package version **1.1.0** (minor: product inject + HF interop surface; registry API rename)
- README status matrix updated for HF keys, inject, parity
- `LoraLayer::from_weights` for fixture/HF inject construction

### Migration
- Callers of the old string-only `get_peft_model(&[&str], …)` must switch to
  `get_peft_model_registry` or the new Linear-based `get_peft_model`.

## [1.0.4] - 2026-07-22

### Changed
- **Skew heal (SoT):** package version set to **1.0.4**, intentionally superseding crates.io / tag **1.0.3** as the forward Source of Truth (see `DECISION.md`).
- Local tree was already a descendant of `v1.0.3` but had regressed `Cargo.toml` to 1.0.1 during workspace/fleet merges; version field corrected without resetting history.
- `cuda` feature remains **candle-core/cuda only** on this tree (no optional `cubecl` / `cubecl-cuda` deps).
- **PR-021 kernels:** fused CubeCL sources **quarantined** under `src/kernels/archive/`; not exported from `lib.rs`; not compiled. README documents no active fused kernels.
- **PR-010 docs honesty:** README product class = Candle PEFT adapter **layer library**; honest status matrix; roadmap no longer claims “success criteria already met”; `METRICS.md` scaffold added.
- **PR-020 flags honesty:**
  - LoRA `dropout` applied in forward when the layer is unfrozen and `dropout > 0`
  - `loftq_iterations` documented as **simplified dual-Gaussian init**, not full SVD/quant LoftQ
  - `Trainable::freeze` documented as a layer flag (gates dropout; does not detach Vars)
  - `DoraLayer` implements `SaveLoad` (lora weights + magnitude)

### Added
- `DECISION.md` — formal restore-vs-supersede decision for ECO-P0-03 / PEFT-P0-02.
- `METRICS.md` — HF peft comparison scaffold (methods listed; numbers not yet measured).
- `src/kernels/archive/README.md` — quarantine notice for historical kernel sources.

### Fixed
- Restored missing changelog sections for **1.0.2** and **1.0.3** that were dropped after the 1.0.3 publish line.
- Corrected historical overclaims about LoftQ “100% Python PEFT parity” and incomplete DoRA SaveLoad notes (see 1.0.0 errata below).

### Notes
- Do not treat crates.io 1.0.3 as SoT for future commits; publish next from this tree as 1.0.4+.

## [1.0.3] - 2026-01-28

### Changed
- Updated safetensors dependency from 0.4 to 0.7
- Updated candle-core and candle-nn dependencies from 0.8 to 0.9
- Added explicit workspace.dependencies section in Cargo.toml
- Optional CubeCL deps under `cuda` feature; `pub mod kernels` gated on `cuda` (published surface on crates.io)

### Fixed
- Clippy lints for `manual_is_multiple_of` in boft.rs, lokr.rs, oft.rs
- Clippy lint for `manual_midpoint` in training.rs

## [1.0.2] - 2026-01-25

### Changed
- Migrated LoRA/DoRA GPU kernels to CubeCL 0.9 API
- Kernel position variables now use correct types
- Added proper usize casts at array index sites
- `sync_cube()` replaces deprecated `sync_units()`
- Wrapped kernel launches in unsafe blocks with SAFETY comments

## [1.0.1] - 2026-01-24

### Added
- CPU fallback warning when CUDA is unavailable

### Changed
- Bumped minimum Rust version to 1.92
- README license link formatting fix

## [1.0.0] - 2026-01-24

### Added
- **SaveLoad trait implementations for adapters** - Persistence support for:
  - LoraLayer (lora.rs) — **done at 1.0.0**
  - DoraLayer — **was claimed here but missing until 1.0.4** (implemented in PR-020)
  - AdaLoraLayer, Ia3Layer, LoHaLayer, LoKrLayer, OftLayer, BoftLayer, VeraLayer
  - PrefixTuningLayer, PromptTuningLayer
- **Examples directory** with 3 runnable examples:
  - `basic_lora.rs` - Simple LoRA adapter usage
  - `multi_adapter.rs` - Using AdapterRegistry with multiple adapters
  - `save_load.rs` - Persisting and loading adapter weights
- `weights()` method for LoRA adapters (from 0.4.1)
- rsLoRA scaling (`use_rslora`) and a **simplified** `loftq_iterations` init path
  - **Errata:** 1.0.0 marketing text claimed “LoftQ support for 100% Python PEFT parity”.
    That was **false**. Full LoftQ (SVD + quantization residual iterations on base weights)
    is **not** implemented; `loftq_iterations > 0` only selects dual-Gaussian A/B init.
- CLAUDE.md for Claude Code development workflow

### Fixed
- Documentation link warnings in adalora.rs and vera.rs
- Clippy warning for inline format args in lora.rs (Rust 1.92 compatibility)

### Changed
- Bumped to stable 1.0.0 release
- All 128 tests passing (at release cut)
- Full clippy compliance with pedantic lints

## [0.4.1] - 2026-01-16

### Added
- `weights()` method and individual weight accessor methods for LoRA

### Fixed
- Clippy lint fixes for Rust 1.92 compatibility

## [0.4.0] - 2026-01-09

### Added
- Development plan and comprehensive roadmap (DEVELOPMENT_PLAN.md)
- CHANGELOG.md for tracking version history
- Cargo-audit integration for CVE checking
- CI/CD pipeline (.github/workflows/ci.yml)
- Quality check scripts (quality-check.sh, pre-commit.sh, setup-dev.sh)
- Testing branch for proper QA workflow
- PLAN_SUMMARY.md for quick reference

### Changed
- Version strategy documented: strict semantic versioning
- Branch strategy formalized: working → dev → testing → main

### Fixed
- All 207 clippy warnings resolved (documentation, code quality, formatting)
- Added proper error documentation to all Result-returning functions
- Fixed comparison warnings (op_ref)
- Updated format strings to inline syntax

## [0.3.0] - 2026-01-08

### Added
- Model integration system (PeftModel wrapper)
- Training utilities (LR schedules, training state)
- Multi-adapter registry with runtime switching
- Enhanced I/O with HuggingFace PEFT format compatibility
- BOFT (Butterfly Orthogonal Fine-Tuning) adapter
- VeRA (Vector-based Random Matrix Adaptation) adapter
- Comprehensive test suite (124 tests)

### Changed
- Improved model.rs with pattern matching
- Enhanced training.rs with multiple LR schedules

## [0.2.0] - 2026-01-07

### Added
- LoHa (Low-Rank Hadamard Product) adapter
- LoKr (Low-Rank Kronecker Product) adapter
- OFT (Orthogonal Fine-Tuning) adapter
- Prefix Tuning adapter
- Prompt Tuning adapter

## [0.1.0] - 2026-01-06

### Added
- Initial implementation of LoRA adapter
- DoRA (Weight-Decomposed Low-Rank Adaptation) support
- AdaLoRA (Adaptive Low-Rank Adaptation) with SVD
- IA³ (Infused Adapter by Inhibiting and Amplifying) adapter
- Core trait system (Adapter, Mergeable, Trainable, AdapterConfig)
- Error handling (PeftError enum)
- Basic I/O with safetensors
- Configuration system
- Initial documentation (README, AGENT_GUIDE, GAP_ANALYSIS, TASK_TRACKER)

[Unreleased]: https://github.com/tzervas/peft-rs/compare/v1.0.4...HEAD
[1.0.4]: https://github.com/tzervas/peft-rs/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/tzervas/peft-rs/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tzervas/peft-rs/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/tzervas/peft-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tzervas/peft-rs/compare/v0.4.1...v1.0.0
[0.4.1]: https://github.com/tzervas/peft-rs/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/tzervas/peft-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/tzervas/peft-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tzervas/peft-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tzervas/peft-rs/releases/tag/v0.1.0
