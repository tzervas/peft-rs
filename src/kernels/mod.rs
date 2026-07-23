//! # Kernel quarantine (PR-021 / PEFT-P0-05)
//!
//! Historical CubeCL fused LoRA/DoRA kernel sources are kept under
//! [`archive/`](archive/) for reference only.
//!
//! ## Status
//!
//! | Surface | Status |
//! |---------|--------|
//! | Compiled into the crate | **No** — this module is not exported from `lib.rs` |
//! | `cuda` feature | Enables **candle-core CUDA only**, not peft-rs fused kernels |
//! | CubeCL deps | **Not** declared in this tree (removed after 1.0.3 skew heal) |
//! | Active fused GPU kernels | **None** |
//!
//! ## Why quarantine (option B)
//!
//! CubeCL 0.9 + candle CUDA feature wiring is environment-fragile and not
//! proven green on fleet CI without a dedicated GPU toolkit matrix. Exporting
//! half-working fused kernels would re-lie about acceleration.
//!
//! See `DECISION.md` (PR-002 deferred kernels) and README **Features**.
//!
//! ## Restoring later
//!
//! A future PR may rewire these sources under an optional `cubecl` feature
//! with best-effort `cargo check --features cuda`. Until then, treat
//! `src/kernels/archive/*` as dead reference code.

// Intentionally empty: no submodules, no re-exports, no CubeCL.
// Archive sources are plain files under archive/ and are not part of the
// Rust module graph.
