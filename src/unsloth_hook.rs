// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Optional consume-path for [unsloth-rs](https://github.com/tzervas/unsloth-rs).
//!
//! Feature `unsloth` (not default). **Does not** un-quarantine `src/kernels/archive`.
//! peft remains adapter math. Fused `LoRA`-add is **not** dispatched (no such
//! kernel in unsloth-rs 1.0.x).

/// Feature compiled in. Does not mean a fused `LoRA` kernel exists.
#[must_use]
pub const fn unsloth_feature_enabled() -> bool {
    cfg!(feature = "unsloth")
}

/// Whether this crate should dispatch an unsloth kernel for `LoRA` add.
///
/// Always `false` in 1.1.x — no fused `LoRA`-add `CustomOp` yet.
#[must_use]
pub const fn should_dispatch_unsloth_lora() -> bool {
    false
}

/// Layer helpers from unsloth-rs (RMSNorm / SwiGLU). Not LoRA kernels.
#[cfg(feature = "unsloth")]
pub use unsloth_rs::kernels::{RmsNorm, SwiGLU};

#[cfg(test)]
mod tests {
    #[test]
    fn does_not_dispatch_lora() {
        assert!(!super::should_dispatch_unsloth_lora());
        assert_eq!(super::unsloth_feature_enabled(), cfg!(feature = "unsloth"));
    }
}
