//! # peft-rs
//!
//! Candle PEFT **adapter layer library** for Rust — modular parameter-efficient
//! fine-tuning *layers* (forward / merge / save), not a drop-in HuggingFace PEFT
//! framework.
//!
//! Implemented layer surfaces (depth varies; see README status matrix):
//! - **`LoRA`** (Low-Rank Adaptation)
//! - **`DoRA`** (Weight-Decomposed Low-Rank Adaptation)
//! - **`AdaLoRA`** (Adaptive Low-Rank Adaptation)
//! - **IA³** (Infused Adapter by Inhibiting and Amplifying Inner Activations)
//! - **`LoHa`** (Low-Rank Hadamard Product)
//! - **`LoKr`** (Low-Rank Kronecker Product)
//! - **OFT** (Orthogonal Fine-Tuning)
//! - **BOFT** (Butterfly Orthogonal Fine-Tuning)
//! - **`VeRA`** (Vector-based Random Matrix Adaptation)
//! - **Prefix Tuning** / **Prompt Tuning** (thin embeddings helpers)
//!
//! ## Non-goals (crate docs)
//!
//! Full base-model injection, HF checkpoint key parity, QLoRA, trainer loops, and
//! fused CUDA kernels are **not** provided as complete features in this release.
//! Historical CubeCL sources are quarantined under `src/kernels/archive/` and are
//! **not** exported from this crate.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use peft_rs::{Adapter, LoraConfig, LoraLayer};
//! use candle_core::{Device, Tensor, DType};
//!
//! let config = LoraConfig {
//!     r: 8,
//!     alpha: 16,
//!     dropout: 0.0,
//!     ..Default::default()
//! };
//! let layer = LoraLayer::new_with_zeros(768, 768, config, &Device::Cpu)?;
//! let input = Tensor::zeros(&[1, 10, 768], DType::F32, &Device::Cpu)?;
//! let output = layer.forward(&input, None)?;
//! ```
//!
//! ## Architecture
//!
//! All adapters implement the [`Adapter`] trait, which provides a common interface
//! for forward passes and weight merging.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod adapters;
pub mod config;
pub mod error;
pub mod io;
pub mod model;
pub mod registry;
pub mod training;
pub mod traits;

// NOTE (PR-021): `kernels` is intentionally NOT a public module.
// Fused CubeCL sources are quarantined under `src/kernels/archive/` and are not
// part of the compile graph. The `cuda` feature enables candle-core CUDA only.

pub use adapters::adalora::{AdaLoraConfig, AdaLoraLayer};
pub use adapters::boft::{BoftConfig, BoftLayer};
pub use adapters::ia3::{Ia3Config, Ia3Layer};
pub use adapters::loha::{LoHaConfig, LoHaLayer};
pub use adapters::lokr::{LoKrConfig, LoKrLayer};
pub use adapters::lora::{DoraLayer, LoraConfig, LoraLayer};
pub use adapters::oft::{OftConfig, OftLayer};
pub use adapters::prefix_tuning::{PrefixTuningConfig, PrefixTuningLayer};
pub use adapters::prompt_tuning::{PromptTuningConfig, PromptTuningLayer};
pub use adapters::vera::{VeraConfig, VeraLayer};
pub use error::{PeftError, Result};
pub use io::{
    load_adapter_config, load_adapter_weights, load_pretrained, save_adapter_config,
    save_adapter_weights, save_pretrained, SaveLoad, ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
};
pub use model::{get_peft_model, ModulePattern, PeftModel};
pub use registry::AdapterRegistry;
pub use training::{
    count_trainable_parameters, format_parameter_count, AdapterTrainingConfig,
    AdapterTrainingState, LrSchedule,
};
pub use traits::{Adapter, AdapterConfig, Mergeable, Trainable};
