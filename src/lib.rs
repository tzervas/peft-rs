//! # peft-rs
//!
//! Candle PEFT **adapter layer library** for Rust — modular parameter-efficient
//! fine-tuning *layers* (forward / merge / save) plus a **Linear inject path**
//! for named modules. Not a drop-in `HuggingFace` PEFT framework for full models.
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
//! - **Prefix Tuning** / **Prompt Tuning** (**experimental** helpers + reparam / prepend)
//!
//! ## Product path (1.1+)
//!
//! - [`LinearWithLora`] / [`PeftLinearModel`] / [`get_peft_model`] — real base
//!   Linear + `LoRA` residual forward for caller-supplied modules
//! - [`hf`] — `HuggingFace` `adapter_config.json` + `lora_A` / `lora_B` key interop
//! - [`training::train_step_mse`] — minimal real `AdamW` train step on inject path
//! - [`AdapterRegistry`] weighted residual composition
//! - [`quant`] — `QLoRA` bridge traits (impl lives in qlora-rs)
//! - Parity fixtures under `tests/parity/` (`LoRA` forward/merge goldens)
//!
//! ## Non-goals (crate docs)
//!
//! Full transformers model zoo, full `QLoRA` codecs, full `PeftTrainer` loops, and
//! fused CUDA kernels are **not** provided as complete features. Historical
//! `CubeCL` sources are quarantined under `src/kernels/archive/` and are **not**
//! exported.
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
pub mod hf;
pub mod inference;
pub mod io;
pub mod model;
pub mod quant;
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
pub use adapters::prompt_tuning::{PromptInit, PromptTuningConfig, PromptTuningLayer};
pub use adapters::vera::{VeraConfig, VeraLayer};
pub use error::{PeftError, Result};
pub use hf::{
    extract_lora_ab, hf_state_dict_to_native, insert_module_lora_weights, load_pretrained_hf,
    native_state_dict_to_hf, pack_lora_state_dict, save_pretrained_hf, slice_module_state_dict,
    HfLoraConfig, LoraKeyStyle, DEFAULT_ADAPTER_NAME, PEFT_TYPE_LORA,
};
pub use inference::{
    validate_adapter_compatibility, BatchAdapterSwitcher, InferenceMetrics, InferenceMode,
};
pub use io::{
    load_adapter_config, load_adapter_weights, load_pretrained, save_adapter_config,
    save_adapter_weights, save_pretrained, SaveLoad, ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
};
pub use model::{
    get_peft_model, get_peft_model_registry, LinearWithLora, ModulePattern, PeftLinearModel,
    PeftModel,
};
pub use quant::{
    forward_quantized_with_adapter, DenseBaseLinear, QuantizedAdapterLayer, QuantizedBaseLinear,
};
pub use registry::{AdapterRegistry, AdapterWeight};
pub use training::{
    count_trainable_parameters, format_parameter_count, train_step_mse, train_step_with_loss,
    AdapterTrainingConfig, AdapterTrainingState, LrSchedule, TrainStepResult,
};
pub use traits::{Adapter, AdapterConfig, Mergeable, Trainable};
