//! # peft-rs
//!
//! Comprehensive PEFT (Parameter-Efficient Fine-Tuning) adapter library for Rust.
//!
//! This crate provides modular implementations of various PEFT methods:
//! - **LoRA** (Low-Rank Adaptation)
//! - **Prefix Tuning**
//! - **Prompt Tuning**
//! - **IA³** (Infused Adapter by Inhibiting and Amplifying Inner Activations)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use peft_rs::{LoraConfig, LoraLayer};
//! use candle_core::{Device, Tensor};
//!
//! // Create a LoRA layer
//! let config = LoraConfig {
//!     r: 8,
//!     alpha: 16,
//!     dropout: 0.0,
//! };
//! let layer = LoraLayer::new(768, 768, config, &Device::Cpu)?;
//!
//! // Apply to input
//! let input = Tensor::zeros(&[1, 10, 768], candle_core::DType::F32, &Device::Cpu)?;
//! let output = layer.forward(&input)?;
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
pub mod traits;

pub use adapters::adalora::{AdaLoraConfig, AdaLoraLayer};
pub use adapters::ia3::{Ia3Config, Ia3Layer};
pub use adapters::lora::{LoraConfig, LoraLayer};
pub use adapters::prefix_tuning::{PrefixTuningConfig, PrefixTuningLayer};
pub use adapters::prompt_tuning::{PromptTuningConfig, PromptTuningLayer};
pub use error::{PeftError, Result};
pub use traits::{Adapter, AdapterConfig, Mergeable, Trainable};
