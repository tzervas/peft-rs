//! Quantization bridge traits for `QLoRA` / low-bit base layers.
//!
//! # Purpose (PEFT-P1-04)
//!
//! peft-rs owns **adapter residual math**. Quantized *base* linears live in
//! companion crates (notably [`qlora-rs`](https://crates.io/crates/qlora-rs)).
//! This module defines a thin, dependency-free trait surface so those backends
//! can plug quantized bases into PEFT-style residual forwards without peft-rs
//! depending on any quantizer.
//!
//! # Backend path (documented, impl lives in qlora)
//!
//! `qlora_rs::QuantizedLinear` should implement:
//! - [`QuantizedBaseLinear`] — dequant-or-cache weight + base matmul
//! - optionally [`QuantizedAdapterLayer`] — full `base(x) + lora(x)`
//!
//! Until that impl lands in qlora, callers can wrap any dequant + matmul via
//! [`forward_quantized_with_adapter`] with a free-function adapter residual.
//!
//! ```text
//! y = QuantizedBaseLinear::forward_base(x) + Adapter::forward(x, None)
//! ```
//!
//! # Non-goals
//!
//! - No NF4/FP4 codecs inside peft-rs
//! - No bitsandbytes / GPTQ / AWQ loaders
//! - No automatic model-wide `QLoRA` conversion

use candle_core::Tensor;
use candle_nn::{Linear, Module};

use crate::error::Result;
use crate::traits::Adapter;

/// Frozen (or inference) quantized base linear interface.
///
/// Implementors dequantize (or use a cached dequant) and run the base matmul.
/// Trainable PEFT adapters are **not** part of this trait — compose them via
/// [`forward_quantized_with_adapter`] or [`QuantizedAdapterLayer`].
pub trait QuantizedBaseLinear: Send + Sync {
    /// Input feature dimension.
    fn in_features(&self) -> usize;

    /// Output feature dimension.
    fn out_features(&self) -> usize;

    /// Materialize (or return cached) dequantized weight `[out, in]`.
    ///
    /// # Errors
    /// Returns an error if dequantization fails.
    fn dequantized_weight(&self) -> Result<Tensor>;

    /// Base forward: typically `x @ W_dequant.T (+ bias)`.
    ///
    /// # Errors
    /// Returns an error if the matmul / dequant fails.
    fn forward_base(&self, input: &Tensor) -> Result<Tensor>;
}

/// Combined quantized base + PEFT residual (QLoRA-style layer).
///
/// `qlora-rs` `QuantizedLinear` is the intended implementor: quantized frozen
/// base + trainable `LoraLayer`.
pub trait QuantizedAdapterLayer: Send + Sync {
    /// Full layer forward: base (quantized) + adapter residual.
    ///
    /// # Errors
    /// Propagates base or adapter errors.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Number of **trainable adapter** parameters (not quantized base storage).
    fn num_adapter_parameters(&self) -> usize;

    /// Input features.
    fn in_features(&self) -> usize;

    /// Output features.
    fn out_features(&self) -> usize;
}

/// Compose a quantized base with any [`Adapter`] residual.
///
/// ```text
/// y = base.forward_base(x) + adapter.forward(x, None)
/// ```
///
/// When `adapter` is a zero-init `LoRA` residual this matches the `QLoRA` product
/// path at the math level (scale lives inside the adapter).
///
/// # Errors
/// Propagates base or adapter forward errors.
pub fn forward_quantized_with_adapter<B, A>(base: &B, adapter: &A, input: &Tensor) -> Result<Tensor>
where
    B: QuantizedBaseLinear + ?Sized,
    A: Adapter,
{
    let base_out = base.forward_base(input)?;
    adapter.forward(input, Some(&base_out))
}

/// Reference / test double: full-precision base linear stored as a dense weight.
///
/// Demonstrates the bridge without a real quantizer. Production `QLoRA` should
/// implement [`QuantizedBaseLinear`] on the quantized type instead.
pub struct DenseBaseLinear {
    /// Dense linear (identity "quant" path).
    linear: Linear,
    in_features: usize,
    out_features: usize,
}

impl DenseBaseLinear {
    /// Wrap a dense weight as a [`QuantizedBaseLinear`] (identity quant path).
    ///
    /// # Errors
    /// Returns an error if weight is not rank-2.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(crate::error::PeftError::InvalidConfig(
                "DenseBaseLinear weight must be rank-2 [out, in]".into(),
            ));
        }
        Ok(Self {
            in_features: dims[1],
            out_features: dims[0],
            linear: Linear::new(weight, bias),
        })
    }
}

impl QuantizedBaseLinear for DenseBaseLinear {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn dequantized_weight(&self) -> Result<Tensor> {
        Ok(self.linear.weight().clone())
    }

    fn forward_base(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.linear.forward(input)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoraConfig, LoraLayer};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn dense_base_plus_lora_residual() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::eye(4, DType::F32, &device)?;
        let base = DenseBaseLinear::new(weight, None)?;
        let cfg = LoraConfig {
            r: 2,
            alpha: 4,
            ..Default::default()
        };
        // Non-zero residual so composition is observable
        let lora_a = Tensor::ones((2, 4), DType::F32, &device)?;
        let lora_b = Tensor::ones((4, 2), DType::F32, &device)?;
        let lora = LoraLayer::from_weights(lora_a, lora_b, cfg)?;

        let input = Tensor::ones((1, 1, 4), DType::F32, &device)?;
        let output = forward_quantized_with_adapter(&base, &lora, &input)?;
        let base_only = base.forward_base(&input)?;
        let diff = output
            .sub(&base_only)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-3,
            "expected non-zero adapter residual, got {diff}"
        );
        assert_eq!(output.dims(), &[1, 1, 4]);
        assert_eq!(base.in_features(), 4);
        assert_eq!(base.out_features(), 4);
        Ok(())
    }

    #[test]
    fn dequantized_weight_shape() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::zeros((8, 16), DType::F32, &device)?;
        let base = DenseBaseLinear::new(w, None)?;
        assert_eq!(base.dequantized_weight()?.dims(), &[8, 16]);
        Ok(())
    }
}
