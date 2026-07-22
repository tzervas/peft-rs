//! Core traits for PEFT adapters.

use candle_core::Tensor;
use candle_nn::VarMap;

use crate::Result;

/// Configuration trait for adapter hyperparameters.
pub trait AdapterConfig: Clone + Send + Sync {
    /// Validate the configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    fn validate(&self) -> Result<()>;
}

/// Core adapter trait for parameter-efficient fine-tuning.
pub trait Adapter: Send + Sync {
    /// The configuration type for this adapter.
    type Config: AdapterConfig;

    /// Forward pass applying the adapter transformation.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `base_output` - Optional output from the base layer (for residual adapters)
    ///
    /// # Returns
    /// Transformed tensor
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    fn forward(&self, input: &Tensor, base_output: Option<&Tensor>) -> Result<Tensor>;

    /// Get the number of trainable parameters.
    #[must_use]
    fn num_parameters(&self) -> usize;

    /// Get the adapter's configuration.
    fn config(&self) -> &Self::Config;
}

/// Trait for adapters that can be merged into base weights.
pub trait Mergeable: Adapter {
    /// Merge adapter weights into base model weights.
    ///
    /// # Arguments
    /// * `base_weight` - The original weight tensor to merge into
    ///
    /// # Returns
    /// New tensor with adapter weights merged
    ///
    /// # Errors
    ///
    /// Returns an error if merging fails.
    fn merge(&self, base_weight: &Tensor) -> Result<Tensor>;

    /// Unmerge adapter weights from merged weights.
    ///
    /// # Arguments
    /// * `merged_weight` - Weight tensor with adapter already merged
    ///
    /// # Returns
    /// Original base weight tensor
    ///
    /// # Errors
    ///
    /// Returns an error if unmerging fails.
    fn unmerge(&self, merged_weight: &Tensor) -> Result<Tensor>;
}

/// Trait for trainable adapters.
///
/// # Freeze honesty
///
/// Implementations set a **layer-level frozen flag**. That flag is used to gate
/// training-only behavior (e.g. LoRA dropout). It does **not** reliably detach
/// Candle `Var`s from autograd: constructors that materialize plain `Tensor`
/// weights (`new_with_zeros`) have no grad membership to clear, and
/// `VarBuilder`/`VarMap` paths still require the optimizer to honor
/// [`Trainable::is_frozen`]. Full grad detach is deferred to a later PR.
pub trait Trainable: Adapter {
    /// Register trainable parameters with the variable map.
    ///
    /// Many adapters treat this as a no-op when weights were already created via
    /// `VarBuilder` during construction.
    ///
    /// # Errors
    ///
    /// Returns an error if parameter registration fails.
    fn register_parameters(&self, var_map: &mut VarMap, prefix: &str) -> Result<()>;

    /// Mark the adapter frozen (training-off / inference-oriented).
    ///
    /// See trait-level freeze honesty: this is a flag, not a full grad detach.
    fn freeze(&mut self);

    /// Mark the adapter unfrozen (training-on).
    fn unfreeze(&mut self);

    /// Check if the adapter is frozen.
    #[must_use]
    fn is_frozen(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trait object safety check
    fn _assert_adapter_object_safe(_: &dyn Adapter<Config = crate::LoraConfig>) {}
}
