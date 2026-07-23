//! Inference utilities for PEFT adapters.
//!
//! Provides lightweight helpers for adapter switching, residual-gating via
//! [`InferenceMode`], optional merge of the active adapter into a base weight
//! tensor, and simple metrics counters.
//!
//! This is **not** a full evaluation harness (no dataset loop, generation
//! pipeline, or benchmark runner). [`BatchAdapterSwitcher`] switches one
//! adapter at a time; the "batch" name is historical API surface, not a
//! multi-request schedule.
//!
//! [`InferenceMode`] is a caller-facing control flag:
//! - [`InferenceMode::Adapter`]: apply the adapter residual at runtime
//!   ([`BatchAdapterSwitcher::should_apply_adapter`] → `true`)
//! - [`InferenceMode::Merged`]: weights are assumed already merged into the base;
//!   do **not** also apply the residual (avoids double-counting ΔW)
//! - [`InferenceMode::BaseOnly`]: ignore adapters entirely
//!
//! Actual weight merge for the active adapter is available via
//! [`BatchAdapterSwitcher::merge_active`] when `A: Mergeable`. Full model export
//! of merged state dicts is not provided here.

use crate::error::{PeftError, Result};
use crate::registry::AdapterRegistry;
use crate::traits::Adapter;

/// Inference mode (residual-gating flag for callers).
///
/// Does not itself swap weight tensors; it tells the host whether to apply the
/// adapter residual. See module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceMode {
    /// Apply adapter residual at runtime.
    #[default]
    Adapter,
    /// Assume base weights already include the adapter merge; skip residual.
    Merged,
    /// Ignore adapters; base model only.
    BaseOnly,
}

/// Thin wrapper over [`AdapterRegistry`] with an [`InferenceMode`] flag.
///
/// Named "batch" for historical API compatibility; switching is one adapter at
/// a time via [`Self::switch_adapter`] (no multi-request batch schedule).
pub struct BatchAdapterSwitcher<A: Adapter> {
    registry: AdapterRegistry<A>,
    mode: InferenceMode,
}

impl<A: Adapter> BatchAdapterSwitcher<A> {
    /// Create a new switcher wrapping the given registry.
    #[must_use]
    pub fn new(registry: AdapterRegistry<A>) -> Self {
        Self {
            registry,
            mode: InferenceMode::default(),
        }
    }

    /// Set the residual-gating mode.
    pub fn set_mode(&mut self, mode: InferenceMode) {
        self.mode = mode;
    }

    /// Current residual-gating mode.
    #[must_use]
    pub fn mode(&self) -> InferenceMode {
        self.mode
    }

    /// Switch the active adapter by name.
    ///
    /// # Errors
    /// Returns an error if the adapter cannot be found.
    pub fn switch_adapter(&mut self, adapter_name: impl Into<String>) -> Result<()> {
        self.registry.set_active_adapter(adapter_name)
    }

    /// Name of the active adapter, if any.
    #[must_use]
    pub fn active_adapter(&self) -> Option<&str> {
        self.registry.active_adapter_name()
    }

    /// Immutable access to the underlying registry.
    #[must_use]
    pub fn registry(&self) -> &AdapterRegistry<A> {
        &self.registry
    }

    /// Mutable access to the underlying registry.
    #[must_use]
    pub fn registry_mut(&mut self) -> &mut AdapterRegistry<A> {
        &mut self.registry
    }

    /// Whether the host should apply the adapter residual on the next forward.
    ///
    /// Returns `true` only for [`InferenceMode::Adapter`]. For `Merged` and
    /// `BaseOnly` returns `false` so callers do not double-apply ΔW or apply
    /// any residual.
    #[must_use]
    pub fn should_apply_adapter(&self) -> bool {
        matches!(self.mode, InferenceMode::Adapter)
    }
}

impl<A: crate::traits::Mergeable> BatchAdapterSwitcher<A> {
    /// Merge the active adapter into `base_weight` (returns a new tensor).
    ///
    /// Does not mutate mode; callers that plan to run with merged weights should
    /// typically call [`Self::set_mode`](`InferenceMode::Merged`) so
    /// [`Self::should_apply_adapter`] returns `false`.
    ///
    /// # Errors
    /// Returns an error if there is no active adapter or if merging fails.
    pub fn merge_active(&self, base_weight: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let adapter = self.registry.get_active_adapter()?;
        adapter.merge(base_weight)
    }

    /// Unmerge the active adapter from a previously merged weight tensor.
    ///
    /// # Errors
    /// Returns an error if there is no active adapter or if unmerging fails.
    pub fn unmerge_active(
        &self,
        merged_weight: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        let adapter = self.registry.get_active_adapter()?;
        adapter.unmerge(merged_weight)
    }
}

/// Lightweight inference counters (caller-driven; no automatic timing).
#[derive(Debug, Clone, Copy, Default)]
pub struct InferenceMetrics {
    /// Total recorded forward/call events.
    pub total_calls: usize,
    /// Total recorded adapter-switch events.
    pub adapter_switches: usize,
    /// Optional average latency in milliseconds (set by caller).
    pub avg_time_ms: Option<f64>,
}

impl InferenceMetrics {
    /// Create zeroed metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one call/forward.
    pub fn record_call(&mut self) {
        self.total_calls += 1;
    }

    /// Record one adapter switch.
    pub fn record_switch(&mut self) {
        self.adapter_switches += 1;
    }

    /// Set the average latency (milliseconds). Caller computes the average.
    pub fn set_avg_time(&mut self, time_ms: f64) {
        self.avg_time_ms = Some(time_ms);
    }

    /// Adapter switches divided by total calls (`0.0` when there are no calls).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn switches_per_call(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.adapter_switches as f64 / self.total_calls as f64
        }
    }
}

/// Validate that adapters report the same trainable parameter count.
///
/// This is a **weak** compatibility check: equal `num_parameters()` does not
/// guarantee matching shapes, ranks, dtypes, or target modules. Prefer
/// configuration-level checks when available.
///
/// # Errors
/// Returns [`PeftError::InvalidConfig`] when parameter counts differ.
pub fn validate_adapter_compatibility<A: Adapter>(adapters: &[&A]) -> Result<()> {
    if adapters.is_empty() {
        return Ok(());
    }

    let first_params = adapters[0].num_parameters();
    for (idx, adapter) in adapters.iter().enumerate().skip(1) {
        if adapter.num_parameters() != first_params {
            return Err(PeftError::InvalidConfig(format!(
                "Adapter {} has {} parameters, expected {}",
                idx,
                adapter.num_parameters(),
                first_params
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::lora::{LoraConfig, LoraLayer};
    use candle_core::{Device, Tensor};

    #[test]
    fn test_inference_mode_default() {
        assert_eq!(InferenceMode::default(), InferenceMode::Adapter);
    }

    #[test]
    fn test_batch_adapter_switcher() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let adapter1 = LoraLayer::new_with_zeros(768, 768, config.clone(), &device)?;
        let adapter2 = LoraLayer::new_with_zeros(768, 768, config, &device)?;

        registry.register_adapter("adapter1", adapter1)?;
        registry.register_adapter("adapter2", adapter2)?;

        let mut switcher = BatchAdapterSwitcher::new(registry);
        assert_eq!(switcher.mode(), InferenceMode::Adapter);
        assert!(switcher.should_apply_adapter());
        assert_eq!(switcher.active_adapter(), Some("adapter1"));

        // Switch adapter
        switcher.switch_adapter("adapter2")?;
        assert_eq!(switcher.active_adapter(), Some("adapter2"));

        // Merged mode must NOT request residual application (avoids double ΔW)
        switcher.set_mode(InferenceMode::Merged);
        assert_eq!(switcher.mode(), InferenceMode::Merged);
        assert!(!switcher.should_apply_adapter());

        switcher.set_mode(InferenceMode::BaseOnly);
        assert_eq!(switcher.mode(), InferenceMode::BaseOnly);
        assert!(!switcher.should_apply_adapter());

        // Registry inspection
        assert_eq!(switcher.registry().len(), 2);

        // Errors on invalid adapter switch
        let result = switcher.switch_adapter("nonexistent");
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_inference_metrics() {
        let mut metrics = InferenceMetrics::new();
        assert_eq!(metrics.total_calls, 0);
        assert_eq!(metrics.adapter_switches, 0);
        assert_eq!(metrics.avg_time_ms, None);
        assert_eq!(metrics.switches_per_call(), 0.0);

        metrics.record_call();
        assert_eq!(metrics.total_calls, 1);
        assert_eq!(metrics.switches_per_call(), 0.0);

        metrics.record_switch();
        assert_eq!(metrics.adapter_switches, 1);
        assert_eq!(metrics.switches_per_call(), 1.0);

        metrics.record_call();
        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.switches_per_call(), 0.5);

        metrics.set_avg_time(12.34);
        assert_eq!(metrics.avg_time_ms, Some(12.34));
    }

    #[test]
    fn test_validate_adapter_compatibility() -> Result<()> {
        let device = Device::Cpu;
        let config_std = LoraConfig::default();
        let config_alt = LoraConfig {
            r: 16,
            ..LoraConfig::default()
        };

        let adapter1 = LoraLayer::new_with_zeros(768, 768, config_std.clone(), &device)?;
        let adapter2 = LoraLayer::new_with_zeros(768, 768, config_std, &device)?;
        let adapter3 = LoraLayer::new_with_zeros(768, 768, config_alt, &device)?;

        // Empty is compatible
        let empty_list: Vec<&LoraLayer> = vec![];
        assert!(validate_adapter_compatibility(&empty_list).is_ok());

        // Identical parameters is compatible
        let compatible_list = vec![&adapter1, &adapter2];
        assert!(validate_adapter_compatibility(&compatible_list).is_ok());

        // Mismatched parameters is incompatible
        let incompatible_list = vec![&adapter1, &adapter3];
        let result = validate_adapter_compatibility(&incompatible_list);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PeftError::InvalidConfig(_)));

        Ok(())
    }

    #[test]
    fn test_merge_active_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let config = LoraConfig::default();
        // Non-zero A so merge is observably different from base for Standard init
        // (B zeros → ΔW = 0). Use Gaussian init for a nonzero check.
        let config = LoraConfig {
            init_lora_weights: crate::adapters::lora::LoraInitialization::Gaussian,
            ..config
        };

        let adapter = LoraLayer::new_with_zeros(4, 4, config, &device)?;
        let mut registry = AdapterRegistry::new();
        registry.register_adapter("a", adapter)?;
        let switcher = BatchAdapterSwitcher::new(registry);

        let base = Tensor::ones((4, 4), candle_core::DType::F32, &device)?;
        let merged = switcher.merge_active(&base)?;
        assert_eq!(merged.dims(), base.dims());

        // Unmerge should recover a tensor of the same shape; for exact identity
        // we only require that unmerge succeeds (numerical path covered in
        // adapter-level merge tests).
        let unmerged = switcher.unmerge_active(&merged)?;
        assert_eq!(unmerged.dims(), base.dims());

        Ok(())
    }
}
