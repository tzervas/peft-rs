//! Inference utilities for PEFT adapters.

use crate::error::{PeftError, Result};
use crate::registry::AdapterRegistry;
use crate::traits::Adapter;

/// Inference mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceMode {
    /// Use adapter
    #[default]
    Adapter,
    /// Use merged weights
    Merged,
    /// Base model only
    BaseOnly,
}

/// Batch adapter switcher.
pub struct BatchAdapterSwitcher<A: Adapter> {
    registry: AdapterRegistry<A>,
    mode: InferenceMode,
}

impl<A: Adapter> BatchAdapterSwitcher<A> {
    /// Create new switcher.
    #[must_use]
    pub fn new(registry: AdapterRegistry<A>) -> Self {
        Self {
            registry,
            mode: InferenceMode::default(),
        }
    }

    /// Set mode.
    pub fn set_mode(&mut self, mode: InferenceMode) {
        self.mode = mode;
    }

    /// Get mode.
    #[must_use]
    pub fn mode(&self) -> InferenceMode {
        self.mode
    }

    /// Switch adapter.
    ///
    /// # Errors
    /// Returns an error if the adapter cannot be found.
    pub fn switch_adapter(&mut self, adapter_name: impl Into<String>) -> Result<()> {
        self.registry.set_active_adapter(adapter_name)
    }

    /// Get active adapter.
    #[must_use]
    pub fn active_adapter(&self) -> Option<&str> {
        self.registry.active_adapter_name()
    }

    /// Get registry.
    #[must_use]
    pub fn registry(&self) -> &AdapterRegistry<A> {
        &self.registry
    }

    /// Get mutable registry.
    #[must_use]
    pub fn registry_mut(&mut self) -> &mut AdapterRegistry<A> {
        &mut self.registry
    }

    /// Should apply adapter.
    #[must_use]
    pub fn should_apply_adapter(&self) -> bool {
        !matches!(self.mode, InferenceMode::BaseOnly)
    }
}

impl<A: crate::traits::Mergeable> BatchAdapterSwitcher<A> {
    /// Merge active adapter weights with base weights.
    ///
    /// # Errors
    /// Returns an error if merging fails.
    pub fn merge_active(&self, base_weight: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let adapter = self.registry.get_active_adapter()?;
        adapter.merge(base_weight)
    }

    /// Unmerge active adapter weights from merged weights.
    ///
    /// # Errors
    /// Returns an error if unmerging fails.
    pub fn unmerge_active(
        &self,
        merged_weight: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        let adapter = self.registry.get_active_adapter()?;
        adapter.unmerge(merged_weight)
    }
}

/// Save a merged model to a safetensors file.
///
/// This is an export utility that merges adapter weights from a `PeftModel`
/// into the provided base weights map, and saves the resulting merged model state dict.
///
/// # Errors
/// Returns an error if merging or saving fails.
#[allow(clippy::implicit_hasher)]
pub fn save_merged_model<A: crate::traits::Mergeable, P: AsRef<std::path::Path>>(
    peft_model: &crate::model::PeftModel<A>,
    base_weights: &std::collections::HashMap<String, candle_core::Tensor>,
    path: P,
) -> Result<()> {
    let merged_weights = peft_model.merge_weights(base_weights)?;
    candle_core::safetensors::save(&merged_weights, path.as_ref())
        .map_err(|e| PeftError::Io(format!("Failed to save merged model: {e}")))?;
    Ok(())
}

/// Inference metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct InferenceMetrics {
    /// Total calls
    pub total_calls: usize,
    /// Adapter switches
    pub adapter_switches: usize,
    /// Average time
    pub avg_time_ms: Option<f64>,
}

impl InferenceMetrics {
    /// Create new metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record call.
    pub fn record_call(&mut self) {
        self.total_calls += 1;
    }

    /// Record switch.
    pub fn record_switch(&mut self) {
        self.adapter_switches += 1;
    }

    /// Set average time.
    pub fn set_avg_time(&mut self, time_ms: f64) {
        self.avg_time_ms = Some(time_ms);
    }

    /// Get switches per call.
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

/// Validate adapter compatibility.
///
/// # Errors
/// Returns an error if compatibility check fails.
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
    use crate::{LoraConfig, LoraLayer, PeftModel};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    #[test]
    fn test_inference_mode() {
        let mode = InferenceMode::default();
        assert_eq!(mode, InferenceMode::Adapter);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_inference_metrics() {
        let mut metrics = InferenceMetrics::new();
        assert_eq!(metrics.total_calls, 0);
        assert_eq!(metrics.adapter_switches, 0);
        assert_eq!(metrics.switches_per_call(), 0.0);

        metrics.record_call();
        metrics.record_call();
        metrics.record_switch();
        metrics.set_avg_time(1.5);

        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.adapter_switches, 1);
        assert_eq!(metrics.switches_per_call(), 0.5);
        assert_eq!(metrics.avg_time_ms, Some(1.5));
    }

    #[test]
    fn test_batch_adapter_switcher() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut registry = AdapterRegistry::new();
        let config = LoraConfig::default();
        let adapter1 = LoraLayer::new_with_zeros(768, 768, config.clone(), &device)?;
        let adapter2 = LoraLayer::new_with_zeros(768, 768, config, &device)?;

        registry.register_adapter("task1", adapter1)?;
        registry.register_adapter("task2", adapter2)?;

        let mut switcher = BatchAdapterSwitcher::new(registry);
        assert_eq!(switcher.mode(), InferenceMode::Adapter);
        assert_eq!(switcher.active_adapter(), Some("task1"));
        assert!(switcher.should_apply_adapter());

        switcher.set_mode(InferenceMode::BaseOnly);
        assert_eq!(switcher.mode(), InferenceMode::BaseOnly);
        assert!(!switcher.should_apply_adapter());

        switcher.switch_adapter("task2")?;
        assert_eq!(switcher.active_adapter(), Some("task2"));

        assert_eq!(switcher.registry().len(), 2);
        assert_eq!(switcher.registry_mut().len(), 2);

        Ok(())
    }

    #[test]
    fn test_validate_adapter_compatibility() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = LoraConfig::default();
        let adapter1 = LoraLayer::new_with_zeros(768, 768, config.clone(), &device)?;
        let adapter2 = LoraLayer::new_with_zeros(768, 768, config, &device)?;
        let adapter3 = LoraLayer::new_with_zeros(
            512,
            512,
            LoraConfig {
                r: 4,
                ..Default::default()
            },
            &device,
        )?;

        // Empty is compatible
        let empty_adapters: Vec<&LoraLayer> = vec![];
        assert!(validate_adapter_compatibility(&empty_adapters).is_ok());

        // Identical shapes compatible
        let compatible = vec![&adapter1, &adapter2];
        assert!(validate_adapter_compatibility(&compatible).is_ok());

        // Different shapes incompatible
        let incompatible = vec![&adapter1, &adapter3];
        let result = validate_adapter_compatibility(&incompatible);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PeftError::InvalidConfig(_)));

        Ok(())
    }

    #[test]
    fn test_switcher_merging() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut registry = AdapterRegistry::new();
        let config = LoraConfig::default();
        let adapter = LoraLayer::new_with_zeros(768, 768, config, &device)?;
        registry.register_adapter("task1", adapter)?;

        let switcher = BatchAdapterSwitcher::new(registry);
        let base_weight = Tensor::zeros(&[768, 768], candle_core::DType::F32, &device)?;

        let merged = switcher.merge_active(&base_weight)?;
        assert_eq!(merged.dims(), &[768, 768]);

        let unmerged = switcher.unmerge_active(&merged)?;
        assert_eq!(unmerged.dims(), &[768, 768]);

        Ok(())
    }

    #[test]
    fn test_peft_model_merging() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let config = LoraConfig::default();

        model.add_adapter("task1", "*", &["layer.0", "layer.1"], |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        let mut base_weights = HashMap::new();
        base_weights.insert(
            "layer.0.weight".to_string(),
            Tensor::zeros(&[768, 768], candle_core::DType::F32, &device)?,
        );
        base_weights.insert(
            "layer.1.weight".to_string(),
            Tensor::zeros(&[768, 768], candle_core::DType::F32, &device)?,
        );
        base_weights.insert(
            "unrelated.weight".to_string(),
            Tensor::zeros(&[100], candle_core::DType::F32, &device)?,
        );

        let merged_weights = model.merge_weights(&base_weights)?;
        assert_eq!(merged_weights.len(), 3);
        assert_eq!(merged_weights["layer.0.weight"].dims(), &[768, 768]);
        assert_eq!(merged_weights["unrelated.weight"].dims(), &[100]);

        let unmerged_weights = model.unmerge_weights(&merged_weights)?;
        assert_eq!(unmerged_weights.len(), 3);
        assert_eq!(unmerged_weights["layer.0.weight"].dims(), &[768, 768]);

        Ok(())
    }

    #[test]
    fn test_save_merged_model_and_export() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let config = LoraConfig::default();

        model.add_adapter("task1", "*", &["layer.0"], |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        let mut base_weights = HashMap::new();
        base_weights.insert(
            "layer.0.weight".to_string(),
            Tensor::zeros(&[768, 768], candle_core::DType::F32, &device)?,
        );

        let temp_dir = tempfile::TempDir::new()?;
        let path = temp_dir.path().join("merged.safetensors");

        save_merged_model(&model, &base_weights, &path)?;
        assert!(path.exists());

        // Load and check
        let loaded = candle_core::safetensors::load(&path, &device)?;
        assert!(loaded.contains_key("layer.0.weight"));
        assert_eq!(loaded["layer.0.weight"].dims(), &[768, 768]);

        Ok(())
    }
}
