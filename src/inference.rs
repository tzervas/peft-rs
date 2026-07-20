//! Inference utilities for PEFT adapters.

use crate::error::{PeftError, Result};
use crate::registry::AdapterRegistry;
use crate::traits::Adapter;

/// Inference mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    /// Use adapter
    Adapter,
    /// Use merged weights
    Merged,
    /// Base model only
    BaseOnly,
}

impl Default for InferenceMode {
    fn default() -> Self {
        Self::Adapter
    }
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

    /// Should apply adapter.
    #[must_use]
    pub fn should_apply_adapter(&self) -> bool {
        !matches!(self.mode, InferenceMode::BaseOnly)
    }
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
    pub fn switches_per_call(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.adapter_switches as f64 / self.total_calls as f64
        }
    }
}

/// Validate adapter compatibility.
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
    use candle_core::Device;

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

        // Change mode
        switcher.set_mode(InferenceMode::Merged);
        assert_eq!(switcher.mode(), InferenceMode::Merged);
        assert!(switcher.should_apply_adapter());

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
}
