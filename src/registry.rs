//! Multi-adapter registry for managing multiple PEFT adapters.
//!
//! This module provides functionality for:
//! - Registering multiple named adapters
//! - Switching between active adapters (single-active mode)
//! - **Weighted residual composition** of multiple adapters (PEFT-P1-01)
//! - Managing adapter lifecycle
//!
//! # Composition (PEFT-P1-01)
//!
//! When weighted composition is set via [`AdapterRegistry::set_weighted_adapters`],
//! [`AdapterRegistry::forward`] computes:
//!
//! ```text
//! y = base + Σ_i w_i * residual_i(x)
//! ```
//!
//! where each residual is `adapter.forward(x, None)`. Callers can also use
//! [`AdapterRegistry::forward_weighted`] for a one-shot composition without
//! changing registry state.

use std::collections::HashMap;

use candle_core::Tensor;

use crate::error::{PeftError, Result};
use crate::traits::Adapter;

/// A named weight for multi-adapter residual composition.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterWeight {
    /// Registered adapter name.
    pub name: String,
    /// Scalar multiplier applied to that adapter's residual.
    pub weight: f64,
}

impl AdapterWeight {
    /// Create a new adapter weight entry.
    #[must_use]
    pub fn new(name: impl Into<String>, weight: f64) -> Self {
        Self {
            name: name.into(),
            weight,
        }
    }
}

/// Registry for managing multiple named adapters.
///
/// Supports single-active switching and weighted multi-adapter residual composition.
pub struct AdapterRegistry<A: Adapter> {
    /// Map of adapter names to adapters
    adapters: HashMap<String, A>,
    /// Currently active adapter name (single-active mode)
    active_adapter: Option<String>,
    /// When `Some`, [`Self::forward`] uses weighted residual composition.
    composition: Option<Vec<AdapterWeight>>,
}

impl<A: Adapter> AdapterRegistry<A> {
    /// Create a new empty adapter registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
            composition: None,
        }
    }

    /// Register a new adapter with the given name.
    ///
    /// # Arguments
    /// * `name` - Unique name for the adapter
    /// * `adapter` - The adapter to register
    ///
    /// # Errors
    /// Returns an error if an adapter with this name already exists
    pub fn register_adapter(&mut self, name: impl Into<String>, adapter: A) -> Result<()> {
        let name = name.into();

        if self.adapters.contains_key(&name) {
            return Err(PeftError::AdapterExists { name });
        }

        self.adapters.insert(name.clone(), adapter);

        // Set as active if it's the first adapter
        if self.active_adapter.is_none() {
            self.active_adapter = Some(name);
        }

        Ok(())
    }

    /// Set the active adapter by name (single-active mode).
    ///
    /// Clears any weighted composition so subsequent [`Self::forward`] calls
    /// use only this adapter.
    ///
    /// # Arguments
    /// * `name` - Name of the adapter to activate
    ///
    /// # Errors
    /// Returns an error if no adapter with this name exists
    pub fn set_active_adapter(&mut self, name: impl Into<String>) -> Result<()> {
        let name = name.into();

        if !self.adapters.contains_key(&name) {
            return Err(PeftError::AdapterNotFound { name });
        }

        self.active_adapter = Some(name);
        self.composition = None;
        Ok(())
    }

    /// Enable weighted multi-adapter residual composition (PEFT-P1-01).
    ///
    /// Subsequent [`Self::forward`] calls compute
    /// `base + Σ w_i * residual_i(x)`. Weights may be any finite `f64`
    /// (including >1 or negative for difference-style mixes).
    ///
    /// # Errors
    /// Returns an error if any named adapter is missing, the list is empty,
    /// or a weight is non-finite.
    pub fn set_weighted_adapters(
        &mut self,
        weights: impl IntoIterator<Item = AdapterWeight>,
    ) -> Result<()> {
        let weights: Vec<AdapterWeight> = weights.into_iter().collect();
        if weights.is_empty() {
            return Err(PeftError::InvalidConfig(
                "weighted composition requires at least one adapter weight".into(),
            ));
        }
        for w in &weights {
            if !w.weight.is_finite() {
                return Err(PeftError::InvalidConfig(format!(
                    "adapter weight for '{}' must be finite, got {}",
                    w.name, w.weight
                )));
            }
            if !self.adapters.contains_key(&w.name) {
                return Err(PeftError::AdapterNotFound {
                    name: w.name.clone(),
                });
            }
        }
        // Keep first weighted name as "active" for introspection convenience.
        self.active_adapter = Some(weights[0].name.clone());
        self.composition = Some(weights);
        Ok(())
    }

    /// Clear weighted composition and return to single-active mode.
    ///
    /// Does not remove adapters. If an active adapter name remains, forward
    /// uses that single adapter.
    pub fn clear_composition(&mut self) {
        self.composition = None;
    }

    /// Whether weighted composition mode is active.
    #[must_use]
    pub fn is_weighted_composition(&self) -> bool {
        self.composition.is_some()
    }

    /// Current composition weights, if any.
    #[must_use]
    pub fn composition_weights(&self) -> Option<&[AdapterWeight]> {
        self.composition.as_deref()
    }

    /// Get a reference to the active adapter.
    ///
    /// # Errors
    /// Returns an error if no adapter is currently active
    pub fn get_active_adapter(&self) -> Result<&A> {
        let name = self
            .active_adapter
            .as_ref()
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: "no active adapter".to_string(),
            })?;

        self.adapters
            .get(name)
            .ok_or_else(|| PeftError::AdapterNotFound { name: name.clone() })
    }

    /// Get a mutable reference to the active adapter.
    ///
    /// # Errors
    /// Returns an error if no adapter is currently active
    pub fn get_active_adapter_mut(&mut self) -> Result<&mut A> {
        let name = self
            .active_adapter
            .as_ref()
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: "no active adapter".to_string(),
            })?
            .clone();

        self.adapters
            .get_mut(&name)
            .ok_or_else(|| PeftError::AdapterNotFound { name })
    }

    /// Get a reference to an adapter by name.
    ///
    /// # Arguments
    /// * `name` - Name of the adapter
    ///
    /// # Errors
    /// Returns an error if no adapter with this name exists
    pub fn get_adapter(&self, name: &str) -> Result<&A> {
        self.adapters
            .get(name)
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: name.to_string(),
            })
    }

    /// Get a mutable reference to an adapter by name.
    ///
    /// # Arguments
    /// * `name` - Name of the adapter
    ///
    /// # Errors
    /// Returns an error if no adapter with this name exists
    pub fn get_adapter_mut(&mut self, name: &str) -> Result<&mut A> {
        self.adapters
            .get_mut(name)
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: name.to_string(),
            })
    }

    /// Check if an adapter with the given name exists.
    #[must_use]
    pub fn contains_adapter(&self, name: &str) -> bool {
        self.adapters.contains_key(name)
    }

    /// Get the name of the currently active adapter.
    #[must_use]
    pub fn active_adapter_name(&self) -> Option<&str> {
        self.active_adapter.as_deref()
    }

    /// Get a list of all registered adapter names.
    #[must_use]
    pub fn adapter_names(&self) -> Vec<&str> {
        self.adapters.keys().map(String::as_str).collect()
    }

    /// Get the number of registered adapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Remove an adapter by name.
    ///
    /// # Arguments
    /// * `name` - Name of the adapter to remove
    ///
    /// # Returns
    /// The removed adapter, if it existed
    ///
    /// # Errors
    /// Returns an error if trying to remove the active adapter or one used in
    /// the current weighted composition.
    pub fn remove_adapter(&mut self, name: &str) -> Result<Option<A>> {
        // Don't allow removing the active adapter
        if self.active_adapter.as_deref() == Some(name) {
            return Err(PeftError::InvalidConfig(
                "Cannot remove the currently active adapter".to_string(),
            ));
        }
        if let Some(comp) = &self.composition {
            if comp.iter().any(|w| w.name == name) {
                return Err(PeftError::InvalidConfig(
                    "Cannot remove an adapter that is part of the active weighted composition"
                        .to_string(),
                ));
            }
        }

        Ok(self.adapters.remove(name))
    }

    /// Clear all adapters from the registry.
    pub fn clear(&mut self) {
        self.adapters.clear();
        self.active_adapter = None;
        self.composition = None;
    }

    /// Apply adapters to an input tensor.
    ///
    /// - **Weighted composition** (if set): `base + Σ w_i * residual_i(x)`
    /// - **Single active**: delegates to the active adapter
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `base_output` - Optional base layer output
    ///
    /// # Errors
    /// Returns an error if no active adapter / composition or forward fails
    pub fn forward(&self, input: &Tensor, base_output: Option<&Tensor>) -> Result<Tensor> {
        if let Some(weights) = &self.composition {
            self.compose_residuals(input, base_output, weights)
        } else {
            let adapter = self.get_active_adapter()?;
            adapter.forward(input, base_output)
        }
    }

    /// One-shot weighted residual composition without mutating registry mode.
    ///
    /// ```text
    /// y = base + Σ w_i * residual_i(x)
    /// ```
    ///
    /// # Errors
    /// Returns an error if any adapter is missing, weights are empty/non-finite,
    /// or a forward pass fails.
    pub fn forward_weighted(
        &self,
        input: &Tensor,
        base_output: Option<&Tensor>,
        weights: &[AdapterWeight],
    ) -> Result<Tensor> {
        if weights.is_empty() {
            return Err(PeftError::InvalidConfig(
                "forward_weighted requires at least one adapter weight".into(),
            ));
        }
        for w in weights {
            if !w.weight.is_finite() {
                return Err(PeftError::InvalidConfig(format!(
                    "adapter weight for '{}' must be finite, got {}",
                    w.name, w.weight
                )));
            }
            if !self.adapters.contains_key(&w.name) {
                return Err(PeftError::AdapterNotFound {
                    name: w.name.clone(),
                });
            }
        }
        self.compose_residuals(input, base_output, weights)
    }

    /// Core residual sum: `base + Σ w_i * adapter_i(x, None)`.
    fn compose_residuals(
        &self,
        input: &Tensor,
        base_output: Option<&Tensor>,
        weights: &[AdapterWeight],
    ) -> Result<Tensor> {
        let mut residual_sum: Option<Tensor> = None;
        for w in weights {
            let adapter = self
                .adapters
                .get(&w.name)
                .ok_or_else(|| PeftError::AdapterNotFound {
                    name: w.name.clone(),
                })?;
            let residual = adapter.forward(input, None)?;
            let scaled = if (w.weight - 1.0).abs() < f64::EPSILON {
                residual
            } else {
                #[allow(clippy::cast_possible_truncation)]
                let w_f32 = w.weight as f32;
                let scale = Tensor::new(w_f32, residual.device())?;
                residual.broadcast_mul(&scale)?
            };
            residual_sum = Some(match residual_sum {
                None => scaled,
                Some(acc) => acc.broadcast_add(&scaled)?,
            });
        }
        let residuals = residual_sum.ok_or_else(|| {
            PeftError::InvalidConfig("weighted composition produced no residuals".into())
        })?;
        match base_output {
            Some(base) => Ok(base.broadcast_add(&residuals)?),
            None => Ok(residuals),
        }
    }
}

impl<A: Adapter> Default for AdapterRegistry<A> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoraConfig, LoraLayer};
    use candle_core::{DType, Device, Tensor};

    fn make_lora(device: &Device, r: usize) -> Result<LoraLayer> {
        let config = LoraConfig {
            r,
            alpha: r * 2,
            dropout: 0.0,
            ..Default::default()
        };
        // new_with_zeros → zero residual; use from_weights for non-zero tests
        LoraLayer::new_with_zeros(4, 4, config, device)
    }

    fn make_nonzero_lora(device: &Device, r: usize, fill: f32) -> Result<LoraLayer> {
        let config = LoraConfig {
            r,
            alpha: r * 2,
            dropout: 0.0,
            ..Default::default()
        };
        let a = Tensor::full(fill, (r, 4), device)?;
        let b = Tensor::full(fill, (4, r), device)?;
        LoraLayer::from_weights(a, b, config)
    }

    #[test]
    fn test_registry_creation() {
        let registry: AdapterRegistry<LoraLayer> = AdapterRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.active_adapter_name().is_none());
        assert!(!registry.is_weighted_composition());
    }

    #[test]
    fn test_register_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        let adapter1 = make_lora(&device, 2)?;
        let adapter2 = make_lora(&device, 2)?;

        registry.register_adapter("adapter1", adapter1)?;
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.active_adapter_name(), Some("adapter1"));

        registry.register_adapter("adapter2", adapter2)?;
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.active_adapter_name(), Some("adapter1"));

        Ok(())
    }

    #[test]
    fn test_register_duplicate_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        let adapter1 = make_lora(&device, 2)?;
        let adapter2 = make_lora(&device, 2)?;

        registry.register_adapter("adapter1", adapter1)?;
        let result = registry.register_adapter("adapter1", adapter2);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PeftError::AdapterExists { .. }
        ));

        Ok(())
    }

    #[test]
    fn test_set_active_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;
        registry.register_adapter("adapter2", make_lora(&device, 2)?)?;

        assert_eq!(registry.active_adapter_name(), Some("adapter1"));

        registry.set_active_adapter("adapter2")?;
        assert_eq!(registry.active_adapter_name(), Some("adapter2"));
        assert!(!registry.is_weighted_composition());

        Ok(())
    }

    #[test]
    fn test_set_nonexistent_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;

        let result = registry.set_active_adapter("nonexistent");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PeftError::AdapterNotFound { .. }
        ));

        Ok(())
    }

    #[test]
    fn test_get_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;

        let retrieved = registry.get_adapter("adapter1")?;
        assert_eq!(retrieved.num_parameters(), 2 * (4 + 4));

        Ok(())
    }

    #[test]
    fn test_get_active_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;

        let active = registry.get_active_adapter()?;
        assert_eq!(active.num_parameters(), 2 * (4 + 4));

        Ok(())
    }

    #[test]
    fn test_contains_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;

        assert!(registry.contains_adapter("adapter1"));
        assert!(!registry.contains_adapter("adapter2"));

        Ok(())
    }

    #[test]
    fn test_adapter_names() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;
        registry.register_adapter("adapter2", make_lora(&device, 2)?)?;

        let mut names = registry.adapter_names();
        names.sort_unstable();
        assert_eq!(names, vec!["adapter1", "adapter2"]);

        Ok(())
    }

    #[test]
    fn test_remove_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;
        registry.register_adapter("adapter2", make_lora(&device, 2)?)?;

        let removed = registry.remove_adapter("adapter2")?;
        assert!(removed.is_some());
        assert_eq!(registry.len(), 1);

        let result = registry.remove_adapter("adapter1");
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_clear() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("adapter1", make_lora(&device, 2)?)?;

        assert_eq!(registry.len(), 1);
        registry.clear();
        assert_eq!(registry.len(), 0);
        assert!(registry.active_adapter_name().is_none());
        assert!(!registry.is_weighted_composition());

        Ok(())
    }

    #[test]
    fn test_forward_with_active_adapter() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("test_adapter", make_lora(&device, 2)?)?;

        let input = Tensor::zeros(&[1, 2, 4], DType::F32, &device)?;
        let output = registry.forward(&input, None)?;

        assert_eq!(output.shape().dims(), &[1, 2, 4]);

        Ok(())
    }

    #[test]
    fn test_weighted_composition_sum() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        // Two non-zero residuals with known fills
        registry.register_adapter("a", make_nonzero_lora(&device, 2, 0.1)?)?;
        registry.register_adapter("b", make_nonzero_lora(&device, 2, 0.2)?)?;

        let input = Tensor::ones(&[1, 1, 4], DType::F32, &device)?;

        let y_a = registry.get_adapter("a")?.forward(&input, None)?;
        let y_b = registry.get_adapter("b")?.forward(&input, None)?;
        let expected = ((&y_a * 0.5)? + (&y_b * 0.5)?)?;

        registry
            .set_weighted_adapters([AdapterWeight::new("a", 0.5), AdapterWeight::new("b", 0.5)])?;
        assert!(registry.is_weighted_composition());

        let y = registry.forward(&input, None)?;
        let diff = (y - expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "weighted residual mismatch, diff={diff}");

        Ok(())
    }

    #[test]
    fn test_forward_weighted_with_base() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;

        registry.register_adapter("a", make_nonzero_lora(&device, 2, 0.1)?)?;
        registry.register_adapter("b", make_nonzero_lora(&device, 2, 0.1)?)?;

        let input = Tensor::ones(&[1, 1, 4], DType::F32, &device)?;
        let base = Tensor::full(2.0f32, (1, 1, 4), &device)?;

        let weights = [AdapterWeight::new("a", 1.0), AdapterWeight::new("b", 1.0)];
        let y = registry.forward_weighted(&input, Some(&base), &weights)?;
        // base is 2.0 + positive residuals → values > 2
        let min_v = y
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_v > 2.0,
            "expected base + positive residual, min={min_v}"
        );
        assert_eq!(y.dims(), &[1, 1, 4]);
        // state unchanged
        assert!(!registry.is_weighted_composition());
        Ok(())
    }

    #[test]
    fn test_set_active_clears_composition() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;
        registry.register_adapter("a", make_lora(&device, 2)?)?;
        registry.register_adapter("b", make_lora(&device, 2)?)?;
        registry
            .set_weighted_adapters([AdapterWeight::new("a", 0.7), AdapterWeight::new("b", 0.3)])?;
        assert!(registry.is_weighted_composition());
        registry.set_active_adapter("b")?;
        assert!(!registry.is_weighted_composition());
        assert_eq!(registry.active_adapter_name(), Some("b"));
        Ok(())
    }

    #[test]
    fn test_weighted_missing_adapter_errors() -> Result<()> {
        let mut registry = AdapterRegistry::new();
        let device = Device::Cpu;
        registry.register_adapter("a", make_lora(&device, 2)?)?;
        let err = registry.set_weighted_adapters([AdapterWeight::new("missing", 1.0)]);
        assert!(matches!(err, Err(PeftError::AdapterNotFound { .. })));
        Ok(())
    }
}
