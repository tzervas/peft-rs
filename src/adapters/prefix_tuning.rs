//! Prefix Tuning implementation (**experimental**).
//!
//! Prefix tuning prepends trainable "prefix" vectors to the keys and values
//! in attention layers, without modifying the original model weights.
//!
//! # Experimental status (PEFT-P1-03)
//!
//! This module is a **layer helper**, not a full transformers attention patch:
//! - Provides prefix K/V tensors and optional MLP reparameterization
//! - Provides [`PrefixTuningLayer::concat_to_kv`] to prepend prefixes to caller K/V
//! - Does **not** inject into candle-transformers models automatically
//! - `Adapter::forward` remains a pass-through (prefixes are consumed via
//!   [`get_prefix_keys`] / [`get_prefix_values`] / [`concat_to_kv`])
//!
//! Reference: <https://arxiv.org/abs/2101.00190>

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{PeftError, Result};
use crate::io::SaveLoad;
use crate::traits::{Adapter, AdapterConfig};

/// Configuration for prefix tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixTuningConfig {
    /// Number of prefix tokens to prepend.
    pub num_prefix_tokens: usize,

    /// Hidden dimension of the reparameterization MLP input (when enabled).
    pub prefix_dim: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of layers to apply prefix to.
    pub num_layers: usize,

    /// Whether to use a reparameterization MLP (Li & Liang style).
    ///
    /// When `true`, trainable parameters are a lower-dim embedding + two-layer
    /// MLP that produces K/V. When `false`, K/V tensors are stored directly.
    #[serde(default = "default_true")]
    pub use_reparameterization: bool,

    /// Hidden size of the reparameterization MLP (only if reparam enabled).
    #[serde(default = "default_reparam_hidden")]
    pub reparam_hidden_size: usize,
}

fn default_true() -> bool {
    true
}

fn default_reparam_hidden() -> usize {
    512
}

impl Default for PrefixTuningConfig {
    fn default() -> Self {
        Self {
            num_prefix_tokens: 20,
            prefix_dim: 512,
            num_heads: 12,
            num_layers: 12,
            use_reparameterization: true,
            reparam_hidden_size: default_reparam_hidden(),
        }
    }
}

impl AdapterConfig for PrefixTuningConfig {
    fn validate(&self) -> Result<()> {
        if self.num_prefix_tokens == 0 {
            return Err(PeftError::InvalidConfig(
                "num_prefix_tokens must be > 0".into(),
            ));
        }
        if self.prefix_dim == 0 {
            return Err(PeftError::InvalidConfig("prefix_dim must be > 0".into()));
        }
        if self.num_heads == 0 {
            return Err(PeftError::InvalidConfig("num_heads must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(PeftError::InvalidConfig("num_layers must be > 0".into()));
        }
        if self.use_reparameterization && self.reparam_hidden_size == 0 {
            return Err(PeftError::InvalidConfig(
                "reparam_hidden_size must be > 0 when reparameterization is enabled".into(),
            ));
        }
        Ok(())
    }
}

/// Optional two-layer MLP that maps prefix embeddings → K/V.
struct ReparamMlp {
    /// W1: [`prefix_dim`, hidden]
    w1: Tensor,
    /// b1: `[hidden]`
    b1: Tensor,
    /// W2: [hidden, 2 * `num_heads` * `head_dim`]
    w2: Tensor,
    /// b2: [2 * `num_heads` * `head_dim`]
    b2: Tensor,
}

/// Prefix tuning layer (**experimental**).
///
/// Stores trainable prefix parameters and materializes per-layer K/V prefixes.
pub struct PrefixTuningLayer {
    /// When reparam: raw embeddings [`num_layers`, `num_prefix_tokens`, `prefix_dim`].
    /// When direct: unused (None).
    prefix_tokens: Option<Tensor>,
    /// Reparameterization MLP (when enabled).
    reparam: Option<ReparamMlp>,
    /// Direct prefix keys (when reparam disabled):
    /// [`num_layers`, `num_prefix_tokens`, `num_heads`, `head_dim`]
    prefix_keys: Option<Tensor>,
    /// Direct prefix values (when reparam disabled).
    prefix_values: Option<Tensor>,
    /// Per-head dimension.
    head_dim: usize,
    /// Configuration
    config: PrefixTuningConfig,
}

impl PrefixTuningLayer {
    /// Create a new prefix tuning layer.
    ///
    /// # Arguments
    /// * `config` - Prefix tuning configuration
    /// * `head_dim` - Dimension per attention head
    /// * `device` - Device to create tensors on
    ///
    /// # Errors
    ///
    /// Returns an error if configuration validation fails or layer construction fails.
    pub fn new(config: PrefixTuningConfig, head_dim: usize, device: &Device) -> Result<Self> {
        config.validate()?;
        if head_dim == 0 {
            return Err(PeftError::InvalidConfig("head_dim must be > 0".into()));
        }

        if config.use_reparameterization {
            let prefix_tokens = Tensor::randn(
                0.0f32,
                0.02,
                (
                    config.num_layers,
                    config.num_prefix_tokens,
                    config.prefix_dim,
                ),
                device,
            )?;
            let hidden = config.reparam_hidden_size;
            let out = 2 * config.num_heads * head_dim;
            let w1 = Tensor::randn(0.0f32, 0.02, (config.prefix_dim, hidden), device)?;
            let b1 = Tensor::zeros(hidden, DType::F32, device)?;
            let w2 = Tensor::randn(0.0f32, 0.02, (hidden, out), device)?;
            let b2 = Tensor::zeros(out, DType::F32, device)?;
            Ok(Self {
                prefix_tokens: Some(prefix_tokens),
                reparam: Some(ReparamMlp { w1, b1, w2, b2 }),
                prefix_keys: None,
                prefix_values: None,
                head_dim,
                config,
            })
        } else {
            let shape = (
                config.num_layers,
                config.num_prefix_tokens,
                config.num_heads,
                head_dim,
            );
            let prefix_keys = Tensor::randn(0.0f32, 0.02, shape, device)?;
            let prefix_values = Tensor::randn(0.0f32, 0.02, shape, device)?;
            Ok(Self {
                prefix_tokens: None,
                reparam: None,
                prefix_keys: Some(prefix_keys),
                prefix_values: Some(prefix_values),
                head_dim,
                config,
            })
        }
    }

    /// Whether the reparameterization MLP is active.
    #[must_use]
    pub fn uses_reparameterization(&self) -> bool {
        self.reparam.is_some()
    }

    /// Head dimension used at construction.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Materialize full prefix keys:
    /// `[num_layers, num_prefix_tokens, num_heads, head_dim]`.
    ///
    /// # Errors
    /// Returns an error if the MLP or reshape fails.
    pub fn materialize_keys(&self) -> Result<Tensor> {
        if let Some(keys) = &self.prefix_keys {
            return Ok(keys.clone());
        }
        let (keys, _) = self.materialize_kv()?;
        Ok(keys)
    }

    /// Materialize full prefix values (same shape as keys).
    ///
    /// # Errors
    /// Returns an error if the MLP or reshape fails.
    pub fn materialize_values(&self) -> Result<Tensor> {
        if let Some(vals) = &self.prefix_values {
            return Ok(vals.clone());
        }
        let (_, vals) = self.materialize_kv()?;
        Ok(vals)
    }

    fn materialize_kv(&self) -> Result<(Tensor, Tensor)> {
        let tokens = self
            .prefix_tokens
            .as_ref()
            .ok_or_else(|| PeftError::InvalidConfig("reparam path missing prefix_tokens".into()))?;
        let mlp = self
            .reparam
            .as_ref()
            .ok_or_else(|| PeftError::InvalidConfig("reparam path missing MLP".into()))?;

        // tokens: [L, P, D] → [L*P, D]
        let l = self.config.num_layers;
        let p = self.config.num_prefix_tokens;
        let d = self.config.prefix_dim;
        let flat = tokens.reshape((l * p, d))?;
        // MLP: relu(x @ W1 + b1) @ W2 + b2
        let h = flat.matmul(&mlp.w1)?.broadcast_add(&mlp.b1)?;
        let h = h.relu()?;
        let out = h.matmul(&mlp.w2)?.broadcast_add(&mlp.b2)?;
        // out: [L*P, 2 * H * head_dim]
        let kv = out.reshape((l, p, 2, self.config.num_heads, self.head_dim))?;
        let keys = kv.i((.., .., 0, .., ..))?;
        let vals = kv.i((.., .., 1, .., ..))?;
        Ok((keys, vals))
    }

    /// Get prefix keys for a specific layer: `[num_prefix_tokens, num_heads, head_dim]`.
    ///
    /// # Errors
    /// Returns an error if the layer index is out of bounds.
    pub fn get_prefix_keys(&self, layer_idx: usize) -> Result<Tensor> {
        if layer_idx >= self.config.num_layers {
            return Err(PeftError::InvalidConfig(format!(
                "layer_idx {layer_idx} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        Ok(self.materialize_keys()?.i(layer_idx)?)
    }

    /// Get prefix values for a specific layer.
    ///
    /// # Errors
    /// Returns an error if the layer index is out of bounds.
    pub fn get_prefix_values(&self, layer_idx: usize) -> Result<Tensor> {
        if layer_idx >= self.config.num_layers {
            return Err(PeftError::InvalidConfig(format!(
                "layer_idx {layer_idx} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        Ok(self.materialize_values()?.i(layer_idx)?)
    }

    /// Prepend prefix K/V to attention keys/values for one layer.
    ///
    /// Expected shapes (batch-first multi-head):
    /// - `keys` / `values`: `[batch, num_heads, seq, head_dim]`
    /// - prefixes expanded to: `[batch, num_heads, num_prefix, head_dim]`
    /// - output: `[batch, num_heads, num_prefix + seq, head_dim]`
    ///
    /// # Errors
    /// Shape or device errors.
    pub fn concat_to_kv(
        &self,
        layer_idx: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let pk = self.get_prefix_keys(layer_idx)?; // [P, H, D]
        let pv = self.get_prefix_values(layer_idx)?;
        let batch = keys.dim(0)?;
        // [P, H, D] → [1, H, P, D] → expand batch
        let pk = pk.transpose(0, 1)?.unsqueeze(0)?; // [1, H, P, D]
        let pv = pv.transpose(0, 1)?.unsqueeze(0)?;
        let pk = pk.expand((
            batch,
            self.config.num_heads,
            self.config.num_prefix_tokens,
            self.head_dim,
        ))?;
        let pv = pv.expand((
            batch,
            self.config.num_heads,
            self.config.num_prefix_tokens,
            self.head_dim,
        ))?;
        // Contiguous for cat
        let pk = pk.contiguous()?;
        let pv = pv.contiguous()?;
        let keys = keys.contiguous()?;
        let values = values.contiguous()?;
        let out_k = Tensor::cat(&[&pk, &keys], 2)?;
        let out_v = Tensor::cat(&[&pv, &values], 2)?;
        Ok((out_k, out_v))
    }
}

impl Adapter for PrefixTuningLayer {
    type Config = PrefixTuningConfig;

    fn forward(&self, input: &Tensor, _base_output: Option<&Tensor>) -> Result<Tensor> {
        // Prefix tuning does not modify token embeddings directly.
        // Use concat_to_kv / get_prefix_* in the attention path.
        Ok(input.clone())
    }

    fn num_parameters(&self) -> usize {
        let mut n = 0;
        if let Some(t) = &self.prefix_tokens {
            n += t.elem_count();
        }
        if let Some(mlp) = &self.reparam {
            n += mlp.w1.elem_count()
                + mlp.b1.elem_count()
                + mlp.w2.elem_count()
                + mlp.b2.elem_count();
        }
        if let Some(k) = &self.prefix_keys {
            n += k.elem_count();
        }
        if let Some(v) = &self.prefix_values {
            n += v.elem_count();
        }
        n
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl SaveLoad for PrefixTuningLayer {
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        if let Some(t) = &self.prefix_tokens {
            state_dict.insert("prefix_tokens".into(), t.clone());
        }
        if let Some(mlp) = &self.reparam {
            state_dict.insert("reparam.w1".into(), mlp.w1.clone());
            state_dict.insert("reparam.b1".into(), mlp.b1.clone());
            state_dict.insert("reparam.w2".into(), mlp.w2.clone());
            state_dict.insert("reparam.b2".into(), mlp.b2.clone());
        }
        if let Some(k) = &self.prefix_keys {
            state_dict.insert("prefix_keys".into(), k.clone());
        }
        if let Some(v) = &self.prefix_values {
            state_dict.insert("prefix_values".into(), v.clone());
        }
        #[allow(clippy::cast_precision_loss)]
        let head_dim_f = self.head_dim as f32;
        state_dict.insert("head_dim".into(), Tensor::new(head_dim_f, &Device::Cpu)?);
        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        if let Some(t) = state_dict.get("prefix_tokens") {
            self.prefix_tokens = Some(t.clone());
        }
        if let Some(mlp) = self.reparam.as_mut() {
            if let Some(t) = state_dict.get("reparam.w1") {
                mlp.w1 = t.clone();
            }
            if let Some(t) = state_dict.get("reparam.b1") {
                mlp.b1 = t.clone();
            }
            if let Some(t) = state_dict.get("reparam.w2") {
                mlp.w2 = t.clone();
            }
            if let Some(t) = state_dict.get("reparam.b2") {
                mlp.b2 = t.clone();
            }
        }
        if let Some(t) = state_dict.get("prefix_keys") {
            self.prefix_keys = Some(t.clone());
        }
        if let Some(t) = state_dict.get("prefix_values") {
            self.prefix_values = Some(t.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_tuning_creation_reparam() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 4,
            num_heads: 2,
            num_layers: 2,
            prefix_dim: 16,
            reparam_hidden_size: 32,
            use_reparameterization: true,
        };
        let device = Device::Cpu;
        let layer = PrefixTuningLayer::new(config, 8, &device).unwrap();
        assert!(layer.uses_reparameterization());
        let keys = layer.get_prefix_keys(0).unwrap();
        assert_eq!(keys.dims(), &[4, 2, 8]);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_prefix_tuning_creation_direct() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 10,
            num_heads: 8,
            num_layers: 6,
            use_reparameterization: false,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = PrefixTuningLayer::new(config, 64, &device).unwrap();
        assert!(!layer.uses_reparameterization());
        let keys = layer.get_prefix_keys(0).unwrap();
        assert_eq!(keys.shape().dims(), &[10, 8, 64]);
    }

    #[test]
    fn test_concat_to_kv() {
        let config = PrefixTuningConfig {
            num_prefix_tokens: 3,
            num_heads: 2,
            num_layers: 1,
            use_reparameterization: false,
            prefix_dim: 8,
            reparam_hidden_size: 8,
        };
        let device = Device::Cpu;
        let layer = PrefixTuningLayer::new(config, 4, &device).unwrap();
        // keys: [batch=2, heads=2, seq=5, dim=4]
        let keys = Tensor::zeros(&[2, 2, 5, 4], DType::F32, &device).unwrap();
        let values = Tensor::zeros(&[2, 2, 5, 4], DType::F32, &device).unwrap();
        let (ok, ov) = layer.concat_to_kv(0, &keys, &values).unwrap();
        assert_eq!(ok.dims(), &[2, 2, 8, 4]); // 3 prefix + 5 seq
        assert_eq!(ov.dims(), &[2, 2, 8, 4]);
    }

    #[test]
    fn test_prefix_layer_oob() {
        let config = PrefixTuningConfig {
            num_layers: 2,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::new(config, 8, &Device::Cpu).unwrap();
        assert!(layer.get_prefix_keys(5).is_err());
    }
}
