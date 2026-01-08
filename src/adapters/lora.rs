//! LoRA (Low-Rank Adaptation) implementation.
//!
//! LoRA reduces the number of trainable parameters by decomposing weight updates
//! into low-rank matrices: `ΔW = BA` where `B ∈ R^{d×r}` and `A ∈ R^{r×k}`.
//!
//! Reference: <https://arxiv.org/abs/2106.09685>

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::error::{PeftError, Result};
use crate::traits::{Adapter, AdapterConfig, Mergeable, Trainable};

/// Configuration for LoRA adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition.
    pub r: usize,
    
    /// Scaling factor (typically `alpha / r`).
    pub alpha: usize,
    
    /// Dropout probability applied to LoRA outputs.
    #[serde(default)]
    pub dropout: f64,
    
    /// Target modules to apply LoRA to.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
    
    /// Initialize A with Gaussian, B with zeros (standard) or vice versa.
    #[serde(default)]
    pub init_lora_weights: LoraInitialization,
}

fn default_target_modules() -> Vec<String> {
    vec!["q_proj".into(), "v_proj".into()]
}

/// Initialization strategy for LoRA weights.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum LoraInitialization {
    /// Standard: A ~ N(0, σ²), B = 0
    #[default]
    Standard,
    /// Gaussian for both: A, B ~ N(0, σ²)
    Gaussian,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16,
            dropout: 0.0,
            target_modules: default_target_modules(),
            init_lora_weights: LoraInitialization::Standard,
        }
    }
}

impl AdapterConfig for LoraConfig {
    fn validate(&self) -> Result<()> {
        if self.r == 0 {
            return Err(PeftError::InvalidConfig("rank must be > 0".into()));
        }
        if self.alpha == 0 {
            return Err(PeftError::InvalidConfig("alpha must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(PeftError::InvalidConfig(
                "dropout must be between 0 and 1".into(),
            ));
        }
        Ok(())
    }
}

/// LoRA layer implementing low-rank adaptation.
///
/// Computes: `output = base_output + (x @ A^T @ B^T) * scaling`
pub struct LoraLayer {
    /// Down projection: in_features → r
    lora_a: Linear,
    /// Up projection: r → out_features  
    lora_b: Linear,
    /// Scaling factor = alpha / r
    scaling: f64,
    /// Configuration
    config: LoraConfig,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Whether gradients are disabled
    frozen: bool,
}

impl LoraLayer {
    /// Create a new LoRA layer.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `config` - LoRA configuration
    /// * `vb` - Variable builder for weight initialization
    ///
    /// # Errors
    /// Returns error if configuration is invalid or weight initialization fails.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: LoraConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        config.validate()?;

        let scaling = config.alpha as f64 / config.r as f64;

        // A: in_features → r (initialized with small random values)
        let lora_a = linear_no_bias(in_features, config.r, vb.pp("lora_a"))?;
        
        // B: r → out_features (initialized to zeros for standard init)
        let lora_b = linear_no_bias(config.r, out_features, vb.pp("lora_b"))?;

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            config,
            in_features,
            out_features,
            frozen: false,
        })
    }

    /// Create a new LoRA layer with zeros initialization for B.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `config` - LoRA configuration
    /// * `device` - Device to create tensors on
    pub fn new_with_zeros(
        in_features: usize,
        out_features: usize,
        config: LoraConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;

        let scaling = config.alpha as f64 / config.r as f64;
        let dtype = DType::F32;

        // Initialize A with small random values (Kaiming uniform)
        let std = (1.0 / in_features as f64).sqrt();
        let a_weight = Tensor::randn(0.0f32, std as f32, (config.r, in_features), device)?;
        
        // Initialize B with zeros
        let b_weight = Tensor::zeros((out_features, config.r), dtype, device)?;

        let lora_a = Linear::new(a_weight, None);
        let lora_b = Linear::new(b_weight, None);

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            config,
            in_features,
            out_features,
            frozen: false,
        })
    }

    /// Get the scaling factor.
    #[must_use]
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Get the rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.config.r
    }
}

impl Adapter for LoraLayer {
    type Config = LoraConfig;

    fn forward(&self, input: &Tensor, base_output: Option<&Tensor>) -> Result<Tensor> {
        // LoRA forward: x @ A^T @ B^T * scaling
        let lora_out = self.lora_a.forward(input)?;
        let lora_out = self.lora_b.forward(&lora_out)?;
        let scaling = Tensor::new(self.scaling as f32, lora_out.device())?;
        let lora_out = lora_out.broadcast_mul(&scaling)?;

        // Add to base output if provided
        match base_output {
            Some(base) => Ok(base.broadcast_add(&lora_out)?),
            None => Ok(lora_out),
        }
    }

    fn num_parameters(&self) -> usize {
        self.config.r * (self.in_features + self.out_features)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Mergeable for LoraLayer {
    fn merge(&self, base_weight: &Tensor) -> Result<Tensor> {
        // ΔW = B @ A * scaling
        // merged = W + ΔW
        let a_weight = self.lora_a.weight();
        let b_weight = self.lora_b.weight();
        
        let delta_w = b_weight.matmul(a_weight)?;
        let scaling = Tensor::new(self.scaling as f32, delta_w.device())?;
        let delta_w = delta_w.broadcast_mul(&scaling)?;
        
        Ok(base_weight.broadcast_add(&delta_w)?)
    }

    fn unmerge(&self, merged_weight: &Tensor) -> Result<Tensor> {
        let a_weight = self.lora_a.weight();
        let b_weight = self.lora_b.weight();
        
        let delta_w = b_weight.matmul(a_weight)?;
        let scaling = Tensor::new(self.scaling as f32, delta_w.device())?;
        let delta_w = delta_w.broadcast_mul(&scaling)?;
        
        Ok(merged_weight.broadcast_sub(&delta_w)?)
    }
}

impl Trainable for LoraLayer {
    fn register_parameters(&self, _var_map: &mut VarMap, _prefix: &str) -> Result<()> {
        // Parameters are already registered via VarBuilder during construction
        Ok(())
    }

    fn freeze(&mut self) {
        self.frozen = true;
    }

    fn unfreeze(&mut self) {
        self.frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.frozen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.r, 8);
        assert_eq!(config.alpha, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lora_config_invalid_rank() {
        let config = LoraConfig {
            r: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lora_layer_creation() {
        let config = LoraConfig::default();
        let device = Device::Cpu;
        let layer = LoraLayer::new_with_zeros(768, 768, config, &device);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_lora_forward_shape() {
        let config = LoraConfig::default();
        let device = Device::Cpu;
        let layer = LoraLayer::new_with_zeros(768, 768, config, &device).unwrap();
        
        let input = Tensor::zeros(&[1, 10, 768], DType::F32, &device).unwrap();
        let output = layer.forward(&input, None).unwrap();
        
        assert_eq!(output.shape().dims(), &[1, 10, 768]);
    }

    #[test]
    fn test_lora_num_parameters() {
        let config = LoraConfig {
            r: 8,
            alpha: 16,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = LoraLayer::new_with_zeros(768, 768, config, &device).unwrap();
        
        // r * (in + out) = 8 * (768 + 768) = 12288
        assert_eq!(layer.num_parameters(), 12288);
    }
}
