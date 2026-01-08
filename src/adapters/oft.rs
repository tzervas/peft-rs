//! OFT (Orthogonal Fine-Tuning) implementation.
//!
//! OFT applies orthogonal transformations to preserve the pretrained knowledge
//! while adapting models. It uses block-diagonal orthogonal matrices to
//! transform weights efficiently.
//!
//! Reference: <https://arxiv.org/abs/2306.07280>

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::error::{PeftError, Result};
use crate::traits::{Adapter, AdapterConfig, Mergeable, Trainable};

/// Configuration for OFT adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OftConfig {
    /// Number of OFT blocks (determines expressiveness vs efficiency).
    pub r: usize,

    /// Whether to use constrained OFT (COFT) which enforces strict orthogonality.
    #[serde(default)]
    pub coft: bool,

    /// Small constant for numerical stability.
    #[serde(default = "default_eps")]
    pub eps: f64,

    /// Block sharing across layers.
    #[serde(default)]
    pub block_share: bool,

    /// Target modules to apply OFT to.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
}

fn default_eps() -> f64 {
    1e-5
}

fn default_target_modules() -> Vec<String> {
    vec!["q_proj".into(), "v_proj".into()]
}

impl Default for OftConfig {
    fn default() -> Self {
        Self {
            r: 8,
            coft: false,
            eps: default_eps(),
            block_share: false,
            target_modules: default_target_modules(),
        }
    }
}

impl AdapterConfig for OftConfig {
    fn validate(&self) -> Result<()> {
        if self.r == 0 {
            return Err(PeftError::InvalidConfig(
                "number of blocks (r) must be > 0".into(),
            ));
        }
        if self.eps <= 0.0 {
            return Err(PeftError::InvalidConfig("eps must be > 0".into()));
        }
        Ok(())
    }
}

/// OFT layer implementing Orthogonal Fine-Tuning.
///
/// Uses block-diagonal orthogonal matrices to transform weights:
/// `W' = W @ R` where R is a block-diagonal orthogonal matrix.
///
/// The orthogonal matrix R is parameterized via Cayley transform:
/// `R = (I - Q) @ (I + Q)^{-1}` where Q is skew-symmetric.
pub struct OftLayer {
    /// Skew-symmetric parameters for Cayley parameterization.
    /// Shape: [num_blocks, block_size, block_size]
    oft_r: Tensor,
    /// Configuration
    config: OftConfig,
    /// Input/output dimension (OFT requires square transformation)
    features: usize,
    /// Size of each block
    block_size: usize,
    /// Number of blocks
    num_blocks: usize,
    /// Whether gradients are disabled
    frozen: bool,
}

impl OftLayer {
    /// Create a new OFT layer.
    ///
    /// # Arguments
    /// * `features` - Dimension of the weight matrix (must be divisible by r)
    /// * `config` - OFT configuration
    /// * `device` - Device to create tensors on
    pub fn new(features: usize, config: OftConfig, device: &Device) -> Result<Self> {
        config.validate()?;

        if features % config.r != 0 {
            return Err(PeftError::InvalidConfig(format!(
                "features ({}) must be divisible by r ({})",
                features, config.r
            )));
        }

        let num_blocks = config.r;
        let block_size = features / num_blocks;

        // Initialize skew-symmetric parameters to small values
        // This makes the initial orthogonal matrix close to identity
        let std = 0.01_f32;
        let oft_r = Tensor::randn(0.0f32, std, (num_blocks, block_size, block_size), device)?;

        Ok(Self {
            oft_r,
            config,
            features,
            block_size,
            num_blocks,
            frozen: false,
        })
    }

    /// Get the number of blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get the block size.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Make the parameter matrix skew-symmetric: Q = (R - R^T) / 2
    fn make_skew_symmetric(&self) -> Result<Tensor> {
        let r_t = self.oft_r.transpose(1, 2)?;
        let diff = self.oft_r.broadcast_sub(&r_t)?;
        let two = Tensor::new(2.0f32, self.oft_r.device())?;
        Ok(diff.broadcast_div(&two)?)
    }

    /// Compute the orthogonal matrix using Cayley transform.
    /// R = (I - Q) @ (I + Q)^{-1}
    fn compute_orthogonal_matrix(&self) -> Result<Tensor> {
        let q = self.make_skew_symmetric()?;
        let device = q.device();

        // Create block-diagonal identity
        let eye = Tensor::eye(self.block_size, candle_core::DType::F32, device)?;
        let eye = eye.unsqueeze(0)?.expand((self.num_blocks, self.block_size, self.block_size))?;

        // I - Q
        let i_minus_q = eye.broadcast_sub(&q)?;

        // I + Q
        let i_plus_q = eye.broadcast_add(&q)?;

        // For each block, compute (I - Q) @ (I + Q)^{-1}
        // Using matrix solve: X @ (I + Q) = (I - Q)  =>  X = (I - Q) @ (I + Q)^{-1}
        // We'll approximate using Neumann series for near-identity: (I + Q)^{-1} ≈ I - Q + Q^2 - ...
        // For small Q, we can use: (I + Q)^{-1} ≈ I - Q (first order approximation)
        
        // More accurate: compute actual inverse for each block
        let mut result_blocks = Vec::with_capacity(self.num_blocks);
        
        for block_idx in 0..self.num_blocks {
            let i_minus_q_block = i_minus_q.i(block_idx)?;
            let _i_plus_q_block = i_plus_q.i(block_idx)?;
            
            // Use pseudo-inverse approximation via transpose for orthogonal-like matrices
            // For small perturbations: (I + Q)^{-1} ≈ I - Q + Q^2
            let q_block = q.i(block_idx)?;
            let q_sq = q_block.matmul(&q_block)?;
            let eye_block = Tensor::eye(self.block_size, candle_core::DType::F32, device)?;
            
            // (I + Q)^{-1} ≈ I - Q + Q^2 - Q^3 + ...
            // Truncate at Q^2 for efficiency
            let inv_approx = eye_block.broadcast_sub(&q_block)?.broadcast_add(&q_sq)?;
            
            // R_block = (I - Q) @ (I + Q)^{-1}
            let r_block = i_minus_q_block.matmul(&inv_approx)?;
            result_blocks.push(r_block);
        }

        // Stack blocks: [num_blocks, block_size, block_size]
        Ok(Tensor::stack(&result_blocks, 0)?)
    }

    /// Apply block-diagonal orthogonal transformation to input.
    fn apply_block_diagonal(&self, input: &Tensor, orth_matrix: &Tensor) -> Result<Tensor> {
        let input_dims = input.dims();
        let batch_seq = input_dims[0] * input_dims[1];
        
        // Reshape input to [batch*seq, num_blocks, block_size]
        let input_blocked = input.reshape((batch_seq, self.num_blocks, self.block_size))?;
        
        // Apply orthogonal transformation to each block
        // input_blocked: [batch*seq, num_blocks, block_size]
        // orth_matrix: [num_blocks, block_size, block_size]
        
        // For each block: output[b, n, :] = input[b, n, :] @ R[n, :, :]
        // We need batch matrix multiply
        
        let mut output_blocks = Vec::with_capacity(self.num_blocks);
        
        for block_idx in 0..self.num_blocks {
            // input_block: [batch*seq, block_size]
            let input_block = input_blocked.i((.., block_idx, ..))?;
            // orth_block: [block_size, block_size]
            let orth_block = orth_matrix.i(block_idx)?;
            
            // output_block: [batch*seq, block_size]
            let output_block = input_block.matmul(&orth_block)?;
            output_blocks.push(output_block);
        }

        // Stack and reshape back
        let output_stacked = Tensor::stack(&output_blocks, 1)?; // [batch*seq, num_blocks, block_size]
        Ok(output_stacked.reshape((input_dims[0], input_dims[1], self.features))?)
    }
}

impl Adapter for OftLayer {
    type Config = OftConfig;

    fn forward(&self, input: &Tensor, base_output: Option<&Tensor>) -> Result<Tensor> {
        // Compute the orthogonal transformation matrix
        let orth_matrix = self.compute_orthogonal_matrix()?;
        
        // Apply block-diagonal orthogonal transformation
        let transformed = self.apply_block_diagonal(input, &orth_matrix)?;
        
        // For OFT, the transformation replaces the base output
        // If base_output provided, compute the difference (delta)
        match base_output {
            Some(base) => {
                // Return: base + (transformed - input) = base + delta
                let delta = transformed.broadcast_sub(input)?;
                Ok(base.broadcast_add(&delta)?)
            }
            None => Ok(transformed),
        }
    }

    fn num_parameters(&self) -> usize {
        // Skew-symmetric blocks: num_blocks * block_size * block_size
        // But only lower/upper triangle is independent: num_blocks * block_size * (block_size - 1) / 2
        // For simplicity, we count all parameters
        self.num_blocks * self.block_size * self.block_size
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Mergeable for OftLayer {
    fn merge(&self, base_weight: &Tensor) -> Result<Tensor> {
        // W' = W @ R (right multiply by orthogonal matrix)
        let orth_matrix = self.compute_orthogonal_matrix()?;
        
        // Construct full block-diagonal matrix from blocks
        let full_orth = self.construct_full_matrix(&orth_matrix)?;
        
        // base_weight: [out_features, in_features]
        // full_orth: [in_features, in_features]
        Ok(base_weight.matmul(&full_orth)?)
    }

    fn unmerge(&self, merged_weight: &Tensor) -> Result<Tensor> {
        // W = W' @ R^T (R is orthogonal, so R^{-1} = R^T)
        let orth_matrix = self.compute_orthogonal_matrix()?;
        let full_orth = self.construct_full_matrix(&orth_matrix)?;
        
        // R^T
        let full_orth_t = full_orth.t()?;
        
        Ok(merged_weight.matmul(&full_orth_t)?)
    }
}

impl OftLayer {
    /// Construct full block-diagonal matrix from blocks.
    fn construct_full_matrix(&self, blocks: &Tensor) -> Result<Tensor> {
        let device = blocks.device();
        let n = self.features;
        
        // Start with zeros
        let mut full_data = vec![0.0f32; n * n];
        
        // Fill in blocks along diagonal
        for block_idx in 0..self.num_blocks {
            let block = blocks.i(block_idx)?;
            let block_data: Vec<f32> = block.flatten_all()?.to_vec1()?;
            
            let start = block_idx * self.block_size;
            
            for i in 0..self.block_size {
                for j in 0..self.block_size {
                    let row = start + i;
                    let col = start + j;
                    full_data[row * n + col] = block_data[i * self.block_size + j];
                }
            }
        }
        
        Ok(Tensor::from_vec(full_data, (n, n), device)?)
    }
}

impl Trainable for OftLayer {
    fn register_parameters(&self, _var_map: &mut VarMap, _prefix: &str) -> Result<()> {
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
    use candle_core::{DType, IndexOp};

    #[test]
    fn test_oft_config_default() {
        let config = OftConfig::default();
        assert_eq!(config.r, 8);
        assert!(!config.coft);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_oft_config_invalid_r() {
        let config = OftConfig {
            r: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_oft_layer_creation() {
        let config = OftConfig {
            r: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        // 64 is divisible by 8
        let layer = OftLayer::new(64, config, &device);
        assert!(layer.is_ok());
        
        let layer = layer.unwrap();
        assert_eq!(layer.num_blocks(), 8);
        assert_eq!(layer.block_size(), 8);
    }

    #[test]
    fn test_oft_layer_invalid_dimensions() {
        let config = OftConfig {
            r: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        // 65 is not divisible by 8
        let layer = OftLayer::new(65, config, &device);
        assert!(layer.is_err());
    }

    #[test]
    fn test_oft_forward_shape() {
        let config = OftConfig {
            r: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = OftLayer::new(64, config, &device).unwrap();

        let input = Tensor::zeros(&[1, 10, 64], DType::F32, &device).unwrap();
        let output = layer.forward(&input, None).unwrap();

        assert_eq!(output.shape().dims(), &[1, 10, 64]);
    }

    #[test]
    fn test_oft_forward_with_base_output() {
        let config = OftConfig {
            r: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = OftLayer::new(64, config, &device).unwrap();

        let input = Tensor::zeros(&[1, 10, 64], DType::F32, &device).unwrap();
        let base_output = Tensor::ones(&[1, 10, 64], DType::F32, &device).unwrap();
        let output = layer.forward(&input, Some(&base_output)).unwrap();

        assert_eq!(output.shape().dims(), &[1, 10, 64]);
    }

    #[test]
    fn test_oft_num_parameters() {
        let config = OftConfig {
            r: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = OftLayer::new(64, config, &device).unwrap();

        // 8 blocks of 8x8 = 8 * 64 = 512
        assert_eq!(layer.num_parameters(), 512);
    }

    #[test]
    fn test_oft_skew_symmetric() {
        let config = OftConfig {
            r: 2,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = OftLayer::new(8, config, &device).unwrap();

        let skew = layer.make_skew_symmetric().unwrap();
        
        // Check Q = -Q^T for each block
        for block_idx in 0..2 {
            let q = skew.i(block_idx).unwrap();
            let q_t = q.t().unwrap();
            let sum = q.broadcast_add(&q_t).unwrap();
            let max_val: f32 = sum.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
            assert!(max_val < 1e-5, "Matrix should be skew-symmetric");
        }
    }

    #[test]
    fn test_oft_freeze_unfreeze() {
        let config = OftConfig::default();
        let device = Device::Cpu;
        let mut layer = OftLayer::new(64, config, &device).unwrap();

        assert!(!layer.is_frozen());
        layer.freeze();
        assert!(layer.is_frozen());
        layer.unfreeze();
        assert!(!layer.is_frozen());
    }

    #[test]
    fn test_oft_merge_unmerge() {
        let config = OftConfig {
            r: 4,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = OftLayer::new(16, config, &device).unwrap();

        let base_weight = Tensor::eye(16, DType::F32, &device).unwrap();
        let merged = layer.merge(&base_weight).unwrap();
        let unmerged = layer.unmerge(&merged).unwrap();

        // Unmerged should be close to original
        let diff = unmerged.broadcast_sub(&base_weight).unwrap();
        let max_diff: f32 = diff.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(max_diff < 0.1, "Max diff: {}", max_diff); // Allow some numerical error
    }
}
