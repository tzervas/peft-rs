//! `AdaLoRA` (Adaptive Low-Rank Adaptation) implementation.
//!
//! `AdaLoRA` dynamically allocates rank budget during training using SVD-based
//! importance scores. It uses a three-phase training schedule:
//! 1. Initial warmup phase (tinit steps)
//! 2. Rank reduction phase (between tinit and `total_step` - tfinal)
//! 3. Final fine-tuning phase (tfinal steps)
//!
//! Reference: <https://arxiv.org/abs/2303.10512>

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::error::{PeftError, Result};
use crate::io::SaveLoad;
use crate::traits::{Adapter, AdapterConfig, Mergeable, Trainable};

/// Configuration for AdaLoRA adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaLoraConfig {
    /// Target average rank after pruning.
    pub target_r: usize,

    /// Initial rank for each incremental matrix (before pruning).
    pub init_r: usize,

    /// Scaling factor (typically alpha / r).
    pub alpha: usize,

    /// Dropout probability applied to outputs.
    #[serde(default)]
    pub dropout: f64,

    /// Target modules to apply AdaLoRA to.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,

    /// Steps of initial warmup (no rank reduction).
    #[serde(default)]
    pub tinit: usize,

    /// Steps of final fine-tuning (no rank reduction).
    #[serde(default)]
    pub tfinal: usize,

    /// Time interval between budget allocations.
    #[serde(default = "default_delta_t")]
    pub delta_t: usize,

    /// Hyperparameter of EMA for sensitivity smoothing.
    #[serde(default = "default_beta")]
    pub beta1: f64,

    /// Hyperparameter of EMA for uncertainty quantification.
    #[serde(default = "default_beta")]
    pub beta2: f64,

    /// Coefficient of orthogonal regularization.
    #[serde(default = "default_orth_reg")]
    pub orth_reg_weight: f64,

    /// Total training steps (required for AdaLoRA).
    pub total_step: usize,
}

fn default_target_modules() -> Vec<String> {
    vec!["q_proj".into(), "v_proj".into()]
}

fn default_delta_t() -> usize {
    1
}

fn default_beta() -> f64 {
    0.85
}

fn default_orth_reg() -> f64 {
    0.5
}

impl Default for AdaLoraConfig {
    fn default() -> Self {
        Self {
            target_r: 8,
            init_r: 12,
            alpha: 16,
            dropout: 0.0,
            target_modules: default_target_modules(),
            tinit: 0,
            tfinal: 0,
            delta_t: default_delta_t(),
            beta1: default_beta(),
            beta2: default_beta(),
            orth_reg_weight: default_orth_reg(),
            total_step: 1000, // Must be set by user
        }
    }
}

impl AdapterConfig for AdaLoraConfig {
    fn validate(&self) -> Result<()> {
        if self.init_r == 0 {
            return Err(PeftError::InvalidConfig("init_r must be > 0".into()));
        }
        if self.target_r == 0 {
            return Err(PeftError::InvalidConfig("target_r must be > 0".into()));
        }
        if self.target_r > self.init_r {
            return Err(PeftError::InvalidConfig(
                "target_r must be <= init_r".into(),
            ));
        }
        if self.alpha == 0 {
            return Err(PeftError::InvalidConfig("alpha must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(PeftError::InvalidConfig(
                "dropout must be between 0 and 1".into(),
            ));
        }
        if self.total_step == 0 {
            return Err(PeftError::InvalidConfig("total_step must be > 0".into()));
        }
        if self.tinit >= self.total_step.saturating_sub(self.tfinal) {
            return Err(PeftError::InvalidConfig(
                "tinit must be < (total_step - tfinal) for budgeting phase".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.beta1) || !(0.0..=1.0).contains(&self.beta2) {
            return Err(PeftError::InvalidConfig(
                "beta1 and beta2 must be between 0 and 1".into(),
            ));
        }
        Ok(())
    }
}

/// AdaLoRA layer using SVD-based parameterization.
///
/// Uses `W = W0 + P * Λ * Q` where:
/// - P: Left singular vectors (out_features × r)
/// - Λ: Diagonal singular values (r)
/// - Q: Right singular vectors (r × in_features)
///
/// This allows for dynamic rank allocation by zeroing out singular values.
pub struct AdaLoraLayer {
    /// Left singular vectors: [out_features, init_r]
    lora_a: Tensor,
    /// Singular values: [init_r]
    lora_e: Tensor,
    /// Right singular vectors: [init_r, in_features]
    lora_b: Tensor,
    /// Scaling factor
    scaling: f64,
    /// Configuration
    config: AdaLoraConfig,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Current rank (may be reduced during training)
    current_rank: usize,
    /// Mask for pruned singular values
    rank_mask: Tensor,
    /// Whether gradients are disabled
    frozen: bool,
}

impl AdaLoraLayer {
    /// Create a new AdaLoRA layer.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `config` - AdaLoRA configuration
    /// * `device` - Device to create tensors on
    ///
    /// # Errors
    /// Returns error if configuration is invalid or tensor initialization fails.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: AdaLoraConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;

        let scaling = config.alpha as f64 / config.init_r as f64;
        let dtype = DType::F32;

        // Initialize A (left singular vectors) with orthogonal-like initialization
        let std_a = (1.0 / out_features as f64).sqrt();
        let lora_a = Tensor::randn(0.0f32, std_a as f32, (out_features, config.init_r), device)?;

        // Initialize E (singular values) to small values
        let lora_e = Tensor::ones(config.init_r, dtype, device)?;
        let lora_e = lora_e.broadcast_mul(&Tensor::new(0.01f32, device)?)?;

        // Initialize B (right singular vectors) with orthogonal-like initialization
        let std_b = (1.0 / in_features as f64).sqrt();
        let lora_b = Tensor::randn(0.0f32, std_b as f32, (config.init_r, in_features), device)?;

        // Initialize rank mask to all ones (all ranks active)
        let rank_mask = Tensor::ones(config.init_r, dtype, device)?;

        let init_r = config.init_r;

        Ok(Self {
            lora_a,
            lora_e,
            lora_b,
            scaling,
            config,
            in_features,
            out_features,
            current_rank: init_r,
            rank_mask,
            frozen: false,
        })
    }

    /// Get the current active rank.
    #[must_use]
    pub fn current_rank(&self) -> usize {
        self.current_rank
    }

    /// Rank mask (`1` = kept singular value, `0` = pruned). Shape: `[init_r]`.
    #[must_use]
    pub fn rank_mask(&self) -> &Tensor {
        &self.rank_mask
    }

    /// Get the target rank.
    #[must_use]
    pub fn target_rank(&self) -> usize {
        self.config.target_r
    }

    /// Get the initial rank.
    #[must_use]
    pub fn init_rank(&self) -> usize {
        self.config.init_r
    }

    /// Get the scaling factor.
    #[must_use]
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Update the rank mask by keeping the **top-`budget`** singular values
    /// ranked by `importance_scores` (PEFT-P1-02).
    ///
    /// Ties are broken by ascending index after a host-side sort (`init_r` is
    /// small — typically ≤ 64). Exactly `budget` entries are kept when
    /// `0 < budget < init_r`.
    ///
    /// # Arguments
    /// * `importance_scores` - Importance score for each rank (length: `init_r`)
    /// * `budget` - Number of ranks to keep
    ///
    /// # Errors
    /// Returns error if tensor operations fail or score length mismatches `init_r`.
    pub fn update_rank_mask(&mut self, importance_scores: &Tensor, budget: usize) -> Result<()> {
        let device = importance_scores.device();
        let scores = importance_scores.flatten_all()?;
        let n = scores.elem_count();
        if n != self.config.init_r {
            return Err(PeftError::ShapeMismatch {
                expected: vec![self.config.init_r],
                actual: vec![n],
            });
        }

        if budget >= self.config.init_r {
            self.rank_mask = Tensor::ones(self.config.init_r, DType::F32, device)?;
            self.current_rank = self.config.init_r;
            return Ok(());
        }
        if budget == 0 {
            self.rank_mask = Tensor::zeros(self.config.init_r, DType::F32, device)?;
            self.current_rank = 0;
            return Ok(());
        }

        // Host-side top-k: exact budget, stable for small init_r.
        let values: Vec<f32> = scores.to_vec1()?;
        let mut idxs: Vec<usize> = (0..values.len()).collect();
        idxs.sort_by(|&i, &j| match values[j].partial_cmp(&values[i]) {
            Some(o) => o.then_with(|| i.cmp(&j)),
            None => i.cmp(&j),
        });
        let mut mask = vec![0.0f32; values.len()];
        for &i in idxs.iter().take(budget) {
            mask[i] = 1.0;
        }
        self.rank_mask = Tensor::new(mask.as_slice(), device)?;
        self.current_rank = budget;
        Ok(())
    }

    /// Cubic rank budget schedule from AdaLoRA (arxiv:2303.10512).
    ///
    /// - `step < tinit`: full `init_r`
    /// - `step >= total_step - tfinal`: `target_r`
    /// - otherwise: cubic interpolate from `init_r` → `target_r`
    #[must_use]
    pub fn rank_budget_at_step(&self, step: usize) -> usize {
        let tinit = self.config.tinit;
        let tfinal = self.config.tfinal;
        let total = self.config.total_step;
        let init_r = self.config.init_r;
        let target_r = self.config.target_r;

        if step < tinit {
            return init_r;
        }
        if total <= tfinal || step >= total.saturating_sub(tfinal) {
            return target_r;
        }
        // Progress in (0, 1] over the budgeting phase.
        let denom = (total - tfinal - tinit) as f64;
        if denom <= 0.0 {
            return target_r;
        }
        let t = (step - tinit) as f64 / denom;
        // Cubic: budget = target + (init - target) * (1 - t)^3
        let frac = (1.0 - t).max(0.0).powi(3);
        let budget = target_r as f64 + (init_r as f64 - target_r as f64) * frac;
        budget.round().clamp(target_r as f64, init_r as f64) as usize
    }

    /// Apply the schedule at `step`: recompute top-k mask from current importance.
    ///
    /// Mid-phase updates honor `delta_t` (plus the first budgeting step).
    /// Warmup / final phases always refresh so rank matches schedule endpoints.
    ///
    /// # Errors
    /// Propagates importance / mask errors.
    pub fn update_rank_from_schedule(&mut self, step: usize) -> Result<()> {
        let budget = self.rank_budget_at_step(step);
        let scores = self.get_importance_scores()?;
        let in_budget_phase = step >= self.config.tinit
            && step < self.config.total_step.saturating_sub(self.config.tfinal);
        if in_budget_phase
            && self.config.delta_t > 1
            && !step.is_multiple_of(self.config.delta_t)
            && step != self.config.tinit
        {
            return Ok(());
        }
        self.update_rank_mask(&scores, budget)
    }

    /// Compute the orthogonal regularization loss.
    ///
    /// Encourages P^T P ≈ I and Q Q^T ≈ I.
    ///
    /// # Errors
    /// Returns error if tensor operations fail.
    pub fn orthogonal_regularization(&self) -> Result<Tensor> {
        // P^T P - I
        let pta = self.lora_a.t()?.matmul(&self.lora_a)?;
        let eye_a = Tensor::eye(self.config.init_r, DType::F32, self.lora_a.device())?;
        let orth_loss_a = pta.broadcast_sub(&eye_a)?.sqr()?.sum_all()?;

        // Q Q^T - I
        let bbt = self.lora_b.matmul(&self.lora_b.t()?)?;
        let eye_b = Tensor::eye(self.config.init_r, DType::F32, self.lora_b.device())?;
        let orth_loss_b = bbt.broadcast_sub(&eye_b)?.sqr()?.sum_all()?;

        Ok(orth_loss_a.broadcast_add(&orth_loss_b)?)
    }

    /// Get the importance scores for rank allocation.
    ///
    /// The importance is based on the magnitude of singular values.
    ///
    /// # Errors
    /// Returns error if tensor operations fail.
    pub fn get_importance_scores(&self) -> Result<Tensor> {
        // Simple importance: absolute value of singular values
        Ok(self.lora_e.abs()?)
    }

    /// Replace singular values `E` (length `init_r`) — useful for tests and
    /// advanced schedulers that write importance-driven values.
    ///
    /// # Errors
    /// Shape mismatch if `values` is not `[init_r]`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn set_singular_values(&mut self, values: Tensor) -> Result<()> {
        let n = values.flatten_all()?.elem_count();
        if n != self.config.init_r {
            return Err(PeftError::ShapeMismatch {
                expected: vec![self.config.init_r],
                actual: vec![n],
            });
        }
        self.lora_e = values.flatten_all()?;
        Ok(())
    }
}

impl Adapter for AdaLoraLayer {
    type Config = AdaLoraConfig;

    fn forward(&self, input: &Tensor, base_output: Option<&Tensor>) -> Result<Tensor> {
        // AdaLoRA forward: x @ B^T @ diag(E * mask) @ A^T * scaling
        // Input shape: [batch, seq, in_features]
        // B shape: [init_r, in_features], B^T shape: [in_features, init_r]
        // A shape: [out_features, init_r], A^T shape: [init_r, out_features]

        // For batched matmul, we need to handle the 3D tensor
        // Use the last dimension for matmul
        let input_dims = input.dims();

        // First: input @ B^T -> [batch, seq, init_r]
        // Reshape input to [batch*seq, in_features] for matmul
        let batch_seq = input_dims[0] * input_dims[1];
        let input_2d = input.reshape((batch_seq, self.in_features))?;

        // Compute input_2d @ B^T = [batch*seq, in_features] @ [in_features, init_r] = [batch*seq, init_r]
        let out = input_2d.matmul(&self.lora_b.t()?)?;

        // Apply singular values with mask: [batch*seq, init_r] * [init_r]
        let masked_e = self.lora_e.broadcast_mul(&self.rank_mask)?;
        let masked_e = masked_e.reshape((1, self.config.init_r))?;
        let out = out.broadcast_mul(&masked_e)?;

        // Then: @ A^T -> [batch*seq, out_features]
        // A^T shape: [init_r, out_features]
        let out = out.matmul(&self.lora_a.t()?)?;

        // Reshape back to [batch, seq, out_features]
        let out = out.reshape((input_dims[0], input_dims[1], self.out_features))?;

        // Apply scaling
        let scaling = Tensor::new(self.scaling as f32, out.device())?;
        let out = out.broadcast_mul(&scaling)?;

        // Add to base output if provided
        match base_output {
            Some(base) => Ok(base.broadcast_add(&out)?),
            None => Ok(out),
        }
    }

    fn num_parameters(&self) -> usize {
        // A: out_features × init_r
        // E: init_r
        // B: init_r × in_features
        self.out_features * self.config.init_r
            + self.config.init_r
            + self.config.init_r * self.in_features
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Mergeable for AdaLoraLayer {
    fn merge(&self, base_weight: &Tensor) -> Result<Tensor> {
        // ΔW = A @ diag(E * mask) @ B * scaling
        // Apply mask to singular values
        let masked_e = self.lora_e.broadcast_mul(&self.rank_mask)?;

        // Compute A @ diag(E) = A * E (broadcast along columns)
        let masked_e_col = masked_e.reshape((self.config.init_r, 1))?;
        let ae = self.lora_a.broadcast_mul(&masked_e_col.t()?)?;

        // Then @ B
        let delta_w = ae.matmul(&self.lora_b)?;

        // Apply scaling
        let scaling = Tensor::new(self.scaling as f32, delta_w.device())?;
        let delta_w = delta_w.broadcast_mul(&scaling)?;

        Ok(base_weight.broadcast_add(&delta_w)?)
    }

    fn unmerge(&self, merged_weight: &Tensor) -> Result<Tensor> {
        let masked_e = self.lora_e.broadcast_mul(&self.rank_mask)?;

        let masked_e_col = masked_e.reshape((self.config.init_r, 1))?;
        let ae = self.lora_a.broadcast_mul(&masked_e_col.t()?)?;
        let delta_w = ae.matmul(&self.lora_b)?;

        let scaling = Tensor::new(self.scaling as f32, delta_w.device())?;
        let delta_w = delta_w.broadcast_mul(&scaling)?;

        Ok(merged_weight.broadcast_sub(&delta_w)?)
    }
}

impl Trainable for AdaLoraLayer {
    fn register_parameters(&self, _var_map: &mut VarMap, _prefix: &str) -> Result<()> {
        // Note: In the current design, tensors are created directly.
        // For full training support, tensors should be created via VarBuilder
        // during construction, which automatically registers them.
        // This is a simplified implementation suitable for inference.
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

impl SaveLoad for AdaLoraLayer {
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        state_dict.insert("lora_e".to_string(), self.lora_e.clone());
        state_dict.insert("lora_a".to_string(), self.lora_a.clone());
        state_dict.insert("lora_b".to_string(), self.lora_b.clone());
        state_dict.insert("rank_mask".to_string(), self.rank_mask.clone());
        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        if let Some(t) = state_dict.get("lora_e") {
            self.lora_e = t.clone();
        }
        if let Some(t) = state_dict.get("lora_a") {
            self.lora_a = t.clone();
        }
        if let Some(t) = state_dict.get("lora_b") {
            self.lora_b = t.clone();
        }
        if let Some(t) = state_dict.get("rank_mask") {
            self.rank_mask = t.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adalora_config_default() {
        let config = AdaLoraConfig::default();
        assert_eq!(config.target_r, 8);
        assert_eq!(config.init_r, 12);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_adalora_config_invalid_rank() {
        let config = AdaLoraConfig {
            target_r: 16,
            init_r: 8, // target > init is invalid
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_adalora_config_invalid_schedule() {
        let config = AdaLoraConfig {
            tinit: 500,
            tfinal: 600,
            total_step: 1000, // tinit >= total_step - tfinal
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_adalora_layer_creation() {
        let config = AdaLoraConfig::default();
        let device = Device::Cpu;
        let layer = AdaLoraLayer::new(768, 768, config, &device);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.init_rank(), 12);
        assert_eq!(layer.target_rank(), 8);
        assert_eq!(layer.current_rank(), 12);
    }

    #[test]
    fn test_adalora_forward_shape() {
        let config = AdaLoraConfig::default();
        let device = Device::Cpu;
        let layer = AdaLoraLayer::new(768, 768, config, &device).unwrap();

        let input = Tensor::zeros(&[1, 10, 768], DType::F32, &device).unwrap();
        let output = layer.forward(&input, None).unwrap();

        assert_eq!(output.shape().dims(), &[1, 10, 768]);
    }

    #[test]
    fn test_adalora_num_parameters() {
        let config = AdaLoraConfig {
            init_r: 12,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = AdaLoraLayer::new(768, 768, config, &device).unwrap();

        // A: 768 × 12 = 9216
        // E: 12
        // B: 12 × 768 = 9216
        // Total: 18444
        assert_eq!(layer.num_parameters(), 768 * 12 + 12 + 12 * 768);
    }

    #[test]
    fn test_adalora_importance_scores() {
        let config = AdaLoraConfig::default();
        let device = Device::Cpu;
        let layer = AdaLoraLayer::new(768, 768, config, &device).unwrap();

        let scores = layer.get_importance_scores().unwrap();
        assert_eq!(scores.dims(), &[12]);
    }

    #[test]
    fn test_adalora_orthogonal_regularization() {
        let config = AdaLoraConfig::default();
        let device = Device::Cpu;
        let layer = AdaLoraLayer::new(64, 64, config, &device).unwrap();

        let orth_loss = layer.orthogonal_regularization().unwrap();
        // Should be a scalar tensor (0-dimensional)
        assert!(orth_loss.dims().is_empty());
    }

    #[test]
    fn test_adalora_topk_mask_exact_budget() {
        let config = AdaLoraConfig {
            init_r: 8,
            target_r: 3,
            total_step: 100,
            tinit: 10,
            tfinal: 10,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut layer = AdaLoraLayer::new(32, 32, config, &device).unwrap();

        // Scores: indices 7,5,3 are top-3
        let scores =
            Tensor::from_slice(&[0.1f32, 0.2, 0.05, 0.9, 0.3, 0.8, 0.4, 1.0], (8,), &device)
                .unwrap();
        layer.update_rank_mask(&scores, 3).unwrap();
        assert_eq!(layer.current_rank(), 3);
        let mask: Vec<f32> = layer.rank_mask().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask.iter().filter(|&&v| v > 0.5).count(), 3);
        // top scores at idx 7, 3, 5
        assert!((mask[7] - 1.0).abs() < 1e-6);
        assert!((mask[3] - 1.0).abs() < 1e-6);
        assert!((mask[5] - 1.0).abs() < 1e-6);
        assert!((mask[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_adalora_rank_budget_schedule() {
        let config = AdaLoraConfig {
            init_r: 12,
            target_r: 4,
            total_step: 100,
            tinit: 10,
            tfinal: 10,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = AdaLoraLayer::new(16, 16, config, &device).unwrap();

        assert_eq!(layer.rank_budget_at_step(0), 12); // warmup
        assert_eq!(layer.rank_budget_at_step(9), 12);
        assert_eq!(layer.rank_budget_at_step(95), 4); // final
        assert_eq!(layer.rank_budget_at_step(100), 4);
        // Mid: strictly between target and init
        let mid = layer.rank_budget_at_step(50);
        assert!((4..=12).contains(&mid), "mid budget={mid}");
    }

    #[test]
    fn test_adalora_update_from_schedule() {
        let config = AdaLoraConfig {
            init_r: 8,
            target_r: 2,
            total_step: 20,
            tinit: 2,
            tfinal: 2,
            delta_t: 1,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut layer = AdaLoraLayer::new(16, 16, config, &device).unwrap();
        // Force distinct singular values for importance ordering
        layer
            .set_singular_values(
                Tensor::from_slice(
                    &[0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                    (8,),
                    &device,
                )
                .unwrap(),
            )
            .unwrap();
        layer.update_rank_from_schedule(0).unwrap(); // warmup → budget 8
        assert_eq!(layer.current_rank(), 8);
        layer.update_rank_from_schedule(19).unwrap(); // final → budget 2
        assert_eq!(layer.current_rank(), 2);
    }
}
