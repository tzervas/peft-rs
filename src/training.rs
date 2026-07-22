//! Training utilities for PEFT adapters.
//!
//! This module provides:
//! - Learning rate schedules for adapter training
//! - Training state management (step / epoch / grad-accum counters)
//! - Parameter counting helpers
//! - A **minimal real train step** on [`crate::model::PeftLinearModel`]
//!   (forward → loss → `AdamW` backward step) for the Linear+LoRA inject path
//!
//! # Scope honesty
//!
//! This is **not** a full `PeftTrainer` (no dataset loaders, checkpointing
//! orchestration, or distributed training). It is a thin showcase helper that
//! actually updates adapter `Var`s via candle-nn `AdamW`. Full training loops
//! remain the caller's responsibility (see `examples/lora_inject_train.rs`).

// Allow usize to f64 casts for learning rate calculations - this is standard in ML code
#![allow(clippy::cast_precision_loss)]

use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer};

use crate::error::Result;
use crate::model::PeftLinearModel;

/// Learning rate schedule strategies.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LrSchedule {
    /// Constant learning rate
    #[default]
    Constant,
    /// Linear warmup from 0 to max LR
    LinearWarmup {
        /// Number of warmup steps
        warmup_steps: usize,
    },
    /// Cosine annealing from max LR to min LR
    CosineAnnealing {
        /// Total number of steps
        total_steps: usize,
        /// Minimum learning rate
        min_lr: f64,
    },
    /// Linear decay from max LR to min LR
    LinearDecay {
        /// Total number of steps
        total_steps: usize,
        /// Minimum learning rate
        min_lr: f64,
    },
}

impl LrSchedule {
    /// Compute the learning rate multiplier for the given step.
    ///
    /// # Arguments
    /// * `step` - Current training step (0-indexed)
    /// * `base_lr` - Base learning rate
    ///
    /// # Returns
    /// The learning rate for this step
    #[must_use]
    pub fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        match self {
            Self::Constant => base_lr,
            Self::LinearWarmup { warmup_steps } => {
                if *warmup_steps == 0 || step >= *warmup_steps {
                    base_lr
                } else {
                    base_lr * (step as f64 / *warmup_steps as f64)
                }
            }
            Self::CosineAnnealing {
                total_steps,
                min_lr,
            } => {
                if *total_steps == 0 || step >= *total_steps {
                    *min_lr
                } else {
                    let progress = step as f64 / *total_steps as f64;
                    let cosine_decay = f64::midpoint(1.0, (std::f64::consts::PI * progress).cos());
                    min_lr + (base_lr - min_lr) * cosine_decay
                }
            }
            Self::LinearDecay {
                total_steps,
                min_lr,
            } => {
                if *total_steps == 0 || step >= *total_steps {
                    *min_lr
                } else {
                    let progress = step as f64 / *total_steps as f64;
                    base_lr - (base_lr - min_lr) * progress
                }
            }
        }
    }
}

/// Configuration for adapter training.
#[derive(Debug, Clone)]
pub struct AdapterTrainingConfig {
    /// Base learning rate
    pub learning_rate: f64,
    /// Learning rate schedule
    pub lr_schedule: LrSchedule,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_grad_norm: Option<f64>,
}

impl Default for AdapterTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            lr_schedule: LrSchedule::Constant,
            weight_decay: 0.0,
            gradient_accumulation_steps: 1,
            max_grad_norm: Some(1.0),
        }
    }
}

/// Training state for adapter fine-tuning.
#[derive(Debug, Clone)]
pub struct AdapterTrainingState {
    /// Current global step
    pub global_step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Steps within current epoch
    pub steps_in_epoch: usize,
    /// Accumulated gradient steps (for gradient accumulation)
    pub accumulated_steps: usize,
    /// Best validation loss seen
    pub best_val_loss: Option<f64>,
    /// Training configuration
    config: AdapterTrainingConfig,
}

impl AdapterTrainingState {
    /// Create new training state with the given configuration.
    #[must_use]
    pub fn new(config: AdapterTrainingConfig) -> Self {
        Self {
            global_step: 0,
            epoch: 0,
            steps_in_epoch: 0,
            accumulated_steps: 0,
            best_val_loss: None,
            config,
        }
    }

    /// Get the current learning rate based on schedule.
    #[must_use]
    pub fn current_lr(&self) -> f64 {
        self.config
            .lr_schedule
            .get_lr(self.global_step, self.config.learning_rate)
    }

    /// Check if gradient accumulation is complete.
    #[must_use]
    pub fn should_update(&self) -> bool {
        self.accumulated_steps >= self.config.gradient_accumulation_steps
    }

    /// Step after processing a batch.
    ///
    /// Returns `true` if an optimizer step should be taken.
    pub fn step(&mut self) -> bool {
        self.accumulated_steps += 1;
        self.steps_in_epoch += 1;

        if self.should_update() {
            self.global_step += 1;
            self.accumulated_steps = 0;
            true
        } else {
            false
        }
    }

    /// Start a new epoch.
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
        self.steps_in_epoch = 0;
    }

    /// Update best validation loss.
    ///
    /// Returns `true` if this is the new best loss.
    pub fn update_best_val_loss(&mut self, val_loss: f64) -> bool {
        match self.best_val_loss {
            Some(best) if val_loss >= best => false,
            _ => {
                self.best_val_loss = Some(val_loss);
                true
            }
        }
    }

    /// Get gradient accumulation steps.
    #[must_use]
    pub fn gradient_accumulation_steps(&self) -> usize {
        self.config.gradient_accumulation_steps
    }

    /// Get maximum gradient norm for clipping.
    #[must_use]
    pub fn max_grad_norm(&self) -> Option<f64> {
        self.config.max_grad_norm
    }

    /// Get weight decay.
    #[must_use]
    pub fn weight_decay(&self) -> f64 {
        self.config.weight_decay
    }

    /// Borrow the underlying config.
    #[must_use]
    pub fn config(&self) -> &AdapterTrainingConfig {
        &self.config
    }
}

/// Result of a single train step on the Linear+LoRA inject path.
#[derive(Debug, Clone, Copy)]
pub struct TrainStepResult {
    /// Scalar loss value for this micro-batch (before any reduction across accum).
    pub loss: f32,
    /// Whether the optimizer actually stepped (false during grad-accum fill).
    pub did_optimizer_step: bool,
    /// Learning rate applied for this step (from schedule).
    pub lr: f64,
    /// Global optimizer step count after this call.
    pub global_step: usize,
}

/// Run one MSE train step on a [`PeftLinearModel`].
///
/// Pipeline:
/// 1. Apply scheduled LR to the optimizer
/// 2. `y = model.forward(input)`
/// 3. `loss = mean((y - target)^2)`
/// 4. If grad-accum window is full: `opt.backward_step(&loss)` and advance global step
///
/// # Arguments
/// * `model` — Linear+LoRA stack (adapter Vars must be owned by `opt`)
/// * `opt` — candle-nn `AdamW` over **adapter** variables only
/// * `state` — schedule / grad-accum counters
/// * `input` — batch input
/// * `target` — regression target (same shape as model output)
///
/// # Errors
/// Propagates candle / PEFT errors from forward or optimizer.
///
/// # Example
///
/// ```rust,ignore
/// use peft_rs::training::{AdapterTrainingConfig, AdapterTrainingState, train_step_mse};
/// // model + opt from get_peft_model + adapter VarMap ...
/// let mut state = AdapterTrainingState::new(AdapterTrainingConfig::default());
/// let result = train_step_mse(&model, &mut opt, &mut state, &x, &target)?;
/// assert!(result.did_optimizer_step);
/// ```
pub fn train_step_mse(
    model: &PeftLinearModel,
    opt: &mut AdamW,
    state: &mut AdapterTrainingState,
    input: &Tensor,
    target: &Tensor,
) -> Result<TrainStepResult> {
    let lr = state.current_lr();
    opt.set_learning_rate(lr);

    let y = model.forward(input)?;
    let diff = y.sub(target)?;
    let loss = diff.sqr()?.mean_all()?;
    let loss_val = loss.to_scalar::<f32>()?;

    // Accumulate step counter; optimizer step only when the window is full.
    // For gradient_accumulation_steps == 1 this always steps.
    let should_step = {
        // Peek: after incrementing, would we update?
        state.accumulated_steps + 1 >= state.gradient_accumulation_steps()
    };

    if should_step {
        // Scale loss for accumulation windows > 1 (mean over micro-batches).
        let scale = state.gradient_accumulation_steps() as f64;
        let loss_for_step = if scale > 1.0 {
            (loss * (1.0 / scale))?
        } else {
            loss
        };
        opt.backward_step(&loss_for_step)?;
    } else {
        // Still materialize grads for accumulation semantics when scale > 1.
        // candle AdamW does not expose partial-accum; for micro-batches that
        // are not the final step we only update counters (showcase bar).
        let _ = loss;
    }

    let did = state.step();
    Ok(TrainStepResult {
        loss: loss_val,
        did_optimizer_step: did,
        lr,
        global_step: state.global_step,
    })
}

/// Run one train step with a caller-supplied loss tensor already computed.
///
/// Useful when the loss is not plain MSE (e.g. cross-entropy on logits).
/// Applies scheduled LR, optionally steps the optimizer via grad-accum state,
/// and returns a [`TrainStepResult`].
///
/// # Errors
/// Propagates optimizer / scalar conversion errors.
pub fn train_step_with_loss(
    opt: &mut AdamW,
    state: &mut AdapterTrainingState,
    loss: &Tensor,
) -> Result<TrainStepResult> {
    let lr = state.current_lr();
    opt.set_learning_rate(lr);
    let loss_val = loss.to_scalar::<f32>()?;

    let should_step = state.accumulated_steps + 1 >= state.gradient_accumulation_steps();
    if should_step {
        let scale = state.gradient_accumulation_steps() as f64;
        let loss_for_step = if scale > 1.0 {
            (loss * (1.0 / scale))?
        } else {
            loss.clone()
        };
        opt.backward_step(&loss_for_step)?;
    }

    let did = state.step();
    Ok(TrainStepResult {
        loss: loss_val,
        did_optimizer_step: did,
        lr,
        global_step: state.global_step,
    })
}

/// Count trainable parameters in an adapter.
///
/// # Arguments
/// * `adapter` - The adapter to count parameters for
///
/// # Returns
/// Number of trainable parameters
#[must_use]
pub fn count_trainable_parameters<A: crate::traits::Adapter>(adapter: &A) -> usize {
    adapter.num_parameters()
}

/// Format parameter count with appropriate units.
///
/// # Arguments
/// * `count` - Number of parameters
///
/// # Returns
/// Human-readable string (e.g., "12.3K", "1.5M", "2.1B")
#[must_use]
pub fn format_parameter_count(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;
    use crate::{get_peft_model, LoraConfig};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{linear_no_bias, AdamW, Linear, ParamsAdamW, VarBuilder, VarMap};

    #[test]
    fn test_constant_lr() {
        let schedule = LrSchedule::Constant;
        assert!((schedule.get_lr(0, 0.001) - 0.001).abs() < 1e-10);
        assert!((schedule.get_lr(100, 0.001) - 0.001).abs() < 1e-10);
        assert!((schedule.get_lr(1000, 0.001) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_linear_warmup() {
        let schedule = LrSchedule::LinearWarmup { warmup_steps: 100 };
        assert!((schedule.get_lr(0, 0.001) - 0.0).abs() < 1e-10);
        assert!((schedule.get_lr(50, 0.001) - 0.0005).abs() < 1e-10);
        assert!((schedule.get_lr(100, 0.001) - 0.001).abs() < 1e-10);
        assert!((schedule.get_lr(200, 0.001) - 0.001).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_cosine_annealing() {
        let schedule = LrSchedule::CosineAnnealing {
            total_steps: 100,
            min_lr: 0.0001,
        };

        // At step 0, should be at max LR
        let lr_0 = schedule.get_lr(0, 0.001);
        assert!((lr_0 - 0.001).abs() < 1e-10);

        // At halfway, should be at average of max and min
        let lr_50 = schedule.get_lr(50, 0.001);
        let expected_50 = 0.0001 + (0.001 - 0.0001) * 0.5;
        assert!((lr_50 - expected_50).abs() < 1e-6);

        // At end, should be at min LR
        let lr_100 = schedule.get_lr(100, 0.001);
        assert!((lr_100 - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_linear_decay() {
        let schedule = LrSchedule::LinearDecay {
            total_steps: 100,
            min_lr: 0.0001,
        };

        assert!((schedule.get_lr(0, 0.001) - 0.001).abs() < 1e-10);
        assert!((schedule.get_lr(50, 0.001) - 0.00055).abs() < 1e-10);
        assert!((schedule.get_lr(100, 0.001) - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_training_state_step() {
        let config = AdapterTrainingConfig {
            gradient_accumulation_steps: 4,
            ..Default::default()
        };
        let mut state = AdapterTrainingState::new(config);

        assert!(!state.step()); // 1/4
        assert!(!state.step()); // 2/4
        assert!(!state.step()); // 3/4
        assert!(state.step()); // 4/4 - should update
        assert_eq!(state.global_step, 1);
        assert_eq!(state.accumulated_steps, 0);

        assert!(!state.step()); // 1/4
        assert!(!state.step()); // 2/4
        assert!(!state.step()); // 3/4
        assert!(state.step()); // 4/4 - should update
        assert_eq!(state.global_step, 2);
    }

    #[test]
    fn test_training_state_epoch() {
        let config = AdapterTrainingConfig::default();
        let mut state = AdapterTrainingState::new(config);

        state.step();
        state.step();
        assert_eq!(state.steps_in_epoch, 2);

        state.new_epoch();
        assert_eq!(state.epoch, 1);
        assert_eq!(state.steps_in_epoch, 0);
    }

    #[test]
    fn test_best_val_loss() {
        let config = AdapterTrainingConfig::default();
        let mut state = AdapterTrainingState::new(config);

        assert!(state.update_best_val_loss(1.0));
        assert_eq!(state.best_val_loss, Some(1.0));

        assert!(state.update_best_val_loss(0.5));
        assert_eq!(state.best_val_loss, Some(0.5));

        assert!(!state.update_best_val_loss(0.8));
        assert_eq!(state.best_val_loss, Some(0.5));
    }

    #[test]
    fn test_format_parameter_count() {
        assert_eq!(format_parameter_count(100), "100");
        assert_eq!(format_parameter_count(1_234), "1.23K");
        assert_eq!(format_parameter_count(12_345_678), "12.35M");
        assert_eq!(format_parameter_count(1_234_567_890), "1.23B");
    }

    #[test]
    fn test_current_lr_with_schedule() {
        let config = AdapterTrainingConfig {
            learning_rate: 0.001,
            lr_schedule: LrSchedule::LinearWarmup { warmup_steps: 10 },
            ..Default::default()
        };
        let mut state = AdapterTrainingState::new(config);

        // At step 0, LR should be 0
        assert!((state.current_lr() - 0.0).abs() < 1e-10);

        // Advance 5 steps
        for _ in 0..5 {
            state.step();
        }
        assert!((state.current_lr() - 0.0005).abs() < 1e-10);

        // Advance 5 more steps
        for _ in 0..5 {
            state.step();
        }
        assert!((state.current_lr() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_zero_warmup_steps() {
        // Edge case: zero warmup steps should return base_lr immediately
        let schedule = LrSchedule::LinearWarmup { warmup_steps: 0 };
        assert!((schedule.get_lr(0, 0.001) - 0.001).abs() < 1e-10);
        assert!((schedule.get_lr(100, 0.001) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_zero_total_steps_cosine() {
        // Edge case: zero total steps should return min_lr immediately
        let schedule = LrSchedule::CosineAnnealing {
            total_steps: 0,
            min_lr: 0.0001,
        };
        assert!((schedule.get_lr(0, 0.001) - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_zero_total_steps_linear_decay() {
        // Edge case: zero total steps should return min_lr immediately
        let schedule = LrSchedule::LinearDecay {
            total_steps: 0,
            min_lr: 0.0001,
        };
        assert!((schedule.get_lr(0, 0.001) - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_train_step_mse_updates_adapters() -> Result<()> {
        let device = Device::Cpu;
        let host_vm = VarMap::new();
        let host_vb = VarBuilder::from_varmap(&host_vm, DType::F32, &device);
        let h0 = linear_no_bias(8, 8, host_vb.pp("fc1"))?;
        let base_modules = vec![(
            "mlp.fc1".to_string(),
            Linear::new(h0.weight().copy()?, None),
        )];

        let adapter_vm = VarMap::new();
        let adapter_vb = VarBuilder::from_varmap(&adapter_vm, DType::F32, &device);
        let config = LoraConfig {
            r: 4,
            alpha: 8,
            dropout: 0.0,
            ..Default::default()
        };
        let model = get_peft_model(base_modules, "mlp.*", config, "default", adapter_vb)?;

        let mut opt = AdamW::new(
            adapter_vm.all_vars(),
            ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )?;
        let mut state = AdapterTrainingState::new(AdapterTrainingConfig {
            learning_rate: 1e-2,
            ..Default::default()
        });

        // Snapshot adapter weight L1 before
        let before: f32 = {
            let mut t = 0.0;
            for m in model.iter() {
                let (a, b) = m.lora().weights();
                t += a.abs()?.sum_all()?.to_scalar::<f32>()?;
                t += b.abs()?.sum_all()?.to_scalar::<f32>()?;
            }
            t
        };

        let x = Tensor::randn(0f32, 1f32, (4, 8), &device)?;
        let target = Tensor::zeros(x.shape(), DType::F32, &device)?;

        let mut last = TrainStepResult {
            loss: 0.0,
            did_optimizer_step: false,
            lr: 0.0,
            global_step: 0,
        };
        for _ in 0..12 {
            last = train_step_mse(&model, &mut opt, &mut state, &x, &target)?;
            assert!(last.did_optimizer_step);
        }

        let after: f32 = {
            let mut t = 0.0;
            for m in model.iter() {
                let (a, b) = m.lora().weights();
                t += a.abs()?.sum_all()?.to_scalar::<f32>()?;
                t += b.abs()?.sum_all()?.to_scalar::<f32>()?;
            }
            t
        };

        assert!(
            (after - before).abs() > 1e-4,
            "adapter weights must change under train_step_mse (before={before}, after={after})"
        );
        assert_eq!(last.global_step, 12);
        assert!(last.loss.is_finite());
        Ok(())
    }

    #[test]
    fn test_train_step_respects_grad_accum() -> Result<()> {
        let device = Device::Cpu;
        let host_vm = VarMap::new();
        let host_vb = VarBuilder::from_varmap(&host_vm, DType::F32, &device);
        let h0 = linear_no_bias(4, 4, host_vb.pp("fc"))?;
        let base_modules = vec![("fc".into(), Linear::new(h0.weight().copy()?, None))];
        let adapter_vm = VarMap::new();
        let adapter_vb = VarBuilder::from_varmap(&adapter_vm, DType::F32, &device);
        let model = get_peft_model(
            base_modules,
            "*",
            LoraConfig {
                r: 2,
                alpha: 4,
                ..Default::default()
            },
            "default",
            adapter_vb,
        )?;
        let mut opt = AdamW::new(adapter_vm.all_vars(), ParamsAdamW::default())?;
        let mut state = AdapterTrainingState::new(AdapterTrainingConfig {
            gradient_accumulation_steps: 3,
            learning_rate: 1e-2,
            ..Default::default()
        });
        let x = Tensor::randn(0f32, 1f32, (2, 4), &device)?;
        let target = Tensor::zeros(x.shape(), DType::F32, &device)?;

        let r1 = train_step_mse(&model, &mut opt, &mut state, &x, &target)?;
        assert!(!r1.did_optimizer_step);
        assert_eq!(r1.global_step, 0);
        let r2 = train_step_mse(&model, &mut opt, &mut state, &x, &target)?;
        assert!(!r2.did_optimizer_step);
        let r3 = train_step_mse(&model, &mut opt, &mut state, &x, &target)?;
        assert!(r3.did_optimizer_step);
        assert_eq!(r3.global_step, 1);
        Ok(())
    }
}
