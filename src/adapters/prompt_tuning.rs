//! Prompt Tuning implementation (**experimental**).
//!
//! Prompt tuning prepends learnable "soft prompt" embeddings to the input,
//! allowing the model to be steered without modifying weights.
//!
//! # Experimental status (PEFT-P1-03)
//!
//! - Critical path: random init + [`PromptTuningLayer::prepend_to_input`]
//! - [`PromptInit::Text`]: **simplified** text seeding without a tokenizer —
//!   UTF-8 bytes are mapped through a fixed 256-row embedding table and
//!   averaged into virtual-token slots. This is **not** HF peft
//!   `PromptTuningInit.TEXT` (which requires the base model embedding matrix
//!   and a real tokenizer). Documented as experimental; tests match this claim.
//!
//! Reference: <https://arxiv.org/abs/2104.08691>

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{PeftError, Result};
use crate::io::SaveLoad;
use crate::traits::{Adapter, AdapterConfig};

/// Configuration for prompt tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTuningConfig {
    /// Number of virtual tokens (soft prompt length).
    pub num_virtual_tokens: usize,

    /// Hidden size of the model embeddings.
    pub hidden_size: usize,

    /// Initialization strategy.
    #[serde(default)]
    pub init_strategy: PromptInit,
}

/// Initialization strategy for soft prompts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum PromptInit {
    /// Random initialization from normal distribution (`N(0, 0.02)`).
    #[default]
    Random,
    /// **Experimental** text seeding without a full tokenizer.
    ///
    /// UTF-8 bytes of the string are projected through a deterministic 256×H
    /// table and folded into `num_virtual_tokens` slots. Not HF TEXT init.
    Text(String),
}

impl Default for PromptTuningConfig {
    fn default() -> Self {
        Self {
            num_virtual_tokens: 20,
            hidden_size: 768,
            init_strategy: PromptInit::Random,
        }
    }
}

impl AdapterConfig for PromptTuningConfig {
    fn validate(&self) -> Result<()> {
        if self.num_virtual_tokens == 0 {
            return Err(PeftError::InvalidConfig(
                "num_virtual_tokens must be > 0".into(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(PeftError::InvalidConfig("hidden_size must be > 0".into()));
        }
        if let PromptInit::Text(s) = &self.init_strategy {
            if s.is_empty() {
                return Err(PeftError::InvalidConfig(
                    "PromptInit::Text string must be non-empty".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Prompt tuning layer (**experimental** helpers + real prepend path).
///
/// Maintains soft prompt embeddings that are prepended to input embeddings.
pub struct PromptTuningLayer {
    /// Soft prompt embeddings: [`num_virtual_tokens`, `hidden_size`]
    soft_prompt: Tensor,
    /// Configuration
    config: PromptTuningConfig,
}

impl PromptTuningLayer {
    /// Create a new prompt tuning layer.
    ///
    /// # Arguments
    /// * `config` - Prompt tuning configuration
    /// * `device` - Device to create tensors on
    ///
    /// # Errors
    ///
    /// Returns an error if configuration validation fails or layer construction fails.
    pub fn new(config: PromptTuningConfig, device: &Device) -> Result<Self> {
        config.validate()?;

        let soft_prompt = match &config.init_strategy {
            PromptInit::Random => Tensor::randn(
                0.0f32,
                0.02,
                (config.num_virtual_tokens, config.hidden_size),
                device,
            )?,
            PromptInit::Text(text) => {
                text_seeded_prompt(text, config.num_virtual_tokens, config.hidden_size, device)?
            }
        };

        Ok(Self {
            soft_prompt,
            config,
        })
    }

    /// Get the soft prompt embeddings.
    #[must_use]
    pub fn soft_prompt(&self) -> &Tensor {
        &self.soft_prompt
    }

    /// Prepend soft prompts to input embeddings.
    ///
    /// # Arguments
    /// * `input_embeds` - Input embeddings `[batch, seq_len, hidden]`
    ///
    /// # Returns
    /// Concatenated embeddings `[batch, num_virtual_tokens + seq_len, hidden]`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn prepend_to_input(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let dims = input_embeds.dims();
        if dims.len() != 3 {
            return Err(PeftError::InvalidConfig(format!(
                "prepend_to_input expects rank-3 [batch, seq, hidden], got {dims:?}"
            )));
        }
        if dims[2] != self.config.hidden_size {
            return Err(PeftError::ShapeMismatch {
                expected: vec![dims[0], dims[1], self.config.hidden_size],
                actual: dims.to_vec(),
            });
        }
        let batch_size = dims[0];

        // Expand soft prompt for batch: [1, num_virtual_tokens, hidden] -> [batch, ...]
        let expanded_prompt = self.soft_prompt.unsqueeze(0)?.expand((
            batch_size,
            self.config.num_virtual_tokens,
            self.config.hidden_size,
        ))?;
        let expanded_prompt = expanded_prompt.contiguous()?;
        let input_embeds = input_embeds.contiguous()?;

        // Concatenate along sequence dimension
        Ok(Tensor::cat(&[&expanded_prompt, &input_embeds], 1)?)
    }
}

/// Deterministic experimental text → soft-prompt init (no tokenizer).
///
/// 1. Build a 256×H embedding table seeded from a fixed LCG over byte values
/// 2. Map each UTF-8 byte to a row
/// 3. Partition bytes across `num_virtual_tokens` slots and mean-pool each slot
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn text_seeded_prompt(
    text: &str,
    num_virtual_tokens: usize,
    hidden_size: usize,
    device: &Device,
) -> Result<Tensor> {
    // Fixed embedding table: row b = deterministic pseudo-random vector from byte b.
    let mut table = vec![0.0f32; 256 * hidden_size];
    for b in 0..256u32 {
        let mut state = b.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
        for h in 0..hidden_size {
            // xorshift-ish → ~N(0, 0.02) via box-scale of uniform
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let u = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            table[b as usize * hidden_size + h] = u * 0.02;
        }
    }

    let bytes = text.as_bytes();
    let mut prompt = vec![0.0f32; num_virtual_tokens * hidden_size];
    if bytes.is_empty() {
        return Err(PeftError::InvalidConfig(
            "text seed produced empty byte sequence".into(),
        ));
    }

    for (slot, chunk) in bytes
        .chunks(bytes.len().div_ceil(num_virtual_tokens))
        .take(num_virtual_tokens)
        .enumerate()
    {
        if chunk.is_empty() {
            continue;
        }
        let inv = 1.0f32 / chunk.len() as f32;
        for &byte in chunk {
            let row = &table[byte as usize * hidden_size..(byte as usize + 1) * hidden_size];
            for h in 0..hidden_size {
                prompt[slot * hidden_size + h] += row[h] * inv;
            }
        }
    }
    // If fewer chunks than slots (short text), fill remaining from cycling bytes.
    let filled = bytes
        .chunks(bytes.len().div_ceil(num_virtual_tokens))
        .take(num_virtual_tokens)
        .count();
    for slot in filled..num_virtual_tokens {
        let byte = bytes[slot % bytes.len()];
        let row = &table[byte as usize * hidden_size..(byte as usize + 1) * hidden_size];
        prompt[slot * hidden_size..(slot + 1) * hidden_size].copy_from_slice(row);
    }

    Ok(Tensor::from_vec(
        prompt,
        (num_virtual_tokens, hidden_size),
        device,
    )?)
}

impl Adapter for PromptTuningLayer {
    type Config = PromptTuningConfig;

    fn forward(&self, input: &Tensor, _base_output: Option<&Tensor>) -> Result<Tensor> {
        self.prepend_to_input(input)
    }

    fn num_parameters(&self) -> usize {
        self.config.num_virtual_tokens * self.config.hidden_size
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl SaveLoad for PromptTuningLayer {
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        state_dict.insert("soft_prompt".to_string(), self.soft_prompt.clone());
        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        if let Some(t) = state_dict.get("soft_prompt") {
            self.soft_prompt = t.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_prompt_tuning_creation() {
        let config = PromptTuningConfig::default();
        let device = Device::Cpu;
        let layer = PromptTuningLayer::new(config, &device);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_prepend_to_input() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 10,
            hidden_size: 768,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = PromptTuningLayer::new(config, &device).unwrap();

        let input = Tensor::zeros(&[2, 20, 768], DType::F32, &device).unwrap();
        let output = layer.prepend_to_input(&input).unwrap();

        // Output should be [2, 10+20, 768] = [2, 30, 768]
        assert_eq!(output.shape().dims(), &[2, 30, 768]);
    }

    #[test]
    fn test_num_parameters() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 20,
            hidden_size: 768,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = PromptTuningLayer::new(config, &device).unwrap();

        assert_eq!(layer.num_parameters(), 20 * 768);
    }

    #[test]
    fn test_text_init_deterministic_and_shape() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 5,
            hidden_size: 32,
            init_strategy: PromptInit::Text("hello peft".into()),
        };
        let device = Device::Cpu;
        let a = PromptTuningLayer::new(config.clone(), &device).unwrap();
        let b = PromptTuningLayer::new(config, &device).unwrap();
        assert_eq!(a.soft_prompt().dims(), &[5, 32]);
        // Same text → same init
        let da = a.soft_prompt().to_vec2::<f32>().unwrap();
        let db = b.soft_prompt().to_vec2::<f32>().unwrap();
        assert_eq!(da, db);
        // Different text → different init
        let other = PromptTuningLayer::new(
            PromptTuningConfig {
                num_virtual_tokens: 5,
                hidden_size: 32,
                init_strategy: PromptInit::Text("different".into()),
            },
            &device,
        )
        .unwrap();
        let dother = other.soft_prompt().to_vec2::<f32>().unwrap();
        assert_ne!(da, dother);
    }

    #[test]
    fn test_text_init_empty_rejected() {
        let config = PromptTuningConfig {
            init_strategy: PromptInit::Text(String::new()),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_forward_is_prepend() {
        let config = PromptTuningConfig {
            num_virtual_tokens: 4,
            hidden_size: 16,
            ..Default::default()
        };
        let device = Device::Cpu;
        let layer = PromptTuningLayer::new(config, &device).unwrap();
        let input = Tensor::zeros(&[1, 8, 16], DType::F32, &device).unwrap();
        let y = layer.forward(&input, None).unwrap();
        assert_eq!(y.dims(), &[1, 12, 16]);
    }
}
