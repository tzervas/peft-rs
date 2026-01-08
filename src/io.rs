//! I/O utilities for saving and loading adapter weights and configurations.
//!
//! This module provides functionality for:
//! - Saving adapter weights to safetensors format
//! - Loading adapter weights from safetensors format
//! - Saving adapter configurations to JSON
//! - Loading adapter configurations from JSON

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{Device, Tensor};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::{PeftError, Result};

/// Trait for adapters that can be saved and loaded.
pub trait SaveLoad {
    /// Get all adapter tensors as a map of name -> tensor.
    fn state_dict(&self) -> Result<HashMap<String, Tensor>>;

    /// Load adapter tensors from a state dict.
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()>;
}

/// Save adapter weights to a safetensors file.
///
/// # Arguments
/// * `adapter` - The adapter implementing SaveLoad trait
/// * `path` - Path to save the safetensors file
///
/// # Errors
/// Returns an error if:
/// - Failed to get state dict from adapter
/// - Failed to serialize tensors to safetensors format
/// - Failed to write file to disk
pub fn save_adapter_weights<P: AsRef<Path>>(
    adapter: &dyn SaveLoad,
    path: P,
) -> Result<()> {
    let state_dict = adapter.state_dict()?;
    
    // Convert HashMap to Vec for safetensors
    let tensors: Vec<(&str, Tensor)> = state_dict
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor.clone()))
        .collect();
    
    // Use candle's built-in safetensors serialization
    safetensors::tensor::serialize_to_file(tensors, &None, path.as_ref())
        .map_err(|e| PeftError::Io(format!("Failed to save safetensors: {e}")))?;
    
    Ok(())
}

/// Load adapter weights from a safetensors file.
///
/// # Arguments
/// * `adapter` - The adapter to load weights into
/// * `path` - Path to the safetensors file
/// * `device` - Device to load tensors on
///
/// # Errors
/// Returns an error if:
/// - Failed to read file from disk
/// - Failed to parse safetensors format
/// - Failed to load tensors into adapter
pub fn load_adapter_weights<P: AsRef<Path>>(
    adapter: &mut dyn SaveLoad,
    path: P,
    device: &Device,
) -> Result<()> {
    // Use candle's built-in safetensors loading
    let tensors = candle_core::safetensors::load(path.as_ref(), device)?;
    
    // Load into adapter
    adapter.load_state_dict(tensors)?;
    
    Ok(())
}

/// Save adapter configuration to a JSON file.
///
/// # Arguments
/// * `config` - The configuration to save
/// * `path` - Path to save the JSON file
///
/// # Errors
/// Returns an error if serialization or file writing fails
pub fn save_adapter_config<T: Serialize, P: AsRef<Path>>(
    config: &T,
    path: P,
) -> Result<()> {
    let json = serde_json::to_string_pretty(config)
        .map_err(|e| PeftError::Io(format!("Failed to serialize config: {e}")))?;
    
    fs::write(path, json)
        .map_err(|e| PeftError::Io(format!("Failed to write config file: {e}")))?;
    
    Ok(())
}

/// Load adapter configuration from a JSON file.
///
/// # Arguments
/// * `path` - Path to the JSON file
///
/// # Errors
/// Returns an error if file reading or deserialization fails
pub fn load_adapter_config<T: DeserializeOwned, P: AsRef<Path>>(
    path: P,
) -> Result<T> {
    let json = fs::read_to_string(path)
        .map_err(|e| PeftError::Io(format!("Failed to read config file: {e}")))?;
    
    let config = serde_json::from_str(&json)
        .map_err(|e| PeftError::Io(format!("Failed to parse config: {e}")))?;
    
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;
    use tempfile::TempDir;

    struct MockAdapter {
        weights: HashMap<String, Tensor>,
    }

    impl SaveLoad for MockAdapter {
        fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
            Ok(self.weights.clone())
        }

        fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
            self.weights = state_dict;
            Ok(())
        }
    }

    #[test]
    fn test_save_load_adapter_weights() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let temp_dir = TempDir::new()?;
        let weights_path = temp_dir.path().join("adapter.safetensors");

        // Create mock adapter with some weights
        let mut weights = HashMap::new();
        weights.insert(
            "lora_a".to_string(),
            Tensor::randn(0f32, 1f32, (64, 8), &device)?,
        );
        weights.insert(
            "lora_b".to_string(),
            Tensor::randn(0f32, 1f32, (8, 64), &device)?,
        );

        let adapter = MockAdapter {
            weights: weights.clone(),
        };

        // Save weights
        save_adapter_weights(&adapter, &weights_path)?;
        assert!(weights_path.exists());

        // Load weights into new adapter
        let mut loaded_adapter = MockAdapter {
            weights: HashMap::new(),
        };
        load_adapter_weights(&mut loaded_adapter, &weights_path, &device)?;

        // Verify loaded weights
        assert_eq!(loaded_adapter.weights.len(), 2);
        assert!(loaded_adapter.weights.contains_key("lora_a"));
        assert!(loaded_adapter.weights.contains_key("lora_b"));

        Ok(())
    }

    #[test]
    fn test_save_load_config() -> anyhow::Result<()> {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        struct TestConfig {
            r: usize,
            alpha: usize,
            dropout: f64,
        }

        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.json");

        let config = TestConfig {
            r: 8,
            alpha: 16,
            dropout: 0.1,
        };

        // Save config
        save_adapter_config(&config, &config_path)?;
        assert!(config_path.exists());

        // Load config
        let loaded_config: TestConfig = load_adapter_config(&config_path)?;
        assert_eq!(config, loaded_config);

        Ok(())
    }
}
