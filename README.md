# peft-rs

Comprehensive PEFT (Parameter-Efficient Fine-Tuning) adapter library for Rust.

[![Crates.io](https://img.shields.io/crates/v/peft-rs.svg)](https://crates.io/crates/peft-rs)
[![Documentation](https://docs.rs/peft-rs/badge.svg)](https://docs.rs/peft-rs)
[![License](https://img.shields.io/crates/l/peft-rs.svg)](LICENSE-MIT)

## Overview

`peft-rs` provides modular implementations of various PEFT methods for fine-tuning large language models efficiently:

- **LoRA** (Low-Rank Adaptation) - Decomposes weight updates into low-rank matrices
- **Prefix Tuning** - Prepends trainable vectors to attention keys/values
- **Prompt Tuning** - Adds learnable soft prompt embeddings

## Features

- 🦀 Pure Rust implementation using [candle](https://github.com/huggingface/candle)
- 🔌 Modular adapter design with common traits
- 📦 Easy integration with existing models
- ⚡ Optional CUDA acceleration
- 📊 Minimal memory overhead

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
peft-rs = "0.1"
```

For CUDA support:

```toml
[dependencies]
peft-rs = { version = "0.1", features = ["cuda"] }
```

## Quick Start

### LoRA Example

```rust
use peft_rs::{LoraConfig, LoraLayer};
use candle_core::{Device, Tensor, DType};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    // Configure LoRA
    let config = LoraConfig {
        r: 8,           // Rank
        alpha: 16,      // Scaling factor
        dropout: 0.0,
        ..Default::default()
    };
    
    // Create LoRA layer for a 768-dim linear layer
    let lora = LoraLayer::new_with_zeros(768, 768, config, &device)?;
    
    // Forward pass
    let input = Tensor::randn(0.0, 1.0, (1, 10, 768), &device)?;
    let base_output = Tensor::zeros(&[1, 10, 768], DType::F32, &device)?;
    let output = lora.forward(&input, Some(&base_output))?;
    
    println!("Output shape: {:?}", output.shape());
    println!("Trainable parameters: {}", lora.num_parameters());
    
    Ok(())
}
```

### Prompt Tuning Example

```rust
use peft_rs::{PromptTuningConfig, PromptTuningLayer};
use candle_core::{Device, Tensor, DType};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    let config = PromptTuningConfig {
        num_virtual_tokens: 20,
        hidden_size: 768,
        ..Default::default()
    };
    
    let prompt_tuning = PromptTuningLayer::new(config, &device)?;
    
    // Prepend soft prompts to input embeddings
    let input_embeds = Tensor::zeros(&[2, 100, 768], DType::F32, &device)?;
    let output = prompt_tuning.prepend_to_input(&input_embeds)?;
    
    // Output: [2, 120, 768] (20 virtual tokens + 100 input tokens)
    println!("Output shape: {:?}", output.shape());
    
    Ok(())
}
```

## Architecture

All adapters implement common traits for consistent usage:

```rust
pub trait Adapter {
    type Config: AdapterConfig;
    
    fn forward(&self, input: &Tensor, base_output: Option<&Tensor>) -> Result<Tensor>;
    fn num_parameters(&self) -> usize;
    fn config(&self) -> &Self::Config;
}

pub trait Mergeable: Adapter {
    fn merge(&self, base_weight: &Tensor) -> Result<Tensor>;
    fn unmerge(&self, merged_weight: &Tensor) -> Result<Tensor>;
}
```

## Comparison with Python PEFT

| Feature | peft-rs | HuggingFace PEFT |
|---------|---------|------------------|
| LoRA | ✅ | ✅ |
| Prefix Tuning | ✅ | ✅ |
| Prompt Tuning | ✅ | ✅ |
| IA³ | 🚧 | ✅ |
| AdaLoRA | 🚧 | ✅ |
| Weight merging | ✅ | ✅ |
| CUDA support | ✅ | ✅ |
| No Python runtime | ✅ | ❌ |

## Contributing

Contributions welcome! Please see the workspace [AGENTS.md](../AGENTS.md) for coding conventions.

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
