# peft-rs

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/peft-rs/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/peft-rs/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/peft-rs/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/peft-rs/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

Candle PEFT **adapter layer library** for Rust.

[![Crates.io](https://img.shields.io/crates/v/peft-rs.svg)](https://crates.io/crates/peft-rs)
[![Documentation](https://docs.rs/peft-rs/badge.svg)](https://docs.rs/peft-rs)
[![License](https://img.shields.io/crates/l/peft-rs.svg)](LICENSE)

> **Honest product class:** modular PEFT *layer math* on
> [candle](https://github.com/huggingface/candle) (forward / merge / safetensors).
> This is **not** a drop-in HuggingFace PEFT framework and does **not** claim full
> Python parity. See [METRICS.md](METRICS.md) and [roadmap.md](roadmap.md).

## What this crate is

- Standalone adapter **layers** (LoRA, DoRA, AdaLoRA, IA³, LoHa, LoKr, OFT, BOFT, VeRA, prefix/prompt tuning)
- Common traits: `Adapter`, `Mergeable`, `Trainable`, `SaveLoad`
- Safetensors + JSON helpers for adapter weights/config (Rust-native keys/schema today)
- Multi-adapter **registry** (single active adapter; no weighted compose yet)

## Non-goals (until later PRs)

| Non-goal | Notes / when |
|----------|----------------|
| Full `PeftModel` base-model injection | Name-list registry only today → PR-041 |
| HuggingFace `adapter_config.json` / weight key interop | Partial filenames only → PR-040 |
| QLoRA / bitsandbytes / GPTQ / AWQ | Out of this crate; see qlora-rs ecosystem |
| PeftTrainer / full training loop | `training` module = LR schedules + counters only |
| Fused CUDA kernels (CubeCL) | **Quarantined** under `src/kernels/archive/` (PR-021) |
| Numerical parity suite vs Python peft | Planned; METRICS.md scaffold only |

## Status matrix (honest)

Legend: **done** = usable layer math / API · **partial** = real code but incomplete vs HF · **missing** = not implemented · **stub** = flag or shell without full behavior

| Feature | Status | Notes |
|---------|--------|-------|
| LoRA linear layer | **partial** | Solid forward/merge/save; no bias / `modules_to_save` / HF keys |
| rsLoRA scaling | **done** | `use_rslora` → `α/√r` |
| LoRA dropout | **done** | Applied in forward when unfrozen and `dropout > 0` |
| DoRA | **partial** | Magnitude/direction; falls back to LoRA without base weight; SaveLoad supported |
| LoftQ init | **stub / simplified** | `loftq_iterations > 0` → dual-Gaussian init; **not** full SVD+quant LoftQ |
| AdaLoRA | **partial** | SVD params + simplified mask (not full top-k budget trainer) |
| IA³ / LoHa / LoKr | **partial** | Linear-shaped layers; limited options vs Python |
| OFT / BOFT | **partial** | Nontrivial math; some options (e.g. BOFT dropout) incomplete |
| VeRA | **partial** | Core frozen-random + scaling |
| Prefix tuning | **partial / thin** | Tensors + getters; reparam unused; no attention inject |
| Prompt tuning | **partial / thin** | Soft prompt prepend; text init unused |
| Weight merge/unmerge | **partial** | Per-layer `Mergeable`; no model-level merge/unload |
| Save/load safetensors | **partial** | Works for implementing adapters; schema ≠ HF peft |
| Multi-adapter registry | **partial** | Switch active; no weighted compose |
| `PeftModel` / `get_peft_model` | **stub** | Module-name map, not base-model wrap |
| Trainable freeze | **partial** | Layer flag; gates dropout; does **not** detach Candle Vars |
| CUDA (candle) | **partial** | Feature enables candle CUDA device path only |
| Fused GPU kernels | **missing (quarantined)** | Archive only; not exported or compiled |
| QLoRA / quant backends | **missing** | Non-goal here |
| HF full parity / trainer | **missing** | Explicit non-goals |

Showcase target later: **parity+** on fair layer fixtures (see METRICS.md). Numbers are **not yet measured**.

## Features (Cargo)

| Feature | Default | Effect |
|---------|---------|--------|
| *(none)* | yes | CPU-friendly candle build; all layer math on host |
| `cuda` | no | Enables **`candle-core/cuda`** only — use `Device::cuda_if_available` |

There is **no** `cubecl` feature on this tree. Historical fused-kernel sources live in
`src/kernels/archive/` and are **not** part of the build (PR-021 quarantine).

```toml
[dependencies]
peft-rs = "1.0"

# Optional: candle CUDA device support (not peft fused kernels)
peft-rs = { version = "1.0", features = ["cuda"] }
```

## Installation

Add to your `Cargo.toml` as above. MSRV: Rust **1.92**.

## Quick Start

### LoRA Example

```rust
use peft_rs::{LoraConfig, LoraLayer};
use candle_core::{Device, Tensor, DType};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    let config = LoraConfig {
        r: 8,
        alpha: 16,
        dropout: 0.0,
        ..Default::default()
    };

    let lora = LoraLayer::new_with_zeros(768, 768, config, &device)?;

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

    let input_embeds = Tensor::zeros(&[2, 100, 768], DType::F32, &device)?;
    let output = prompt_tuning.prepend_to_input(&input_embeds)?;

    // Output: [2, 120, 768] (20 virtual tokens + 100 input tokens)
    println!("Output shape: {:?}", output.shape());

    Ok(())
}
```

## Saving and Loading Adapters

Adapters that implement `SaveLoad` can be stored with safetensors (Rust key names
such as `lora_a.weight` — **not** HF `…lora_A.default.weight` yet):

```rust
use peft_rs::{LoraLayer, save_adapter_weights, load_adapter_weights, save_adapter_config, load_adapter_config};

// Save adapter weights and config
save_adapter_weights(&lora_layer, "adapter_weights.safetensors")?;
save_adapter_config(&config, "adapter_config.json")?;

// Load adapter weights and config
let loaded_config = load_adapter_config("adapter_config.json")?;
let mut loaded_layer = LoraLayer::new_with_zeros(768, 768, loaded_config, &device)?;
load_adapter_weights(&mut loaded_layer, "adapter_weights.safetensors", &device)?;
```

## Multi-Adapter Support

Manage multiple adapters and switch between them at runtime (one active adapter):

```rust
use peft_rs::{AdapterRegistry, LoraLayer, LoraConfig};

let mut registry = AdapterRegistry::new();

let task1_adapter = LoraLayer::new_with_zeros(768, 768, config1, &device)?;
let task2_adapter = LoraLayer::new_with_zeros(768, 768, config2, &device)?;

registry.register_adapter("task1", task1_adapter)?;
registry.register_adapter("task2", task2_adapter)?;

registry.set_active_adapter("task1")?;
let output1 = registry.forward(&input, None)?;

registry.set_active_adapter("task2")?;
let output2 = registry.forward(&input, None)?;
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

This table is **not** a claim of full feature parity. Prefer the status matrix above.

| Capability | peft-rs (honest) | HuggingFace PEFT |
|------------|------------------|------------------|
| LoRA layer math | Partial (usable) | Full framework |
| Other tuners | Partial linear layers | Broad model integration |
| Base-model inject | Stub | Yes |
| HF checkpoint interop | Partial / planned | Native |
| QLoRA | No (this crate) | Yes |
| Trainer | Schedulers only | Yes |
| Fused CUDA kernels | Quarantined / none active | Ecosystem CUDA |
| No Python runtime | Yes | No |

## Metrics & roadmap

- [METRICS.md](METRICS.md) — comparison plan vs HuggingFace peft (**not yet measured**)
- [roadmap.md](roadmap.md) — remaining work; success criteria **not** “already met”
- [DECISION.md](DECISION.md) — SoT vs crates.io 1.0.3 skew heal
- [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) — historical gap notes (may lag; prefer this README + METRICS)

## Contributing

Contributions welcome. Prefer honesty over vanity claims. Run:

```bash
cargo test --lib
```

## License

MIT Licensed — see [LICENSE](LICENSE).
