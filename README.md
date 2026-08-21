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
> [candle](https://github.com/huggingface/candle) (forward / merge / safetensors)
> plus a **Linear inject path** (`LinearWithLora` / `get_peft_model`).
> This is **not** a drop-in HuggingFace PEFT framework and does **not** claim full
> Python parity for every tuner.
>
> **Docs:** [METRICS.md](METRICS.md) · [roadmap.md](roadmap.md) · [CHANGELOG.md](CHANGELOG.md) ·
> [docs/TASK_TRACKER.md](docs/TASK_TRACKER.md) · [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) ·
> [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) (no circular deps)
## What this crate is

- Standalone adapter **layers** (LoRA, DoRA, AdaLoRA, IA³, LoHa, LoKr, OFT, BOFT, VeRA, prefix/prompt tuning)
- Common traits: `Adapter`, `Mergeable`, `Trainable`, `SaveLoad`
- **HF LoRA interop** (`hf` module): `adapter_config.json` core fields + `lora_A`/`lora_B` key mapping
- **`LinearWithLora` / `PeftLinearModel`**: real base Linear + LoRA residual forward for named modules
- Multi-adapter **registry** with single-active switch **and** weighted residual composition
- **LoRA parity fixtures** under `tests/parity/` (forward/merge allclose)

## Non-goals

| Non-goal | Notes |
|----------|--------|
| Full transformers model zoo / AutoModel | Caller supplies named `Linear`s |
| Automatic `modules_to_save` training | Field preserved on `HfLoraConfig` only (see below) |
| Full QLoRA codecs (NF4/FP4) / bnb / GPTQ / AWQ | Out of this crate; `quant` bridge traits only — see qlora-rs |
| Full PeftTrainer / dataset loop | Thin `train_step_mse` helper; full loops are caller's (see example) |
| Fused CUDA kernels (CubeCL) | **Quarantined** under `src/kernels/archive/` (PR-021) |
| Full multi-tuner HF parity | LoRA is the product interop surface |

## Status matrix (honest)

Legend: **done** = usable · **partial** = real code but incomplete vs HF · **missing** = not implemented · **stub** = flag/shell

| Feature | Status | Notes |
|---------|--------|-------|
| LoRA linear layer | **done** (core) | Forward/merge/save; bias LoRA still missing |
| rsLoRA scaling | **done** | `use_rslora` → `α/√r` |
| LoRA dropout | **done** | Applied when unfrozen and `dropout > 0` |
| HF `adapter_config.json` | **done** (LoRA core) | `peft_type`, `r`, `lora_alpha`, `target_modules`, optional base/task |
| HF LoRA weight keys | **done** | `lora_A/B.default.weight` + module / `base_model.model` prefixes; native keys still default on save |
| `LinearWithLora` inject | **done** | Base Linear + LoRA residual; base frozen if only adapter Vars optimized |
| `get_peft_model` | **done** (Linear path) | Builds wrappers for pattern-matched modules; legacy registry → `get_peft_model_registry` |
| LoRA parity fixtures | **done** | `tests/parity` allclose atol/rtol `1e-5` |
| `modules_to_save` | **non-goal / config-only** | Serialized on HF config; not auto-trained |
| DoRA | **partial** | Magnitude/direction; SaveLoad supported |
| LoftQ init | **stub / simplified** | Dual-Gaussian; not full SVD+quant LoftQ |
| AdaLoRA | **partial** | SVD param + **top-k** rank mask + cubic budget schedule; no HF key suite |
| IA³ / LoHa / LoKr / OFT / BOFT / VeRA | **partial** | Layer math; no HF key suite |
| Prefix / Prompt tuning | **experimental** | Reparam MLP + `concat_to_kv`; prompt prepend + simplified text init |
| Multi-adapter registry | **done** (core) | Switch active + weighted residual compose (`AdapterWeight`) |
| Train step helper | **done** (minimal) | `train_step_mse` / `train_step_with_loss` on inject path |
| Quant bridge traits | **done** (bridge) | `QuantizedBaseLinear` / compose helper; codecs in qlora-rs |
| Trainable freeze | **partial** | Layer flag; gates dropout; does **not** detach Vars |
| Criterion benches | **done** (LoRA) | Real LoRA forward/merge benches; numbers in METRICS.md (CPU) |
| CUDA (candle) | **partial** | Feature enables candle CUDA device path only |
| Fused GPU kernels | **missing (quarantined)** | Archive only |
| Full QLoRA / HF trainer | **missing** | Codecs + full trainer remain non-goals |

Showcase: LoRA **correctness** goldens are green; CPU wall-time baselines in METRICS.md (not yet vs HF peft).

## Features (Cargo)

| Feature | Default | Effect |
|---------|---------|--------|
| *(none)* | yes | CPU-friendly candle build; all layer math on host |
| `cuda` | no | Enables **`candle-core/cuda`** only — use `Device::cuda_if_available` |

There is **no** `cubecl` feature on this tree. Historical fused-kernel sources live in
`src/kernels/archive/` and are **not** part of the build (PR-021 quarantine).

```toml
[dependencies]
peft-rs = "1.1.0"

# Optional: candle CUDA device support (not peft fused kernels)
peft-rs = { version = "1.1.0", features = ["cuda"] }
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

### Native keys (default `SaveLoad`)

| Tensor | peft-rs native key |
|--------|-------------------|
| A | `lora_a.weight` |
| B | `lora_b.weight` |

```rust
use peft_rs::{LoraLayer, save_adapter_weights, load_adapter_weights, save_adapter_config, load_adapter_config};

save_adapter_weights(&lora_layer, "adapter_weights.safetensors")?;
save_adapter_config(&config, "adapter_config.json")?;
```

### HuggingFace PEFT LoRA interop (`hf` module)

**Config** (`adapter_config.json`) via `HfLoraConfig`:

| HF field | Maps to |
|----------|---------|
| `peft_type` | `"LORA"` |
| `r` | `LoraConfig::r` |
| `lora_alpha` | `LoraConfig::alpha` |
| `target_modules` | `LoraConfig::target_modules` |
| `base_model_name_or_path` | optional Hub/path metadata |
| `task_type` | optional (e.g. `CAUSAL_LM`) |
| `lora_dropout` | `LoraConfig::dropout` |
| `modules_to_save` | **config-only** (not auto-trained) |

**Weight keys** (load accepts all; HF save chooses style):

| Style | Example |
|-------|---------|
| Native | `lora_a.weight` |
| HF bare | `lora_A.default.weight` |
| HF module | `layers.0.q_proj.lora_A.default.weight` |
| HF full | `base_model.model.… .lora_A.default.weight` |

```rust
use peft_rs::{
    save_pretrained_hf, load_pretrained_hf, HfLoraConfig, LoraKeyStyle, LoraLayer, LoraConfig,
};

let hf_cfg = HfLoraConfig::from_lora_config(&config, Some("org/model".into()), Some("CAUSAL_LM".into()));
save_pretrained_hf(&layer, &hf_cfg, "out_dir", &LoraKeyStyle::hf_module("model.layers.0.q_proj"))?;

let mut layer2 = LoraLayer::new_with_zeros(768, 768, config, &device)?;
let loaded = load_pretrained_hf(&mut layer2, "out_dir", &device, Some("model.layers.0.q_proj"))?;
```

### Linear inject + train (product path)

```rust
use peft_rs::{get_peft_model, LoraConfig};
// base_modules: Vec<(String, candle_nn::Linear)>
// adapter_vb: VarBuilder over a dedicated VarMap (base weights NOT in that map → frozen)
let model = get_peft_model(base_modules, "mlp.*", config, "default", adapter_vb)?;
let y = model.forward(&x)?;
// AdamW on adapter_vm.all_vars() — see examples/lora_inject_train.rs
```

Legacy name-only registry (no base Linear): `get_peft_model_registry`.

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

// Weighted residual composition
use peft_rs::AdapterWeight;
registry.set_weighted_adapters([
    AdapterWeight::new("task1", 0.7),
    AdapterWeight::new("task2", 0.3),
])?;
let mixed = registry.forward(&input, None)?;
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
