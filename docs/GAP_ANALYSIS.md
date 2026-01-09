# PEFT-RS Gap Analysis

This document analyzes the gaps between the Rust `peft-rs` implementation and the official HuggingFace Python PEFT library.

## Current State of peft-rs

### Implemented Adapters
| Adapter | Status | Notes |
|---------|--------|-------|
| LoRA | ✅ Basic | Core functionality implemented |
| Prefix Tuning | ✅ Basic | Core functionality implemented |
| Prompt Tuning | ✅ Basic | Core functionality implemented |

### Core Infrastructure
| Component | Status | Notes |
|-----------|--------|-------|
| `Adapter` trait | ✅ | Forward pass and parameter counting |
| `Mergeable` trait | ✅ | Weight merging/unmerging |
| `Trainable` trait | ✅ | Parameter registration, freeze/unfreeze |
| `AdapterConfig` trait | ✅ | Configuration validation |
| Error handling | ✅ | PeftError enum with variants |

## Gaps Identified

### Priority 1: Missing Core Adapters

#### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
**Python PEFT Reference:** `src/peft/tuners/ia3/`

IA³ is a highly parameter-efficient method that learns rescaling vectors for keys, values, and feedforward layers.

**Key features to implement:**
- `IA3Config` with fields:
  - `target_modules`: Modules to apply IA³ to
  - `feedforward_modules`: Modules treated as feedforward (multiplied on input)
  - `fan_in_fan_out`: Weight storage convention
  - `init_ia3_weights`: Whether to initialize to ones
- `IA3Layer` with:
  - Learnable scaling vectors `ia3_l` (initialized to ones)
  - Forward: `output = base_output * ia3_scaling` (non-feedforward) or `output = base(input * ia3_scaling)` (feedforward)
  - Merge/unmerge: multiply/divide base weights by scaling

**Parameters:** Only `in_features` or `out_features` depending on feedforward flag.

#### AdaLoRA (Adaptive Low-Rank Adaptation)
**Python PEFT Reference:** `src/peft/tuners/adalora/`

AdaLoRA dynamically allocates rank budget during training using SVD-based importance scores.

**Key features to implement:**
- `AdaLoraConfig` extending `LoraConfig` with:
  - `target_r`: Target average rank
  - `init_r`: Initial rank (higher than target)
  - `tinit`, `tfinal`, `total_step`: Training schedule phases
  - `deltaT`: Interval between budget allocations
  - `beta1`, `beta2`: EMA hyperparameters for sensitivity smoothing
  - `orth_reg_weight`: Orthogonal regularization coefficient
- `AdaLoraLayer` with SVD parameterization:
  - `lora_E`: Singular values (diagonal)
  - `lora_A`: Right singular vectors
  - `lora_B`: Left singular vectors
  - `ranknum`: Current rank allocation
- Rank allocation based on importance scores

### Priority 2: LoRA Variants

#### DoRA (Weight-Decomposed Low-Rank Adaptation)
**Python PEFT Reference:** `src/peft/tuners/lora/dora.py`

DoRA decomposes weight updates into magnitude and direction components.

**Key features to implement:**
- Add `use_dora: bool` to `LoraConfig`
- `DoraLayer` wrapper that:
  - Maintains `magnitude` vector
  - Applies weight decomposition: `W = m * (W0 + ΔW) / ||W0 + ΔW||`
  - Handles column-wise normalization

#### QLoRA Support
**Python PEFT Reference:** `src/peft/tuners/lora/bnb.py`

Quantized LoRA for 4-bit/8-bit training.

**Key features to implement:**
- Integration with quantization backend
- `Linear4bit` and `Linear8bitLt` layer support
- Dequantization during forward pass

### Priority 3: Additional Adapter Types

#### LoHa (Low-Rank Hadamard Product)
**Python PEFT Reference:** `src/peft/tuners/loha/`

Uses Hadamard product of two low-rank matrices.

**Key features:**
- `ΔW = (A1 ⊗ B1) ⊙ (A2 ⊗ B2)` where ⊙ is Hadamard product
- More expressive than standard LoRA

#### LoKr (Low-Rank Kronecker Product)
**Python PEFT Reference:** `src/peft/tuners/lokr/`

Uses Kronecker product for weight decomposition.

#### OFT (Orthogonal Fine-Tuning)
**Python PEFT Reference:** `src/peft/tuners/oft/`

Applies orthogonal transformations to preserve pretrained knowledge.

#### BOFT (Butterfly Orthogonal Fine-Tuning)
**Python PEFT Reference:** `src/peft/tuners/boft/`

Uses butterfly factorization for efficient orthogonal transformations.

#### VeRA (Vector-based Random Matrix Adaptation)
**Python PEFT Reference:** `src/peft/tuners/vera/`

Uses frozen random matrices with trainable scaling vectors.

### Priority 4: Infrastructure Improvements

#### Weight Loading/Saving
**Gaps:**
- No safetensors integration (Python PEFT uses safetensors extensively)
- No adapter weight file format support
- No state dict save/load utilities

**To implement:**
- `save_pretrained()` / `load_pretrained()` methods
- Safetensors format support
- Adapter config serialization

#### Model Integration
**Gaps:**
- No `get_peft_model()` equivalent
- No automatic module injection
- No module matching by regex

**To implement:**
- `inject_adapter()` function to wrap model layers
- Module name matching with glob/regex patterns
- `PeftModel` wrapper struct

#### Multi-Adapter Support
**Gaps:**
- No multiple adapter management
- No adapter switching
- No adapter composition

**To implement:**
- Adapter registry
- `set_adapter()` / `get_adapter()` methods
- Adapter merging utilities

### Priority 5: Advanced Features

#### Quantization Support
- 4-bit quantization (bitsandbytes equivalent)
- 8-bit quantization
- GPTQ/AWQ integration

#### Training Utilities
- Gradient checkpointing integration
- Mixed precision training support
- Learning rate scheduling for adapters

#### Evaluation/Inference
- Batch adapter switching
- Merged inference mode
- Export to standard formats

## Recommended Implementation Order

### Phase 1: Core Adapters (High Impact)
1. **IA³** - Simple, highly efficient, good for benchmarking
2. **AdaLoRA** - Advanced LoRA variant with practical benefits

### Phase 2: LoRA Enhancements
3. **DoRA** - Popular enhancement to LoRA
4. **Quantization infrastructure** - Essential for large models

### Phase 3: Model Integration
5. **Weight loading/saving** - Required for practical use
6. **Model injection** - User-friendly API

### Phase 4: Additional Methods
7. **LoHa/LoKr** - LyCORIS family adapters
8. **OFT/BOFT** - Orthogonal methods
9. **VeRA** - Ultra-efficient adaptation

## API Design Recommendations

### Consistent Trait Pattern
All adapters should implement:
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

pub trait Trainable: Adapter {
    fn register_parameters(&self, var_map: &mut VarMap, prefix: &str) -> Result<()>;
    fn freeze(&mut self);
    fn unfreeze(&mut self);
    fn is_frozen(&self) -> bool;
}
```

### Configuration Pattern
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub target_modules: Vec<String>,
    pub modules_to_save: Vec<String>,
    // ... adapter-specific fields
}
```

### Model Integration Pattern
```rust
pub fn get_peft_model<M: Model>(model: M, config: impl AdapterConfig) -> Result<PeftModel<M>> {
    // Inject adapters into target modules
}
```

## References

- [HuggingFace PEFT Repository](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [IA³ Paper](https://arxiv.org/abs/2205.05638)
- [AdaLoRA Paper](https://arxiv.org/abs/2303.10512)
- [DoRA Paper](https://arxiv.org/abs/2402.09353)
- [Prefix Tuning Paper](https://arxiv.org/abs/2101.00190)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
