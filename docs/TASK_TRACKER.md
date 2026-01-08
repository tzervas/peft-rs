# PEFT-RS Task Tracker

This document tracks the implementation progress of features in peft-rs based on the [Gap Analysis](GAP_ANALYSIS.md).

## Implementation Status Overview

### ✅ Completed Features

| Feature | Status | Tests | Notes |
|---------|--------|-------|-------|
| **LoRA** | ✅ Complete | 5 tests | Core implementation with VarBuilder support |
| **DoRA** | ✅ Complete | 5 tests | Magnitude/direction decomposition |
| **AdaLoRA** | ✅ Complete | 7 tests | SVD-based rank allocation |
| **IA³** | ✅ Complete | 8 tests | Learned rescaling vectors |
| **Prefix Tuning** | ✅ Complete | 2 tests | Trainable prefix vectors |
| **Prompt Tuning** | ✅ Complete | 3 tests | Soft prompt embeddings |
| **Adapter trait** | ✅ Complete | - | Forward pass, parameter counting |
| **Mergeable trait** | ✅ Complete | - | Weight merging/unmerging |
| **Trainable trait** | ✅ Complete | - | Parameter registration, freeze/unfreeze |
| **AdapterConfig trait** | ✅ Complete | - | Configuration validation |
| **Error handling** | ✅ Complete | - | PeftError enum |

### 🚧 In Progress / Pending Features

---

## Phase 1: Additional Adapter Types (Priority 3)

### 1.1 LoHa (Low-Rank Hadamard Product)
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 2-3 days

**Description:**  
Uses Hadamard product of two low-rank matrices for more expressive weight updates.

**Key Implementation Details:**
- `LoHaConfig` with fields:
  - `r1`, `r2`: Ranks for the two low-rank decompositions
  - `alpha`: Scaling factor
  - `target_modules`: Target modules
- `LoHaLayer` with:
  - Four matrices: `A1`, `B1`, `A2`, `B2`
  - Forward: `ΔW = (A1 ⊗ B1) ⊙ (A2 ⊗ B2)` where ⊙ is Hadamard product
  - Merge/unmerge support

**Tasks:**
- [ ] Create `src/adapters/loha.rs`
- [ ] Implement `LoHaConfig` with validation
- [ ] Implement `LoHaLayer` struct
- [ ] Implement `Adapter` trait
- [ ] Implement `Mergeable` trait
- [ ] Implement `Trainable` trait
- [ ] Add unit tests
- [ ] Export from `src/adapters/mod.rs`
- [ ] Add to `src/lib.rs` public exports
- [ ] Update README.md

---

### 1.2 LoKr (Low-Rank Kronecker Product)
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 2-3 days

**Description:**  
Uses Kronecker product for weight decomposition.

**Key Implementation Details:**
- `LoKrConfig` with fields:
  - `r`: Rank
  - `alpha`: Scaling factor
  - `factor`: Kronecker factorization parameter
  - `target_modules`: Target modules
- `LoKrLayer` with:
  - Kronecker factored matrices
  - Forward: `ΔW = A ⊗ B` where ⊗ is Kronecker product
  - Efficient computation using decomposition

**Tasks:**
- [ ] Create `src/adapters/lokr.rs`
- [ ] Implement `LoKrConfig` with validation
- [ ] Implement `LoKrLayer` struct
- [ ] Implement `Adapter` trait
- [ ] Implement `Mergeable` trait
- [ ] Implement `Trainable` trait
- [ ] Add unit tests
- [ ] Export from `src/adapters/mod.rs`
- [ ] Add to `src/lib.rs` public exports
- [ ] Update README.md

---

### 1.3 OFT (Orthogonal Fine-Tuning)
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 2-3 days

**Description:**  
Applies orthogonal transformations to preserve pretrained knowledge.

**Key Implementation Details:**
- `OftConfig` with fields:
  - `r`: Number of blocks
  - `coft`: Whether to use constrained OFT
  - `eps`: Numerical stability epsilon
  - `target_modules`: Target modules
- `OftLayer` with:
  - Block-diagonal orthogonal matrix parameterization
  - Forward: `W' = W @ R` where R is orthogonal
  - Cayley parameterization for guaranteed orthogonality

**Tasks:**
- [ ] Create `src/adapters/oft.rs`
- [ ] Implement `OftConfig` with validation
- [ ] Implement `OftLayer` struct
- [ ] Implement Cayley transform helper
- [ ] Implement `Adapter` trait
- [ ] Implement `Mergeable` trait
- [ ] Implement `Trainable` trait
- [ ] Add unit tests
- [ ] Export from `src/adapters/mod.rs`
- [ ] Add to `src/lib.rs` public exports
- [ ] Update README.md

---

### 1.4 BOFT (Butterfly Orthogonal Fine-Tuning)
**Status:** ❌ Not Started  
**Priority:** Low  
**Estimated Effort:** 3-4 days

**Description:**  
Uses butterfly factorization for efficient orthogonal transformations.

**Key Implementation Details:**
- `BoftConfig` with fields:
  - `boft_block_size`: Size of butterfly blocks
  - `boft_block_num`: Number of butterfly blocks
  - `boft_n_butterfly_factor`: Number of butterfly factors
  - `target_modules`: Target modules
- `BoftLayer` with:
  - Butterfly matrix parameterization
  - Efficient O(n log n) multiplication

**Tasks:**
- [ ] Create `src/adapters/boft.rs`
- [ ] Implement `BoftConfig` with validation
- [ ] Implement `BoftLayer` struct
- [ ] Implement butterfly matrix operations
- [ ] Implement `Adapter` trait
- [ ] Implement `Mergeable` trait
- [ ] Implement `Trainable` trait
- [ ] Add unit tests
- [ ] Export from `src/adapters/mod.rs`
- [ ] Add to `src/lib.rs` public exports
- [ ] Update README.md

---

### 1.5 VeRA (Vector-based Random Matrix Adaptation)
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 2-3 days

**Description:**  
Uses frozen random matrices with trainable scaling vectors for ultra-efficient adaptation.

**Key Implementation Details:**
- `VeraConfig` with fields:
  - `r`: Rank
  - `d_initial`: Initial value for scaling vector d
  - `target_modules`: Target modules
  - `projection_prng_key`: Seed for random projection
- `VeraLayer` with:
  - Frozen random matrices `A`, `B`
  - Trainable scaling vectors `d`, `b`
  - Forward: `ΔW = B @ diag(d) @ A` (B, A frozen)

**Tasks:**
- [ ] Create `src/adapters/vera.rs`
- [ ] Implement `VeraConfig` with validation
- [ ] Implement `VeraLayer` struct
- [ ] Implement frozen random matrix initialization
- [ ] Implement `Adapter` trait
- [ ] Implement `Mergeable` trait
- [ ] Implement `Trainable` trait
- [ ] Add unit tests
- [ ] Export from `src/adapters/mod.rs`
- [ ] Add to `src/lib.rs` public exports
- [ ] Update README.md

---

## Phase 2: Infrastructure Improvements (Priority 4)

### 2.1 Weight Loading/Saving
**Status:** ❌ Not Started  
**Priority:** High  
**Estimated Effort:** 3-4 days

**Description:**  
Safetensors integration for saving and loading adapter weights.

**Key Implementation Details:**
- `save_pretrained()` method for adapters
- `load_pretrained()` method for adapters
- Safetensors file format support
- Config JSON serialization

**Tasks:**
- [ ] Add `safetensors` dependency to Cargo.toml
- [ ] Create `src/io.rs` or `src/persistence.rs` module
- [ ] Implement `save_adapter_weights()` function
- [ ] Implement `load_adapter_weights()` function
- [ ] Implement `save_adapter_config()` function
- [ ] Implement `load_adapter_config()` function
- [ ] Add `SaveLoad` trait for adapters
- [ ] Add unit tests
- [ ] Add integration tests with actual files
- [ ] Update documentation

---

### 2.2 Model Integration
**Status:** ❌ Not Started  
**Priority:** High  
**Estimated Effort:** 4-5 days

**Description:**  
Automatic adapter injection into models.

**Key Implementation Details:**
- `get_peft_model()` function
- `inject_adapter()` for wrapping layers
- Module name matching with glob/regex
- `PeftModel` wrapper struct

**Tasks:**
- [ ] Create `src/model.rs` module
- [ ] Define `PeftModel` wrapper struct
- [ ] Implement module name pattern matching
- [ ] Implement `inject_adapter()` function
- [ ] Implement `get_peft_model()` function
- [ ] Add adapter management methods
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Update documentation

---

### 2.3 Multi-Adapter Support
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 3-4 days

**Description:**  
Support for multiple adapters with switching and composition.

**Key Implementation Details:**
- Adapter registry
- `set_adapter()` / `get_adapter()` methods
- Adapter merging utilities
- Named adapter management

**Tasks:**
- [ ] Create `src/registry.rs` module
- [ ] Implement `AdapterRegistry` struct
- [ ] Implement `register_adapter()` method
- [ ] Implement `set_active_adapter()` method
- [ ] Implement `get_adapter()` method
- [ ] Implement adapter merging utilities
- [ ] Add unit tests
- [ ] Update documentation

---

## Phase 3: Advanced Features (Priority 5)

### 3.1 Quantization Support
**Status:** ❌ Not Started  
**Priority:** Low  
**Estimated Effort:** 5-7 days

**Description:**  
4-bit and 8-bit quantization support for QLoRA.

**Tasks:**
- [ ] Research quantization options in Rust (candle quantization)
- [ ] Create `src/quantization.rs` module
- [ ] Implement 4-bit linear layer wrapper
- [ ] Implement 8-bit linear layer wrapper
- [ ] Add `QuantizedLoraLayer` implementation
- [ ] Add unit tests
- [ ] Update documentation

---

### 3.2 Training Utilities
**Status:** ❌ Not Started  
**Priority:** Low  
**Estimated Effort:** 3-4 days

**Tasks:**
- [ ] Gradient checkpointing helpers
- [ ] Mixed precision support utilities
- [ ] Learning rate scheduling for adapters

---

### 3.3 Evaluation/Inference Utilities
**Status:** ❌ Not Started  
**Priority:** Low  
**Estimated Effort:** 2-3 days

**Tasks:**
- [ ] Batch adapter switching
- [ ] Merged inference mode
- [ ] Export utilities

---

## Implementation Plan

### Recommended Order of Implementation

1. **Phase 1a - LyCORIS Family (LoHa, LoKr)**
   - Complete the LyCORIS family adapters
   - These are commonly requested and well-documented

2. **Phase 1b - Orthogonal Methods (OFT, BOFT)**
   - Implement orthogonal fine-tuning methods
   - More complex but valuable for specific use cases

3. **Phase 1c - VeRA**
   - Implement ultra-efficient VeRA adapter
   - Very parameter efficient

4. **Phase 2a - Weight Loading/Saving**
   - Essential for practical usage
   - Enables model sharing and persistence

5. **Phase 2b - Model Integration**
   - Creates user-friendly API
   - Reduces boilerplate for users

6. **Phase 2c - Multi-Adapter Support**
   - Enables advanced use cases
   - Builds on registry pattern

7. **Phase 3 - Advanced Features**
   - Quantization, training utilities
   - Lower priority, implement as needed

---

## Testing Strategy

Each new adapter should include:
1. **Configuration tests** - Validate config defaults and validation
2. **Creation tests** - Test layer instantiation
3. **Forward pass tests** - Verify output shapes
4. **Parameter count tests** - Ensure correct counting
5. **Merge/unmerge tests** - For Mergeable adapters
6. **Edge case tests** - Boundary conditions

---

## References

- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) - LoHa, LoKr implementations
- [OFT Paper](https://arxiv.org/abs/2306.07280)
- [BOFT Paper](https://arxiv.org/abs/2311.06243)
- [VeRA Paper](https://arxiv.org/abs/2310.11454)

---

*Last Updated: 2025-01-08*
