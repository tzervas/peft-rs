# PEFT-RS Task Tracker

Honest implementation status for **peft-rs 1.1.0**. Prefer [README.md](../README.md)
status matrix and [METRICS.md](../METRICS.md) as the user-facing source of truth.

> **Product class:** Candle PEFT adapter **layer library** + Linear inject path —
> not full HuggingFace PEFT.

## Status overview (1.1.0)

| Feature | Status | Notes |
|---------|--------|-------|
| **LoRA** | **done** (core) | Forward/merge/save; dropout when unfrozen; rsLoRA |
| **DoRA** | **partial** | Layer math + SaveLoad; simplified without base |
| **AdaLoRA** | **partial** | SVD params + **top-k** rank mask + cubic schedule |
| **IA³ / LoHa / LoKr / OFT / BOFT / VeRA** | **partial** | Layer math; no HF key suite |
| **Prefix / Prompt** | **experimental** | Reparam MLP + prepend helpers |
| **Adapter / Mergeable / Trainable / SaveLoad** | **done** | Core traits |
| **Trainable freeze** | **partial** | Flag + dropout gate; does not detach Vars |
| **Weight I/O (native)** | **done** | safetensors helpers |
| **HF adapter_config + LoRA keys** | **done** (LoRA) | `hf` module; product interop surface |
| **`LinearWithLora` / `get_peft_model`** | **done** | Real residual forward; legacy → `get_peft_model_registry` |
| **Multi-adapter registry** | **done** | Switch + weighted residual compose |
| **Training utilities** | **done** (minimal) | `train_step_mse` / `train_step_with_loss` — not full PeftTrainer |
| **Inference utilities** | **partial** | `BatchAdapterSwitcher`, residual-gating `InferenceMode`, metrics, `merge_active`; not an eval harness |
| **Quant bridge** | **done** (traits only) | `quant` module; codecs in qlora-rs |
| **LoRA parity fixtures** | **done** | `tests/parity` allclose 1e-5 |
| **Criterion benches** | **done** (LoRA) | Numbers in METRICS.md (CPU baselines) |
| **Fused CUDA kernels** | **quarantined** | `src/kernels/archive/` — not built |
| **METRICS vs Python peft** | **partial** | Correctness done; wall-time vs peft not measured |

## Closed in 1.1.0

- PEFT-P0-07/08 — HF config + weight keys  
- PEFT-P0-09 — Real Linear inject  
- PEFT-P0-10 — Numerical parity fixtures  
- PEFT-P0-11 — `modules_to_save` policy (config-only non-goal)  
- PEFT-P0-12 — Real train step  
- PEFT-P1-01 — Weighted multi-adapter  
- PEFT-P1-02 — AdaLoRA top-k  
- PEFT-P1-03 — Prefix reparam + prompt text init  
- PEFT-P1-04 — Quant bridge traits  
- PEFT-P2-02 — Non-empty LoRA benches  

## Open work

| Gap | Topic |
|-----|--------|
| PEFT-P1-05 | Optional CubeCL kernel restore **or** leave quarantined |
| PEFT-P1-06 | Conv2d / Embedding LoRA targets |
| PEFT-P2-01 | Additional tuners (p-tuning, X-LoRA, …) after core |
| PEFT-P2-03 | Wall-time / RSS / throughput vs Python peft |
| PEFT-P2-04 | Inference eval harness / merged full-model export beyond `merge_active` |

## Non-goals

Full transformers zoo, full QLoRA codecs, full PeftTrainer/datasets, drop-in HF PEFT.
