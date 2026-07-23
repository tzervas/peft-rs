# PEFT-RS Gap Analysis

Comparison between Rust **peft-rs 1.1.0** and HuggingFace Python PEFT.

> **Honesty note (2026-07-22):** peft-rs is a **Candle adapter layer library** with a
> Linear inject path and LoRA HF interop — **not** a full PEFT framework.
> User-facing truth: [README.md](../README.md), [roadmap.md](../roadmap.md),
> [METRICS.md](../METRICS.md), [TASK_TRACKER.md](TASK_TRACKER.md).

## Product class

| Claim | Truth (1.1.0) |
|-------|----------------|
| Drop-in HF PEFT | **No** |
| Candle adapter layers | **Yes** (varying depth) |
| Linear inject + LoRA residual | **Yes** (`get_peft_model` / `LinearWithLora`) |
| HF LoRA config + weight keys | **Yes** (LoRA product surface) |
| Full Python parity (all tuners) | **No** |
| Showcase metrics vs peft wall-time | **Not measured** (correctness goldens yes) |

## Adapter surface

| Adapter | Status | Notes |
|---------|--------|-------|
| LoRA | **done** (core) | Best of suite; HF keys; inject; parity fixtures |
| DoRA | **partial** | Magnitude/direction; SaveLoad; simplified without base |
| AdaLoRA | **partial** | SVD + top-k mask + schedule; no full HF suite |
| IA³ / LoHa / LoKr / OFT / BOFT / VeRA | **partial** | Layer math only |
| Prefix / Prompt | **experimental** | Helpers; not full HF prefix-tuning stack |
| p-tuning / X-LoRA / FourierFT / … | **missing** | Out of 1.1.0 |

## Infrastructure

| Component | Status |
|-----------|--------|
| Traits (`Adapter`, `Mergeable`, `Trainable`, `SaveLoad`) | **done** |
| Native safetensors I/O | **done** |
| HF `adapter_config` + LoRA keys | **done** (LoRA) |
| `PeftLinearModel` / `get_peft_model` | **done** |
| Weighted multi-adapter | **done** |
| `train_step_mse` | **done** (minimal) |
| `quant` bridge traits | **done** (no codecs) |
| Fused CUDA kernels | **quarantined** |

## Remaining gaps (prioritized)

1. Wall-time / RSS METRICS vs Python peft
2. Embedding / Conv2d LoRA targets
3. Optional kernel restore under feature
4. Broader HF key suites for non-LoRA adapters
5. Additional tuners only after above

## References

- [HuggingFace PEFT](https://github.com/huggingface/peft)
- LoRA / DoRA / AdaLoRA / IA³ papers (see historical README links)
