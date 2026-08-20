# PEFT-RS Gap Analysis

Comparison between Rust **peft-rs 1.2.0** and HuggingFace Python PEFT.

> **Honesty note (2026-08-20):** peft-rs is a **Candle adapter layer library** with a
> Linear inject path and LoRA HF *key mapping* — **not** a full PEFT framework.
> User-facing truth: [README.md](../README.md), [roadmap.md](../roadmap.md),
> [METRICS.md](../METRICS.md), [TASK_TRACKER.md](TASK_TRACKER.md).

## Product class

| Claim | Truth (1.2.0) |
|-------|----------------|
| Drop-in HF PEFT | **No** |
| Candle adapter layers | **Yes** (varying depth) |
| Linear inject + LoRA residual | **Yes** (`get_peft_model` / `LinearWithLora`) |
| HF LoRA config + weight keys | **Yes** via `save_pretrained_hf` (LoRA product surface) |
| Hub / Python peft / mistral.rs roundtrip | **Not verified** (self-roundtrip only) |
| Full Python parity (all tuners) | **No** |
| Showcase metrics vs peft wall-time | **Not measured** (correctness goldens yes) |
| Candle | **0.11** |

## Adapter surface

| Adapter | Status | Notes |
|---------|--------|-------|
| LoRA | **done** (core) | Best of suite; HF keys; inject; parity fixtures |
| DoRA | **partial** | Magnitude/direction; SaveLoad; simplified without base; key is `magnitude` not HF `lora_magnitude_vector` |
| AdaLoRA | **partial** | SVD + top-k mask + schedule; no full HF suite |
| IA³ / LoHa / LoKr / OFT / BOFT / VeRA | **partial** | Layer math only |
| Prefix / Prompt | **experimental** | Helpers; not full HF prefix-tuning stack |
| p-tuning / X-LoRA / FourierFT / … | **missing** | Out of 1.2.0 |

## Infrastructure

| Component | Status |
|-----------|--------|
| Traits (`Adapter`, `Mergeable`, `Trainable`, `SaveLoad`) | **done** |
| Native safetensors I/O | **done** |
| HF `adapter_config` + LoRA keys | **done** (LoRA) — **`save_pretrained` is not Hub-safe** |
| `PeftLinearModel` / `get_peft_model` | **done** (Linear only) |
| Weighted multi-adapter | **done** |
| `train_step_mse` | **done** (minimal) |
| `quant` bridge traits | **done** (no codecs) |
| Fused CUDA kernels | **quarantined** |

## Remaining gaps (prioritized)

1. Cross-runtime golden: Python `peft` save → peft-rs load, and peft-rs `save_pretrained_hf` → `PeftModel.from_pretrained` / mistral.rs `--lora`
2. Pack all target modules (full Llama prefix) into one `adapter_model.safetensors`
3. Wall-time / RSS METRICS vs Python peft
4. Embedding / Conv2d LoRA targets
5. Optional kernel restore under feature
6. Broader HF key suites for non-LoRA adapters

## References

- [HuggingFace PEFT](https://github.com/huggingface/peft)
- LoRA / DoRA / AdaLoRA / IA³ papers (see historical README links)
