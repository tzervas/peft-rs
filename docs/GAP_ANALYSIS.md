# PEFT-RS Gap Analysis

Comparison between the Rust `peft-rs` crate and HuggingFace Python PEFT.

> **Honesty note (2026-07-22):** peft-rs is a **layer-math library**, not a full PEFT
> framework. See [README.md](../README.md), [roadmap.md](../roadmap.md), and
> [METRICS.md](../METRICS.md). This document is retained for historical detail;
> status rows below are corrected to match code depth.

## Product class

| Claim | Truth |
|-------|--------|
| Drop-in HF PEFT | **No** |
| Candle adapter layers | **Yes** (varying completeness) |
| Full Python parity | **No** |
| Showcase metrics measured | **Not yet** (scaffold only) |

## Adapter surface

| Adapter | Status | Notes |
|---------|--------|-------|
| LoRA | **partial** | Best of suite; dropout when unfrozen; simplified loftq init |
| DoRA | **partial** | SaveLoad present (1.0.4); simplified path without base |
| AdaLoRA | **partial** | Not full top-k budget trainer |
| IA³ | **partial** | Layer only |
| LoHa | **partial** | Linear; some config flags unused |
| LoKr | **partial** | CPU Kronecker path simplified |
| OFT | **partial** | Square features; approx/exact Cayley options |
| BOFT | **partial** | boft_dropout incomplete |
| VeRA | **partial** | Core path |
| Prefix Tuning | **thin** | No reparam / attention inject |
| Prompt Tuning | **thin** | No text init / model hook |

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| `Adapter` trait | done | Forward + param count |
| `Mergeable` trait | done | Per-layer merge/unmerge |
| `Trainable` trait | partial | freeze is a flag (see traits docs) |
| `SaveLoad` trait | partial | Most adapters; Dora included as of 1.0.4 |
| Weight I/O | partial | safetensors; not HF schema/keys |
| `PeftModel` | stub | Name list, not base wrap |
| Registry | partial | Single active adapter |
| Training | stub | Schedules only |
| Kernels | quarantined | archive only; no fused CUDA in build |
| `cuda` feature | partial | candle CUDA device only |

## Major gaps vs Python PEFT

### Framework (P0/P1)

1. **Base-model injection** — `get_peft_model` does not replace modules in a real model.
2. **HF checkpoint interop** — config schema and weight key names differ.
3. **modules_to_save / target auto-detect** — caller supplies names; no transformers introspection.
4. **QLoRA / quant backends** — out of scope for this crate.
5. **Trainer loop** — not implemented.
6. **Numerical parity tests** — missing (METRICS scaffold only).
7. **Multi-adapter composition** — no weighted combine.
8. **Fused GPU kernels** — quarantined; candle CUDA optional only.

### Missing tuners (P2)

p-tuning, adaption_prompt, LN-tuning, FourierFT, HRA, Bone, Poly, C3A, CAR, Shira,
VB-LoRA, RandLoRA, X-LoRA, mixed multi-type models, Conv2d/Embedding LoRA, etc.

## Priority guidance

Do not expand tuner surface until P0/P1 framework honesty items land (HF I/O,
model inject, parity fixtures). Prefer implementing real behavior or documenting
non-goals over green checkmarks.
